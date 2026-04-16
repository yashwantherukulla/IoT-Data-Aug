"""Tests for model components."""

import pathlib
import shutil

import pytest
import torch
from omegaconf import OmegaConf

from cgdap.augmentation.engine import augment_domain_instruction
from cgdap.data.raw_loader import run_cleaning_pipeline
from cgdap.models.condition import CrossAttentionBlock, CrossAttentionConditionEmbedder
from cgdap.models.ddpm import DDPMSchedule
from cgdap.models.unet import ConditionalUNet


def test_cross_attn_block():
    blk = CrossAttentionBlock(d_model=64, n_heads=4)
    x = torch.rand(2, 16, 64)       # [B, H*W, d_model]
    ctx = torch.rand(2, 6, 64)      # [B, n_tokens, d_model]
    out = blk(x, ctx)
    assert out.shape == x.shape


def test_condition_embedder():
    emb = CrossAttentionConditionEmbedder(n_metrics=5, n_classes=5, d_model=64, n_heads=4, n_cond_tokens=6)
    metrics = torch.rand(2, 5)
    labels = torch.zeros(2, 5)
    labels[:, 0] = 1.0
    out = emb(metrics, labels)
    assert out.shape == (2, 6, 64)


def test_unet_shape_preservation():
    net = ConditionalUNet(in_channels=3, base_channels=16, channel_multipliers=[1, 2], n_res_blocks=1, temb_dim=64, d_model=32, n_heads=2, cross_attn_depths=[0])
    x = torch.rand(2, 3, 32, 64)
    t = torch.randint(0, 1000, (2,))
    cond = torch.rand(2, 6, 32)
    out = net(x, t, cond)
    assert out.shape == x.shape, f"U-Net shape mismatch: {out.shape} vs {x.shape}"


def test_ddpm_q_sample():
    sched = DDPMSchedule(train_timesteps=100)
    x0 = torch.rand(2, 3, 32, 64)
    t = torch.randint(0, 100, (2,))
    x_t, noise = sched.q_sample(x0, t)
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape


def test_ddpm_predict_x0():
    sched = DDPMSchedule(train_timesteps=100)
    x0 = torch.rand(2, 3, 32, 64)
    t = torch.randint(0, 100, (2,))
    x_t, noise = sched.q_sample(x0, t)
    x0_hat = sched.predict_x0(x_t, t, noise)
    assert x0_hat.shape == x0.shape


def test_cgdap_forward_backward():
    from cgdap.models.cgdap import MultimodalCGDAP

    mods = ["acc", "gyr"]
    model = MultimodalCGDAP(
        modalities=mods,
        denoiser_cls=ConditionalUNet,
        schedule_cls=DDPMSchedule,
        embedder_cls=CrossAttentionConditionEmbedder,
        denoiser_kwargs=dict(in_channels=3, base_channels=16, channel_multipliers=[1, 2], n_res_blocks=1, temb_dim=64, d_model=32, n_heads=2, cross_attn_depths=[0]),
        schedule_kwargs=dict(train_timesteps=100, num_train_steps=2, num_infer_steps=5),
        embedder_kwargs=dict(n_metrics=5, n_classes=5, d_model=32, n_heads=2, n_cond_tokens=6),
        n_metrics=5,
    )

    B, C, F, T = 2, 3, 32, 64
    batch = {
        "label": torch.randint(0, 5, (B,)),
        "acc": {"spectrogram": torch.rand(B, C, F, T), "metrics": torch.rand(B, 5)},
        "gyr": {"spectrogram": torch.rand(B, C, F, T), "metrics": torch.rand(B, 5)},
    }
    loss_dict = model(batch, n_classes=5)
    assert "L_total" in loss_dict
    assert "L_G_acc" in loss_dict
    assert "L_G_gyr" in loss_dict
    assert "L_metric_acc_temporal_range" in loss_dict
    assert "L_metric_gyr_entropy" in loss_dict
    assert "metric_weight_temporal_range" in loss_dict
    assert not torch.isnan(loss_dict["L_total"])
    loss_dict["L_total"].backward()


def test_metric_weights_do_not_update_in_eval():
    from cgdap.models.cgdap import MultimodalCGDAP

    mods = ["acc", "gyr"]
    model = MultimodalCGDAP(
        modalities=mods,
        denoiser_cls=ConditionalUNet,
        schedule_cls=DDPMSchedule,
        embedder_cls=CrossAttentionConditionEmbedder,
        denoiser_kwargs=dict(in_channels=3, base_channels=16, channel_multipliers=[1, 2], n_res_blocks=1, temb_dim=64, d_model=32, n_heads=2, cross_attn_depths=[0]),
        schedule_kwargs=dict(train_timesteps=100, num_train_steps=2, num_infer_steps=5),
        embedder_kwargs=dict(n_metrics=5, n_classes=5, d_model=32, n_heads=2, n_cond_tokens=6),
        adaptive_start_epoch=0,
        n_metrics=5,
    )
    model.current_epoch = 3
    model.eval()

    B, C, F, T = 2, 3, 32, 64
    batch = {
        "label": torch.randint(0, 5, (B,)),
        "acc": {"spectrogram": torch.rand(B, C, F, T), "metrics": torch.rand(B, 5)},
        "gyr": {"spectrogram": torch.rand(B, C, F, T), "metrics": torch.rand(B, 5)},
    }
    original_weights = model.metric_weights.clone()
    _ = model(batch, n_classes=5)
    assert torch.equal(model.metric_weights, original_weights)


def test_cgdap_sample_offsets_seeds_per_modality(monkeypatch):
    from cgdap.models.cgdap import MultimodalCGDAP

    mods = ["acc", "gyr"]
    model = MultimodalCGDAP(
        modalities=mods,
        denoiser_cls=ConditionalUNet,
        schedule_cls=DDPMSchedule,
        embedder_cls=CrossAttentionConditionEmbedder,
        denoiser_kwargs=dict(in_channels=3, base_channels=16, channel_multipliers=[1, 2], n_res_blocks=1, temb_dim=64, d_model=32, n_heads=2, cross_attn_depths=[0]),
        schedule_kwargs=dict(train_timesteps=100, num_train_steps=2, num_infer_steps=5),
        embedder_kwargs=dict(n_metrics=5, n_classes=5, d_model=32, n_heads=2, n_cond_tokens=6),
        n_metrics=5,
    )

    seen_seeds = []

    def fake_sample_loop(*, seed=None, shape, device, **kwargs):
        seen_seeds.append(seed)
        return torch.zeros(shape, device=device)

    monkeypatch.setattr(model.schedule, "sample_loop", fake_sample_loop)
    metric_targets = {mod: torch.zeros(1, 5) for mod in mods}
    labels = torch.zeros(1, dtype=torch.long)
    _ = model.sample(
        metric_targets,
        labels,
        n_classes=5,
        spec_shape=(3, 8, 8),
        device=torch.device("cpu"),
        seed=123,
    )
    assert seen_seeds == [123, 124]


def test_domain_instruction_samples_per_modality_ranges():
    cfg = OmegaConf.create(
        {
            "augmentation": {
                "domain_instruction": {
                    "walking": {
                        "acc": {
                            "temporal_range": [1.0, 1.0],
                            "f0_amplitude": [1.0, 1.0],
                            "contrast": [1.0, 1.0],
                            "flatness": [1.0, 1.0],
                            "entropy": [1.0, 1.0],
                        },
                        "gyr": {
                            "temporal_range": [2.0, 2.0],
                            "f0_amplitude": [2.0, 2.0],
                            "contrast": [2.0, 2.0],
                            "flatness": [2.0, 2.0],
                            "entropy": [2.0, 2.0],
                        },
                    }
                }
            }
        }
    )
    out = augment_domain_instruction("walking", ["acc", "gyr"], cfg, {"walking": 0})
    assert torch.equal(out["acc"], torch.ones(5))
    assert torch.equal(out["gyr"], torch.full((5,), 2.0))


def test_raw_cleaning_pipeline_is_deprecated_noop():
    test_root = pathlib.Path("tmp_raw_loader_test")
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True)
    sentinel = test_root / ".cleaned"
    try:
        run_cleaning_pipeline(test_root, sentinel)
        assert sentinel.exists()
    finally:
        if test_root.exists():
            shutil.rmtree(test_root)
