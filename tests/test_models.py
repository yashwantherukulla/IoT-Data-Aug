"""Tests for model components."""
import torch, pytest
from cgdap.models.condition import CrossAttentionConditionEmbedder, CrossAttentionBlock
from cgdap.models.unet import ConditionalUNet
from cgdap.models.ddpm import DDPMSchedule


def test_cross_attn_block():
    blk = CrossAttentionBlock(d_model=64, n_heads=4)
    x = torch.rand(2, 16, 64)       # [B, H*W, d_model]
    ctx = torch.rand(2, 6, 64)      # [B, n_tokens, d_model]
    out = blk(x, ctx)
    assert out.shape == x.shape


def test_condition_embedder():
    emb = CrossAttentionConditionEmbedder(n_metrics=5, n_classes=5, d_model=64, n_heads=4, n_cond_tokens=6)
    metrics = torch.rand(2, 5)
    labels = torch.zeros(2, 5); labels[:, 0] = 1.0
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
    from cgdap.models.unet import ConditionalUNet
    from cgdap.models.ddpm import DDPMSchedule
    from cgdap.models.condition import CrossAttentionConditionEmbedder

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
    assert not torch.isnan(loss_dict["L_total"])
    loss_dict["L_total"].backward()
