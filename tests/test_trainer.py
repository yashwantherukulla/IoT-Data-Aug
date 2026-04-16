from __future__ import annotations

import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cgdap.training.trainer import CGDAPTrainer, ExperimentLogger, compute_grad_norm


def _write_sample(path: Path, *, label: int, activity: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spectrogram": torch.rand(3, 32, 32),
        "metrics": torch.rand(5),
        "label": int(label),
        "activity": activity,
        "subject": "trainer_test",
        "window_index": 0,
        "sample_rate_hz": 100.0,
        "freq_axis_hz": torch.linspace(0, 50, 32),
        "time_axis_s": torch.linspace(0, 2.5, 32),
    }
    torch.save(payload, path)


def _build_processed_root(tmp_path: Path) -> Path:
    processed_root = tmp_path / "processed" / "HAR"
    for split in ("train", "val"):
        for activity, label in (("running", 0), ("walking", 1)):
            for modality in ("acc", "gyr"):
                _write_sample(
                    processed_root / split / modality / activity / f"{activity}_{split}.pt",
                    label=label,
                    activity=activity,
                )
    return processed_root


def _build_cfg(processed_root: Path, tmp_path: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 13,
            "experiment_name": "trainer_test",
            "dataset": {
                "paths": {"processed": str(processed_root)},
                "modalities": ["acc", "gyr"],
                "activities": ["running", "walking"],
                "loader": {"num_workers": 0, "pin_memory": False, "drop_last": False},
                "metrics": {
                    "names": ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"],
                },
            },
            "model": {
                "modalities": ["acc", "gyr"],
                "denoiser": "unet",
                "schedule": "ddpm",
                "embedder": "cross_attention",
                "in_channels": 3,
                "condition": {"d_model": 32, "n_heads": 2, "n_cond_tokens": 6, "dropout": 0.1},
                "unet": {
                    "base_channels": 16,
                    "channel_multipliers": [1, 2],
                    "n_res_blocks": 1,
                    "temb_dim": 64,
                    "cross_attn_depths": [0],
                    "dropout": 0.1,
                },
                "ddpm": {
                    "train_timesteps": 50,
                    "beta_start": 1.0e-4,
                    "beta_end": 2.0e-2,
                    "num_train_steps": 2,
                    "num_infer_steps": 4,
                },
            },
            "training": {
                "max_epochs": 2,
                "batch_size": 1,
                "optimizer": {
                    "name": "adamw",
                    "lr": 1.0e-4,
                    "betas": [0.9, 0.999],
                    "weight_decay": 0.0,
                    "eps": 1.0e-8,
                },
                "scheduler": {"name": "none"},
                "loss": {
                    "metric_weight_init": 0.1,
                    "target_ratio": 10.0,
                    "adaptive_start_epoch": 1,
                    "metric_weight_ema_decay": 0.9,
                    "weight_min": 0.01,
                    "weight_max": 10.0,
                },
                "log_every_n_steps": 1,
                "save_every_n_epochs": 100,
                "val_every_n_epochs": 1,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "resume": False,
                "resume_checkpoint": None,
                "restore_rng_state": True,
            },
            "logging": {"backend": "console", "log_every_n_steps": 1, "save_dir": str(tmp_path / "logs")},
            "augmentation": {
                "mode": "disturbance",
                "disturbance": {
                    "temporal_range": 0.0,
                    "f0_amplitude": 0.0,
                    "contrast": 0.0,
                    "flatness": 0.0,
                    "entropy": 0.0,
                },
                "interpolation": {"beta_mean": 0.5, "beta_std": 0.1, "beta_low": 0.0, "beta_high": 1.0},
                "domain_instruction": {},
            },
            "evaluation": {
                "augmentation": {"num_steps": None},
                "product_eval": {
                    "enabled": True,
                    "every_n_epochs": 1,
                    "split": "val",
                    "samples_per_activity": 1,
                    "samples_per_probe": 1,
                    "seed": 13,
                    "num_steps": None,
                    "z_score_threshold": 2.0,
                    "log_histograms": False,
                    "log_scatters": False,
                },
            },
        }
    )


def test_compute_grad_norm_tracks_preclip_value():
    layer = torch.nn.Linear(4, 1, bias=False)
    loss = layer(torch.ones(1, 4)).sum()
    loss.backward()

    preclip_norm = compute_grad_norm(layer.parameters())
    torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=0.01)
    postclip_norm = compute_grad_norm(layer.parameters())

    assert preclip_norm > postclip_norm


def test_experiment_logger_console_accepts_non_scalar_payload():
    logger = ExperimentLogger(OmegaConf.create({"logging": {"backend": "console", "save_dir": "tmp_logs"}}))
    logger.log({"chart": object()})


def test_trainer_runs_product_eval_each_epoch_without_checkpoint_reload():
    root = Path("tmp_trainer_product_eval")
    if root.exists():
        shutil.rmtree(root)
    try:
        processed_root = _build_processed_root(root)
        cfg = _build_cfg(processed_root, root)
        trainer = CGDAPTrainer(cfg)

        logged_payloads: list[dict[str, object]] = []

        class DummyEvaluator:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate(self, model: object, *, enable_artifacts: bool = False) -> dict[str, float]:
                assert model is trainer.model
                assert enable_artifacts is False
                self.calls += 1
                return {
                    "pair_rmse": 0.25,
                    "nn_distance_val_mean": 0.3,
                    "nn_distance_train_mean": 0.2,
                    "coverage_unique_nn_ratio": 0.1,
                }

        trainer.train_epoch = lambda epoch: {
            "L_total": 1.0,
            "L_G": 0.5,
            "L_metric": 0.25,
            "L_metric_temporal_range": 0.1,
        }
        trainer.val_epoch = lambda epoch: {"L_total": 0.9, "L_G": 0.45, "L_metric": 0.2}
        trainer.save_checkpoint = lambda epoch, metrics: None
        trainer.experiment_logger.log = lambda payload, step=None: logged_payloads.append(payload)
        trainer.experiment_logger.finish = lambda: None
        trainer.product_evaluator = DummyEvaluator()

        trainer.run()

        assert trainer.product_evaluator.calls == 2
        assert any("product_eval/pair_rmse" in payload for payload in logged_payloads)
        assert any("train_epoch/L_metric_temporal_range" in payload for payload in logged_payloads)
    finally:
        if root.exists():
            shutil.rmtree(root)
