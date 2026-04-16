from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from cgdap.evaluation.product_eval import ProductEvaluator


class IdentityMetricExtractor(torch.nn.Module):
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return spec.reshape(spec.shape[0], -1).float()


class StubGenerator:
    def __init__(self, modalities: list[str], offsets: dict[str, torch.Tensor] | None = None) -> None:
        self.modalities = modalities
        self.metric_extractor = IdentityMetricExtractor()
        self.training = True
        self.offsets = offsets or {modality: torch.zeros(5) for modality in modalities}

    def eval(self) -> "StubGenerator":
        self.training = False
        return self

    def train(self) -> "StubGenerator":
        self.training = True
        return self

    def sample(
        self,
        *,
        metric_targets: dict[str, torch.Tensor],
        labels: torch.Tensor,
        n_classes: int,
        spec_shape: tuple[int, ...],
        device: torch.device,
        seed: int | None = None,
        num_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> dict[str, torch.Tensor]:
        del labels, n_classes, spec_shape, seed, num_steps, return_trajectory
        return {
            modality: (targets + self.offsets[modality].to(device)).reshape(targets.shape[0], 1, 1, 5)
            for modality, targets in metric_targets.items()
        }


def _write_sample(path: Path, *, metrics: torch.Tensor, activity: str, label: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spectrogram": metrics.reshape(1, 1, 5).float(),
        "metrics": metrics.float(),
        "label": int(label),
        "activity": activity,
        "subject": "test_subject",
        "window_index": 0,
        "sample_rate_hz": 100.0,
        "freq_axis_hz": torch.arange(1, dtype=torch.float32),
        "time_axis_s": torch.arange(5, dtype=torch.float32),
    }
    torch.save(payload, path)


def _build_processed_root(root: Path) -> Path:
    processed_root = root / "processed" / "HAR"
    activities = ["walking", "running"]
    base_metrics = {
        "walking": {
            "acc": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "gyr": torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5]),
        },
        "running": {
            "acc": torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]),
            "gyr": torch.tensor([2.5, 3.5, 4.5, 5.5, 6.5]),
        },
    }
    label_map = {"running": 0, "walking": 1}

    for activity in activities:
        for modality in ["acc", "gyr"]:
            for idx, delta in enumerate((0.0, 0.2)):
                metrics = base_metrics[activity][modality] + delta
                _write_sample(
                    processed_root / "train" / modality / activity / f"{activity}_{idx}.pt",
                    metrics=metrics,
                    activity=activity,
                    label=label_map[activity],
                )
            _write_sample(
                processed_root / "val" / modality / activity / f"{activity}_val.pt",
                metrics=base_metrics[activity][modality],
                activity=activity,
                label=label_map[activity],
            )

    return processed_root


def _build_cfg(processed_root: Path, *, mode: str) -> OmegaConf:
    domain_instruction = {
        "running": {
            "acc": {name: [float(v), float(v)] for name, v in zip(
                ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                strict=False,
            )},
            "gyr": {name: [float(v), float(v)] for name, v in zip(
                ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"],
                [2.5, 3.5, 4.5, 5.5, 6.5],
                strict=False,
            )},
        },
        "walking": {
            "acc": {name: [float(v), float(v)] for name, v in zip(
                ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                strict=False,
            )},
            "gyr": {name: [float(v), float(v)] for name, v in zip(
                ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"],
                [1.5, 2.5, 3.5, 4.5, 5.5],
                strict=False,
            )},
        },
    }
    return OmegaConf.create(
        {
            "seed": 7,
            "experiment_name": "test_run",
            "dataset": {
                "paths": {"processed": str(processed_root)},
                "modalities": ["acc", "gyr"],
                "activities": ["running", "walking"],
            },
            "augmentation": {
                "mode": mode,
                "disturbance": {
                    "temporal_range": 0.0,
                    "f0_amplitude": 0.0,
                    "contrast": 0.0,
                    "flatness": 0.0,
                    "entropy": 0.0,
                },
                "interpolation": {
                    "beta_mean": 0.5,
                    "beta_std": 0.0,
                    "beta_low": 0.5,
                    "beta_high": 0.5,
                },
                "domain_instruction": domain_instruction,
            },
            "evaluation": {
                "augmentation": {"num_steps": None},
                "product_eval": {
                    "enabled": True,
                    "every_n_epochs": 1,
                    "split": "val",
                    "samples_per_activity": 1,
                    "samples_per_probe": 1,
                    "seed": 11,
                    "num_steps": None,
                    "z_score_threshold": 2.0,
                    "log_histograms": False,
                    "log_scatters": False,
                },
            },
        }
    )


def test_product_evaluator_probe_selection_is_deterministic():
    root = Path("tmp_product_eval_deterministic")
    if root.exists():
        shutil.rmtree(root)
    try:
        processed_root = _build_processed_root(root)
        cfg = _build_cfg(processed_root, mode="disturbance")
        evaluator_a = ProductEvaluator(cfg, label_map={"running": 0, "walking": 1}, device=torch.device("cpu"))
        evaluator_b = ProductEvaluator(cfg, label_map={"running": 0, "walking": 1}, device=torch.device("cpu"))

        probe_a = [(sample["activity"], int(sample["label"])) for sample in evaluator_a.probe_samples]
        probe_b = [(sample["activity"], int(sample["label"])) for sample in evaluator_b.probe_samples]
        assert probe_a == probe_b
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_product_evaluator_reports_expected_pair_rmse_for_disturbance():
    root = Path("tmp_product_eval_disturbance")
    if root.exists():
        shutil.rmtree(root)
    try:
        processed_root = _build_processed_root(root)
        cfg = _build_cfg(processed_root, mode="disturbance")
        evaluator = ProductEvaluator(cfg, label_map={"running": 0, "walking": 1}, device=torch.device("cpu"))
        offsets = {
            "acc": torch.full((5,), 0.1),
            "gyr": torch.full((5,), -0.2),
        }
        model = StubGenerator(["acc", "gyr"], offsets=offsets)

        metrics = evaluator.evaluate(model)
        expected_rmse = torch.sqrt(torch.mean(torch.cat([offsets["acc"], offsets["gyr"]]) ** 2)).item()

        assert metrics["pair_rmse"] == pytest.approx(expected_rmse)
        assert metrics["activity/running_count"] == 1.0
        assert metrics["activity/walking_count"] == 1.0
        assert metrics["fidelity/acc_temporal_range_bias"] == pytest.approx(0.1)
        assert metrics["fidelity/gyr_temporal_range_bias"] == pytest.approx(-0.2)
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_product_evaluator_supports_all_augmentation_modes():
    root = Path("tmp_product_eval_modes")
    if root.exists():
        shutil.rmtree(root)
    try:
        processed_root = _build_processed_root(root)

        for mode in ("disturbance", "interpolation", "domain_instruction"):
            random.seed(0)
            cfg = _build_cfg(processed_root, mode=mode)
            evaluator = ProductEvaluator(cfg, label_map={"running": 0, "walking": 1}, device=torch.device("cpu"))
            metrics = evaluator.evaluate(StubGenerator(["acc", "gyr"]))
            activity_count_keys = [key for key in metrics if key.startswith("activity/") and key.endswith("_count")]
            assert sum(float(metrics[key]) for key in activity_count_keys) == 2.0
            assert "nn_distance_val_mean" in metrics
    finally:
        if root.exists():
            shutil.rmtree(root)
