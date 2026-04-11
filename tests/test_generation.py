"""Tests for standalone generation helpers."""

from __future__ import annotations

import pathlib
import shutil

import torch

from cgdap.generation import load_reference_pair, save_generated_outputs


def _sample_payload(label: int = 1) -> dict[str, object]:
    return {
        "spectrogram": torch.rand(3, 8, 12),
        "metrics": torch.rand(5),
        "label": label,
        "activity": "walking",
        "subject": "proband1",
        "window_index": 7,
        "sample_rate_hz": 100.0,
        "freq_axis_hz": torch.linspace(0, 50, 8),
        "time_axis_s": torch.linspace(0, 2.5, 12),
    }


def test_load_reference_pair_from_processed_layout():
    processed_root = pathlib.Path("tmp_generation_reference_test") / "processed" / "HAR"
    acc_path = processed_root / "train" / "acc" / "walking" / "demo.pt"
    gyr_path = processed_root / "train" / "gyr" / "walking" / "demo.pt"
    if processed_root.parent.parent.exists():
        shutil.rmtree(processed_root.parent.parent)
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    gyr_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(_sample_payload(), acc_path)
        torch.save(_sample_payload(), gyr_path)

        paired = load_reference_pair(acc_path, processed_root, ["acc", "gyr"])

        assert paired["split"] == "train"
        assert paired["activity"] == "walking"
        assert paired["stem"] == "demo"
        assert paired["label"] == 1
        assert set(paired["paths"]) == {"acc", "gyr"}
        assert paired["acc"]["spectrogram"].shape == (3, 8, 12)
        assert paired["gyr"]["metrics"].shape == (5,)
    finally:
        if processed_root.parent.parent.exists():
            shutil.rmtree(processed_root.parent.parent)


def test_save_generated_outputs_writes_expected_files():
    output_root = pathlib.Path("tmp_generation_output_test") / "generated"
    if output_root.parent.exists():
        shutil.rmtree(output_root.parent)
    reference_sample = {
        "paths": {
            "acc": pathlib.Path("data/processed/HAR/train/acc/walking/demo.pt"),
            "gyr": pathlib.Path("data/processed/HAR/train/gyr/walking/demo.pt"),
        },
        "acc": _sample_payload(),
        "gyr": _sample_payload(),
    }
    generated = {
        "acc": torch.rand(1, 3, 8, 12),
        "gyr": torch.rand(1, 3, 8, 12),
    }
    metric_targets = {
        "acc": torch.rand(5),
        "gyr": torch.rand(5),
    }

    try:
        saved = save_generated_outputs(
            output_root=output_root,
            sample_name="demo_0000",
            activity="walking",
            label=1,
            generated=generated,
            metric_targets=metric_targets,
            checkpoint_path=pathlib.Path("outputs/checkpoints/test_run/ckpt_epoch0000.pt"),
            augmentation_mode="disturbance",
            sample_index=0,
            reference_sample=reference_sample,
        )

        assert saved["acc_sample"].exists()
        assert saved["gyr_sample"].exists()
        assert saved["paired_bundle"].exists()

        acc_payload = torch.load(saved["acc_sample"], map_location="cpu", weights_only=True)
        bundle_payload = torch.load(saved["paired_bundle"], map_location="cpu", weights_only=True)

        assert acc_payload["label"] == 1
        assert acc_payload["activity"] == "walking"
        assert acc_payload["spectrogram"].shape == (3, 8, 12)
        assert bundle_payload["sample_name"] == "demo_0000"
        assert set(bundle_payload["modalities"]) == {"acc", "gyr"}
    finally:
        if output_root.parent.exists():
            shutil.rmtree(output_root.parent)
