"""Visualize RealWorld-HAR spectrograms from precomputed sample files."""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


METRIC_NAMES = ["temp_range", "f0_amp_hps", "contrast", "flatness", "entropy"]


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    """Load config from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sample(path: pathlib.Path) -> dict[str, Any]:
    """Load one sample from .pt file."""
    data = torch.load(path, weights_only=True)
    return {
        "spectrogram": data["spectrogram"].numpy(),
        "freq": data["freq_axis_hz"].numpy(),
        "time": data["time_axis_s"].numpy(),
        "metrics": data.get("metrics", torch.zeros(5)).numpy(),
        "label": int(data.get("label", -1)),
        "activity": data.get("activity", path.parent.name),
    }


def get_samples(split_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Get one sample per activity class."""
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    activities = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    samples: list[dict[str, Any]] = []

    for activity in activities:
        act_dir = split_dir / activity
        sample_files = sorted(act_dir.glob("*.pt"))
        if sample_files:
            sample = load_sample(sample_files[0])
            sample["activity"] = activity
            samples.append(sample)

    return samples


def plot_spectrogram(
    ax: plt.Axes,
    spec: np.ndarray,
    freq: np.ndarray,
    time: np.ndarray,
    title: str | None = None,
) -> None:
    """Plot one 2D spectrogram."""
    # Collapse multi-channel spectrograms (C, F, T) -> (F, T) by averaging
    if spec.ndim == 3:
        spec = spec.mean(axis=0)

    vmin, vmax = np.percentile(spec, [2, 98])

    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[time[0], time[-1], freq[0], freq[-1]],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (Hz)")

    if title:
        ax.set_title(title, fontsize=10)


def _format_metrics(metrics: np.ndarray) -> str:
    values = ", ".join(f"{name}={value:.3f}" for name, value in zip(METRIC_NAMES, metrics))
    return values


def main() -> None:
    """Main visualization function."""
    config_path = pathlib.Path(__file__).parents[1] / "configs" / "dataset" / "har_dataset.yaml"
    cfg = load_config(config_path)

    processed_root = pathlib.Path(cfg["dataset"]["paths"]["processed"])
    split = "train"

    split_dir = processed_root / split
    samples = get_samples(split_dir)

    if not samples:
        print("No samples found")
        return

    n_classes = len(samples)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4), squeeze=False)

    for col, sample in enumerate(samples):
        plot_spectrogram(
            axes[0, col],
            sample["spectrogram"],
            sample["freq"],
            sample["time"],
            title=f"{sample['activity']} (label={sample['label']})",
        )
        axes[0, col].text(
            0.01,
            -0.24,
            _format_metrics(sample["metrics"]),
            transform=axes[0, col].transAxes,
            fontsize=8,
            va="top",
        )

    plt.suptitle(f"RealWorld HAR Spectrograms + Metrics ({split})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
