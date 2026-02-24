"""Visualize RealWorld-HAR spectrograms."""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    """Load config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_spectrogram(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spectrogram and axes from .pt file."""
    data = torch.load(path, weights_only=True)
    return (
        data["spectrogram"].numpy(),
        data["freq_axis_hz"].numpy(),
        data["time_axis_s"].numpy(),
    )


def get_samples(split_dir: pathlib.Path) -> list[dict]:
    """Get one paired sample per activity class."""
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    activities = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    samples: list[dict] = []
    
    for activity in activities:
        act_dir = split_dir / activity
        acc_files = list(act_dir.glob("*_acc.pt"))
        
        if acc_files:
            acc_path = acc_files[0]
            gyr_path = acc_path.with_name(acc_path.name.replace("_acc.pt", "_gyr.pt"))
            
            if gyr_path.exists():
                acc_spec, acc_freq, acc_time = load_spectrogram(acc_path)
                gyr_spec, gyr_freq, gyr_time = load_spectrogram(gyr_path)
                
                samples.append({
                    "activity": activity,
                    "acc": (acc_spec, acc_freq, acc_time),
                    "gyr": (gyr_spec, gyr_freq, gyr_time),
                })
    
    return samples


def plot_spectrogram(
    ax: plt.Axes,
    spec: np.ndarray,
    freq: np.ndarray,
    time: np.ndarray,
    channel: int,
    title: str | None = None,
    show_labels: bool = False,
) -> None:
    """Plot single channel spectrogram."""
    # Use percentile for robust color scaling
    vmin, vmax = np.percentile(spec[channel], [2, 98])
    
    ax.imshow(
        spec[channel],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[time[0], time[-1], freq[0], freq[-1]],
    )
    
    if title:
        ax.set_title(title, fontsize=9)
    if show_labels:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (Hz)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def main() -> None:
    """Main visualization function."""
    # Load config to get paths
    config_path = pathlib.Path(__file__).parents[1] / "configs" / "dataset" / "har_dataset.yaml"
    cfg = load_config(config_path)
    
    processed_root = pathlib.Path(cfg["dataset"]["paths"]["processed"])
    split = "train"  # Could be made configurable
    
    split_dir = processed_root / split
    samples = get_samples(split_dir)
    
    if not samples:
        print("No samples found")
        return
    
    n_classes = len(samples)
    n_rows = 6  # acc_x/y/z + gyr_x/y/z
    
    fig, axes = plt.subplots(n_rows, n_classes, figsize=(3 * n_classes, 10))
    if n_classes == 1:
        axes = axes.reshape(-1, 1)
    
    for col, sample in enumerate(samples):
        acc_spec, acc_freq, acc_time = sample["acc"]
        gyr_spec, gyr_freq, gyr_time = sample["gyr"]
        
        # Plot acc channels (rows 0-2)
        for ch in range(3):
            plot_spectrogram(
                axes[ch, col],
                acc_spec, acc_freq, acc_time,
                channel=ch,
                title=sample["activity"] if ch == 0 else None,
                show_labels=(col == 0 and ch == 2)
            )
            if col == 0:
                axes[ch, 0].set_ylabel(f"Acc {['x','y','z'][ch]}")
        
        # Plot gyr channels (rows 3-5)
        for ch in range(3):
            plot_spectrogram(
                axes[ch + 3, col],
                gyr_spec, gyr_freq, gyr_time,
                channel=ch,
                show_labels=(col == 0 and ch == 2)
            )
            if col == 0:
                axes[ch + 3, 0].set_ylabel(f"Gyr {['x','y','z'][ch]}")
    
    plt.suptitle(f"RealWorld HAR Spectrograms ({split})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()