"""Summary statistics for the RealWorld-HAR processed dataset via PyTorch DataLoaders."""

from __future__ import annotations

import json
import os
import pathlib
from collections import defaultdict
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    """Load config from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HARSpectrogramDataset(Dataset):
    """Loads processed HAR samples saved as single .pt files.

    Each sample returns a dict:
        spectrogram: FloatTensor [F, T]
        label: int
        metrics: FloatTensor [5]
        activity: str
    """

    def __init__(self, split_dir: pathlib.Path, label_map: dict[str, int] | None = None) -> None:
        self.label_map = label_map or {}
        self.samples: list[dict[str, Any]] = []

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for act_dir in sorted(split_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            activity = act_dir.name
            fallback_label = self.label_map.get(activity)
            for sample_path in sorted(act_dir.glob("*.pt")):
                self.samples.append(
                    {
                        "sample_path": sample_path,
                        "activity": activity,
                        "fallback_label": fallback_label,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        item = torch.load(s["sample_path"], weights_only=True)
        label = int(item.get("label", s["fallback_label"]))
        metrics = item.get("metrics", torch.zeros(5, dtype=torch.float32)).to(torch.float32)
        return {
            "spectrogram": item["spectrogram"].float(),
            "label": label,
            "metrics": metrics,
            "activity": item.get("activity", s["activity"]),
        }


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def build_label_map(split_dir: pathlib.Path) -> dict[str, int]:
    activities = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    return {act: i for i, act in enumerate(activities)}


def compute_split_stats(loader: DataLoader, split: str) -> None:
    """Iterate a DataLoader and print class + spectrogram + metric statistics."""
    class_counts: dict[str, int] = defaultdict(int)
    global_sum = 0.0
    global_sq_sum = 0.0
    global_min = float("inf")
    global_max = float("-inf")
    n_pixels = 0
    spec_shape: tuple[int, ...] | None = None
    nan_count = 0
    inf_count = 0

    metric_sum = torch.zeros(5)
    metric_sq_sum = torch.zeros(5)
    metric_min = torch.full((5,), float("inf"))
    metric_max = torch.full((5,), float("-inf"))
    metric_count = 0

    for batch in loader:
        specs: torch.Tensor = batch["spectrogram"]  # [B, F, T]
        labels: list[str] = batch["activity"]
        metrics: torch.Tensor = batch["metrics"]  # [B, 5]

        if spec_shape is None:
            spec_shape = tuple(specs.shape[1:])

        for act in labels:
            class_counts[act] += 1

        flat = specs.view(specs.size(0), -1)
        bsz, n = flat.shape

        nan_count += flat.isnan().sum().item()
        inf_count += flat.isinf().sum().item()

        clean = flat.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        global_sum += clean.sum().item()
        global_sq_sum += clean.pow(2).sum().item()
        global_min = min(global_min, clean.min().item())
        global_max = max(global_max, clean.max().item())
        n_pixels += bsz * n

        metric_sum += metrics.sum(dim=0)
        metric_sq_sum += metrics.pow(2).sum(dim=0)
        metric_min = torch.minimum(metric_min, metrics.min(dim=0).values)
        metric_max = torch.maximum(metric_max, metrics.max(dim=0).values)
        metric_count += metrics.shape[0]

    total = sum(class_counts.values())
    mean = global_sum / n_pixels
    std = max(global_sq_sum / n_pixels - mean**2, 0.0) ** 0.5

    metric_mean = metric_sum / metric_count
    metric_std = (metric_sq_sum / metric_count - metric_mean**2).clamp(min=0).sqrt()

    metric_names = ["temp_range", "f0_amp_hps", "contrast", "flatness", "entropy"]

    print(f"\n{'='*60}")
    print(f"  Split: {split.upper()}   |   Total samples: {total}")
    print(f"{'='*60}")

    print(f"\n  Spectrogram shape (F x T): {spec_shape}")
    print(f"  NaN values : {int(nan_count)}")
    print(f"  Inf values : {int(inf_count)}")

    print("\n  Class distribution:")
    for act in sorted(class_counts):
        pct = 100 * class_counts[act] / total
        print(f"    {act:<20s}  {class_counts[act]:>5d}  ({pct:5.1f}%)")

    print("\n  Spectrogram statistics:")
    print(f"    Mean: {mean:8.4f}  Std: {std:8.4f}  Min: {global_min:8.4f}  Max: {global_max:8.4f}")

    print("\n  Metric statistics:")
    header = f"  {'Metric':<12}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}"
    print(header)
    print(f"  {'-'*62}")
    for i, name in enumerate(metric_names):
        print(
            f"  {name:<12}  {metric_mean[i].item():>10.4f}  {metric_std[i].item():>10.4f}"
            f"  {metric_min[i].item():>10.4f}  {metric_max[i].item():>10.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    repo_root = pathlib.Path(__file__).parents[1]
    config_path = repo_root / "configs" / "dataset" / "har_dataset.yaml"
    cfg = load_config(config_path)
    loader_cfg = cfg["dataset"]["loader"]

    processed_root = repo_root / pathlib.Path(cfg["dataset"]["paths"]["processed"])
    metadata_path = processed_root / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        print("\nMetadata")
        print(json.dumps(meta, indent=2))

    train_dir = processed_root / "train"
    label_map = build_label_map(train_dir)
    print(f"\nLabel map: {label_map}")

    for split in ("train", "val"):
        split_dir = processed_root / split
        if not split_dir.exists():
            print(f"\nSkipping {split}: directory not found")
            continue

        dataset = HARSpectrogramDataset(split_dir, label_map)
        loader = DataLoader(
            dataset,
            batch_size=loader_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=min(loader_cfg.get("num_workers", 4), os.cpu_count() or 0),
            pin_memory=bool(loader_cfg.get("pin_memory", False)),
        )
        compute_split_stats(loader, split)

    print()


if __name__ == "__main__":
    main()
