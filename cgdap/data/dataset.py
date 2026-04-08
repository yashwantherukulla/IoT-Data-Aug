"""Dataset classes for CGDAP v2.1 HAR pipeline.

Two dataset classes:

ModalityDataset
    Loads samples from a single modality folder:
        processed/HAR/<split>/<modality>/<activity>/<sample>.pt
    Each item: {spectrogram[3,F,T], metrics[5], label, activity, ...}
    Use this for independent single-modality training (larger batches).

PairedDataset
    Pairs acc and gyr by matching filenames across both modality trees.
    Each item: {acc: {...}, gyr: {...}, label, activity, ...}
    Use this for the augmentation engine (needs synchronized pairs).
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Single-modality dataset
# ---------------------------------------------------------------------------


class ModalityDataset(Dataset):
    """Load samples for one modality from processed/<split>/<modality>/."""

    REQUIRED_KEYS = {"spectrogram", "metrics", "label"}

    def __init__(
        self,
        split_dir: pathlib.Path,
        modality: str,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.modality = modality
        self.label_map = label_map or {}
        self.samples: list[dict[str, Any]] = []

        modality_dir = split_dir / modality
        if not modality_dir.exists():
            raise FileNotFoundError(f"Modality directory not found: {modality_dir}")

        for act_dir in sorted(modality_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            activity = act_dir.name
            fallback_label = self.label_map.get(activity)
            for sample_path in sorted(act_dir.glob("*.pt")):
                self.samples.append(
                    {
                        "path": sample_path,
                        "activity": activity,
                        "fallback_label": fallback_label,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        item: dict[str, Any] = torch.load(s["path"], weights_only=True)

        # Validate contract v2.1
        missing = self.REQUIRED_KEYS - item.keys()
        if missing:
            raise KeyError(
                f"Sample {s['path']} missing v2.1 keys: {missing}. "
                "Regenerate the dataset with scripts/prepare_dataset.py."
            )

        label = int(item.get("label", s["fallback_label"]))
        return {
            "spectrogram": item["spectrogram"].float(),   # [3, F, T]
            "metrics":     item["metrics"].float(),       # [5]
            "label":       label,
            "activity":    item.get("activity", s["activity"]),
            "subject":     item.get("subject", "unknown"),
            "window_index": int(item.get("window_index", -1)),
        }


# ---------------------------------------------------------------------------
# Paired dataset (acc + gyr by matching filenames)
# ---------------------------------------------------------------------------


class PairedDataset(Dataset):
    """Pair acc and gyr samples by matching their file paths.

    Both modality trees must share the same directory structure and filenames:
        <split>/acc/<activity>/<stem>.pt
        <split>/gyr/<activity>/<stem>.pt  <- same <activity>/<stem>
    """

    def __init__(
        self,
        split_dir: pathlib.Path,
        modalities: list[str] | None = None,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.modalities = modalities or ["acc", "gyr"]
        self.label_map = label_map or {}
        self.pairs: list[dict[str, Any]] = []

        # Build index from first modality; check others exist
        primary = self.modalities[0]
        primary_dir = split_dir / primary

        if not primary_dir.exists():
            raise FileNotFoundError(f"Primary modality dir not found: {primary_dir}")

        for act_dir in sorted(primary_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            activity = act_dir.name
            fallback_label = self.label_map.get(activity)

            for primary_path in sorted(act_dir.glob("*.pt")):
                stem = primary_path.stem  # e.g. "proband11_climbingdown_00000"
                paths: dict[str, pathlib.Path] = {primary: primary_path}
                all_found = True

                for mod in self.modalities[1:]:
                    mod_path = split_dir / mod / activity / f"{stem}.pt"
                    if not mod_path.exists():
                        all_found = False
                        break
                    paths[mod] = mod_path

                if all_found:
                    self.pairs.append(
                        {
                            "paths":         paths,
                            "activity":      activity,
                            "fallback_label": fallback_label,
                        }
                    )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        p = self.pairs[idx]
        result: dict[str, Any] = {"activity": p["activity"]}
        label: int | None = None

        for mod, path in p["paths"].items():
            item: dict[str, Any] = torch.load(path, weights_only=True)
            result[mod] = {
                "spectrogram": item["spectrogram"].float(),
                "metrics":     item["metrics"].float(),
                "subject":     item.get("subject", "unknown"),
                "window_index": int(item.get("window_index", -1)),
            }
            if label is None:
                label = int(item.get("label", p["fallback_label"]))

        result["label"] = label
        return result


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------


def build_label_map(split_dir: pathlib.Path, modality: str = "acc") -> dict[str, int]:
    """Build label map from activity directory names."""
    modality_dir = split_dir / modality
    if not modality_dir.exists():
        raise FileNotFoundError(f"Modality dir not found: {modality_dir}")
    activities = sorted(d.name for d in modality_dir.iterdir() if d.is_dir())
    return {act: i for i, act in enumerate(activities)}


def make_modality_loader(
    split_dir: pathlib.Path,
    modality: str,
    label_map: dict[str, int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    dataset = ModalityDataset(split_dir, modality, label_map)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def make_paired_loader(
    split_dir: pathlib.Path,
    modalities: list[str],
    label_map: dict[str, int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    dataset = PairedDataset(split_dir, modalities, label_map)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
