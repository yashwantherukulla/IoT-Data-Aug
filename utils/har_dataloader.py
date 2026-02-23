"""har_dataloader.py
================
PyTorch Dataset and DataLoader factory for the RealWorld-HAR dataset
prepared by prepare_dataset.py.

Expected layout (all under dataset.paths.processed):

    PROCESSED_ROOT/
    ├── metadata.json                                    ← written by prepare_dataset.py
    ├── train/
    │   ├── climbing_down/
    │   │   ├── proband10_climbingdown_00_00000_acc.pt   # (3, T)
    │   │   ├── proband10_climbingdown_00_00000_gyr.pt   # (3, T)
    │   │   └── ...
    │   └── ...
    └── val/
        └── ... (same structure)

File naming convention (produced by prepare_dataset.py):
    {subject_name}_{record_key}_{pair_idx:02d}_{window_idx:05d}_{modality}.pt
    Example: proband10_climbingdown_00_00003_acc.pt

Usage:
    from har_dataloader import HARDataset, get_dataloaders
    from omegaconf import DictConfig

    # cfg is the Hydra DictConfig from har_dataset.yaml (top-level)
    train_loader, val_loader, meta = get_dataloaders(cfg)

    for acc, gyr, label in train_loader:
        # acc  : (B, 3, T)  — accelerometer, channels-first
        # gyr  : (B, 3, T)  — gyroscope, channels-first
        # label: (B,)        — integer class index
        ...
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# ================================================================= #
#  Dataset                                                           #
# ================================================================= #

@dataclass
class Sample:
    acc_path: pathlib.Path
    gyr_path: pathlib.Path
    label: int


class HARDataset(Dataset):
    """
    Paired accelerometer + gyroscope segment dataset.

    __getitem__ returns (acc_tensor, gyr_tensor, label):
        acc_tensor : FloatTensor of shape (3, T)
        gyr_tensor : FloatTensor of shape (3, T)
        label      : int  (class index)

    Class index mapping (sorted alphabetically by directory name):
        0 — climbing_down
        1 — climbing_up
        2 — jumping
        3 — running
        4 — walking

    File pairs are matched by stripping the terminal '_acc' / '_gyr' suffix
    from each .pt stem, following the naming convention used by prepare_dataset.py:
        {subject_name}_{record_key}_{pair_idx:02d}_{window_idx:05d}_{modality}.pt

    Args:
        root:       Path to the split directory, e.g. '.../HAR/train'
        transform:  Optional callable applied identically to both tensors.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.root = pathlib.Path(root)
        self.transform = transform

        if not self.root.is_dir():
            raise FileNotFoundError(f"HARDataset root not found: {self.root}")

        self.classes: list[str] = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[Sample] = []
        self._build_index()

        log.info(
            "HARDataset  root=%s  classes=%s  n_samples=%d",
            self.root, self.classes, len(self.samples),
        )

    def _build_index(self) -> None:
        """
        Pair every *_acc.pt with its matching *_gyr.pt.

        Naming convention set by prepare_dataset.py:
            {subject_name}_{record_key}_{pair_idx:02d}_{window_idx:05d}_acc.pt
            {subject_name}_{record_key}_{pair_idx:02d}_{window_idx:05d}_gyr.pt

        The pair key is everything before the terminal '_acc' / '_gyr' token.
        """
        for activity in self.classes:
            label = self.class_to_idx[activity]
            act_dir = self.root / activity

            acc_map: dict[str, pathlib.Path] = {}
            gyr_map: dict[str, pathlib.Path] = {}

            for pt in sorted(act_dir.glob("*.pt")):
                if pt.stem.endswith("_acc"):
                    acc_map[pt.stem[:-4]] = pt
                elif pt.stem.endswith("_gyr"):
                    gyr_map[pt.stem[:-4]] = pt

            for key in sorted(acc_map):
                if key in gyr_map:
                    self.samples.append(Sample(acc_map[key], gyr_map[key], label))
                else:
                    log.warning("No matching gyr for %s", acc_map[key])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        s = self.samples[idx]
        acc = torch.load(s.acc_path, weights_only=True)   # (3, T)
        gyr = torch.load(s.gyr_path, weights_only=True)   # (3, T)

        if self.transform is not None:
            acc = self.transform(acc)
            gyr = self.transform(gyr)

        return acc, gyr, s.label

    def __repr__(self) -> str:
        return (
            f"HARDataset(root={self.root}, "
            f"classes={self.classes}, "
            f"n_samples={len(self)})"
        )


# ================================================================= #
#  DataLoader metadata                                               #
# ================================================================= #

@dataclass
class DataLoaderMeta:
    """Metadata returned alongside the loaders.

    Populated first from the filesystem (classes discovered at runtime),
    then enriched with the contents of metadata.json written by
    prepare_dataset.py (split subjects, window counts, sensor info, etc.).
    """

    # ---- from filesystem ----
    classes: list[str]
    class_to_idx: dict[str, int]
    n_train: int
    n_val: int
    # ---- from metadata.json (None / empty when file is absent) ----
    seed: Optional[int] = None
    window_seconds: Optional[float] = None
    train_subjects: list[str] = field(default_factory=list)
    val_subjects: list[str] = field(default_factory=list)
    window_counts: dict[str, int] = field(default_factory=dict)
    sensors: list[str] = field(default_factory=list)
    placement: Optional[str] = None
    raw_root: Optional[str] = None


# ================================================================= #
#  DataLoader factory                                                #
# ================================================================= #

def _load_metadata(processed_root: pathlib.Path) -> dict[str, Any]:
    """Read metadata.json written by prepare_dataset.py, if present."""
    meta_path = processed_root / "metadata.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as fh:
            return json.load(fh)
    log.warning("metadata.json not found at %s — skipping enrichment", meta_path)
    return {}


def get_dataloaders(
    cfg: DictConfig,
    transform: Optional[Callable] = None,
) -> tuple[DataLoader, DataLoader, DataLoaderMeta]:
    """
    Build train and val DataLoaders directly from the Hydra config.

    All settings (root path, batch size, num_workers, etc.) are read from
    cfg.dataset so no values need to be passed individually.

    Also reads metadata.json produced by prepare_dataset.py to populate
    DataLoaderMeta with split subjects, window counts, and sensor info.

    Args:
        cfg:        Full Hydra DictConfig (top-level, from har_dataset.yaml).
        transform:  Optional tensor transform applied identically to both modalities.

    Returns:
        (train_loader, val_loader, meta)
    """
    ds_cfg = cfg.dataset
    dl_cfg = cfg.dataset.dataloader

    processed_root = pathlib.Path(ds_cfg.paths.processed)
    if not processed_root.is_dir():
        raise FileNotFoundError(
            f"Processed dataset root not found: {processed_root}. "
            "Run prepare_dataset.py first."
        )

    train_ds = HARDataset(processed_root / "train", transform=transform)
    val_ds   = HARDataset(processed_root / "val",   transform=transform)

    if train_ds.classes != val_ds.classes:
        log.warning(
            "Class mismatch between splits — train: %s  val: %s",
            train_ds.classes, val_ds.classes,
        )

    loader_kwargs: dict[str, Any] = dict(
        batch_size         = dl_cfg.batch_size,
        num_workers        = dl_cfg.num_workers,
        pin_memory         = dl_cfg.pin_memory,
        drop_last          = dl_cfg.drop_last,
        prefetch_factor    = dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
        persistent_workers = dl_cfg.num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # Enrich metadata from metadata.json written by prepare_dataset.py.
    raw_meta   = _load_metadata(processed_root)
    split_info = raw_meta.get("split", {})

    meta = DataLoaderMeta(
        classes        = train_ds.classes,
        class_to_idx   = train_ds.class_to_idx,
        n_train        = len(train_ds),
        n_val          = len(val_ds),
        seed           = raw_meta.get("seed"),
        window_seconds = raw_meta.get("window_seconds"),
        train_subjects = split_info.get("train_subjects", []),
        val_subjects   = split_info.get("val_subjects", []),
        window_counts  = raw_meta.get("window_counts", {}),
        sensors        = raw_meta.get("sensors", []),
        placement      = raw_meta.get("placement"),
        raw_root       = raw_meta.get("raw_root"),
    )

    log.info(
        "DataLoaders ready  —  train: %d samples  val: %d samples  batch_size: %d",
        meta.n_train, meta.n_val, dl_cfg.batch_size,
    )
    log.info(
        "Split  —  train subjects (%d): %s  |  val subjects (%d): %s",
        len(meta.train_subjects), meta.train_subjects,
        len(meta.val_subjects),   meta.val_subjects,
    )

    return train_loader, val_loader, meta


# ================================================================= #
#  Standalone entry point (Hydra)                                    #
# ================================================================= #

@hydra.main(config_path="../configs/dataset", config_name="har_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    """Smoke-test the dataloader by pulling one batch from each split."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    train_loader, val_loader, meta = get_dataloaders(cfg)

    log.info("Classes        : %s", meta.classes)
    log.info("Train subjects : %s", meta.train_subjects)
    log.info("Val subjects   : %s", meta.val_subjects)
    log.info("Window counts  : %s", meta.window_counts)
    log.info("Sensors        : %s  placement=%s", meta.sensors, meta.placement)

    acc, gyr, labels = next(iter(train_loader))
    log.info("Train batch  acc=%s  gyr=%s  labels=%s", acc.shape, gyr.shape, labels.shape)

    acc, gyr, labels = next(iter(val_loader))
    log.info("Val   batch  acc=%s  gyr=%s  labels=%s", acc.shape, gyr.shape, labels.shape)


if __name__ == "__main__":
    main()
