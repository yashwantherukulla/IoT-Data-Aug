"""Prepare RealWorld HAR dataset: raw CSVs → spectrogram tensors.

Pipeline:
  1. Extract upperarm acc/gyr CSVs from activity zips
  2. Segment into non-overlapping windows (2.5s @ 50Hz = 125 samples)
  3. Compute STFT spectrograms (6 channels: acc_x/y/z + gyr_x/y/z saved separately)
  4. Save as .pt files with freq/time axes

Output structure:
  processed/
  ├── train/
  │   ├── climbing_down/
  │   │   ├── proband1_activity_00_00000_acc.pt
  │   │   ├── proband1_activity_00_00000_gyr.pt
  │   │   └── ...
  │   └── ...
  ├── val/
  └── metadata.json
"""

from __future__ import annotations

import csv
import io
import json
import logging
import pathlib
import random
import shutil
import zipfile
from dataclasses import dataclass
from typing import Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class STFTParams:
    """STFT parameters derived from config."""
    win_length: int
    n_fft: int
    hop_length: int
    window_samples: int


# =============================================================================
# Config-derived constants (computed once at import)
# =============================================================================

def _compute_stft_params(cfg: DictConfig) -> STFTParams:
    """Compute STFT parameters from config."""
    sr: float = cfg.dataset.sample_rate_hz
    fft_ms: float = cfg.dataset.spectrogram.fft_window_ms
    hop_ms: float = cfg.dataset.spectrogram.hop_ms
    win_sec: float = cfg.dataset.window_seconds
    
    win_length = round(sr * fft_ms / 1000)
    n_fft = 1 << (win_length - 1).bit_length()
    hop_length = max(1, round(sr * hop_ms / 1000))
    window_samples = round(sr * win_sec)
    
    # Validate: hop should not exceed window
    if hop_length > win_length:
        log.warning(f"hop_length ({hop_length}) > win_length ({win_length})")
    
    log.info(
        f"STFT params @ {sr}Hz: win_length={win_length}, n_fft={n_fft}, "
        f"hop_length={hop_length}, window_samples={window_samples}"
    )
    return STFTParams(win_length, n_fft, hop_length, window_samples)


def _make_window(window_type: str, length: int) -> torch.Tensor:
    """Create window tensor for STFT."""
    if window_type == "hann":
        return torch.hann_window(length)
    if window_type == "hamming":
        return torch.hamming_window(length)
    raise ValueError(f"Unknown window: {window_type}")


# =============================================================================
# Data extraction
# =============================================================================

def extract_csv_from_zip(zip_path: pathlib.Path, placement: str = "upperarm") -> np.ndarray:
    """Extract (N, 3) xyz array from acc/gyr zip file.
    
    Handles both flat zips (CSV directly) and nested zips (CSV inside inner zip).
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        
        # Detect nested structure
        if all(m.endswith(".zip") for m in members):
            # Nested: unzip the first inner zip and read CSV from there
            inner_zip_bytes = zf.read(members[0])
            with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zf:
                csv_name = _find_csv(inner_zf.namelist(), placement)
                csv_bytes = inner_zf.read(csv_name)
        else:
            # Flat: read CSV directly
            csv_name = _find_csv(members, placement)
            csv_bytes = zf.read(csv_name)
    
    return _parse_csv(csv_bytes)


def _find_csv(members: list[str], placement: str) -> str:
    """Find the CSV file for specific placement."""
    matches = [m for m in members if m.endswith(f"_{placement}.csv")]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 {placement} CSV, found {len(matches)}: {matches}")
    return matches[0]


def _parse_csv(csv_bytes: bytes) -> np.ndarray:
    """Parse CSV bytes to (N, 3) float32 array."""
    text = csv_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(text.splitlines())
    xyz = [[float(r["attr_x"]), float(r["attr_y"]), float(r["attr_z"])] for r in reader]
    return np.asarray(xyz, dtype=np.float32)


# =============================================================================
# Spectrogram computation
# =============================================================================

def compute_spectrogram(
    signal: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool,
    power: float,
    use_log1p: bool,
) -> torch.Tensor:
    """Compute (3, freq_bins, time_frames) spectrogram."""
    stft = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    spec = stft.abs()
    if power != 1.0:
        spec = spec.pow(power)
    if use_log1p:
        spec = torch.log1p(spec)
    return spec.to(torch.float32)


def make_axes(
    sample_rate: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_freq: int,
    n_time: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create frequency (Hz) and time (s) axis tensors."""
    freq = torch.linspace(0, sample_rate / 2, n_freq)
    window_duration = ((n_time - 1) * hop_length) / sample_rate
    time = torch.linspace(0.0, window_duration, n_time)
    return freq, time


def segment_windows(data: np.ndarray, window_size: int) -> Iterator[np.ndarray]:
    """Yield non-overlapping windows of size window_size."""
    for start in range(0, len(data) - window_size + 1, window_size):
        yield data[start : start + window_size]


# =============================================================================
# Dataset splitting
# =============================================================================

def split_subjects(
    subject_dirs: list[pathlib.Path], 
    n_train: int, 
    seed: int
) -> tuple[list[str], list[str]]:
    """Split subjects into train/val."""
    names = sorted([d.name for d in subject_dirs])
    rng = random.Random(seed)
    rng.shuffle(names)
    
    train = sorted(names[:n_train])
    val = sorted(names[n_train:])
    return train, val


# =============================================================================
# Main pipeline
# =============================================================================

def process_subject(
    subject_dir: pathlib.Path,
    activities: list[str],
    stft_params: STFTParams,
    cfg: DictConfig,
    out_root: pathlib.Path,
    split: str,
) -> int:
    """Process all activities for one subject. Returns window count."""
    window = _make_window(cfg.dataset.spectrogram.window, stft_params.win_length)
    sr: float = cfg.dataset.sample_rate_hz
    
    total_windows = 0
    
    for activity in activities:
        acc_zip = subject_dir / "data" / f"acc_{activity}_csv.zip"
        gyr_zip = subject_dir / "data" / f"gyr_{activity}_csv.zip"
        
        if not (acc_zip.exists() and gyr_zip.exists()):
            log.warning(f"Missing zip for {subject_dir.name}/{activity}, skipping")
            continue
        
        try:
            acc_data = extract_csv_from_zip(acc_zip)
            gyr_data = extract_csv_from_zip(gyr_zip)
        except Exception as e:
            log.error(f"Failed to extract {subject_dir.name}/{activity}: {e}")
            continue
        
        # Truncate to same length
        n_samples = min(len(acc_data), len(gyr_data))
        if n_samples < stft_params.window_samples:
            log.warning(
                f"Insufficient samples for {subject_dir.name}/{activity}: "
                f"{n_samples} < {stft_params.window_samples}"
            )
            continue
        
        acc_data = acc_data[:n_samples]
        gyr_data = gyr_data[:n_samples]
        
        # Create output dir
        label: str = cfg.dataset.activity_map.get(activity, activity)
        out_dir = out_root / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Process windows
        window_count = 0
        for w_idx, (acc_win, gyr_win) in enumerate(zip(
            segment_windows(acc_data, stft_params.window_samples),
            segment_windows(gyr_data, stft_params.window_samples)
        )):
            # Convert to tensors (3, T)
            acc_t = torch.from_numpy(acc_win.T)
            gyr_t = torch.from_numpy(gyr_win.T)
            
            # Compute spectrograms
            spec_kwargs = {
                "n_fft": stft_params.n_fft,
                "hop_length": stft_params.hop_length,
                "win_length": stft_params.win_length,
                "window": window,
                "center": cfg.dataset.spectrogram.center,
                "power": cfg.dataset.spectrogram.power,
                "use_log1p": cfg.dataset.spectrogram.log1p,
            }
            
            acc_spec = compute_spectrogram(acc_t, **spec_kwargs)
            gyr_spec = compute_spectrogram(gyr_t, **spec_kwargs)
            
            # Create axes
            n_freq, n_time = acc_spec.shape[1], acc_spec.shape[2]
            freq_axis, time_axis = make_axes(sr, stft_params.n_fft, stft_params.hop_length, stft_params.win_length, n_freq, n_time)
            
            # Save
            base_name = f"{subject_dir.name}_{activity}_{w_idx:05d}"
            torch.save({
                "spectrogram": acc_spec,
                "sample_rate_hz": sr,
                "freq_axis_hz": freq_axis,
                "time_axis_s": time_axis,
            }, out_dir / f"{base_name}_acc.pt")
            
            torch.save({
                "spectrogram": gyr_spec,
                "sample_rate_hz": sr,
                "freq_axis_hz": freq_axis,
                "time_axis_s": time_axis,
            }, out_dir / f"{base_name}_gyr.pt")
            
            window_count += 1
        
        total_windows += window_count
        log.info(f"[{split}] {subject_dir.name}/{activity}: {window_count} windows")
    
    return total_windows


@hydra.main(config_path="../configs/dataset", config_name="har_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Validate config
    if cfg.dataset.spectrogram.hop_ms > cfg.dataset.spectrogram.fft_window_ms:
        log.warning("hop_ms > fft_window_ms may cause information loss")
    
    # Setup paths
    raw_root = pathlib.Path(cfg.dataset.paths.raw)
    processed_root = pathlib.Path(cfg.dataset.paths.processed)
    
    if processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True)
    
    # Get subjects
    subject_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and "proband" in p.name])
    if len(subject_dirs) != 15:
        raise ValueError(f"Expected 15 subjects, found {len(subject_dirs)}")
    
    # Split
    train_subjects, val_subjects = split_subjects(
        subject_dirs, 
        cfg.dataset.split.num_train_subjects,
        cfg.seed
    )
    log.info(f"Train: {train_subjects}, Val: {val_subjects}")
    
    # Pre-compute STFT params (fixed for all data since sr is fixed)
    stft_params = _compute_stft_params(cfg)
    
    # Process all subjects
    window_counts: dict[str, int] = {"train": 0, "val": 0}
    
    for subj_dir in subject_dirs:
        split = "train" if subj_dir.name in train_subjects else "val"
        count = process_subject(
            subj_dir,
            cfg.dataset.activities,
            stft_params,
            cfg,
            processed_root,
            split
        )
        window_counts[split] += count
    
    # Save metadata
    metadata = {
        "seed": cfg.seed,
        "sample_rate_hz": cfg.dataset.sample_rate_hz,
        "window_seconds": cfg.dataset.window_seconds,
        "stft_params": {
            "win_length": stft_params.win_length,
            "n_fft": stft_params.n_fft,
            "hop_length": stft_params.hop_length,
            "window_samples": stft_params.window_samples,
        },
        "spectrogram_config": OmegaConf.to_container(cfg.dataset.spectrogram),
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
        },
        "window_counts": window_counts,
    }
    
    with open(processed_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    log.info("=" * 50)
    log.info(f"Done. Total windows: {window_counts}")
    log.info(f"Output: {processed_root}")


if __name__ == "__main__":
    main()