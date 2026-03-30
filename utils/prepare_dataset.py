"""Prepare RealWorld HAR dataset: raw CSVs -> STFT tensors + precomputed metrics.

Pipeline:
  1. Extract upperarm acc/gyr CSVs from activity zips
  2. Segment into non-overlapping windows (window_seconds * sample_rate_hz samples)
  3. Compute STFT spectrograms (acc xyz + gyr xyz)
  4. Build one 2D spectrogram per window by averaging 6 channels
  5. Compute 5 differentiable metrics with PyTorch tensor ops
  6. Save one .pt per sample containing:
      - spectrogram: FloatTensor [F, T]
      - label: int
      - metrics: FloatTensor [5]
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

    if hop_length > win_length:
        log.warning("hop_length (%s) > win_length (%s)", hop_length, win_length)

    log.info(
        "STFT params @ %sHz: win_length=%s, n_fft=%s, hop_length=%s, window_samples=%s",
        sr,
        win_length,
        n_fft,
        hop_length,
        window_samples,
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

        if all(m.endswith(".zip") for m in members):
            inner_zip_bytes = zf.read(members[0])
            with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zf:
                csv_name = _find_csv(inner_zf.namelist(), placement)
                csv_bytes = inner_zf.read(csv_name)
        else:
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
        center=True,
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
    n_freq: int,
    n_time: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create frequency (Hz) and time (s) axis tensors."""
    del n_fft
    freq = torch.linspace(0, sample_rate / 2, n_freq)
    window_duration = ((n_time - 1) * hop_length) / sample_rate
    time = torch.linspace(0.0, window_duration, n_time)
    return freq, time


def segment_windows(data: np.ndarray, window_size: int) -> Iterator[np.ndarray]:
    """Yield non-overlapping windows of size window_size."""
    for start in range(0, len(data) - window_size + 1, window_size):
        yield data[start : start + window_size]


# =============================================================================
# Differentiable metric extraction (PyTorch only)
# =============================================================================


def metric_temporal_amplitude_range(spec_2d: torch.Tensor) -> torch.Tensor:
    """Max-min of mean amplitude over time; spec_2d shape [F, T]."""
    mean_over_time = spec_2d.mean(dim=-1)
    return mean_over_time.max() - mean_over_time.min()


def metric_f0_amplitude_hps(
    spec_2d: torch.Tensor,
    downsample_factor: int = 2,
    softmax_beta: float = 20.0,
) -> torch.Tensor:
    """F0 amplitude via Harmonic Product Spectrum with differentiable soft-argmax."""
    downsample_factor = max(2, int(downsample_factor))
    mean_over_time = spec_2d.mean(dim=-1)
    hps = mean_over_time.clone()

    for d in range(2, downsample_factor + 1):
        down = mean_over_time[::d]
        hps = hps[: down.numel()] * down

    weights = torch.softmax(hps * softmax_beta, dim=0)
    return (spec_2d[: hps.numel()] * weights.unsqueeze(-1)).sum(dim=0).mean()


def metric_contrast(spec_2d: torch.Tensor, tail_ratio: float = 0.05) -> torch.Tensor:
    """Mean(top-k) - mean(bottom-k) over all spectrogram bins."""
    flat = spec_2d.reshape(-1)
    n = flat.numel()
    k = max(1, int(n * tail_ratio))
    sorted_vals, _ = torch.sort(flat)
    valleys = sorted_vals[:k].mean()
    peaks = sorted_vals[-k:].mean()
    return peaks - valleys


def metric_flatness(spec_2d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Geometric mean / arithmetic mean."""
    x = spec_2d.clamp_min(eps)
    gmean = torch.exp(torch.log(x).mean())
    amean = x.mean()
    return gmean / (amean + eps)


def metric_entropy(spec_2d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy base-2 over normalized spectrogram bins."""
    x = spec_2d.clamp_min(eps)
    p = x / x.sum()
    return -(p * torch.log2(p)).sum()


def compute_metrics(spec_2d: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Compute the 5 required scalar metrics, returned as [5]."""
    mcfg = cfg.dataset.metrics
    metrics = [
        metric_temporal_amplitude_range(spec_2d),
        metric_f0_amplitude_hps(
            spec_2d,
            downsample_factor=mcfg.hps_downsample_factor,
            softmax_beta=mcfg.hps_softmax_beta,
        ),
        metric_contrast(spec_2d, tail_ratio=mcfg.contrast_tail_ratio),
        metric_flatness(spec_2d, eps=mcfg.eps),
        metric_entropy(spec_2d, eps=mcfg.eps),
    ]
    return torch.stack(metrics).to(torch.float32)


# =============================================================================
# Dataset splitting
# =============================================================================


def split_subjects(
    subject_dirs: list[pathlib.Path],
    n_train: int,
    seed: int,
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
    label_to_idx: dict[str, int],
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
            log.warning("Missing zip for %s/%s, skipping", subject_dir.name, activity)
            continue

        try:
            acc_data = extract_csv_from_zip(acc_zip)
            gyr_data = extract_csv_from_zip(gyr_zip)
        except Exception as e:  # noqa: BLE001
            log.error("Failed to extract %s/%s: %s", subject_dir.name, activity, e)
            continue

        n_samples = min(len(acc_data), len(gyr_data))
        if n_samples < stft_params.window_samples:
            log.warning(
                "Insufficient samples for %s/%s: %s < %s",
                subject_dir.name,
                activity,
                n_samples,
                stft_params.window_samples,
            )
            continue

        acc_data = acc_data[:n_samples]
        gyr_data = gyr_data[:n_samples]

        label_name = cfg.dataset.activity_map.get(activity, activity)
        label_idx = label_to_idx[label_name]
        out_dir = out_root / split / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        window_count = 0
        for w_idx, (acc_win, gyr_win) in enumerate(
            zip(
                segment_windows(acc_data, stft_params.window_samples),
                segment_windows(gyr_data, stft_params.window_samples),
            )
        ):
            acc_t = torch.from_numpy(acc_win.T)
            gyr_t = torch.from_numpy(gyr_win.T)

            spec_kwargs = {
                "n_fft": stft_params.n_fft,
                "hop_length": stft_params.hop_length,
                "win_length": stft_params.win_length,
                "window": window,
                "power": cfg.dataset.spectrogram.power,
                "use_log1p": cfg.dataset.spectrogram.log1p,
            }

            acc_spec = compute_spectrogram(acc_t, **spec_kwargs)
            gyr_spec = compute_spectrogram(gyr_t, **spec_kwargs)

            spec_2d = torch.cat([acc_spec, gyr_spec], dim=0).mean(dim=0)
            metrics = compute_metrics(spec_2d, cfg)

            n_freq, n_time = spec_2d.shape
            freq_axis, time_axis = make_axes(
                sr,
                stft_params.n_fft,
                stft_params.hop_length,
                n_freq,
                n_time,
            )

            base_name = f"{subject_dir.name}_{activity}_{w_idx:05d}"
            torch.save(
                {
                    "spectrogram": spec_2d.to(torch.float32),
                    "label": int(label_idx),
                    "metrics": metrics,
                    "activity": label_name,
                    "sample_rate_hz": sr,
                    "freq_axis_hz": freq_axis,
                    "time_axis_s": time_axis,
                },
                out_dir / f"{base_name}.pt",
            )

            window_count += 1

        total_windows += window_count
        log.info("[%s] %s/%s: %s windows", split, subject_dir.name, activity, window_count)

    return total_windows


@hydra.main(config_path="../configs/dataset", config_name="har_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.dataset.spectrogram.hop_ms > cfg.dataset.spectrogram.fft_window_ms:
        log.warning("hop_ms > fft_window_ms may cause information loss")

    raw_root = pathlib.Path(cfg.dataset.paths.raw)
    processed_root = pathlib.Path(cfg.dataset.paths.processed)

    if processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True)

    subject_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and "proband" in p.name])
    if len(subject_dirs) != 15:
        raise ValueError(f"Expected 15 subjects, found {len(subject_dirs)}")

    train_subjects, val_subjects = split_subjects(
        subject_dirs,
        cfg.dataset.split.num_train_subjects,
        cfg.seed,
    )
    log.info("Train: %s, Val: %s", train_subjects, val_subjects)

    class_names = sorted(set(cfg.dataset.activity_map.get(a, a) for a in cfg.dataset.activities))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    log.info("Label map: %s", label_to_idx)

    stft_params = _compute_stft_params(cfg)

    window_counts: dict[str, int] = {"train": 0, "val": 0}

    for subj_dir in subject_dirs:
        split = "train" if subj_dir.name in train_subjects else "val"
        count = process_subject(
            subj_dir,
            cfg.dataset.activities,
            label_to_idx,
            stft_params,
            cfg,
            processed_root,
            split,
        )
        window_counts[split] += count

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
        "metrics_config": OmegaConf.to_container(cfg.dataset.metrics),
        "label_map": label_to_idx,
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
        },
        "window_counts": window_counts,
    }

    with open(processed_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("%s", "=" * 50)
    log.info("Done. Total windows: %s", window_counts)
    log.info("Output: %s", processed_root)


if __name__ == "__main__":
    main()
