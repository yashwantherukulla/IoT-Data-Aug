"""Preprocessing: raw HAR zip files -> per-modality .pt files.

Output layout:
    processed/HAR/
    ├── metadata.json
    ├── train/
    │   ├── acc/<activity>/<subject>_<activity>_<idx:05d>.pt
    │   └── gyr/<activity>/...
    └── val/
        ├── acc/...
        └── gyr/...

Each .pt payload:
    {
        "spectrogram": Tensor[3, F, T],   # 3-channel xyz log-magnitude
        "metrics":     Tensor[5],          # per-modality differentiable metrics
        "label":       int,
        "activity":    str,
        "subject":     str,
        "window_index": int,
        "sample_rate_hz": float,
        "freq_axis_hz":   Tensor[F],
        "time_axis_s":    Tensor[T],
    }
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

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from cgdap.metrics.extractor import compute_metrics_fn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STFT parameter dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class STFTParams:
    win_length: int
    n_fft: int
    hop_length: int
    window_samples: int


def build_stft_params(cfg: DictConfig) -> STFTParams:
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
        "STFT params @ %.0fHz: win=%d n_fft=%d hop=%d window_samples=%d",
        sr, win_length, n_fft, hop_length, window_samples,
    )
    return STFTParams(win_length, n_fft, hop_length, window_samples)


# ---------------------------------------------------------------------------
# Window factory
# ---------------------------------------------------------------------------


def _make_window(window_type: str, length: int) -> torch.Tensor:
    if window_type == "hann":
        return torch.hann_window(length)
    if window_type == "hamming":
        return torch.hamming_window(length)
    raise ValueError(f"Unknown window type: {window_type!r}")


# ---------------------------------------------------------------------------
# CSV extraction from zip
# ---------------------------------------------------------------------------


def extract_csv_from_zip(zip_path: pathlib.Path, placement: str = "upperarm") -> np.ndarray:
    """Return (N, 3) float32 array from acc or gyr zip."""
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if all(m.endswith(".zip") for m in members):
            inner_bytes = zf.read(members[0])
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zf:
                csv_name = _find_csv(inner_zf.namelist(), placement)
                csv_bytes = inner_zf.read(csv_name)
        else:
            csv_name = _find_csv(members, placement)
            csv_bytes = zf.read(csv_name)
    return _parse_csv(csv_bytes)


def _find_csv(members: list[str], placement: str) -> str:
    matches = [m for m in members if m.endswith(f"_{placement}.csv")]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 {placement} CSV, found {len(matches)}: {matches}")
    return matches[0]


def _parse_csv(csv_bytes: bytes) -> np.ndarray:
    text = csv_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(text.splitlines())
    xyz = [[float(r["attr_x"]), float(r["attr_y"]), float(r["attr_z"])] for r in reader]
    return np.asarray(xyz, dtype=np.float32)


# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------


def compute_spectrogram(
    signal: torch.Tensor,      # [3, N]
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    power: float,
    use_log1p: bool,
) -> torch.Tensor:
    """Return [3, F, T] spectrogram tensor."""
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
    hop_length: int,
    n_freq: int,
    n_time: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    freq = torch.linspace(0.0, sample_rate / 2.0, n_freq)
    window_dur = ((n_time - 1) * hop_length) / sample_rate
    time = torch.linspace(0.0, window_dur, n_time)
    return freq, time


# ---------------------------------------------------------------------------
# Windowing helper
# ---------------------------------------------------------------------------


def segment_windows(data: np.ndarray, window_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(data) - window_size + 1, window_size):
        yield data[start : start + window_size]


# ---------------------------------------------------------------------------
# Subject splitting
# ---------------------------------------------------------------------------


def split_subjects(
    subject_dirs: list[pathlib.Path],
    n_train: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    names = sorted(d.name for d in subject_dirs)
    rng = random.Random(seed)
    rng.shuffle(names)
    train = sorted(names[:n_train])
    val = sorted(names[n_train:])
    return train, val


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------


def process_subject(
    subject_dir: pathlib.Path,
    activities: list[str],
    label_to_idx: dict[str, int],
    stft_params: STFTParams,
    cfg: DictConfig,
    out_root: pathlib.Path,
    split: str,
    modalities: list[str],
) -> dict[str, int]:
    """Process all activities for one subject. Returns {modality: window_count}."""
    window = _make_window(cfg.dataset.spectrogram.window, stft_params.win_length)
    sr: float = cfg.dataset.sample_rate_hz
    metric_cfg = cfg.dataset.metrics

    counts: dict[str, int] = {m: 0 for m in modalities}

    spec_kwargs = dict(
        n_fft=stft_params.n_fft,
        hop_length=stft_params.hop_length,
        win_length=stft_params.win_length,
        window=window,
        power=cfg.dataset.spectrogram.power,
        use_log1p=cfg.dataset.spectrogram.log1p,
    )

    for activity in activities:
        acc_zip = subject_dir / "data" / f"acc_{activity}_csv.zip"
        gyr_zip = subject_dir / "data" / f"gyr_{activity}_csv.zip"

        if not (acc_zip.exists() and gyr_zip.exists()):
            log.warning("Missing zip for %s/%s, skipping", subject_dir.name, activity)
            continue

        try:
            acc_data = extract_csv_from_zip(acc_zip)
            gyr_data = extract_csv_from_zip(gyr_zip)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to extract %s/%s: %s", subject_dir.name, activity, exc)
            continue

        n_samples = min(len(acc_data), len(gyr_data))
        if n_samples < stft_params.window_samples:
            log.warning(
                "Insufficient samples for %s/%s: %d < %d",
                subject_dir.name, activity, n_samples, stft_params.window_samples,
            )
            continue

        acc_data = acc_data[:n_samples]
        gyr_data = gyr_data[:n_samples]

        label_name = cfg.dataset.activity_map.get(activity, activity)
        label_idx = label_to_idx[label_name]

        # Create output dirs for each modality
        modality_data = {"acc": acc_data, "gyr": gyr_data}
        out_dirs: dict[str, pathlib.Path] = {}
        for mod in modalities:
            d = out_root / split / mod / label_name
            d.mkdir(parents=True, exist_ok=True)
            out_dirs[mod] = d

        raw_windows = list(
            zip(
                segment_windows(acc_data, stft_params.window_samples),
                segment_windows(gyr_data, stft_params.window_samples),
            )
        )

        for w_idx, (acc_win, gyr_win) in enumerate(raw_windows):
            raw_by_mod = {"acc": acc_win, "gyr": gyr_win}
            base_name = f"{subject_dir.name}_{activity}_{w_idx:05d}.pt"

            for mod in modalities:
                win_np = raw_by_mod[mod]
                signal = torch.from_numpy(win_np.T)           # [3, window_samples]
                spec = compute_spectrogram(signal, **spec_kwargs)  # [3, F, T]

                n_freq, n_time = spec.shape[1], spec.shape[2]
                freq_axis, time_axis = make_axes(sr, stft_params.hop_length, n_freq, n_time)

                metrics = compute_metrics_fn(spec.mean(0), metric_cfg)   # [5]

                torch.save(
                    {
                        "spectrogram": spec,           # [3, F, T]
                        "metrics": metrics,            # [5]
                        "label": int(label_idx),
                        "activity": label_name,
                        "subject": subject_dir.name,
                        "window_index": w_idx,
                        "sample_rate_hz": float(sr),
                        "freq_axis_hz": freq_axis,
                        "time_axis_s": time_axis,
                    },
                    out_dirs[mod] / base_name,
                )
                counts[mod] += 1

        log.info("[%s] %s/%s: %d windows", split, subject_dir.name, activity, len(raw_windows))

    return counts


# ---------------------------------------------------------------------------
# Main entry (called from scripts/prepare_dataset.py)
# ---------------------------------------------------------------------------


def run_preprocessing(cfg: DictConfig) -> None:
    """Full preprocessing pipeline: raw -> processed per-modality .pt files."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    raw_root = pathlib.Path(cfg.dataset.paths.raw)
    processed_root = pathlib.Path(cfg.dataset.paths.processed)
    modalities: list[str] = list(cfg.dataset.modalities)

    # Optionally run raw data cleaning
    if cfg.dataset.pipeline.run_clean:
        from cgdap.data.raw_loader import run_cleaning_pipeline
        sentinel = pathlib.Path(cfg.dataset.pipeline.clean_sentinel)
        run_cleaning_pipeline(raw_root, sentinel)

    # Clear stale processed artifacts
    if processed_root.exists() and cfg.dataset.pipeline.force_regenerate:
        log.info("Removing stale processed data: %s", processed_root)
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir() and "proband" in p.name)
    if len(subject_dirs) != 15:
        raise ValueError(f"Expected 15 subjects, found {len(subject_dirs)}")

    train_subjects, val_subjects = split_subjects(
        subject_dirs,
        cfg.dataset.split.num_train_subjects,
        cfg.seed,
    )
    log.info("Train subjects: %s", train_subjects)
    log.info("Val   subjects: %s", val_subjects)

    class_names = sorted(set(cfg.dataset.activity_map.get(a, a) for a in cfg.dataset.activities))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    log.info("Label map: %s", label_to_idx)

    stft_params = build_stft_params(cfg)

    total_counts: dict[str, dict[str, int]] = {"train": {m: 0 for m in modalities}, "val": {m: 0 for m in modalities}}

    for subj_dir in subject_dirs:
        split = "train" if subj_dir.name in train_subjects else "val"
        mod_counts = process_subject(
            subj_dir,
            list(cfg.dataset.activities),
            label_to_idx,
            stft_params,
            cfg,
            processed_root,
            split,
            modalities,
        )
        for mod, cnt in mod_counts.items():
            total_counts[split][mod] += cnt

    # Compute STFT output dimensions from one sample
    _sample_t = torch.zeros(stft_params.window_samples)
    _sample_stft = torch.stft(
        _sample_t,
        n_fft=stft_params.n_fft,
        hop_length=stft_params.hop_length,
        win_length=stft_params.win_length,
        window=torch.hann_window(stft_params.win_length),
        center=True,
        return_complex=True,
    )
    n_freq, n_time = _sample_stft.shape

    metadata = {
        "contract_version": "2.1",
        "seed": int(cfg.seed),
        "sample_rate_hz": float(cfg.dataset.sample_rate_hz),
        "window_seconds": float(cfg.dataset.window_seconds),
        "modalities": modalities,
        "metric_names": list(cfg.dataset.metrics.names),
        "stft_params": {
            "win_length": stft_params.win_length,
            "n_fft": stft_params.n_fft,
            "hop_length": stft_params.hop_length,
            "window_samples": stft_params.window_samples,
            "n_freq": n_freq,
            "n_time": n_time,
        },
        "spectrogram_config": OmegaConf.to_container(cfg.dataset.spectrogram),
        "metrics_config": OmegaConf.to_container(cfg.dataset.metrics),
        "label_map": label_to_idx,
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
        },
        "window_counts": {
            split: dict(counts)
            for split, counts in total_counts.items()
        },
    }

    with open(processed_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("=" * 60)
    log.info("Done. Window counts: %s", total_counts)
    log.info("Output: %s", processed_root)
