"""RealWorld HAR preprocessing pipeline.

This script reproduces the paper's dataset construction constraints:
- Dataset: RealWorld HAR (15 subjects)
- Sensors: accelerometer + gyroscope
- Placement: upperarm
- Activities: climbingdown, climbingup, jumping, running, walking
- Split: subject-wise, 10 train / 5 val, reproducible
- Segmentation: 2.5-second windows, using sampling rate parsed from README
- Representation: each window is transformed via torch.stft before saving

Important implementation note:
Some RealWorld archives contain nested zip files (e.g. proband4 climbingup/down).
The pipeline handles both direct CSV archives and nested recording archives.

STFT output shape per saved .pt file: (3, n_fft // 2 + 1, time_frames) complex64.
STFT parameters are read from dataset.segmentation.stft in the Hydra config.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import pathlib
import random
import re
import shutil
import zipfile
from dataclasses import dataclass
from statistics import median

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


PLACEMENT = "upperarm"


@dataclass(frozen=True)
class ArchiveRecord:
    """A single recording inside an activity archive.

    outer_zip: path to activity zip, e.g. acc_walking_csv.zip
    inner_zip_name: optional nested zip member name when the outer zip contains zips
    csv_name: CSV filename for upperarm sensor stream inside the selected archive level
    readme_name: README filename used to parse sampling frequency
    record_key: canonical key used to pair acc/gyr variants (e.g. "walking_2")
    """

    outer_zip: pathlib.Path
    inner_zip_name: str | None
    csv_name: str
    readme_name: str
    record_key: str


def _subject_id(subject_dir: pathlib.Path) -> int:
    """Extract the integer subject ID from a proband directory path."""
    return int(subject_dir.name.replace("proband", ""))


def _list_subject_dirs(raw_root: pathlib.Path) -> list[pathlib.Path]:
    """Return all 15 proband subject directories sorted by subject ID.

    Raises ValueError if the number of found subject directories is not exactly 15.
    """
    subjects = sorted(
        [
            p
            for p in raw_root.iterdir()
            if p.is_dir() and p.name.startswith("proband") and p.name.replace("proband", "").isdigit()
        ],
        key=_subject_id,
    )
    if len(subjects) != 15:
        raise ValueError(f"Expected 15 subjects in RealWorld HAR, found {len(subjects)} at {raw_root}")
    return subjects


def _find_readme_name(member_names: list[str]) -> str:
    """Return the README filename from a list of archive member names.

    Raises ValueError if no README-like file is found.
    """
    for name in member_names:
        if "read" in name.lower():
            return name
    raise ValueError("No README file found inside archive")


def _normalize_record_key(csv_name: str) -> str:
    """Derive a canonical lowercase record key from a CSV filename.

    Strips sensor-type prefixes (``acc_``, ``Gyroscope_``) and the
    ``_upperarm`` placement suffix so that accelerometer and gyroscope
    files for the same recording share the same key.
    """
    stem = pathlib.Path(csv_name).stem
    lower_stem = stem.lower()
    if lower_stem.startswith("acc_"):
        core = stem[4:]
    elif lower_stem.startswith("gyroscope_"):
        core = stem[len("Gyroscope_") :]
    else:
        core = stem
    suffix = f"_{PLACEMENT}"
    if core.endswith(suffix):
        core = core[: -len(suffix)]
    return core.lower()


def _find_upperarm_csv(member_names: list[str]) -> str:
    """Return the single upperarm CSV filename from a list of archive members.

    Raises ValueError if the number of matching entries is not exactly one.
    """
    matches = [n for n in member_names if n.lower().endswith(f"_{PLACEMENT}.csv")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one upperarm csv, found {len(matches)}: {matches}")
    return matches[0]


def _iter_archive_records(activity_zip: pathlib.Path) -> list[ArchiveRecord]:
    """Parse an activity zip and return one ArchiveRecord per recording.

    Handles two archive layouts:
    - Flat: the zip contains CSV and README files directly.
    - Nested: the zip contains inner zip files (one per recording), each
      holding its own CSV and README (e.g. proband4 climbingup/down).
    """
    records: list[ArchiveRecord] = []
    with zipfile.ZipFile(activity_zip) as outer:
        outer_members = outer.namelist()
        if outer_members and all(n.lower().endswith(".zip") for n in outer_members):
            for nested_name in sorted(outer_members):
                nested_bytes = outer.read(nested_name)
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                    members = inner.namelist()
                    csv_name = _find_upperarm_csv(members)
                    readme_name = _find_readme_name(members)
                    records.append(
                        ArchiveRecord(
                            outer_zip=activity_zip,
                            inner_zip_name=nested_name,
                            csv_name=csv_name,
                            readme_name=readme_name,
                            record_key=_normalize_record_key(csv_name),
                        )
                    )
        else:
            csv_name = _find_upperarm_csv(outer_members)
            readme_name = _find_readme_name(outer_members)
            records.append(
                ArchiveRecord(
                    outer_zip=activity_zip,
                    inner_zip_name=None,
                    csv_name=csv_name,
                    readme_name=readme_name,
                    record_key=_normalize_record_key(csv_name),
                )
            )
    return records


def _read_member_text(record: ArchiveRecord, member_name: str) -> str:
    """Read a text file member from an archive record and return its contents.

    Navigates the outer zip directly or through the nested inner zip
    depending on ``record.inner_zip_name``.
    """
    with zipfile.ZipFile(record.outer_zip) as outer:
        if record.inner_zip_name is None:
            data = outer.read(member_name)
            return data.decode("utf-8", errors="ignore")
        nested_bytes = outer.read(record.inner_zip_name)
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
            data = inner.read(member_name)
            return data.decode("utf-8", errors="ignore")


def _read_xyz_array(record: ArchiveRecord) -> np.ndarray:
    """Read the upperarm CSV from an archive record and return an (N, 3) float32 array.

    Parses ``attr_x``, ``attr_y``, and ``attr_z`` columns from the CSV.
    Raises ValueError if no samples are found.
    """
    with zipfile.ZipFile(record.outer_zip) as outer:
        if record.inner_zip_name is None:
            csv_bytes = outer.read(record.csv_name)
        else:
            nested_bytes = outer.read(record.inner_zip_name)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                csv_bytes = inner.read(record.csv_name)

    rows = csv_bytes.decode("utf-8", errors="ignore").splitlines()
    reader = csv.DictReader(rows)
    xyz = []
    for row in reader:
        xyz.append([float(row["attr_x"]), float(row["attr_y"]), float(row["attr_z"])])
    if not xyz:
        raise ValueError(f"No samples found in {record.outer_zip}::{record.csv_name}")
    return np.asarray(xyz, dtype=np.float32)


def _extract_frequency_hz(record: ArchiveRecord) -> float:
    """Parse and return the sampling frequency (Hz) from the archive's README.

    Scans all upperarm CSV entries in the README and matches the entry
    whose key corresponds to ``record.csv_name``. Falls back to the sole
    entry when only one upperarm stream is listed.

    Raises ValueError if no matching frequency can be determined.
    """
    readme = _read_member_text(record, record.readme_name)
    expected_key = _normalize_record_key(record.csv_name)

    # Parse all upperarm entries from README and select the best key match.
    pattern = r"(?P<fname>[A-Za-z0-9_]+_upperarm\.csv)\s*(?:\r?\n)+(?P<block>.*?)(?:\r?\n\r?\n|$)"
    matches = list(re.finditer(pattern, readme, flags=re.IGNORECASE | re.DOTALL))
    if not matches:
        raise ValueError(f"Could not find any upperarm entry in README for {record.outer_zip}")

    freq_by_key: dict[str, float] = {}
    for match in matches:
        fname = match.group("fname")
        freq_match = re.search(r"frequency:\s*([\d.]+)\s*Hz", match.group("block"), flags=re.IGNORECASE)
        if freq_match:
            freq_by_key[_normalize_record_key(fname)] = float(freq_match.group(1))

    if expected_key in freq_by_key:
        return freq_by_key[expected_key]
    if len(freq_by_key) == 1:
        # Assumption: README has a single upperarm stream for this nested recording.
        return next(iter(freq_by_key.values()))
    raise ValueError(f"Could not match frequency key '{expected_key}' in README entries {list(freq_by_key)}")


def _build_subject_split(subjects: list[pathlib.Path], seed: int, n_train: int, n_val: int) -> tuple[list[str], list[str]]:
    """Randomly split subjects into train and validation sets.

    Uses a seeded RNG for reproducibility. Both returned lists are sorted
    by subject ID. Raises ValueError on size mismatches or subject overlap.

    Returns:
        Tuple of (train_subject_names, val_subject_names).
    """
    subject_names = [s.name for s in subjects]
    rng = random.Random(seed)
    rng.shuffle(subject_names)

    train_subjects = sorted(subject_names[:n_train], key=lambda n: int(n.replace("proband", "")))
    val_subjects = sorted(subject_names[n_train : n_train + n_val], key=lambda n: int(n.replace("proband", "")))

    if len(train_subjects) != n_train or len(val_subjects) != n_val:
        raise ValueError(f"Invalid split sizes: train={len(train_subjects)} val={len(val_subjects)}")
    if set(train_subjects) & set(val_subjects):
        raise ValueError("Subject leakage detected between train and val splits")
    if len(train_subjects) + len(val_subjects) != len(subject_names):
        raise ValueError("Split does not cover all subjects")

    return train_subjects, val_subjects


def _sliding_windows(data: np.ndarray, win: int) -> list[np.ndarray]:
    """Segment a time-series array into non-overlapping windows of length ``win``.

    Uses a stride equal to ``win`` (no overlap), matching the paper's
    2.5-second segmentation. Returns an empty list when the data is
    shorter than one window.
    """
    # Paper specifies 2.5-second segmentation; overlap is not specified.
    # Use non-overlapping windows (stride == win).
    if len(data) < win:
        return []
    return [data[s : s + win] for s in range(0, len(data) - win + 1, win)]


def _apply_stft(
    data: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """Apply STFT to a (C, T) float32 sensor tensor.

    Applies ``torch.stft`` independently over the C channel dimension
    (torch handles 2-D batch input natively) and returns a complex64
    tensor of shape ``(C, n_fft // 2 + 1, time_frames)``.

    Args:
        data:        Sensor tensor of shape ``(C, T)``.
        n_fft:       FFT size.
        hop_length:  Hop length between successive frames.
        win_length:  Window length (must be ≤ n_fft).
        window:      Analysis window tensor of length ``win_length``.

    Returns:
        Complex STFT tensor of shape ``(C, n_fft // 2 + 1, time_frames)``.
    """
    return torch.stft(
        data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )


def _clean_activity_label(activity: str, activity_name_map: dict[str, str]) -> str:
    """Map a raw activity string to its cleaned label using the provided name map.

    Returns the original string unchanged if no mapping is found.
    """
    return activity_name_map.get(activity, activity)


def _pair_records(acc_records: list[ArchiveRecord], gyr_records: list[ArchiveRecord]) -> list[tuple[ArchiveRecord, ArchiveRecord]]:
    """Match accelerometer and gyroscope records by their canonical record key.

    Returns a list of (acc_record, gyr_record) pairs for keys present in
    both modalities. Raises ValueError on duplicate keys or no shared keys.
    """
    acc_by_key: dict[str, ArchiveRecord] = {}
    for rec in acc_records:
        if rec.record_key in acc_by_key:
            raise ValueError(f"Duplicate accelerometer record key: {rec.record_key}")
        acc_by_key[rec.record_key] = rec

    gyr_by_key: dict[str, ArchiveRecord] = {}
    for rec in gyr_records:
        if rec.record_key in gyr_by_key:
            raise ValueError(f"Duplicate gyroscope record key: {rec.record_key}")
        gyr_by_key[rec.record_key] = rec
    shared = sorted(set(acc_by_key) & set(gyr_by_key))
    if not shared:
        raise ValueError("No matching acc/gyr upperarm recordings found")
    return [(acc_by_key[k], gyr_by_key[k]) for k in shared]


def preprocess_dataset(cfg: DictConfig) -> None:
    """Run the full RealWorld HAR preprocessing pipeline.

    For each subject and activity:
    - Locates accelerometer and gyroscope zip archives.
    - Pairs recordings by canonical key.
    - Segments raw sensor data into non-overlapping windows.
    - Applies ``torch.stft`` to each window, storing a complex64 tensor of
      shape ``(3, n_fft // 2 + 1, time_frames)`` per .pt file.
    - Writes a ``metadata.json`` summary to the processed root.

    The processed directory is cleared and rebuilt on each run.
    Raw data is never modified.
    """
    raw_root = pathlib.Path(cfg.dataset.paths.raw)
    processed_root = pathlib.Path(cfg.dataset.paths.processed)

    activities = list(cfg.dataset.activities)
    placement = cfg.dataset.sensor.placement
    modalities = sorted(list(cfg.dataset.sensor.modalities))
    window_seconds = float(cfg.dataset.segmentation.window_seconds)
    n_train = int(cfg.dataset.split.num_train_subjects)
    n_val = int(cfg.dataset.split.num_val_subjects)

    stft_cfg = cfg.dataset.segmentation.stft
    stft_n_fft = int(stft_cfg.n_fft)
    stft_hop_length = int(stft_cfg.hop_length)
    stft_win_length = int(stft_cfg.win_length)
    stft_window_fn: str = str(stft_cfg.window)
    if stft_window_fn == "hann":
        stft_window = torch.hann_window(stft_win_length)
    elif stft_window_fn == "hamming":
        stft_window = torch.hamming_window(stft_win_length)
    else:
        raise ValueError(f"Unsupported STFT window function: '{stft_window_fn}'")

    subjects = _list_subject_dirs(raw_root)
    train_subjects, val_subjects = _build_subject_split(subjects, int(cfg.seed), n_train, n_val)
    split_by_subject = {s: "train" for s in train_subjects}
    split_by_subject.update({s: "val" for s in val_subjects})

    if processed_root.exists():
        # Keep behavior explicit and safe: remove and rebuild processed output.
        # Raw data is never modified.
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    activity_name_map = OmegaConf.to_container(cfg.dataset.activity_name_map, resolve=True)
    if not isinstance(activity_name_map, dict):
        raise ValueError("dataset.activity_name_map must be a mapping")

    per_subject_freqs: dict[str, list[float]] = {s.name: [] for s in subjects}
    total_windows = {"train": 0, "val": 0}
    total_pairs = {"train": 0, "val": 0}

    for subject_dir in subjects:
        subject_name = subject_dir.name
        split = split_by_subject[subject_name]

        for activity in activities:
            acc_zip = subject_dir / "data" / f"acc_{activity}_csv.zip"
            gyr_zip = subject_dir / "data" / f"gyr_{activity}_csv.zip"
            if not acc_zip.exists() or not gyr_zip.exists():
                raise FileNotFoundError(f"Missing required zip for {subject_name} {activity}")

            acc_records = _iter_archive_records(acc_zip)
            gyr_records = _iter_archive_records(gyr_zip)
            paired_records = _pair_records(acc_records, gyr_records)
            clean_activity = _clean_activity_label(activity, activity_name_map)
            out_dir = processed_root / split / clean_activity
            out_dir.mkdir(parents=True, exist_ok=True)

            for pair_idx, (acc_record, gyr_record) in enumerate(paired_records):
                freq_hz = _extract_frequency_hz(acc_record)
                per_subject_freqs[subject_name].append(freq_hz)
                win_samples = int(round(window_seconds * freq_hz))
                if win_samples <= 0:
                    raise ValueError(f"Invalid window samples ({win_samples}) for {subject_name} {activity}")

                acc_xyz = _read_xyz_array(acc_record)
                gyr_xyz = _read_xyz_array(gyr_record)
                n = min(len(acc_xyz), len(gyr_xyz))
                if n < win_samples:
                    continue

                acc_wins = _sliding_windows(acc_xyz[:n], win_samples)
                gyr_wins = _sliding_windows(gyr_xyz[:n], win_samples)
                n_wins = min(len(acc_wins), len(gyr_wins))
                if n_wins == 0:
                    continue

                total_pairs[split] += 1
                for w_idx in range(n_wins):
                    stem = f"{subject_name}_{acc_record.record_key}_{pair_idx:02d}_{w_idx:05d}"
                    acc_tensor = torch.from_numpy(acc_wins[w_idx].T)  # (3, T)
                    gyr_tensor = torch.from_numpy(gyr_wins[w_idx].T)  # (3, T)
                    acc_stft = _apply_stft(acc_tensor, stft_n_fft, stft_hop_length, stft_win_length, stft_window)
                    gyr_stft = _apply_stft(gyr_tensor, stft_n_fft, stft_hop_length, stft_win_length, stft_window)
                    torch.save(acc_stft, out_dir / f"{stem}_acc.pt")
                    torch.save(gyr_stft, out_dir / f"{stem}_gyr.pt")
                total_windows[split] += n_wins

                log.info(
                    "%s %-8s %-12s key=%-15s freq=%.2fHz win=%d windows=%d",
                    split,
                    subject_name,
                    activity,
                    acc_record.record_key,
                    freq_hz,
                    win_samples,
                    n_wins,
                )

    metadata = {
        "seed": int(cfg.seed),
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "activities": activities,
        "sensors": modalities,
        "placement": placement,
        "window_seconds": window_seconds,
        "stft": {
            "n_fft": stft_n_fft,
            "hop_length": stft_hop_length,
            "win_length": stft_win_length,
            "window": stft_window_fn,
            "freq_bins": stft_n_fft // 2 + 1,
        },
        "split": {"train_subjects": train_subjects, "val_subjects": val_subjects},
        "window_counts": total_windows,
        "paired_recording_counts": total_pairs,
        "subject_frequency_summary_hz": {
            subject: {
                "n_values": len(freqs),
                "min": min(freqs) if freqs else None,
                "median": median(freqs) if freqs else None,
                "max": max(freqs) if freqs else None,
            }
            for subject, freqs in per_subject_freqs.items()
        },
    }
    (processed_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log.info("Done. Processed dataset written to: %s", processed_root)
    log.info("Train subjects: %s", train_subjects)
    log.info("Val subjects: %s", val_subjects)
    log.info("Window counts: train=%d val=%d", total_windows["train"], total_windows["val"])


@hydra.main(config_path="../configs/dataset", config_name="har_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point: log the resolved config and run the preprocessing pipeline."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    preprocess_dataset(cfg)


if __name__ == "__main__":
    main()
