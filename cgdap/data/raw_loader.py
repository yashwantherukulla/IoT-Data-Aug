"""Raw HAR data cleaning and reorganization.

Run this once after a fresh dataset download by setting:
    dataset.pipeline.run_clean: true

Steps performed (idempotent -- skipped if sentinel file exists):
    1. delete images and videos subdirectories per proband
    2. keep only acc_* / gyr_* csv zips in each proband/data/ folder
    3. filter to allowed activities only
    4. filter to upperarm placement only
    5. extract frequency info from readMe.txt -> info.json
    6. write sentinel file so the step is skipped on re-runs
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import shutil

log = logging.getLogger(__name__)

ALLOWED_ACTIVITIES = frozenset(
    {"climbingup", "climbingdown", "jumping", "running", "walking"}
)


def clean_data(data_path: pathlib.Path) -> None:
    """Delete images and videos dirs; keep only acc/gyr csv zips."""
    for proband_dir in sorted(data_path.iterdir()):
        if not proband_dir.is_dir():
            continue
        for dir_name in ("images", "videos"):
            target = proband_dir / dir_name
            if target.exists():
                shutil.rmtree(target)
                log.info("Deleted: %s", target)
        data_dir = proband_dir / "data"
        if data_dir.exists():
            for entry in list(data_dir.iterdir()):
                keep = entry.name.startswith(("acc", "gyr")) and "csv" in entry.name
                if not keep:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                    log.info("Deleted: %s", entry)


def reorganize_data(data_path: pathlib.Path) -> None:
    """Move acc_*/gyr_* zips from proband/data/ up to proband/ directly."""
    for proband_dir in sorted(data_path.iterdir()):
        if not proband_dir.is_dir():
            continue
        data_dir = proband_dir / "data"
        if not data_dir.exists():
            continue
        for entry in list(data_dir.iterdir()):
            if entry.name.startswith(("acc", "gyr")):
                target = proband_dir / entry.name
                shutil.move(str(entry), str(target))
                log.info("Moved: %s -> %s", entry, target)
        if data_dir.exists() and not any(data_dir.iterdir()):
            data_dir.rmdir()
            log.info("Deleted empty dir: %s", data_dir)


def filter_activities(data_path: pathlib.Path) -> None:
    """Keep only zip folders whose activity name is in ALLOWED_ACTIVITIES."""
    for proband_dir in sorted(data_path.iterdir()):
        if not proband_dir.is_dir():
            continue
        for entry in sorted(proband_dir.iterdir()):
            if not entry.is_dir() or not entry.name.startswith(("acc_", "gyr_")):
                continue
            parts = entry.name.split("_")
            activity = parts[1] if len(parts) >= 2 else ""
            if activity not in ALLOWED_ACTIVITIES:
                shutil.rmtree(entry)
                log.info("Deleted: %s", entry)


def filter_upperarm(data_path: pathlib.Path) -> None:
    """Inside each acc_*_csv folder, keep only *_upperarm.csv and readMe."""
    for proband_dir in sorted(data_path.iterdir()):
        if not proband_dir.is_dir():
            continue
        for sensor_dir in sorted(proband_dir.iterdir()):
            if not sensor_dir.is_dir() or not sensor_dir.name.startswith("acc_"):
                continue
            for entry in list(sensor_dir.iterdir()):
                if entry.is_dir():
                    continue
                is_upperarm = entry.suffix == ".csv" and entry.stem.endswith("_upperarm")
                is_readme = entry.name.lower().startswith("readme")
                if not is_upperarm and not is_readme:
                    entry.unlink()
                    log.info("Deleted: %s", entry)
                elif is_readme and entry.name != "readMe.txt":
                    new_path = entry.parent / "readMe.txt"
                    entry.rename(new_path)
                    log.info("Renamed: %s -> readMe.txt", entry.name)


def extract_freq_info(data_path: pathlib.Path) -> None:
    """Read readMe.txt to find upperarm sample frequency; write info.json."""
    pattern = re.compile(
        r"acc_[^_]+_upperarm\.csv\s*\n(?:.*\n)*?>\s*frequency:\s*([\d.]+)\s*Hz"
    )
    for proband_dir in sorted(data_path.iterdir()):
        if not proband_dir.is_dir():
            continue
        for acc_dir in sorted(proband_dir.iterdir()):
            if not acc_dir.is_dir() or not acc_dir.name.startswith("acc_"):
                continue
            readme_path = acc_dir / "readMe.txt"
            if not readme_path.exists():
                log.warning("No readMe.txt in %s, skipping", acc_dir)
                continue
            content = readme_path.read_text(encoding="utf-8")
            match = pattern.search(content)
            if match:
                freq = float(match.group(1))
                info_path = acc_dir / "info.json"
                info_path.write_text(json.dumps({"freq": freq}, indent=4), encoding="utf-8")
                log.info("Written: %s (freq=%.1f Hz)", info_path, freq)
                readme_path.unlink()
            else:
                log.warning("Could not find upperarm frequency in %s", readme_path)


def run_cleaning_pipeline(raw_path: pathlib.Path, sentinel_path: pathlib.Path) -> None:
    """Run all cleaning steps in order. Writes sentinel on success."""
    if sentinel_path.exists():
        log.info("Raw data already cleaned (sentinel: %s). Skipping.", sentinel_path)
        return
    log.info("=== Starting raw data cleaning pipeline ===")
    log.info("Step 1/5: clean_data")
    clean_data(raw_path)
    log.info("Step 2/5: reorganize_data")
    reorganize_data(raw_path)
    log.info("Step 3/5: filter_activities")
    filter_activities(raw_path)
    log.info("Step 4/5: filter_upperarm")
    filter_upperarm(raw_path)
    log.info("Step 5/5: extract_freq_info")
    extract_freq_info(raw_path)
    sentinel_path.touch()
    log.info("=== Raw data cleaning complete. Sentinel written: %s ===", sentinel_path)
