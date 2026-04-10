"""Deprecated raw-data cleaning shim.

The active preprocessing pipeline reads the RealWorld HAR sensor archives
directly from the raw `.zip` layout via `extract_csv_from_zip()`. The older
clean/reorganize flow expected extracted directories and is intentionally kept
as a no-op shim so stale config flags do not crash preprocessing.
"""

from __future__ import annotations

import logging
import pathlib

log = logging.getLogger(__name__)


def run_cleaning_pipeline(raw_path: pathlib.Path, sentinel_path: pathlib.Path) -> None:
    """Deprecated no-op kept for backward compatibility.

    The repository now processes the raw zip payloads in place, so there is no
    separate cleaning step to execute here.
    """
    log.warning(
        "dataset.pipeline.run_clean is deprecated and no longer mutates %s. "
        "Preprocessing now consumes the raw zip layout directly; skipping the "
        "legacy cleaning pipeline.",
        raw_path,
    )
    if sentinel_path.parent.exists():
        sentinel_path.touch(exist_ok=True)
