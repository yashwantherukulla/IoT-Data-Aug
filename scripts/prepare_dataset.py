"""Dataset preparation entry point.

Usage:
    # First-time setup (clean raw data + preprocess):
    uv run python scripts/prepare_dataset.py dataset.pipeline.run_clean=true

    # Re-preprocess only (raw already cleaned):
    uv run python scripts/prepare_dataset.py

    # Dry-run without deleting old data:
    uv run python scripts/prepare_dataset.py dataset.pipeline.force_regenerate=false
"""

from __future__ import annotations
import logging
import hydra
from omegaconf import DictConfig
from cgdap.data.preprocessing import run_preprocessing

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_preprocessing(cfg)

if __name__ == "__main__":
    main()
