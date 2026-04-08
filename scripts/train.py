"""Training entry point.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py training.max_epochs=10
    uv run python scripts/train.py model.unet.base_channels=32 training.batch_size=4
    uv run python scripts/train.py logging=wandb
"""

from __future__ import annotations
import logging
import hydra
from omegaconf import DictConfig
from cgdap.training.trainer import CGDAPTrainer

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    trainer = CGDAPTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
