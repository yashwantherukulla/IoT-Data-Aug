"""Hydra-driven CGDAP training loop."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

from cgdap.data.dataset import build_label_map, make_modality_loader, make_paired_loader
from cgdap.models.cgdap import MultimodalCGDAP

log = logging.getLogger(__name__)


def build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> optim.Optimizer:
    opt_cfg = cfg.training.optimizer
    name = opt_cfg.name.lower()
    params = model.parameters()
    if name == "adam":
        return optim.Adam(
            params,
            lr=float(opt_cfg.lr),
            betas=tuple(opt_cfg.betas),
            eps=float(opt_cfg.eps),
            weight_decay=float(opt_cfg.weight_decay),
        )
    if name == "adamw":
        return optim.AdamW(
            params,
            lr=float(opt_cfg.lr),
            weight_decay=float(opt_cfg.weight_decay),
        )
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig):
    sch_cfg = cfg.training.scheduler
    name = sch_cfg.name.lower()
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(sch_cfg.T_max),
            eta_min=float(sch_cfg.eta_min),
        )
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name!r}")


class CGDAPTrainer:
    """End-to-end CGDAP trainer.

    Handles:
        - Dataset / DataLoader construction
        - Model construction from config
        - Training loop with adaptive metric weighting
        - Periodic checkpointing and logging
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s", self.device)

        # Paths
        processed_root = pathlib.Path(cfg.dataset.paths.processed)
        train_dir = processed_root / "train"
        val_dir = processed_root / "val"

        modalities: list[str] = list(cfg.dataset.modalities)
        n_classes = len(cfg.dataset.activities)

        # Label map from first modality of train
        self.label_map = build_label_map(train_dir, modality=modalities[0])
        log.info("Label map: %s", self.label_map)

        ldr_cfg = cfg.dataset.loader
        # Paired loaders (for training synchronization)
        self.train_loader = make_paired_loader(
            train_dir, modalities, self.label_map,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=ldr_cfg.num_workers,
            pin_memory=ldr_cfg.pin_memory,
            drop_last=ldr_cfg.drop_last,
        )
        self.val_loader = make_paired_loader(
            val_dir, modalities, self.label_map,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=ldr_cfg.num_workers,
            pin_memory=ldr_cfg.pin_memory,
            drop_last=False,
        )

        # Model
        self.model = MultimodalCGDAP.from_config(cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("Model parameters: %s", f"{n_params:,}")

        # Optimizer / scheduler
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        self.max_epochs: int = cfg.training.max_epochs
        self.log_every: int = cfg.training.log_every_n_steps
        self.save_every: int = cfg.training.save_every_n_epochs
        self.val_every: int = cfg.training.val_every_n_epochs
        self.n_classes: int = n_classes

        ckpt_dir = pathlib.Path(cfg.training.checkpoint_dir) / cfg.experiment_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.model.current_epoch = epoch
        totals: dict[str, float] = {}
        steps = 0

        for batch in self.train_loader:
            # Move to device
            for mod in self.cfg.dataset.modalities:
                batch[mod]["spectrogram"] = batch[mod]["spectrogram"].to(self.device)
                batch[mod]["metrics"] = batch[mod]["metrics"].to(self.device)
            batch["label"] = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            loss_dict = self.model(batch, n_classes=self.n_classes)
            loss_dict["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            steps += 1

            if steps % self.log_every == 0:
                log.info(
                    "Epoch %d step %d | L_total=%.4f L_G=%.4f L_metric=%.4f",
                    epoch, steps,
                    loss_dict["L_total"].item(),
                    loss_dict["L_G"].item(),
                    loss_dict["L_metric"].item(),
                )

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: v / steps for k, v in totals.items()}

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        steps = 0

        for batch in self.val_loader:
            for mod in self.cfg.dataset.modalities:
                batch[mod]["spectrogram"] = batch[mod]["spectrogram"].to(self.device)
                batch[mod]["metrics"] = batch[mod]["metrics"].to(self.device)
            batch["label"] = batch["label"].to(self.device)

            loss_dict = self.model(batch, n_classes=self.n_classes)
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            steps += 1

        return {k: v / steps for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        path = self.ckpt_dir / f"ckpt_epoch{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        log.info("Checkpoint saved: %s", path)

    def run(self) -> None:
        log.info("=" * 60)
        log.info("Starting CGDAP training | experiment: %s", self.cfg.experiment_name)
        log.info("Config:\n%s", OmegaConf.to_yaml(self.cfg))
        log.info("=" * 60)

        for epoch in range(self.max_epochs):
            train_metrics = self.train_epoch(epoch)
            log.info(
                "Epoch %d | Train L_total=%.4f L_G=%.4f L_metric=%.4f",
                epoch, train_metrics["L_total"], train_metrics["L_G"], train_metrics["L_metric"],
            )

            if epoch % self.val_every == 0:
                val_metrics = self.val_epoch(epoch)
                log.info(
                    "Epoch %d | Val   L_total=%.4f L_G=%.4f L_metric=%.4f",
                    epoch, val_metrics["L_total"], val_metrics["L_G"], val_metrics["L_metric"],
                )

            if epoch % self.save_every == 0 or epoch == self.max_epochs - 1:
                self.save_checkpoint(epoch, train_metrics)

        log.info("Training complete.")
