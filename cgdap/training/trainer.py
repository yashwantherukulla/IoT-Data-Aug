"""Hydra-driven CGDAP training loop."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

from cgdap.data.dataset import build_label_map, make_paired_loader
from cgdap.models.cgdap import MultimodalCGDAP

try:
    from tqdm.auto import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    class _NullProgress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *_args, **_kwargs) -> None:
            return None

    def tqdm(iterable, *args, **kwargs):  # type: ignore[no-redef]
        return _NullProgress(iterable)

    def trange(*args, **kwargs):  # type: ignore[no-redef]
        return _NullProgress(range(*args))

log = logging.getLogger(__name__)


class ExperimentLogger:
    """Thin logging wrapper that keeps console and W&B runs on one code path."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.backend = str(cfg.logging.backend).lower()
        self.run: Any | None = None

        if self.backend == "console":
            return
        if self.backend != "wandb":
            raise ValueError(f"Unknown logging backend: {self.backend!r}")

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging requested, but `wandb` is not installed. "
                "Run `uv sync` after pulling the latest changes."
            ) from exc

        save_dir = pathlib.Path(cfg.logging.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        init_kwargs = {
            "project": cfg.logging.project,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "dir": str(save_dir),
            "mode": str(cfg.logging.get("mode", "online")),
            "name": cfg.logging.get("name") or cfg.experiment_name,
        }
        for field in ("entity", "group", "notes", "tags"):
            value = cfg.logging.get(field)
            if value not in (None, ""):
                init_kwargs[field] = value

        self.run = wandb.init(**init_kwargs)
        wandb.define_metric("epoch")
        wandb.define_metric("train_step/global_step")
        wandb.define_metric("train_step/*", step_metric="train_step/global_step")
        wandb.define_metric("train_epoch/*", step_metric="epoch")
        wandb.define_metric("val_epoch/*", step_metric="epoch")

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        import wandb

        wandb.log(metrics, step=step)

    def finish(self) -> None:
        if not self.enabled:
            return
        import wandb

        wandb.finish()
        self.run = None


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
        if not TQDM_AVAILABLE:
            log.warning("`tqdm` is not installed; progress bars are disabled until you run `uv sync`.")
        self.experiment_logger = ExperimentLogger(cfg)
        self.global_step = 0

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
        self.log_every: int = int(cfg.logging.get("log_every_n_steps", cfg.training.log_every_n_steps))
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

        progress = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"Train {epoch + 1}/{self.max_epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in progress:
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
            self.global_step += 1
            lr = float(self.optimizer.param_groups[0]["lr"])
            progress.set_postfix(
                loss=f"{loss_dict['L_total'].item():.4f}",
                lg=f"{loss_dict['L_G'].item():.4f}",
                lm=f"{loss_dict['L_metric'].item():.4f}",
                lr=f"{lr:.2e}",
            )

            self.experiment_logger.log(
                {
                    "epoch": epoch,
                    "train_step/global_step": self.global_step,
                    "train_step/L_total": loss_dict["L_total"].item(),
                    "train_step/L_G": loss_dict["L_G"].item(),
                    "train_step/L_metric": loss_dict["L_metric"].item(),
                    "train_step/lr": lr,
                },
                step=self.global_step,
            )

        if self.scheduler is not None:
            self.scheduler.step()

        if steps == 0:
            raise RuntimeError("Training loader produced zero batches.")
        return {k: v / steps for k, v in totals.items()}

    def _build_validation_controls(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = int(batch["label"].shape[0])
        generator = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed) + batch_idx)
        timesteps = torch.randint(
            0,
            self.model.schedule.train_timesteps,
            (batch_size,),
            generator=generator,
            device=self.device,
        )
        noises = {
            mod: torch.randn(
                batch[mod]["spectrogram"].shape,
                generator=generator,
                device=self.device,
                dtype=batch[mod]["spectrogram"].dtype,
            )
            for mod in self.cfg.dataset.modalities
        }
        return timesteps, noises

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        steps = 0

        progress = tqdm(
            self.val_loader,
            total=len(self.val_loader),
            desc=f"Val {epoch + 1}/{self.max_epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(progress):
            for mod in self.cfg.dataset.modalities:
                batch[mod]["spectrogram"] = batch[mod]["spectrogram"].to(self.device)
                batch[mod]["metrics"] = batch[mod]["metrics"].to(self.device)
            batch["label"] = batch["label"].to(self.device)

            timesteps, noises = self._build_validation_controls(batch, batch_idx)
            loss_dict = self.model(
                batch,
                n_classes=self.n_classes,
                timesteps=timesteps,
                noises=noises,
            )
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            steps += 1
            progress.set_postfix(
                loss=f"{loss_dict['L_total'].item():.4f}",
                lg=f"{loss_dict['L_G'].item():.4f}",
                lm=f"{loss_dict['L_metric'].item():.4f}",
            )

        if steps == 0:
            raise RuntimeError("Validation loader produced zero batches.")
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
        epoch_progress = trange(self.max_epochs, desc="Epochs", dynamic_ncols=True)
        try:
            for epoch in epoch_progress:
                train_metrics = self.train_epoch(epoch)
                train_log = {f"train_epoch/{k}": v for k, v in train_metrics.items()}
                train_log["epoch"] = epoch
                train_log["train_epoch/lr"] = float(self.optimizer.param_groups[0]["lr"])
                self.experiment_logger.log(train_log, step=self.global_step)
                log.info(
                    "Epoch %d | Train L_total=%.4f L_G=%.4f L_metric=%.4f",
                    epoch, train_metrics["L_total"], train_metrics["L_G"], train_metrics["L_metric"],
                )

                display_metrics = {"train": f"{train_metrics['L_total']:.4f}"}
                if epoch % self.val_every == 0:
                    val_metrics = self.val_epoch(epoch)
                    val_log = {f"val_epoch/{k}": v for k, v in val_metrics.items()}
                    val_log["epoch"] = epoch
                    self.experiment_logger.log(val_log, step=self.global_step)
                    log.info(
                        "Epoch %d | Val   L_total=%.4f L_G=%.4f L_metric=%.4f",
                        epoch, val_metrics["L_total"], val_metrics["L_G"], val_metrics["L_metric"],
                    )
                    display_metrics["val"] = f"{val_metrics['L_total']:.4f}"

                epoch_progress.set_postfix(display_metrics)

                if epoch % self.save_every == 0 or epoch == self.max_epochs - 1:
                    self.save_checkpoint(epoch, train_metrics)
        finally:
            self.experiment_logger.finish()

        log.info("Training complete.")
