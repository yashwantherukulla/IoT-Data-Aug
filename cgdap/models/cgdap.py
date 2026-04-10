"""MultimodalCGDAP: the top-level training/inference wrapper.

Composes:
    - MetricExtractor         : differentiable metrics from spectrogram
    - BaseConditionEmbedder   : encode metrics + labels into condition tokens
    - BaseDenoiser (x2)       : one per modality (acc, gyr)
    - BaseNoiseSchedule       : forward/reverse diffusion

Training forward pass:
    1. Sample shared timestep t for each batch
    2. For each modality:
        a. Embed condition (metrics from batch + one-hot labels)
        b. q_sample: add noise -> x_t
        c. Denoiser: predict noise
        d. L_G = MSE(pred_noise, noise)   [diffusion loss]
        e. predict_x0 from pred_noise
        f. Re-extract metrics from x0_hat
        g. L_metric_i = MSE(extracted_i, target_i) per metric
    3. Adaptive weighting: L_total = L_G + sum_i w_i * L_metric_i

Sampling:
    Given metric targets + label per modality, run reverse loop.
    Reproducible but decorrelated noise streams across modalities.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from cgdap.metrics.extractor import MetricExtractor
from cgdap.models.base import (
    BaseConditionEmbedder,
    BaseDenoiser,
    BaseNoiseSchedule,
    get_denoiser,
    get_embedder,
    get_schedule,
)

# Ensure registrations are loaded
import cgdap.models.condition  # noqa: F401
import cgdap.models.ddpm       # noqa: F401
import cgdap.models.unet       # noqa: F401

log = logging.getLogger(__name__)


class MultimodalCGDAP(nn.Module):
    """Architecture-agnostic multimodal diffusion wrapper.

    Creates separate denoiser instances for each modality; they do NOT share
    weights but share the same noise schedule and condition embedder.

    Args:
        modalities:     list of modality names, e.g. ["acc", "gyr"]
        denoiser_cls:   BaseDenoiser subclass
        schedule_cls:   BaseNoiseSchedule subclass
        embedder_cls:   BaseConditionEmbedder subclass
        denoiser_kwargs: passed to each denoiser constructor
        schedule_kwargs: passed to schedule constructor
        embedder_kwargs: passed to embedder constructor
        metric_weight_init: initial w_i for each metric loss term
        target_ratio:   L_G : L_metric target ratio for adaptive weighting
        adaptive_start_epoch: epoch at which adaptive reweighting begins
        weight_min/max: clamp range for dynamic weights
        n_metrics:      number of metric scalars per modality
    """

    def __init__(
        self,
        modalities: list[str],
        denoiser_cls: type[BaseDenoiser],
        schedule_cls: type[BaseNoiseSchedule],
        embedder_cls: type[BaseConditionEmbedder],
        denoiser_kwargs: dict[str, Any],
        schedule_kwargs: dict[str, Any],
        embedder_kwargs: dict[str, Any],
        metric_weight_init: float = 0.1,
        target_ratio: float = 10.0,
        adaptive_start_epoch: int = 1,
        metric_weight_ema_decay: float = 0.9,
        weight_min: float = 0.01,
        weight_max: float = 10.0,
        n_metrics: int = 5,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.n_metrics = n_metrics
        self.target_ratio = target_ratio
        self.adaptive_start_epoch = adaptive_start_epoch
        self.metric_weight_ema_decay = metric_weight_ema_decay
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.current_epoch: int = 0

        # One denoiser per modality (independent weights)
        self.denoisers = nn.ModuleDict({
            mod: denoiser_cls(**denoiser_kwargs)
            for mod in modalities
        })

        # Shared schedule and embedder
        self.schedule: BaseNoiseSchedule = schedule_cls(**schedule_kwargs)
        self.embedder: BaseConditionEmbedder = embedder_cls(**embedder_kwargs)

        # Shared metric extractor
        self.metric_extractor = MetricExtractor(
            harmonics=embedder_kwargs.get("harmonics", None),
            softmax_temp=embedder_kwargs.get("softmax_temp", 0.1),
        )

        # Adaptive metric weights (one per metric, shared across modalities)
        self.register_buffer(
            "metric_weights",
            torch.full((n_metrics,), metric_weight_init),
        )
        self.register_buffer(
            "metric_loss_ema",
            torch.full((n_metrics,), float("nan")),
        )

    # ------------------------------------------------------------------
    # Class method factory from Hydra config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MultimodalCGDAP":
        """Build from full Hydra config."""
        from cgdap.models.condition import CrossAttentionConditionEmbedder
        from cgdap.models.ddpm import DDPMSchedule
        from cgdap.models.unet import ConditionalUNet

        modalities = list(cfg.model.modalities)
        n_metrics = len(cfg.dataset.metrics.names)
        n_classes = len(cfg.dataset.activities)

        denoiser_cls = get_denoiser(cfg.model.denoiser)
        schedule_cls = get_schedule(cfg.model.schedule)
        embedder_cls = get_embedder(cfg.model.embedder)

        u = cfg.model.unet
        c = cfg.model.condition
        d = cfg.model.ddpm
        t = cfg.training

        denoiser_kwargs = dict(
            in_channels=int(cfg.model.in_channels),
            base_channels=int(u.base_channels),
            channel_multipliers=list(u.channel_multipliers),
            n_res_blocks=int(u.n_res_blocks),
            temb_dim=int(u.temb_dim),
            d_model=int(c.d_model),
            n_heads=int(c.n_heads),
            cross_attn_depths=list(u.cross_attn_depths),
            dropout=float(u.dropout),
        )
        schedule_kwargs = dict(
            train_timesteps=int(d.train_timesteps),
            beta_start=float(d.beta_start),
            beta_end=float(d.beta_end),
            num_train_steps=int(d.num_train_steps),
            num_infer_steps=int(d.num_infer_steps),
        )
        embedder_kwargs = dict(
            n_metrics=n_metrics,
            n_classes=n_classes,
            d_model=int(c.d_model),
            n_heads=int(c.n_heads),
            n_cond_tokens=int(c.n_cond_tokens),
            dropout=float(c.dropout),
        )

        loss_cfg = t.loss
        return cls(
            modalities=modalities,
            denoiser_cls=denoiser_cls,
            schedule_cls=schedule_cls,
            embedder_cls=embedder_cls,
            denoiser_kwargs=denoiser_kwargs,
            schedule_kwargs=schedule_kwargs,
            embedder_kwargs=embedder_kwargs,
            metric_weight_init=float(loss_cfg.metric_weight_init),
            target_ratio=float(loss_cfg.target_ratio),
            adaptive_start_epoch=int(loss_cfg.adaptive_start_epoch),
            metric_weight_ema_decay=float(loss_cfg.metric_weight_ema_decay),
            weight_min=float(loss_cfg.weight_min),
            weight_max=float(loss_cfg.weight_max),
            n_metrics=n_metrics,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _labels_to_onehot(self, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
        return F.one_hot(labels, num_classes=n_classes).float()

    def _update_metric_weights(self, l_g: torch.Tensor, per_metric_losses: torch.Tensor) -> None:
        """Adaptively rescale metric weights after adaptive_start_epoch."""
        if not self.training:
            return
        if self.current_epoch < self.adaptive_start_epoch:
            return
        detached_losses = per_metric_losses.detach()
        if torch.isnan(self.metric_loss_ema).any():
            self.metric_loss_ema.copy_(detached_losses)
        else:
            self.metric_loss_ema.mul_(self.metric_weight_ema_decay).add_(
                detached_losses,
                alpha=1.0 - self.metric_weight_ema_decay,
            )

        stable_losses = self.metric_loss_ema.clamp(min=1e-12)
        weights = (l_g.detach() / (stable_losses * self.target_ratio)).clamp(
            self.weight_min,
            self.weight_max,
        )
        self.metric_weights.copy_(weights)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Any],
        n_classes: int,
        timesteps: torch.Tensor | None = None,
        noises: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss over all modalities.

        Args:
            batch: from PairedDataset or dict with per-modality tensors
                   Expected keys per modality: spectrogram [B,3,F,T], metrics [B,5]
                   Plus: label [B]

        Returns:
            loss_dict with keys: L_G, L_metric, L_total, and per-metric losses
        """
        device = next(self.parameters()).device
        labels = batch["label"].to(device)               # [B]
        n_classes_actual = n_classes
        labels_oh = self._labels_to_onehot(labels, n_classes_actual)

        B = labels.shape[0]
        # Sample shared timestep across modalities unless validation overrides it.
        t = timesteps if timesteps is not None else self.schedule.sample_timesteps(B, device)
        t = t.to(device)

        accum_l_g = torch.zeros(1, device=device)
        accum_metric_losses = torch.zeros(self.n_metrics, device=device)

        for mod in self.modalities:
            mod_data = batch[mod]
            x0 = mod_data["spectrogram"].to(device).float()       # [B,3,F,T]
            metrics_target = mod_data["metrics"].to(device).float()  # [B,5]

            # Embed condition
            condition = self.embedder(metrics_target, labels_oh)   # [B, n_tokens, d_model]

            # Forward diffusion
            noise_override = None if noises is None else noises.get(mod)
            if noise_override is not None:
                noise_override = noise_override.to(device)
            x_t, noise = self.schedule.q_sample(x0, t, noise=noise_override)

            # Predict noise
            pred_noise = self.denoisers[mod](x_t, t, condition)

            # Diffusion loss
            l_g = F.mse_loss(pred_noise, noise)
            accum_l_g = accum_l_g + l_g

            # Metric consistency loss via x0_hat
            x0_hat = self.schedule.predict_x0(x_t, t, pred_noise)
            extracted = self.metric_extractor(x0_hat)    # [B, 5]

            for i in range(self.n_metrics):
                l_m_i = F.mse_loss(extracted[:, i], metrics_target[:, i])
                accum_metric_losses[i] = accum_metric_losses[i] + l_m_i

        # Average over modalities
        n_mod = len(self.modalities)
        l_g_avg = accum_l_g / n_mod
        metric_losses_avg = accum_metric_losses / n_mod

        # Update adaptive weights
        self._update_metric_weights(l_g_avg, metric_losses_avg)

        # Total loss
        l_metric = (self.metric_weights * metric_losses_avg).sum()
        l_total = l_g_avg + l_metric

        loss_dict: dict[str, torch.Tensor] = {
            "L_G": l_g_avg,
            "L_metric": l_metric,
            "L_total": l_total,
        }
        for i, name in enumerate(MetricExtractor.METRIC_NAMES):
            loss_dict[f"L_metric_{name}"] = metric_losses_avg[i]

        return loss_dict

    # ------------------------------------------------------------------
    # Inference / sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        metric_targets: dict[str, torch.Tensor],  # {mod: [B, 5]}
        labels: torch.Tensor,                      # [B] int
        n_classes: int,
        spec_shape: tuple[int, int, int],          # (C, F, T)
        device: torch.device,
        seed: int | None = None,
        num_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate spectrograms for all modalities.

        A base seed can be provided for reproducibility; each modality receives
        a derived offset seed so the reverse-process noise streams stay
        decorrelated.

        Returns:
            {modality: [B, C, F, T]}
        """
        B = labels.shape[0]
        labels_oh = self._labels_to_onehot(labels.to(device), n_classes)

        generated: dict[str, torch.Tensor] = {}
        for mod_idx, mod in enumerate(self.modalities):
            mets = metric_targets[mod].to(device).float()
            condition = self.embedder(mets, labels_oh)
            mod_seed = None if seed is None else seed + mod_idx
            x_gen = self.schedule.sample_loop(
                denoiser=self.denoisers[mod],
                shape=(B, *spec_shape),
                condition=condition,
                device=device,
                seed=mod_seed,
                num_steps=num_steps,
            )
            generated[mod] = x_gen

        return generated
