"""Base abstract classes for all CGDAP model components.

These interfaces define the contracts that every architecture must fulfill.
To add a new architecture (DiT, LDM, Flow Matching, etc.), sub-class the
appropriate base and register it in the model registry.

Hierarchy:
    BaseNoiseSchedule   -- DDPM | FlowMatchSchedule (future)
    BaseConditionEmbedder -- CrossAttentionConditionEmbedder | AdaLNEmbedder (future)
    BaseDenoiser        -- ConditionalUNet | DiTDenoiser (future) | LatentDenoiser (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------


class BaseNoiseSchedule(nn.Module, ABC):
    """Abstract noise / flow schedule.

    Implementations: DDPMSchedule, FlowMatchSchedule (future).
    """

    @abstractmethod
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t from x0 at timestep t.

        Returns:
            (x_t, noise) -- noisy sample and the noise that was added
        """

    @abstractmethod
    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct x0 estimate from x_t and predicted noise."""

    @abstractmethod
    def sample_loop(
        self,
        denoiser: "BaseDenoiser",
        shape: tuple[int, ...],
        condition: torch.Tensor,
        device: torch.device,
        seed: int | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Full reverse / sampling loop. Returns x0 estimate."""

    @abstractmethod
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random training timesteps [B]."""


# ---------------------------------------------------------------------------
# Condition Embedder
# ---------------------------------------------------------------------------


class BaseConditionEmbedder(nn.Module, ABC):
    """Abstract condition embedder.

    Takes per-modality metrics and class labels; produces a condition
    representation that BaseDenoiser can consume.

    Implementations: CrossAttentionConditionEmbedder, AdaLNEmbedder (future).
    """

    @abstractmethod
    def forward(
        self,
        metrics: torch.Tensor,       # [B, n_metrics]
        labels_onehot: torch.Tensor, # [B, n_classes]
    ) -> torch.Tensor:
        """Return condition tokens / embedding.

        Returns:
            For cross-attention: [B, n_tokens, d_model]
            For AdaLN / label emb: [B, d_model]
        """


# ---------------------------------------------------------------------------
# Denoiser (the noise-prediction network)
# ---------------------------------------------------------------------------


class BaseDenoiser(nn.Module, ABC):
    """Abstract denoising network.

    Implementations: ConditionalUNet, DiTDenoiser (future), LatentDenoiser (future).
    """

    @abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,          # [B, C, F, T] noisy spectrogram
        t: torch.Tensor,             # [B] timestep indices
        condition: torch.Tensor,     # output of BaseConditionEmbedder
    ) -> torch.Tensor:
        """Predict noise epsilon. Output shape must match x_t."""


# ---------------------------------------------------------------------------
# Simple model registry
# ---------------------------------------------------------------------------

_DENOISER_REGISTRY: dict[str, type[BaseDenoiser]] = {}
_SCHEDULE_REGISTRY: dict[str, type[BaseNoiseSchedule]] = {}
_EMBEDDER_REGISTRY: dict[str, type[BaseConditionEmbedder]] = {}


def register_denoiser(name: str):
    """Decorator to register a BaseDenoiser subclass."""
    def decorator(cls: type[BaseDenoiser]) -> type[BaseDenoiser]:
        _DENOISER_REGISTRY[name] = cls
        return cls
    return decorator


def register_schedule(name: str):
    """Decorator to register a BaseNoiseSchedule subclass."""
    def decorator(cls: type[BaseNoiseSchedule]) -> type[BaseNoiseSchedule]:
        _SCHEDULE_REGISTRY[name] = cls
        return cls
    return decorator


def register_embedder(name: str):
    """Decorator to register a BaseConditionEmbedder subclass."""
    def decorator(cls: type[BaseConditionEmbedder]) -> type[BaseConditionEmbedder]:
        _EMBEDDER_REGISTRY[name] = cls
        return cls
    return decorator


def get_denoiser(name: str) -> type[BaseDenoiser]:
    if name not in _DENOISER_REGISTRY:
        raise KeyError(f"Denoiser {name!r} not registered. Available: {list(_DENOISER_REGISTRY)}")
    return _DENOISER_REGISTRY[name]


def get_schedule(name: str) -> type[BaseNoiseSchedule]:
    if name not in _SCHEDULE_REGISTRY:
        raise KeyError(f"Schedule {name!r} not registered. Available: {list(_SCHEDULE_REGISTRY)}")
    return _SCHEDULE_REGISTRY[name]


def get_embedder(name: str) -> type[BaseConditionEmbedder]:
    if name not in _EMBEDDER_REGISTRY:
        raise KeyError(f"Embedder {name!r} not registered. Available: {list(_EMBEDDER_REGISTRY)}")
    return _EMBEDDER_REGISTRY[name]
