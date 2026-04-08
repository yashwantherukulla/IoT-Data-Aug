"""Differentiable metric extraction for HAR spectrograms.

MetricExtractor (nn.Module)
    Batched forward: spec [B, 3, F, T] -> metrics [B, 5]
    Differentiable through all five metrics.

compute_metrics_fn
    Functional API for use in preprocessing (no batch dim).
    spec_2d [F, T] -> metrics [5]

Metric definitions (paper-aligned):
    0  temporal_range  : max - min of frequency-mean amplitude over time
    1  f0_amplitude    : HPS with bilinear downsampling + soft-argmax
    2  contrast        : mean(top-5%) - mean(bottom-5%) of all bins
    3  flatness        : geometric_mean / arithmetic_mean
    4  entropy         : Shannon entropy H = -sum(p * log2(p))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

EPS_DEFAULT: float = 1e-10


# ---------------------------------------------------------------------------
# Functional metric implementations (operate on [F, T] tensors)
# ---------------------------------------------------------------------------


def metric_temporal_range(spec_2d: torch.Tensor, eps: float = EPS_DEFAULT) -> torch.Tensor:
    """Max - min of mean amplitude over frequency bins, across time.

    spec_2d: [F, T]
    """
    mean_over_freq = spec_2d.mean(dim=0)  # [T]
    return mean_over_freq.max() - mean_over_freq.min()


def metric_f0_amplitude(
    spec_2d: torch.Tensor,
    harmonics: list[float] | None = None,
    softmax_temp: float = 0.1,
    eps: float = EPS_DEFAULT,
) -> torch.Tensor:
    """F0 amplitude via Harmonic Product Spectrum with bilinear downsampling.

    Uses harmonic ratios (default [1.0, 0.5, 0.25]) to downsample via
    bilinear interpolation -- preserving gradient flow -- and multiplies spectra.
    Soft-argmax finds the peak frequency; returns the weighted average amplitude.

    spec_2d: [F, T]
    """
    if harmonics is None:
        harmonics = [1.0, 0.5, 0.25]

    mean_over_time = spec_2d.mean(dim=-1)  # [F]
    hps = mean_over_time.clone()
    F_orig = mean_over_time.shape[0]

    for ratio in harmonics[1:]:
        target_len = max(1, int(round(F_orig * ratio)))
        # Bilinear interp (4D input required: [B, C, H, W])
        down = F.interpolate(
            mean_over_time.view(1, 1, F_orig, 1),
            size=(target_len, 1),
            mode="bilinear",
            align_corners=False,
        ).view(target_len)
        L = min(hps.shape[0], down.shape[0])
        hps = hps[:L] * down[:L]

    # Soft-argmax over hps to get differentiable frequency peak index
    weights = torch.softmax(hps / (softmax_temp + eps), dim=0)  # [L]
    # Weighted average of frequency-bin amplitudes from original spec
    L = weights.shape[0]
    return (spec_2d[:L] * weights.unsqueeze(-1)).sum(dim=0).mean()


def metric_contrast(
    spec_2d: torch.Tensor,
    tail_ratio: float = 0.05,
    eps: float = EPS_DEFAULT,
) -> torch.Tensor:
    """Mean(top-k%) - mean(bottom-k%) over all spectrogram bins.

    spec_2d: [F, T]
    """
    flat = spec_2d.reshape(-1)
    n = flat.numel()
    k = max(1, int(n * tail_ratio))
    sorted_vals, _ = torch.sort(flat)
    valleys = sorted_vals[:k].mean()
    peaks = sorted_vals[-k:].mean()
    return peaks - valleys


def metric_flatness(spec_2d: torch.Tensor, eps: float = EPS_DEFAULT) -> torch.Tensor:
    """Spectral flatness: geometric_mean / arithmetic_mean.

    spec_2d: [F, T]
    """
    x = spec_2d.clamp_min(eps)
    g_mean = torch.exp(torch.log(x).mean())
    a_mean = x.mean()
    return g_mean / (a_mean + eps)


def metric_entropy(spec_2d: torch.Tensor, eps: float = EPS_DEFAULT) -> torch.Tensor:
    """Shannon entropy (base-2) over normalized spectrogram bins.

    spec_2d: [F, T]
    """
    x = spec_2d.clamp_min(eps)
    p = x / x.sum()
    return -(p * torch.log2(p + eps)).sum()


# ---------------------------------------------------------------------------
# Functional interface for use in preprocessing (no nn.Module overhead)
# ---------------------------------------------------------------------------


def compute_metrics_fn(spec_2d: torch.Tensor, metric_cfg: DictConfig) -> torch.Tensor:
    """Compute all 5 metrics from a [F, T] spectrogram. Returns [5] tensor."""
    harmonics = list(metric_cfg.hps_harmonics) if hasattr(metric_cfg, "hps_harmonics") else [1.0, 0.5, 0.25]
    temp = float(getattr(metric_cfg, "hps_softmax_temp", 0.1))
    tail = float(getattr(metric_cfg, "contrast_tail_ratio", 0.05))
    eps = float(getattr(metric_cfg, "eps", EPS_DEFAULT))

    return torch.stack([
        metric_temporal_range(spec_2d, eps=eps),
        metric_f0_amplitude(spec_2d, harmonics=harmonics, softmax_temp=temp, eps=eps),
        metric_contrast(spec_2d, tail_ratio=tail, eps=eps),
        metric_flatness(spec_2d, eps=eps),
        metric_entropy(spec_2d, eps=eps),
    ]).to(torch.float32)


# ---------------------------------------------------------------------------
# nn.Module interface (batched, differentiable, used in training)
# ---------------------------------------------------------------------------


class MetricExtractor(nn.Module):
    """Batched differentiable metric extraction.

    Args:
        harmonics: HPS harmonic ratios, default [1.0, 0.5, 0.25]
        softmax_temp: temperature for soft-argmax F0 peak finding
        contrast_tail_ratio: fraction used for contrast top/bottom
        eps: numerical stability epsilon for log/division ops
    """

    METRIC_NAMES = ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"]

    def __init__(
        self,
        harmonics: list[float] | None = None,
        softmax_temp: float = 0.1,
        contrast_tail_ratio: float = 0.05,
        eps: float = EPS_DEFAULT,
    ) -> None:
        super().__init__()
        self.harmonics = harmonics or [1.0, 0.5, 0.25]
        self.softmax_temp = softmax_temp
        self.contrast_tail_ratio = contrast_tail_ratio
        self.eps = eps

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MetricExtractor":
        mcfg = cfg.dataset.metrics
        return cls(
            harmonics=list(mcfg.hps_harmonics),
            softmax_temp=float(mcfg.hps_softmax_temp),
            contrast_tail_ratio=float(mcfg.contrast_tail_ratio),
            eps=float(mcfg.eps),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Extract per-sample metrics.

        Args:
            spec: [B, C, F, T]  (C channels, e.g. 3 for xyz)

        Returns:
            metrics: [B, 5]
        """
        # Average over channels -> [B, F, T]
        spec_2d = spec.mean(dim=1)

        B = spec_2d.shape[0]
        results = []
        for b in range(B):
            s = spec_2d[b]  # [F, T]
            m = torch.stack([
                metric_temporal_range(s, eps=self.eps),
                metric_f0_amplitude(s, harmonics=self.harmonics, softmax_temp=self.softmax_temp, eps=self.eps),
                metric_contrast(s, tail_ratio=self.contrast_tail_ratio, eps=self.eps),
                metric_flatness(s, eps=self.eps),
                metric_entropy(s, eps=self.eps),
            ])
            results.append(m)

        return torch.stack(results).to(torch.float32)   # [B, 5]

    def extra_repr(self) -> str:
        return (
            f"harmonics={self.harmonics}, softmax_temp={self.softmax_temp}, "
            f"contrast_tail={self.contrast_tail_ratio}, eps={self.eps}"
        )
