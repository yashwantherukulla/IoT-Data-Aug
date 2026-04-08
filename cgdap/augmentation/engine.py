"""Augmentation engine: three metric-space augmentation modes.

Modes
-----
interpolation
    Sample two same-label samples; draw mixing coefficient beta ~ TruncNormal.
    Interpolate per-modality metric vectors: m_aug = beta*m1 + (1-beta)*m2.

disturbance
    Take real-sample metrics; perturb each dimension by U(-r_i, r_i)%.

domain_instruction
    Sample metric targets directly from expert-defined ranges per activity.
    Does NOT reference real samples.

All modes produce:
    {"acc": Tensor[5], "gyr": Tensor[5], "label": int}
Which is passed to MultimodalCGDAP.sample(...) to generate spectrograms.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
from omegaconf import DictConfig

METRIC_NAMES = ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"]


# ---------------------------------------------------------------------------
# Truncated normal sampler (for interpolation beta)
# ---------------------------------------------------------------------------


def _truncated_normal(mean: float, std: float, low: float, high: float) -> float:
    """Sample from TruncNormal(mean, std, [low, high]) via rejection."""
    for _ in range(100):
        v = random.gauss(mean, std)
        if low <= v <= high:
            return v
    return mean  # fallback


# ---------------------------------------------------------------------------
# Per-mode augmentation functions
# ---------------------------------------------------------------------------


def augment_interpolation(
    samples_by_label: dict[int, list[dict[str, Any]]],
    modalities: list[str],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Sample two same-label samples, interpolate per-modality metrics."""
    aug_cfg = cfg.augmentation.interpolation
    # Pick a random label that has at least 2 samples
    eligible = [lbl for lbl, s in samples_by_label.items() if len(s) >= 2]
    if not eligible:
        raise ValueError("Need >= 2 samples per label for interpolation augmentation.")

    label = random.choice(eligible)
    s1, s2 = random.sample(samples_by_label[label], 2)

    beta = _truncated_normal(
        mean=float(aug_cfg.beta_mean),
        std=float(aug_cfg.beta_std),
        low=float(aug_cfg.beta_low),
        high=float(aug_cfg.beta_high),
    )

    metrics_aug: dict[str, torch.Tensor] = {}
    for mod in modalities:
        m1 = s1[mod]["metrics"].float()
        m2 = s2[mod]["metrics"].float()
        metrics_aug[mod] = beta * m1 + (1.0 - beta) * m2

    return {**metrics_aug, "label": label}


def augment_disturbance(
    sample: dict[str, Any],
    modalities: list[str],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Perturb each metric of a real sample by a configured range."""
    dist_cfg = cfg.augmentation.disturbance

    ranges = [
        float(getattr(dist_cfg, name, 0.1))
        for name in METRIC_NAMES
    ]

    metrics_aug: dict[str, torch.Tensor] = {}
    for mod in modalities:
        m = sample[mod]["metrics"].float().clone()
        for i, r in enumerate(ranges):
            noise = torch.empty(1).uniform_(-r, r).item()
            m[i] = m[i] * (1.0 + noise)
        metrics_aug[mod] = m

    return {**metrics_aug, "label": int(sample["label"])}


def augment_domain_instruction(
    activity: str,
    modalities: list[str],
    cfg: DictConfig,
    label_map: dict[str, int],
) -> dict[str, Any]:
    """Sample metrics directly from expert-defined domain ranges."""
    domain_cfg = cfg.augmentation.domain_instruction

    if activity not in domain_cfg:
        available = list(domain_cfg.keys())
        raise ValueError(f"Activity {activity!r} not in domain_instruction config. Available: {available}")

    act_ranges = domain_cfg[activity]
    metric_sample_targets = []
    for name in METRIC_NAMES:
        lo, hi = act_ranges[name][0], act_ranges[name][1]
        val = random.uniform(float(lo), float(hi))
        metric_sample_targets.append(val)

    metrics_tensor = torch.tensor(metric_sample_targets, dtype=torch.float32)

    # Same targets for all modalities (domain ranges are activity-level)
    metrics_aug = {mod: metrics_tensor.clone() for mod in modalities}
    return {**metrics_aug, "label": label_map[activity]}


# ---------------------------------------------------------------------------
# AugmentationEngine
# ---------------------------------------------------------------------------


class AugmentationEngine:
    """Unified interface for all three augmentation modes.

    Args:
        cfg:       full Hydra config (augmentation sub-tree used)
        modalities: list of modality names
        label_map:  {activity_name: label_int}
    """

    def __init__(
        self,
        cfg: DictConfig,
        modalities: list[str],
        label_map: dict[str, int],
    ) -> None:
        self.cfg = cfg
        self.modalities = modalities
        self.label_map = label_map
        self.mode: str = cfg.augmentation.mode

        # For interpolation mode: samples indexed by (label, modality)
        self._sample_cache: dict[int, list[dict[str, Any]]] = {}

    def register_samples(self, samples: list[dict[str, Any]]) -> None:
        """Pre-load samples for interpolation mode (call after dataset loaded).

        Each sample should be a PairedDataset item with both modality keys.
        """
        self._sample_cache.clear()
        for s in samples:
            lbl = int(s["label"])
            if lbl not in self._sample_cache:
                self._sample_cache[lbl] = []
            self._sample_cache[lbl].append(s)

    def generate_targets(
        self,
        sample: dict[str, Any] | None = None,
        activity: str | None = None,
    ) -> dict[str, Any]:
        """Generate metric targets according to the configured mode.

        Args:
            sample:   real paired sample (required for interpolation / disturbance)
            activity: activity label string (required for domain_instruction)

        Returns:
            {modality: Tensor[5], "label": int}
        """
        if self.mode == "interpolation":
            if not self._sample_cache:
                raise RuntimeError("Call register_samples() before using interpolation mode.")
            return augment_interpolation(self._sample_cache, self.modalities, self.cfg)

        elif self.mode == "disturbance":
            if sample is None:
                raise ValueError("disturbance mode requires a real sample.")
            return augment_disturbance(sample, self.modalities, self.cfg)

        elif self.mode == "domain_instruction":
            if activity is None:
                raise ValueError("domain_instruction mode requires an activity name.")
            return augment_domain_instruction(activity, self.modalities, self.cfg, self.label_map)

        else:
            raise ValueError(f"Unknown augmentation mode: {self.mode!r}")
