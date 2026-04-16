"""Lightweight per-epoch augmentation quality diagnostics."""

from __future__ import annotations

import logging
import pathlib
import random
from collections import defaultdict
from typing import Any

import torch
from omegaconf import DictConfig

from cgdap.augmentation.engine import AugmentationEngine
from cgdap.data.dataset import PairedDataset
from cgdap.metrics.extractor import MetricExtractor

log = logging.getLogger(__name__)


def select_stratified_indices(
    activities: list[str],
    *,
    samples_per_activity: int,
    seed: int,
) -> list[int]:
    """Select a deterministic stratified subset of dataset indices."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, activity in enumerate(activities):
        grouped[str(activity)].append(idx)

    rng = random.Random(seed)
    selected: list[int] = []
    for activity in sorted(grouped):
        candidates = grouped[activity]
        k = min(samples_per_activity, len(candidates))
        if k == len(candidates):
            chosen = candidates
        else:
            chosen = rng.sample(candidates, k)
        selected.extend(sorted(chosen))
    return selected


def _standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1.0e-6)


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.mean().item())


def _safe_std_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0e-6)


def _concat_pair_metrics(sample: dict[str, Any], modalities: list[str]) -> torch.Tensor:
    return torch.cat([sample[mod]["metrics"].float() for mod in modalities], dim=0)


class ProductEvaluator:
    """Probe-based augmentation diagnostics for the live training model."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        label_map: dict[str, int],
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.modalities = list(cfg.dataset.modalities)
        self.label_map = dict(label_map)
        self.label_to_activity = {label: activity for activity, label in self.label_map.items()}

        self.eval_cfg = cfg.evaluation.product_eval
        self.split = str(self.eval_cfg.split)
        self.samples_per_probe = int(self.eval_cfg.samples_per_probe)
        self.z_score_threshold = float(self.eval_cfg.z_score_threshold)
        self.seed = int(self.eval_cfg.seed)
        self.num_steps = self._resolve_num_steps()

        processed_root = pathlib.Path(cfg.dataset.paths.processed)
        self.train_dataset = PairedDataset(processed_root / "train", self.modalities, self.label_map)
        self.split_dataset = PairedDataset(processed_root / self.split, self.modalities, self.label_map)

        if len(self.train_dataset) == 0:
            raise RuntimeError("Training dataset is empty; product evaluation cannot be initialized.")
        if len(self.split_dataset) == 0:
            raise RuntimeError(f"Product-eval dataset split {self.split!r} is empty.")

        self.real_banks = {
            "train": self._build_real_bank(self.train_dataset, keep_metric_samples=True),
            self.split: self._build_real_bank(self.split_dataset, keep_metric_samples=False),
        }
        self.probe_samples = self._build_probe_samples(self.split_dataset)
        self.augmentation_engine = AugmentationEngine(cfg, self.modalities, self.label_map)
        if str(cfg.augmentation.mode) == "interpolation":
            self.augmentation_engine.register_samples(self.real_banks["train"]["metric_samples"])

        log.info(
            "Initialized product evaluator | split=%s probe_samples=%d num_steps=%d",
            self.split,
            len(self.probe_samples),
            self.num_steps,
        )

    def _resolve_num_steps(self) -> int:
        raw_num_steps = self.eval_cfg.get("num_steps")
        if raw_num_steps is None:
            raw_num_steps = self.cfg.evaluation.augmentation.get("num_steps")
        return int(raw_num_steps) if raw_num_steps is not None else 25

    def _build_real_bank(
        self,
        dataset: PairedDataset,
        *,
        keep_metric_samples: bool,
    ) -> dict[str, Any]:
        pair_metrics: list[torch.Tensor] = []
        labels: list[int] = []
        activities: list[str] = []
        metric_samples: list[dict[str, Any]] = []

        for idx in range(len(dataset)):
            sample = dataset[idx]
            pair_metrics.append(_concat_pair_metrics(sample, self.modalities))
            labels.append(int(sample["label"]))
            activities.append(str(sample["activity"]))
            if keep_metric_samples:
                metric_samples.append(
                    {
                        "label": int(sample["label"]),
                        "activity": str(sample["activity"]),
                        **{
                            modality: {"metrics": sample[modality]["metrics"].float().clone()}
                            for modality in self.modalities
                        },
                    }
                )

        pair_tensor = torch.stack(pair_metrics).float()
        label_tensor = torch.tensor(labels, dtype=torch.long)
        global_mean = pair_tensor.mean(dim=0)
        global_std = pair_tensor.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        pair_standardized = _standardize(pair_tensor, global_mean, global_std)

        label_indices: dict[int, torch.Tensor] = {}
        label_means: dict[int, torch.Tensor] = {}
        label_stds: dict[int, torch.Tensor] = {}
        label_centroids: dict[int, torch.Tensor] = {}
        for label in sorted(set(labels)):
            indices = torch.nonzero(label_tensor == label, as_tuple=False).squeeze(1)
            label_indices[label] = indices
            label_pairs = pair_tensor.index_select(0, indices)
            label_means[label] = label_pairs.mean(dim=0)
            label_stds[label] = label_pairs.std(dim=0, unbiased=False).clamp_min(1.0e-6)
            label_centroids[label] = pair_standardized.index_select(0, indices).mean(dim=0)

        return {
            "pairs": pair_tensor,
            "pairs_standardized": pair_standardized,
            "labels": label_tensor,
            "activities": activities,
            "mean": global_mean,
            "std": global_std,
            "label_indices": label_indices,
            "label_means": label_means,
            "label_stds": label_stds,
            "label_centroids": label_centroids,
            "metric_samples": metric_samples,
        }

    def _build_probe_samples(self, dataset: PairedDataset) -> list[dict[str, Any]]:
        activities = [str(dataset[idx]["activity"]) for idx in range(len(dataset))]
        probe_indices = select_stratified_indices(
            activities,
            samples_per_activity=int(self.eval_cfg.samples_per_activity),
            seed=self.seed,
        )
        return [dataset[idx] for idx in probe_indices]

    def _generate_targets(self, probe_sample: dict[str, Any]) -> dict[str, Any]:
        mode = str(self.cfg.augmentation.mode)
        if mode == "disturbance":
            return self.augmentation_engine.generate_targets(sample=probe_sample)
        if mode == "domain_instruction":
            return self.augmentation_engine.generate_targets(activity=str(probe_sample["activity"]))
        return self.augmentation_engine.generate_targets()

    def _nearest_neighbor_stats(
        self,
        synth_standardized: torch.Tensor,
        synth_labels: torch.Tensor,
        bank: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distances = torch.empty(synth_standardized.shape[0], dtype=torch.float32)
        nearest_indices = torch.empty(synth_standardized.shape[0], dtype=torch.long)

        for label in torch.unique(synth_labels, sorted=True):
            mask = synth_labels == label
            synth_subset = synth_standardized[mask]
            bank_indices = bank["label_indices"][int(label.item())]
            bank_subset = bank["pairs_standardized"].index_select(0, bank_indices)
            pairwise = torch.cdist(synth_subset, bank_subset)
            min_dist, min_local = pairwise.min(dim=1)
            distances[mask] = min_dist.cpu()
            nearest_indices[mask] = bank_indices[min_local].cpu()
        return distances, nearest_indices

    def _build_wandb_payload(
        self,
        *,
        pair_error: torch.Tensor,
        val_nn_distances: torch.Tensor,
        train_nn_distances: torch.Tensor,
    ) -> dict[str, Any]:
        try:
            import wandb
        except ImportError:
            return {}

        payload: dict[str, Any] = {}

        if bool(self.eval_cfg.log_histograms):
            payload["hist/nn_distance_val"] = wandb.Histogram(val_nn_distances.numpy())
            payload["hist/nn_distance_train"] = wandb.Histogram(train_nn_distances.numpy())
            payload["hist/pair_error"] = wandb.Histogram(pair_error.reshape(-1).numpy())

        return payload

    @torch.no_grad()
    def evaluate(
        self,
        model: Any,
        *,
        enable_artifacts: bool = False,
    ) -> dict[str, Any]:
        if not self.probe_samples:
            raise RuntimeError("Product evaluator probe set is empty.")

        was_training = bool(model.training)
        model.eval()

        pair_targets: list[torch.Tensor] = []
        pair_generated: list[torch.Tensor] = []
        synth_labels: list[int] = []

        for sample_idx, probe_sample in enumerate(self.probe_samples):
            for repeat_idx in range(self.samples_per_probe):
                targets = self._generate_targets(probe_sample)
                label = int(targets["label"])
                activity = self.label_to_activity[label]
                spec_shape = tuple(probe_sample[self.modalities[0]]["spectrogram"].shape)
                metric_targets = {
                    modality: targets[modality].unsqueeze(0).to(self.device)
                    for modality in self.modalities
                }
                generated = model.sample(
                    metric_targets=metric_targets,
                    labels=torch.tensor([label], dtype=torch.long, device=self.device),
                    n_classes=len(self.label_map),
                    spec_shape=spec_shape,
                    device=self.device,
                    seed=self.seed + sample_idx * self.samples_per_probe + repeat_idx,
                    num_steps=self.num_steps,
                )

                target_parts: list[torch.Tensor] = []
                generated_parts: list[torch.Tensor] = []
                for modality in self.modalities:
                    generated_metrics = model.metric_extractor(generated[modality]).squeeze(0).detach().cpu().float()
                    target_metrics = targets[modality].detach().cpu().float()
                    target_parts.append(target_metrics)
                    generated_parts.append(generated_metrics)

                pair_targets.append(torch.cat(target_parts, dim=0))
                pair_generated.append(torch.cat(generated_parts, dim=0))
                synth_labels.append(label)

        target_pairs = torch.stack(pair_targets).float()
        generated_pairs = torch.stack(pair_generated).float()
        labels_tensor = torch.tensor(synth_labels, dtype=torch.long)

        split_bank = self.real_banks[self.split]
        train_bank = self.real_banks["train"]
        generated_pairs_standardized = _standardize(generated_pairs, split_bank["mean"], split_bank["std"])

        val_nn_distances, val_nn_indices = self._nearest_neighbor_stats(
            generated_pairs_standardized,
            labels_tensor,
            split_bank,
        )
        train_nn_distances, _ = self._nearest_neighbor_stats(
            _standardize(generated_pairs, train_bank["mean"], train_bank["std"]),
            labels_tensor,
            train_bank,
        )

        within_band_masks: list[torch.Tensor] = []
        centroid_drifts: list[float] = []
        activity_pair_rmse: dict[str, float] = {}
        for label in torch.unique(labels_tensor, sorted=True):
            mask = labels_tensor == label
            activity = self.label_to_activity[int(label.item())]
            label_generated = generated_pairs[mask]
            label_targets = target_pairs[mask]
            label_mean = split_bank["label_means"][int(label.item())]
            label_std = split_bank["label_stds"][int(label.item())]
            band_lo = label_mean - self.z_score_threshold * label_std
            band_hi = label_mean + self.z_score_threshold * label_std
            within_band_masks.append((label_generated >= band_lo) & (label_generated <= band_hi))
            label_generated_std = generated_pairs_standardized[mask]
            label_centroid = label_generated_std.mean(dim=0)
            centroid_drifts.append(
                float(torch.norm(label_centroid - split_bank["label_centroids"][int(label.item())], p=2).item())
            )
            activity_pair_rmse[activity] = float(
                torch.sqrt(torch.mean((label_generated - label_targets) ** 2)).item()
            )

        within_band_mask = torch.cat(within_band_masks, dim=0)
        diversity_distances = torch.pdist(generated_pairs_standardized, p=2)
        diversity_mean = _safe_mean(diversity_distances)

        pair_error = generated_pairs - target_pairs
        synth_std = generated_pairs.std(dim=0, unbiased=False)
        val_std_ratio = _safe_std_ratio(synth_std, split_bank["std"])
        mean_abs_error = float(pair_error.abs().mean().item())
        std_ratio_mean = float(val_std_ratio.mean().item())
        std_ratio_drift_mean = float((val_std_ratio - 1.0).abs().mean().item())
        worst_activity_pair_rmse = max(activity_pair_rmse.values()) if activity_pair_rmse else 0.0

        metrics: dict[str, Any] = {
            "pair_rmse": float(torch.sqrt(torch.mean(pair_error ** 2)).item()),
            "metric_mae_mean": mean_abs_error,
            "nn_distance_val_mean": float(val_nn_distances.mean().item()),
            "nn_distance_gap_val_minus_train": float((val_nn_distances.mean() - train_nn_distances.mean()).item()),
            "centroid_drift_mean": sum(centroid_drifts) / max(len(centroid_drifts), 1),
            "within_band_vector_rate": float(within_band_mask.all(dim=1).float().mean().item()),
            "diversity_pairwise_distance_mean": diversity_mean,
            "std_ratio_mean": std_ratio_mean,
            "std_ratio_drift_mean": std_ratio_drift_mean,
            "coverage_unique_nn_ratio": float(torch.unique(val_nn_indices).numel() / max(len(split_bank["pairs"]), 1)),
            "worst_activity_pair_rmse": float(worst_activity_pair_rmse),
        }

        if enable_artifacts:
            metrics.update(
                self._build_wandb_payload(
                    pair_error=pair_error,
                    val_nn_distances=val_nn_distances,
                    train_nn_distances=train_nn_distances,
                )
            )

        if was_training:
            model.train()
        return metrics
