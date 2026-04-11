"""Standalone synthetic sample generation entry point.

Usage:
    uv run python scripts/generate.py generation.reference_pt="data/processed/HAR/train/acc/walking/example.pt" generation.checkpoint_path="outputs/checkpoints/test_run/ckpt_epoch0000.pt"
    uv run python scripts/generate.py generation.reference_pt="data/processed/HAR/train/acc/walking/example.pt" generation.num_samples=8 augmentation.mode=disturbance
    uv run python scripts/generate.py generation.reference_pt="data/processed/HAR/train/acc/walking/example.pt" augmentation.mode=domain_instruction
"""

from __future__ import annotations

import logging
import pathlib
import random

import hydra
import torch
from omegaconf import DictConfig

from cgdap.augmentation.engine import AugmentationEngine
from cgdap.data.dataset import PairedDataset, build_label_map
from cgdap.generation import (
    load_generator_model,
    load_reference_pair,
    save_generated_outputs,
)

log = logging.getLogger(__name__)


def _load_split_samples(
    cfg: DictConfig,
    split_name: str,
    label_map: dict[str, int],
    reference_label: int | None = None,
) -> list[dict[str, object]]:
    processed_root = pathlib.Path(cfg.dataset.paths.processed)
    dataset = PairedDataset(processed_root / split_name, list(cfg.dataset.modalities), label_map)
    samples = [dataset[idx] for idx in range(len(dataset))]
    if reference_label is not None and bool(cfg.generation.match_reference_label):
        samples = [sample for sample in samples if int(sample["label"]) == int(reference_label)]
    if not samples:
        raise RuntimeError(f"No paired samples available in split {split_name!r} for generation.")
    return samples


def _resolve_spec_shape(
    reference_sample: dict[str, object] | None,
    fallback_samples: list[dict[str, object]] | None,
    modalities: list[str],
) -> tuple[int, int, int]:
    if reference_sample is not None:
        return tuple(reference_sample[modalities[0]]["spectrogram"].shape)  # type: ignore[index]
    if fallback_samples:
        return tuple(fallback_samples[0][modalities[0]]["spectrogram"].shape)  # type: ignore[index]
    raise RuntimeError("Could not resolve a spectrogram shape for sampling.")


def _build_targets(
    cfg: DictConfig,
    engine: AugmentationEngine,
    inverse_label_map: dict[int, str],
    reference_sample: dict[str, object] | None,
) -> tuple[dict[str, torch.Tensor | int], str]:
    mode = str(cfg.augmentation.mode)
    if mode == "disturbance":
        if reference_sample is None:
            raise ValueError("generation.reference_pt is required when augmentation.mode=disturbance.")
        targets = engine.generate_targets(sample=reference_sample)
        return targets, str(reference_sample["activity"])

    if mode == "domain_instruction":
        activity = str(cfg.generation.get("activity") or (reference_sample or {}).get("activity"))
        if not activity:
            raise ValueError(
                "Provide generation.activity or generation.reference_pt when augmentation.mode=domain_instruction."
            )
        targets = engine.generate_targets(activity=activity)
        return targets, activity

    if mode == "interpolation":
        targets = engine.generate_targets()
        label = int(targets["label"])
        activity = str(reference_sample["activity"]) if reference_sample and int(reference_sample["label"]) == label else inverse_label_map[label]
        return targets, activity

    raise ValueError(f"Unsupported augmentation.mode for generation: {mode!r}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    random.seed(int(cfg.generation.seed))
    torch.manual_seed(int(cfg.generation.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_root = pathlib.Path(cfg.dataset.paths.processed)
    modalities = list(cfg.dataset.modalities)
    label_map = build_label_map(processed_root / "train", modality=modalities[0])
    inverse_label_map = {label: activity for activity, label in label_map.items()}

    reference_sample = None
    if cfg.generation.reference_pt:
        reference_sample = load_reference_pair(cfg.generation.reference_pt, processed_root, modalities)
        log.info("Loaded reference pair from %s", cfg.generation.reference_pt)

    split_name = str(
        cfg.generation.get("dataset_split")
        or (reference_sample or {}).get("split")
        or "train"
    )

    split_samples = None
    mode = str(cfg.augmentation.mode)
    if mode == "interpolation" or reference_sample is None:
        reference_label = None if reference_sample is None else int(reference_sample["label"])
        split_samples = _load_split_samples(
            cfg,
            split_name,
            label_map,
            reference_label=reference_label if mode == "interpolation" else None,
        )

    spec_shape = _resolve_spec_shape(reference_sample, split_samples, modalities)
    model, checkpoint_path = load_generator_model(cfg, device, explicit_path=cfg.generation.get("checkpoint_path"))
    engine = AugmentationEngine(cfg, modalities, label_map)

    if mode == "interpolation" and split_samples is not None:
        engine.register_samples(split_samples)

    output_dir = pathlib.Path(cfg.generation.output_dir)
    num_samples = int(cfg.generation.num_samples)
    if num_samples <= 0:
        raise ValueError("generation.num_samples must be greater than zero.")

    log.info(
        "Generating %d synthetic sample(s) with mode=%s into %s",
        num_samples,
        cfg.augmentation.mode,
        output_dir,
    )

    for sample_idx in range(num_samples):
        targets, activity = _build_targets(cfg, engine, inverse_label_map, reference_sample)
        label = int(targets["label"])
        metric_targets = {
            modality: targets[modality].unsqueeze(0).to(device)  # type: ignore[index]
            for modality in modalities
        }
        labels = torch.tensor([label], device=device)
        generated = model.sample(
            metric_targets=metric_targets,
            labels=labels,
            n_classes=len(label_map),
            spec_shape=spec_shape,
            device=device,
            seed=int(cfg.generation.seed) + sample_idx,
            num_steps=cfg.generation.num_steps,
        )

        prefix = str(cfg.generation.get("sample_prefix") or "synthetic")
        if reference_sample is not None:
            prefix = f"{reference_sample['stem']}__{prefix}"
        sample_name = f"{prefix}_{sample_idx:04d}"
        saved_paths = save_generated_outputs(
            output_root=output_dir,
            sample_name=sample_name,
            activity=activity,
            label=label,
            generated=generated,
            metric_targets={mod: tensor.squeeze(0).cpu() for mod, tensor in metric_targets.items()},
            checkpoint_path=checkpoint_path,
            augmentation_mode=str(cfg.augmentation.mode),
            sample_index=sample_idx,
            reference_sample=reference_sample,
            save_bundle=bool(cfg.generation.save_bundle),
            save_modalities=bool(cfg.generation.save_modalities),
            save_plots=bool(cfg.generation.save_plots),
        )
        log.info("Saved %s -> %s", sample_name, {k: str(v) for k, v in saved_paths.items()})


if __name__ == "__main__":
    main()
