"""Evaluation entry point: classifiers trained on real vs real+augmented data.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py evaluation.augmentation.enabled=false
    uv run python scripts/evaluate.py evaluation.classifiers=[transformer]
"""

from __future__ import annotations

import copy
import logging
import pathlib
import random
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from cgdap.augmentation.engine import AugmentationEngine
from cgdap.data.dataset import PairedDataset, build_label_map
from cgdap.evaluation.deepsense import DeepSenseClassifier
from cgdap.evaluation.transformer import HATransformerClassifier
from cgdap.models.cgdap import MultimodalCGDAP
from cgdap.utils import batch_to_device

log = logging.getLogger(__name__)


class SyntheticPairedDataset(Dataset):
    """In-memory paired dataset for generated spectrogram samples."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        result: dict[str, Any] = {
            "label": int(sample["label"]),
            "activity": sample["activity"],
        }
        for mod, payload in sample["modalities"].items():
            result[mod] = {
                "spectrogram": payload["spectrogram"].clone(),
                "metrics": payload["metrics"].clone(),
            }
        return result

def evaluate_accuracy(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_to_device(batch, device)
            logits = model(batch)
            labels = batch["label"]
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.shape[0]
    if total == 0:
        raise RuntimeError("Validation loader produced zero samples.")
    return correct / total


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
) -> float:
    opt_cfg = cfg.evaluation.optimizer
    if str(opt_cfg.name).lower() != "adam":
        raise ValueError(f"Unsupported evaluation optimizer: {opt_cfg.name!r}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(opt_cfg.lr),
        weight_decay=float(opt_cfg.weight_decay),
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(int(cfg.evaluation.n_epochs)):
        model.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch_to_device(batch, device)
            labels = batch["label"]
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1

        val_acc = evaluate_accuracy(model, val_loader, device)
        train_loss = running_loss / max(steps, 1)
        log.info(
            "%s epoch %d/%d | train_loss=%.4f val_acc=%.4f",
            model.__class__.__name__,
            epoch + 1,
            int(cfg.evaluation.n_epochs),
            train_loss,
            val_acc,
        )

        if val_acc > best_acc + float(cfg.evaluation.min_delta):
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(cfg.evaluation.patience):
                break

    if best_state is None:
        raise RuntimeError("Classifier training completed without a valid checkpoint.")
    model.load_state_dict(best_state)
    return best_acc


def resolve_checkpoint_path(cfg: DictConfig) -> pathlib.Path:
    aug_cfg = cfg.evaluation.augmentation
    raw_path = aug_cfg.get("checkpoint_path")
    if raw_path:
        candidate = pathlib.Path(str(raw_path))
    else:
        candidate = pathlib.Path(cfg.training.checkpoint_dir) / cfg.experiment_name

    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find a CGDAP checkpoint at {candidate}. "
            "Train the diffusion model first or set evaluation.augmentation.checkpoint_path."
        )

    checkpoints = sorted(candidate.glob("ckpt_epoch*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching ckpt_epoch*.pt found under {candidate}.")
    return checkpoints[-1]


def load_generator_model(cfg: DictConfig, device: torch.device) -> MultimodalCGDAP:
    checkpoint_path = resolve_checkpoint_path(cfg)
    payload = torch.load(checkpoint_path, map_location=device)
    model = MultimodalCGDAP.from_config(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    log.info("Loaded generator checkpoint: %s", checkpoint_path)
    return model


def build_classifier(
    name: str,
    cfg: DictConfig,
    modalities: list[str],
    n_classes: int,
    freq_bins: int,
) -> torch.nn.Module:
    name = name.lower()
    if name == "deepsense":
        return DeepSenseClassifier.from_config(cfg)
    if name == "transformer":
        return HATransformerClassifier.from_config(cfg, freq_bins=freq_bins)
    raise ValueError(f"Unknown classifier: {name!r}")


def make_loader(
    dataset: Dataset,
    cfg: DictConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    batch_size = cfg.evaluation.batch_size or cfg.training.batch_size or cfg.dataset.loader.batch_size
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.dataset.loader.num_workers),
        pin_memory=bool(cfg.dataset.loader.pin_memory),
        drop_last=False,
    )


def build_synthetic_dataset(
    cfg: DictConfig,
    train_dataset: PairedDataset,
    label_map: dict[str, int],
    device: torch.device,
) -> SyntheticPairedDataset:
    model = load_generator_model(cfg, device)
    modalities = list(cfg.dataset.modalities)
    engine = AugmentationEngine(cfg, modalities, label_map)
    real_samples = [train_dataset[idx] for idx in range(len(train_dataset))]

    if cfg.augmentation.mode == "interpolation":
        engine.register_samples(real_samples)

    total_samples = len(real_samples) * int(cfg.evaluation.augmentation.samples_per_real)
    if total_samples <= 0:
        return SyntheticPairedDataset([])

    spec_shape = tuple(real_samples[0][modalities[0]]["spectrogram"].shape)
    seed_base = int(cfg.evaluation.augmentation.seed)
    generated_samples: list[dict[str, Any]] = []

    for sample_idx in range(total_samples):
        real_sample = real_samples[sample_idx % len(real_samples)]
        activity = str(real_sample["activity"])
        if cfg.augmentation.mode == "disturbance":
            targets = engine.generate_targets(sample=real_sample)
        elif cfg.augmentation.mode == "domain_instruction":
            targets = engine.generate_targets(activity=activity)
        else:
            targets = engine.generate_targets()

        labels = torch.tensor([int(targets["label"])], device=device)
        metric_targets = {
            mod: targets[mod].unsqueeze(0).to(device)
            for mod in modalities
        }
        generated = model.sample(
            metric_targets=metric_targets,
            labels=labels,
            n_classes=len(label_map),
            spec_shape=spec_shape,
            device=device,
            seed=seed_base + sample_idx,
            num_steps=cfg.evaluation.augmentation.num_steps,
        )
        generated_samples.append(
            {
                "label": int(targets["label"]),
                "activity": activity,
                "modalities": {
                    mod: {
                        "spectrogram": generated[mod].squeeze(0).cpu(),
                        "metrics": targets[mod].cpu(),
                    }
                    for mod in modalities
                },
            }
        )

    log.info("Generated %d synthetic paired samples for evaluation.", len(generated_samples))
    return SyntheticPairedDataset(generated_samples)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_root = pathlib.Path(cfg.dataset.paths.processed)
    modalities = list(cfg.dataset.modalities)
    label_map = build_label_map(processed_root / "train", modality=modalities[0])

    train_dataset = PairedDataset(processed_root / "train", modalities, label_map)
    val_dataset = PairedDataset(processed_root / "val", modalities, label_map)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Processed train/val datasets are empty. Re-run preprocessing first.")

    freq_bins = int(train_dataset[0][modalities[0]]["spectrogram"].shape[1])
    real_train_loader = make_loader(train_dataset, cfg, shuffle=True)
    val_loader = make_loader(val_dataset, cfg, shuffle=False)

    synthetic_dataset: SyntheticPairedDataset | None = None
    if bool(cfg.evaluation.augmentation.enabled):
        synthetic_dataset = build_synthetic_dataset(cfg, train_dataset, label_map, device)

    for classifier_name in cfg.evaluation.classifiers:
        real_model = build_classifier(classifier_name, cfg, modalities, len(label_map), freq_bins).to(device)
        real_acc = train_classifier(real_model, real_train_loader, val_loader, cfg, device)
        log.info("%s val accuracy (real-only): %.4f", classifier_name, real_acc)

        if synthetic_dataset is not None and len(synthetic_dataset) > 0:
            augmented_dataset = ConcatDataset([train_dataset, synthetic_dataset])
            augmented_loader = make_loader(augmented_dataset, cfg, shuffle=True)
            augmented_model = build_classifier(
                classifier_name,
                cfg,
                modalities,
                len(label_map),
                freq_bins,
            ).to(device)
            augmented_acc = train_classifier(augmented_model, augmented_loader, val_loader, cfg, device)
            log.info("%s val accuracy (real+augmented): %.4f", classifier_name, augmented_acc)


if __name__ == "__main__":
    main()
