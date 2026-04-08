"""Evaluation entry point: classifiers trained on real vs real+augmented data.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py evaluation.classifier=transformer
"""

from __future__ import annotations
import logging
import pathlib
import torch
import hydra
from omegaconf import DictConfig
from cgdap.data.dataset import build_label_map, make_paired_loader
from cgdap.evaluation.deepsense import DeepSenseClassifier
from cgdap.evaluation.transformer import HATransformerClassifier

log = logging.getLogger(__name__)

def train_classifier(model, train_loader, val_loader, n_epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            for mod in [k for k in batch if k not in ("label", "activity")]:
                batch[mod]["spectrogram"] = batch[mod]["spectrogram"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
    # Validation accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            for mod in [k for k in batch if k not in ("label", "activity")]:
                batch[mod]["spectrogram"] = batch[mod]["spectrogram"].to(device)
            labels = batch["label"].to(device)
            logits = model(batch)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.shape[0]
    return correct / total

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_root = pathlib.Path(cfg.dataset.paths.processed)
    modalities = list(cfg.dataset.modalities)
    label_map = build_label_map(processed_root / "train", modality=modalities[0])
    n_classes = len(label_map)
    loader_cfg = dict(batch_size=cfg.dataset.loader.batch_size, num_workers=cfg.dataset.loader.num_workers, pin_memory=cfg.dataset.loader.pin_memory)
    train_loader = make_paired_loader(processed_root / "train", modalities, label_map, **loader_cfg)
    val_loader = make_paired_loader(processed_root / "val", modalities, label_map, shuffle=False, **loader_cfg)
    for cls_name, cls_cls in [("DeepSense", DeepSenseClassifier), ("Transformer", HATransformerClassifier)]:
        if cls_name == "DeepSense":
            model = DeepSenseClassifier(n_classes=n_classes, modalities=modalities).to(device)
        else:
            model = HATransformerClassifier(n_classes=n_classes, modalities=modalities).to(device)
        acc = train_classifier(model, train_loader, val_loader, n_epochs=20, device=device)
        log.info("%s val accuracy (real-only): %.4f", cls_name, acc)

if __name__ == "__main__":
    main()
