"""Device transfer helpers for nested training and evaluation batches."""

from __future__ import annotations

from typing import Any

import torch


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move paired batch tensors to the target device in place."""
    label = batch.get("label")
    if torch.is_tensor(label):
        batch["label"] = label.to(device)

    for payload in batch.values():
        if not isinstance(payload, dict):
            continue
        for key in ("spectrogram", "metrics"):
            value = payload.get(key)
            if torch.is_tensor(value):
                payload[key] = value.to(device)

    return batch
