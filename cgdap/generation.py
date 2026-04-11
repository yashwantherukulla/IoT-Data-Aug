"""Utilities for standalone synthetic sample generation."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from cgdap.models.cgdap import MultimodalCGDAP

log = logging.getLogger(__name__)

REQUIRED_SAMPLE_KEYS = {"spectrogram", "metrics", "label"}
METRIC_NAMES = ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"]


def resolve_checkpoint_path(
    cfg: DictConfig,
    explicit_path: str | None = None,
) -> pathlib.Path:
    """Resolve a generator checkpoint from config or an explicit path."""
    raw_path = (
        explicit_path
        or cfg.generation.get("checkpoint_path")
        or cfg.evaluation.augmentation.get("checkpoint_path")
    )
    if raw_path:
        candidate = pathlib.Path(str(raw_path))
    else:
        candidate = pathlib.Path(cfg.training.checkpoint_dir) / cfg.experiment_name

    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find a CGDAP checkpoint at {candidate}. "
            "Train the diffusion model first or set generation.checkpoint_path."
        )

    checkpoints = sorted(candidate.glob("ckpt_epoch*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching ckpt_epoch*.pt found under {candidate}.")
    return checkpoints[-1]


def load_generator_model(
    cfg: DictConfig,
    device: torch.device,
    explicit_path: str | None = None,
) -> tuple[MultimodalCGDAP, pathlib.Path]:
    """Load a trained MultimodalCGDAP checkpoint."""
    checkpoint_path = resolve_checkpoint_path(cfg, explicit_path=explicit_path)
    payload = torch.load(checkpoint_path, map_location=device)
    model = MultimodalCGDAP.from_config(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    log.info("Loaded generator checkpoint: %s", checkpoint_path)
    return model, checkpoint_path


def _load_sample_payload(sample_path: pathlib.Path) -> dict[str, Any]:
    item = torch.load(sample_path, map_location="cpu", weights_only=True)
    missing = REQUIRED_SAMPLE_KEYS - item.keys()
    if missing:
        raise KeyError(f"Sample {sample_path} is missing required keys: {sorted(missing)}")
    return item


def load_reference_pair(
    reference_path: str | pathlib.Path,
    processed_root: str | pathlib.Path,
    modalities: list[str],
) -> dict[str, Any]:
    """Load a processed sample and its paired modality companions.

    Expected layout:
        <processed_root>/<split>/<modality>/<activity>/<sample>.pt
    """
    reference = pathlib.Path(reference_path).resolve()
    processed = pathlib.Path(processed_root).resolve()

    if not reference.is_file():
        raise FileNotFoundError(f"Reference sample not found: {reference}")

    try:
        rel = reference.relative_to(processed)
    except ValueError as exc:
        raise ValueError(
            f"Reference sample {reference} must live under processed root {processed}."
        ) from exc

    if len(rel.parts) != 4:
        raise ValueError(
            "Reference sample must follow <split>/<modality>/<activity>/<file>.pt under the processed root."
        )

    split, source_modality, activity, filename = rel.parts
    if source_modality not in modalities:
        raise ValueError(f"Reference modality {source_modality!r} is not in configured modalities {modalities}.")

    paired: dict[str, Any] = {
        "split": split,
        "activity": activity,
        "stem": pathlib.Path(filename).stem,
        "source_modality": source_modality,
        "paths": {},
    }

    label: int | None = None
    for modality in modalities:
        sample_path = processed / split / modality / activity / filename
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Missing paired sample for modality {modality!r}: expected {sample_path}"
            )
        item = _load_sample_payload(sample_path)
        paired["paths"][modality] = sample_path
        paired[modality] = {
            "spectrogram": item["spectrogram"].float(),
            "metrics": item["metrics"].float(),
            "subject": item.get("subject", "unknown"),
            "window_index": int(item.get("window_index", -1)),
            "sample_rate_hz": item.get("sample_rate_hz"),
            "freq_axis_hz": item.get("freq_axis_hz"),
            "time_axis_s": item.get("time_axis_s"),
        }
        if label is None:
            label = int(item["label"])

    paired["label"] = label
    return paired


def save_generated_outputs(
    output_root: str | pathlib.Path,
    sample_name: str,
    activity: str,
    label: int,
    generated: dict[str, torch.Tensor],
    metric_targets: dict[str, torch.Tensor],
    checkpoint_path: pathlib.Path,
    augmentation_mode: str,
    sample_index: int,
    reference_sample: dict[str, Any] | None = None,
    *,
    save_bundle: bool = True,
    save_modalities: bool = True,
    save_plots: bool = True,
) -> dict[str, pathlib.Path]:
    """Persist generated samples to disk for demo or downstream inspection."""
    output_dir = pathlib.Path(output_root)
    saved_paths: dict[str, pathlib.Path] = {}

    reference_paths = {}
    if reference_sample is not None:
        reference_paths = {
            mod: str(path)
            for mod, path in reference_sample["paths"].items()
        }

    if save_modalities:
        for modality, tensor in generated.items():
            modality_dir = output_dir / modality / activity
            modality_dir.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "spectrogram": tensor.squeeze(0).detach().cpu(),
                "metrics": metric_targets[modality].detach().cpu(),
                "label": int(label),
                "activity": activity,
                "subject": "synthetic",
                "window_index": int(sample_index),
                "checkpoint_path": str(checkpoint_path),
                "generation_mode": augmentation_mode,
            }
            if reference_sample is not None:
                payload["reference_path"] = str(reference_sample["paths"][modality])
                payload["reference_metrics"] = reference_sample[modality]["metrics"].detach().cpu()
                for key in ("sample_rate_hz", "freq_axis_hz", "time_axis_s"):
                    value = reference_sample[modality].get(key)
                    if value is not None:
                        payload[key] = value
            sample_path = modality_dir / f"{sample_name}.pt"
            torch.save(payload, sample_path)
            saved_paths[f"{modality}_sample"] = sample_path
            if save_plots:
                plot_path = modality_dir / f"{sample_name}.png"
                _save_spectrogram_plot(
                    plot_path=plot_path,
                    spectrogram=payload["spectrogram"],
                    metrics=payload["metrics"],
                    activity=activity,
                    modality=modality,
                    label=label,
                    reference_spectrogram=None if reference_sample is None else reference_sample[modality]["spectrogram"],
                    reference_metrics=None if reference_sample is None else reference_sample[modality]["metrics"],
                    freq_axis=payload.get("freq_axis_hz"),
                    time_axis=payload.get("time_axis_s"),
                )
                saved_paths[f"{modality}_plot"] = plot_path

    if save_bundle:
        paired_dir = output_dir / "paired" / activity
        paired_dir.mkdir(parents=True, exist_ok=True)
        bundle_payload = {
            "label": int(label),
            "activity": activity,
            "sample_name": sample_name,
            "checkpoint_path": str(checkpoint_path),
            "generation_mode": augmentation_mode,
            "reference_paths": reference_paths,
            "modalities": {
                modality: {
                    "spectrogram": tensor.squeeze(0).detach().cpu(),
                    "metrics": metric_targets[modality].detach().cpu(),
                }
                for modality, tensor in generated.items()
            },
        }
        bundle_path = paired_dir / f"{sample_name}.pt"
        torch.save(bundle_payload, bundle_path)
        saved_paths["paired_bundle"] = bundle_path
        if save_plots:
            bundle_plot_path = paired_dir / f"{sample_name}.png"
            _save_paired_plot(
                plot_path=bundle_plot_path,
                generated=generated,
                metric_targets=metric_targets,
                activity=activity,
                label=label,
                reference_sample=reference_sample,
            )
            saved_paths["paired_plot"] = bundle_plot_path

    return saved_paths


def _collapse_spectrogram(spectrogram: torch.Tensor | np.ndarray) -> np.ndarray:
    spec = spectrogram.detach().cpu().numpy() if isinstance(spectrogram, torch.Tensor) else np.asarray(spectrogram)
    if spec.ndim == 4:
        spec = spec[0]
    if spec.ndim == 3:
        spec = spec.mean(axis=0)
    return spec


def _metrics_text(metrics: torch.Tensor | np.ndarray | None) -> str:
    if metrics is None:
        return ""
    values = metrics.detach().cpu().numpy() if isinstance(metrics, torch.Tensor) else np.asarray(metrics)
    return "\n".join(f"{name}: {value:.3f}" for name, value in zip(METRIC_NAMES, values))


def _plot_on_axis(
    ax: plt.Axes,
    spectrogram: torch.Tensor | np.ndarray,
    *,
    title: str,
    freq_axis: torch.Tensor | np.ndarray | None = None,
    time_axis: torch.Tensor | np.ndarray | None = None,
) -> None:
    spec = _collapse_spectrogram(spectrogram)
    vmin, vmax = np.percentile(spec, [2, 98])

    if freq_axis is not None and time_axis is not None:
        freq = freq_axis.detach().cpu().numpy() if isinstance(freq_axis, torch.Tensor) else np.asarray(freq_axis)
        time = time_axis.detach().cpu().numpy() if isinstance(time_axis, torch.Tensor) else np.asarray(time_axis)
        extent = [float(time[0]), float(time[-1]), float(freq[0]), float(freq[-1])]
    else:
        extent = None

    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Freq")


def _save_spectrogram_plot(
    plot_path: pathlib.Path,
    spectrogram: torch.Tensor,
    metrics: torch.Tensor,
    activity: str,
    modality: str,
    label: int,
    *,
    reference_spectrogram: torch.Tensor | None = None,
    reference_metrics: torch.Tensor | None = None,
    freq_axis: torch.Tensor | None = None,
    time_axis: torch.Tensor | None = None,
) -> None:
    has_reference = reference_spectrogram is not None
    ncols = 2 if has_reference else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)

    if has_reference:
        _plot_on_axis(
            axes[0, 0],
            reference_spectrogram,
            title=f"Reference {modality}",
            freq_axis=freq_axis,
            time_axis=time_axis,
        )
        ref_text = _metrics_text(reference_metrics)
        if ref_text:
            axes[0, 0].text(1.02, 0.98, ref_text, transform=axes[0, 0].transAxes, va="top", fontsize=8)
        target_ax = axes[0, 1]
    else:
        target_ax = axes[0, 0]

    _plot_on_axis(
        target_ax,
        spectrogram,
        title=f"Generated {modality}",
        freq_axis=freq_axis,
        time_axis=time_axis,
    )
    target_ax.text(
        1.02,
        0.98,
        _metrics_text(metrics),
        transform=target_ax.transAxes,
        va="top",
        fontsize=8,
    )

    fig.suptitle(f"{activity} | label={label} | {modality}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_paired_plot(
    plot_path: pathlib.Path,
    generated: dict[str, torch.Tensor],
    metric_targets: dict[str, torch.Tensor],
    activity: str,
    label: int,
    reference_sample: dict[str, Any] | None,
) -> None:
    modalities = list(generated.keys())
    has_reference = reference_sample is not None
    nrows = 2 if has_reference else 1
    fig, axes = plt.subplots(nrows, len(modalities), figsize=(5 * len(modalities), 4 * nrows), squeeze=False)

    for col, modality in enumerate(modalities):
        if has_reference:
            ref_ax = axes[0, col]
            _plot_on_axis(
                ref_ax,
                reference_sample[modality]["spectrogram"],
                title=f"Reference {modality}",
                freq_axis=reference_sample[modality].get("freq_axis_hz"),
                time_axis=reference_sample[modality].get("time_axis_s"),
            )
            ref_text = _metrics_text(reference_sample[modality]["metrics"])
            if ref_text:
                ref_ax.text(1.02, 0.98, ref_text, transform=ref_ax.transAxes, va="top", fontsize=8)
            gen_ax = axes[1, col]
        else:
            gen_ax = axes[0, col]

        freq_axis = None if reference_sample is None else reference_sample[modality].get("freq_axis_hz")
        time_axis = None if reference_sample is None else reference_sample[modality].get("time_axis_s")
        _plot_on_axis(
            gen_ax,
            generated[modality],
            title=f"Generated {modality}",
            freq_axis=freq_axis,
            time_axis=time_axis,
        )
        gen_ax.text(
            1.02,
            0.98,
            _metrics_text(metric_targets[modality]),
            transform=gen_ax.transAxes,
            va="top",
            fontsize=8,
        )

    fig.suptitle(f"Generated Pair | {activity} | label={label}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
