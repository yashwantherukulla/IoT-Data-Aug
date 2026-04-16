"""Full validation evaluation report for CGDAP.

Runs generation on *all* val samples, computes ProductEvaluator diagnostics,
and saves a self-contained HTML report plus individual PNG figures.

Usage (from project root):
    uv run python scripts/run_eval_report.py
    uv run python scripts/run_eval_report.py experiment_name=test_run
    uv run python scripts/run_eval_report.py experiment_name=test_run evaluation.product_eval.num_steps=50
    uv run python scripts/run_eval_report.py evaluation.augmentation.checkpoint_path=outputs/checkpoints/test_run/ckpt_epoch0039.pt

Outputs (under outputs/eval_report/<experiment_name>/<timestamp>/):
    report.html                   - Self-contained HTML report
    figures/01_summary_table.png  - Top-level ProductEvaluator metric table
    figures/02_metric_scatter.png - Target vs Generated scatter per metric
    figures/03_nn_distance_hist.png - NN distance histogram: val vs train bank
    figures/04_radar_chart.png    - Per-activity metric fidelity radar
    figures/05_std_ratio_bars.png - Std-ratio and drift bars per metric
    figures/06_spectrogram_gallery.png - Side-by-side reference / generated
    figures/07_per_activity_rmse.png - Per-activity pair RMSE bar chart
"""

from __future__ import annotations

import base64
import copy
import datetime
import io
import logging
import pathlib
import random
import textwrap
from typing import Any

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from omegaconf import DictConfig

from cgdap.augmentation.engine import AugmentationEngine
from cgdap.data.dataset import PairedDataset, build_label_map
from cgdap.evaluation.product_eval import select_stratified_indices
from cgdap.generation import load_generator_model, _collapse_spectrogram, _metrics_text
from cgdap.metrics.extractor import MetricExtractor

log = logging.getLogger(__name__)

METRIC_NAMES = MetricExtractor.METRIC_NAMES
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
FIG_DPI = 150

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    """Encode a matplotlib Figure as a base64 PNG string (for HTML embedding)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _save_figure(fig: plt.Figure, path: pathlib.Path) -> str:
    """Save figure to disk and return its base64 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    log.info("Saved figure -> %s", path)
    return _fig_to_b64(fig)


def _standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1e-6)


def _activity_color(activities: list[str]) -> dict[str, str]:
    unique = sorted(set(activities))
    return {act: PALETTE[i % len(PALETTE)] for i, act in enumerate(unique)}


# ─────────────────────────────────────────────────────────────────────────────
# Data collection: generate for every val sample
# ─────────────────────────────────────────────────────────────────────────────

def collect_eval_data(
    cfg: DictConfig,
    val_dataset: PairedDataset,
    label_map: dict[str, int],
    model: Any,
    device: torch.device,
    *,
    num_steps: int,
    seed: int,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic sample for every val entry; extract metrics.

    Returns a dict with:
        targets       : list of per-modality target metric tensors [5]
        generated_specs : list of per-modality generated spectrograms [C,F,T]
        extracted_metrics : list of per-modality extracted metric tensors [5]
        labels        : list of int labels
        activities    : list of activity strings
        reference_specs : list of per-modality reference spectrograms [C,F,T]
    """
    modalities = list(cfg.dataset.modalities)
    label_to_activity = {v: k for k, v in label_map.items()}
    engine = AugmentationEngine(cfg, modalities, label_map)

    mode = str(cfg.augmentation.mode)
    if mode == "interpolation":
        all_samples = [val_dataset[i] for i in range(len(val_dataset))]
        engine.register_samples(all_samples)

    n = len(val_dataset) if max_samples is None else min(max_samples, len(val_dataset))
    indices = list(range(n))

    records: list[dict[str, Any]] = []
    model.eval()
    for sample_i, ds_idx in enumerate(indices):
        probe = val_dataset[ds_idx]
        activity = str(probe["activity"])
        label = int(probe["label"])

        # Choose targets via the augmentation engine
        if mode == "disturbance":
            targets = engine.generate_targets(sample=probe)
        elif mode == "domain_instruction":
            targets = engine.generate_targets(activity=activity)
        else:
            targets = engine.generate_targets()

        target_label = int(targets["label"])
        target_activity = label_to_activity[target_label]

        spec_shape = tuple(probe[modalities[0]]["spectrogram"].shape)
        metric_targets = {
            mod: targets[mod].unsqueeze(0).to(device)
            for mod in modalities
        }
        labels_t = torch.tensor([target_label], device=device)

        with torch.no_grad():
            generated = model.sample(
                metric_targets=metric_targets,
                labels=labels_t,
                n_classes=len(label_map),
                spec_shape=spec_shape,
                device=device,
                seed=seed + sample_i,
                num_steps=num_steps,
            )

        # Extract metrics from generated spectrogram
        extracted: dict[str, torch.Tensor] = {}
        for mod in modalities:
            gen_spec = generated[mod]  # [1, C, F, T]
            with torch.no_grad():
                ext = model.metric_extractor(gen_spec).squeeze(0).cpu().float()
            extracted[mod] = ext

        records.append({
            "label": target_label,
            "activity": target_activity,
            "targets": {mod: targets[mod].cpu().float() for mod in modalities},
            "generated": {mod: generated[mod].squeeze(0).cpu() for mod in modalities},
            "extracted": extracted,
            "reference": {mod: probe[mod]["spectrogram"].cpu() for mod in modalities},
            "reference_metrics": {mod: probe[mod]["metrics"].cpu().float() for mod in modalities},
        })

        if (sample_i + 1) % 50 == 0 or sample_i == n - 1:
            log.info("  Generated %d / %d val samples", sample_i + 1, n)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Build per-metric arrays
# ─────────────────────────────────────────────────────────────────────────────

def build_metric_arrays(
    records: list[dict[str, Any]],
    modalities: list[str],
) -> dict[str, Any]:
    """Stack per-record data into numpy arrays for plotting."""
    n = len(records)
    n_metrics = len(METRIC_NAMES)
    n_mod = len(modalities)

    # [n, n_mod, n_metrics]
    targets_arr = np.zeros((n, n_mod, n_metrics), dtype=np.float32)
    extracted_arr = np.zeros((n, n_mod, n_metrics), dtype=np.float32)
    labels_arr = np.array([r["label"] for r in records], dtype=np.int64)
    activities_arr = [r["activity"] for r in records]

    for i, rec in enumerate(records):
        for j, mod in enumerate(modalities):
            targets_arr[i, j] = rec["targets"][mod].numpy()
            extracted_arr[i, j] = rec["extracted"][mod].numpy()

    return {
        "targets": targets_arr,
        "extracted": extracted_arr,
        "labels": labels_arr,
        "activities": activities_arr,
        "modalities": modalities,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ProductEvaluator-style aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregate_metrics(
    arrays: dict[str, Any],
    train_dataset: PairedDataset,
) -> dict[str, Any]:
    """Compute aggregate diagnostic metrics mirroring ProductEvaluator."""
    modalities = arrays["modalities"]
    n_mod = len(modalities)
    n_metrics = len(METRIC_NAMES)
    n = arrays["targets"].shape[0]

    # Concatenate modalities for cross-modal metric vector [n, n_mod * n_metrics]
    targets_cat = arrays["targets"].reshape(n, n_mod * n_metrics)
    extracted_cat = arrays["extracted"].reshape(n, n_mod * n_metrics)
    labels = arrays["labels"]
    activities = arrays["activities"]
    unique_activities = sorted(set(activities))

    t_tensor = torch.from_numpy(targets_cat)
    e_tensor = torch.from_numpy(extracted_cat)
    labels_t = torch.from_numpy(labels)

    # Overall pair error
    pair_error = e_tensor - t_tensor
    pair_rmse = float(torch.sqrt((pair_error ** 2).mean()).item())
    metric_mae = float(pair_error.abs().mean().item())

    # Per-activity RMSE
    activity_rmse: dict[str, float] = {}
    for act in unique_activities:
        mask = np.array(activities) == act
        if mask.sum() == 0:
            continue
        ae = e_tensor[mask]
        at = t_tensor[mask]
        activity_rmse[act] = float(torch.sqrt(((ae - at) ** 2).mean()).item())

    # Per-metric MAE (averaged over modalities)
    per_metric_mae: dict[str, float] = {}
    for mi, name in enumerate(METRIC_NAMES):
        # Each modality contributes one column of length n_metrics
        cols = [mi + j * n_metrics for j in range(n_mod)]
        err_cols = pair_error[:, cols].abs().mean().item()
        per_metric_mae[name] = float(err_cols)

    # Std ratio: synth std vs val real std
    val_std = t_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    synth_std = e_tensor.std(dim=0, unbiased=False)
    std_ratio = synth_std / val_std
    std_ratio_mean = float(std_ratio.mean().item())
    std_ratio_drift_mean = float((std_ratio - 1.0).abs().mean().item())

    # Per-metric std ratio (averaged over modalities)
    per_metric_std_ratio: dict[str, float] = {}
    for mi, name in enumerate(METRIC_NAMES):
        cols = [mi + j * n_metrics for j in range(n_mod)]
        per_metric_std_ratio[name] = float(std_ratio[cols].mean().item())

    # Build val real bank for NN distances
    val_real_metrics = []
    val_labels_list = []
    for ds_idx in range(len(train_dataset)):
        s = train_dataset[ds_idx]
        parts = [s[mod]["metrics"].float() for mod in modalities]
        val_real_metrics.append(torch.cat(parts, dim=0))
        val_labels_list.append(int(s["label"]))
    real_tensor = torch.stack(val_real_metrics)
    real_labels_t = torch.tensor(val_labels_list)
    real_mean = real_tensor.mean(dim=0)
    real_std = real_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)

    # Standardize both
    synth_std_tensor = _standardize(e_tensor, real_mean, real_std)
    real_std_tensor = _standardize(real_tensor, real_mean, real_std)

    # NN distances (same-label)
    nn_distances = torch.zeros(n)
    label_indices_real: dict[int, torch.Tensor] = {}
    for lbl in torch.unique(real_labels_t):
        label_indices_real[int(lbl.item())] = torch.where(real_labels_t == lbl)[0]

    for lbl in torch.unique(labels_t):
        mask_s = labels_t == lbl
        label_int = int(lbl.item())
        if label_int not in label_indices_real:
            continue
        bank_idx = label_indices_real[label_int]
        synth_sub = synth_std_tensor[mask_s]
        real_sub = real_std_tensor[bank_idx]
        dist_mat = torch.cdist(synth_sub, real_sub)
        min_dist, _ = dist_mat.min(dim=1)
        nn_distances[mask_s] = min_dist

    nn_dist_mean = float(nn_distances.mean().item())
    diversity = float(torch.pdist(synth_std_tensor, p=2).mean().item()) if n > 1 else 0.0

    return {
        "pair_rmse": pair_rmse,
        "metric_mae": metric_mae,
        "nn_distance_mean": nn_dist_mean,
        "std_ratio_mean": std_ratio_mean,
        "std_ratio_drift_mean": std_ratio_drift_mean,
        "diversity_mean": diversity,
        "per_activity_rmse": activity_rmse,
        "per_metric_mae": per_metric_mae,
        "per_metric_std_ratio": per_metric_std_ratio,
        "nn_distances": nn_distances.numpy(),
        "std_ratio_vec": std_ratio.numpy(),
        "pair_error": pair_error.numpy(),
        "e_tensor": e_tensor.numpy(),
        "t_tensor": t_tensor.numpy(),
        "labels": labels,
        "activities": activities,
        "worst_activity_rmse": max(activity_rmse.values()) if activity_rmse else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Summary table
# ─────────────────────────────────────────────────────────────────────────────

def fig_summary_table(agg: dict[str, Any], checkpoint_path: str, n_val: int) -> plt.Figure:
    rows = [
        ("Val samples evaluated", f"{n_val}"),
        ("Checkpoint", pathlib.Path(checkpoint_path).name),
        ("", ""),
        ("── Core Quality ──", ""),
        ("Pair RMSE (target vs extracted)", f"{agg['pair_rmse']:.4f}"),
        ("Metric MAE (mean)", f"{agg['metric_mae']:.4f}"),
        ("Worst-activity pair RMSE", f"{agg['worst_activity_rmse']:.4f}"),
        ("", ""),
        ("── Distribution ──", ""),
        ("NN distance (synth → val real)", f"{agg['nn_distance_mean']:.4f}"),
        ("Std-ratio mean", f"{agg['std_ratio_mean']:.4f}"),
        ("Std-ratio drift |σ_synth/σ_real − 1|", f"{agg['std_ratio_drift_mean']:.4f}"),
        ("Diversity (pairwise dist mean)", f"{agg['diversity_mean']:.4f}"),
        ("", ""),
        ("── Per-metric MAE ──", ""),
    ]
    for name in METRIC_NAMES:
        rows.append((f"  {name}", f"{agg['per_metric_mae'].get(name, 0):.4f}"))

    rows += [
        ("", ""),
        ("── Per-activity RMSE ──", ""),
    ]
    for act, rmse in sorted(agg["per_activity_rmse"].items()):
        rows.append((f"  {act}", f"{rmse:.4f}"))

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.28 + 1)))
    ax.axis("off")
    col_labels = ["Metric", "Value"]
    table_data = [[r[0], r[1]] for r in rows]

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Style header
    for j in range(2):
        tbl[(0, j)].set_facecolor("#2d3748")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style section headers
    for i, (label, _) in enumerate(rows, start=1):
        if label.startswith("──"):
            for j in range(2):
                tbl[(i, j)].set_facecolor("#ebf8ff")
                tbl[(i, j)].set_text_props(fontweight="bold", color="#2b6cb0")
        elif i % 2 == 0:
            for j in range(2):
                tbl[(i, j)].set_facecolor("#f7fafc")

    fig.suptitle("CGDAP Validation Evaluation — Summary", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Target vs Generated scatter (per metric, color by activity)
# ─────────────────────────────────────────────────────────────────────────────

def fig_metric_scatter(
    arrays: dict[str, Any],
    agg: dict[str, Any],
) -> plt.Figure:
    modalities = arrays["modalities"]
    activities = agg["activities"]
    act_color = _activity_color(activities)
    colors = [act_color[a] for a in activities]

    n_mod = len(modalities)
    n_metrics = len(METRIC_NAMES)
    fig, axes = plt.subplots(n_mod, n_metrics, figsize=(n_metrics * 3.2, n_mod * 3.0), squeeze=False)

    for j, mod in enumerate(modalities):
        for i, mname in enumerate(METRIC_NAMES):
            ax = axes[j, i]
            t_vals = arrays["targets"][:, j, i]
            e_vals = arrays["extracted"][:, j, i]
            ax.scatter(t_vals, e_vals, c=colors, s=14, alpha=0.65, linewidths=0)
            lo = min(t_vals.min(), e_vals.min())
            hi = max(t_vals.max(), e_vals.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            mae = float(np.abs(e_vals - t_vals).mean())
            ax.set_title(f"{mod} / {mname}\nMAE={mae:.4f}", fontsize=8)
            ax.set_xlabel("Target", fontsize=7)
            ax.set_ylabel("Extracted", fontsize=7)
            ax.tick_params(labelsize=7)

    # Legend for activities
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=act_color[a],
                      markersize=7, label=a) for a in sorted(act_color)]
    fig.legend(handles=handles, title="Activity", loc="lower center",
               ncol=min(5, len(act_color)), fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Target vs Extracted Metrics — All Val Samples", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: NN distance histogram
# ─────────────────────────────────────────────────────────────────────────────

def fig_nn_histogram(agg: dict[str, Any]) -> plt.Figure:
    nn_dists = agg["nn_distances"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nn_dists, bins=40, color=PALETTE[0], alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(nn_dists.mean(), color="#e53e3e", lw=1.5, linestyle="--", label=f"Mean={nn_dists.mean():.3f}")
    ax.set_xlabel("NN Distance (synth → val real, standardized metric space)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Nearest-Neighbor Distance: Synthetic → Real Val Bank", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Per-activity metric fidelity radar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_radar_chart(arrays: dict[str, Any]) -> plt.Figure:
    modalities = arrays["modalities"]
    activities_arr = np.array(arrays["activities"])
    unique_acts = sorted(set(arrays["activities"]))
    act_color = _activity_color(unique_acts)
    n_metrics = len(METRIC_NAMES)

    # Per-activity, per-modality mean MAE (normalized 0-1 within each metric for radar)
    # We use 1 - normalized_MAE so higher = better
    per_act_mae = {}
    for act in unique_acts:
        mask = activities_arr == act
        t = arrays["targets"][mask]   # [k, n_mod, n_metrics]
        e = arrays["extracted"][mask]
        mae_mod = np.abs(t - e).mean(axis=(0, 1))  # [n_metrics]
        per_act_mae[act] = mae_mod

    # Normalize MAE across activities per metric (min-max)
    mae_matrix = np.stack([per_act_mae[a] for a in unique_acts])  # [n_act, n_metrics]
    col_min = mae_matrix.min(axis=0, keepdims=True)
    col_max = mae_matrix.max(axis=0, keepdims=True)
    norm_mat = 1.0 - (mae_matrix - col_min) / (col_max - col_min + 1e-9)  # higher = better

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    for ai, act in enumerate(unique_acts):
        vals = norm_mat[ai].tolist() + [norm_mat[ai][0]]
        ax.plot(angles, vals, color=act_color[act], lw=1.8, label=act)
        ax.fill(angles, vals, color=act_color[act], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in METRIC_NAMES], fontsize=9)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Per-activity Metric Fidelity\n(1 = lowest MAE, normalized)", fontsize=11, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Std-ratio bars per metric
# ─────────────────────────────────────────────────────────────────────────────

def fig_std_ratio(agg: dict[str, Any], modalities: list[str]) -> plt.Figure:
    n_metrics = len(METRIC_NAMES)
    n_mod = len(modalities)
    # Reshape std_ratio_vec: [n_mod * n_metrics] → per metric averaged over modalities
    ratio_vec = agg["std_ratio_vec"]  # [n_mod * n_metrics]
    per_metric_ratios = np.array([
        ratio_vec[[mi + j * n_metrics for j in range(n_mod)]].mean()
        for mi in range(n_metrics)
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(METRIC_NAMES, per_metric_ratios, color=PALETTE[:n_metrics], alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="black", lw=1.2, linestyle="--", label="Ideal ratio = 1.0")
    for bar, val in zip(bars, per_metric_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Metric", fontsize=10)
    ax.set_ylabel("σ_synth / σ_val", fontsize=10)
    ax.set_title("Per-metric Std Ratio (synthetic ÷ val real)\nCloser to 1.0 = better variance matching",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(per_metric_ratios.max() * 1.2, 1.4))
    plt.xticks(rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Spectrogram gallery (reference vs generated)
# ─────────────────────────────────────────────────────────────────────────────

def fig_spectrogram_gallery(
    records: list[dict[str, Any]],
    modalities: list[str],
    *,
    n_show: int = 4,
    seed: int = 42,
) -> plt.Figure:
    """Show n_show samples: each row is reference | generated for each modality."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), min(n_show, len(records)))
    n_rows = len(indices)
    n_cols = len(modalities) * 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.8), squeeze=False)

    for row_i, rec_idx in enumerate(indices):
        rec = records[rec_idx]
        for col_j, mod in enumerate(modalities):
            ref_spec = _collapse_spectrogram(rec["reference"][mod])
            gen_spec = _collapse_spectrogram(rec["generated"][mod])
            vmin = min(ref_spec.min(), gen_spec.min())
            vmax = max(ref_spec.max(), gen_spec.max())

            ax_ref = axes[row_i, col_j * 2]
            ax_gen = axes[row_i, col_j * 2 + 1]

            ax_ref.imshow(ref_spec, origin="lower", aspect="auto", cmap="viridis",
                          vmin=vmin, vmax=vmax, interpolation="nearest")
            ax_ref.set_title(f"Ref {mod}\n[{rec['activity']}]", fontsize=8)
            ax_ref.axis("off")

            ax_gen.imshow(gen_spec, origin="lower", aspect="auto", cmap="viridis",
                          vmin=vmin, vmax=vmax, interpolation="nearest")
            ax_gen.set_title(f"Gen {mod}", fontsize=8)
            ax_gen.axis("off")

            # Metric annotations
            ref_txt = _metrics_text(rec["reference_metrics"][mod])
            gen_txt = _metrics_text(rec["extracted"][mod])
            ax_ref.text(1.0, 0.5, ref_txt, transform=ax_ref.transAxes,
                        fontsize=6, va="center", ha="left", color="#2d3748",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            ax_gen.text(1.0, 0.5, gen_txt, transform=ax_gen.transAxes,
                        fontsize=6, va="center", ha="left", color="#2d3748",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.suptitle("Spectrogram Gallery — Reference vs Generated (sampled val pairs)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Per-activity RMSE bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_per_activity_rmse(agg: dict[str, Any]) -> plt.Figure:
    act_rmse = agg["per_activity_rmse"]
    activities = sorted(act_rmse.keys())
    values = [act_rmse[a] for a in activities]
    act_color = _activity_color(activities)
    colors = [act_color[a] for a in activities]

    fig, ax = plt.subplots(figsize=(max(6, len(activities) * 1.3), 4.5))
    bars = ax.bar(activities, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axhline(agg["pair_rmse"], color="black", lw=1.2, linestyle="--",
               label=f"Overall RMSE={agg['pair_rmse']:.4f}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Activity", fontsize=10)
    ax.set_ylabel("Pair RMSE (target vs extracted metrics)", fontsize=10)
    ax.set_title("Per-activity Metric Pair RMSE", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML report builder
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CGDAP Evaluation Report — {experiment}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Inter',sans-serif;background:#f0f4f8;color:#2d3748;}}
  header{{background:linear-gradient(135deg,#1a365d 0%,#2b6cb0 60%,#4299e1 100%);
          color:white;padding:2.5rem 3rem;}}
  header h1{{font-size:2rem;font-weight:700;letter-spacing:-0.5px;}}
  header p{{opacity:0.85;margin-top:0.4rem;font-size:0.95rem;}}
  .badge{{display:inline-block;background:rgba(255,255,255,0.2);border-radius:6px;
          padding:0.25rem 0.75rem;font-size:0.82rem;margin-top:0.6rem;margin-right:0.5rem;}}
  main{{max-width:1200px;margin:2rem auto;padding:0 1.5rem;}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:2rem;}}
  .kpi{{background:white;border-radius:12px;padding:1.4rem 1.6rem;
        box-shadow:0 1px 3px rgba(0,0,0,0.08);border-left:4px solid {accent};}}
  .kpi .label{{font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;color:#718096;}}
  .kpi .value{{font-size:1.9rem;font-weight:700;color:#1a365d;margin-top:0.25rem;}}
  .kpi .sub{{font-size:0.75rem;color:#a0aec0;margin-top:0.2rem;}}
  .section{{background:white;border-radius:12px;padding:1.8rem;
            box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:1.8rem;}}
  .section h2{{font-size:1.15rem;font-weight:600;color:#2b6cb0;border-bottom:2px solid #bee3f8;
               padding-bottom:0.5rem;margin-bottom:1.2rem;}}
  .fig-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:1.2rem;}}
  .fig-card{{border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;}}
  .fig-card img{{width:100%;display:block;}}
  .fig-caption{{padding:0.6rem 0.9rem;font-size:0.8rem;color:#718096;background:#f7fafc;}}
  footer{{text-align:center;color:#a0aec0;font-size:0.78rem;padding:2rem 0 3rem;}}
</style>
</head>
<body>
<header>
  <h1>🔬 CGDAP Evaluation Report</h1>
  <p>Generative model quality assessment on the full val split</p>
  <span class="badge">Experiment: {experiment}</span>
  <span class="badge">Checkpoint: {checkpoint}</span>
  <span class="badge">Generated: {timestamp}</span>
  <span class="badge">Val samples: {n_val}</span>
  <span class="badge">Mode: {aug_mode}</span>
</header>
<main>

<div class="kpi-grid">
  <div class="kpi" style="border-left-color:#e53e3e;">
    <div class="label">Pair RMSE</div>
    <div class="value">{pair_rmse}</div>
    <div class="sub">Target vs extracted metrics</div>
  </div>
  <div class="kpi" style="border-left-color:#dd6b20;">
    <div class="label">Metric MAE</div>
    <div class="value">{metric_mae}</div>
    <div class="sub">Mean absolute error</div>
  </div>
  <div class="kpi" style="border-left-color:#38a169;">
    <div class="label">Std-ratio</div>
    <div class="value">{std_ratio}</div>
    <div class="sub">σ_synth / σ_real (1.0 = perfect)</div>
  </div>
  <div class="kpi" style="border-left-color:#3182ce;">
    <div class="label">NN Distance</div>
    <div class="value">{nn_dist}</div>
    <div class="sub">Synth → val real bank (metric space)</div>
  </div>
  <div class="kpi" style="border-left-color:#805ad5;">
    <div class="label">Diversity</div>
    <div class="value">{diversity}</div>
    <div class="sub">Mean pairwise distance (synth)</div>
  </div>
  <div class="kpi" style="border-left-color:#d69e2e;">
    <div class="label">Worst-activity RMSE</div>
    <div class="value">{worst_rmse}</div>
    <div class="sub">Highest per-activity pair RMSE</div>
  </div>
</div>

<div class="section">
  <h2>📊 Summary Metrics Table</h2>
  <div class="fig-grid">
    <div class="fig-card">
      <img src="data:image/png;base64,{b64_table}" alt="Summary Table"/>
      <div class="fig-caption">All aggregate diagnostic metrics from the evaluation pass.</div>
    </div>
  </div>
</div>

<div class="section">
  <h2>🎯 Target vs Extracted Metrics (Scatter)</h2>
  <div class="fig-card">
    <img src="data:image/png;base64,{b64_scatter}" alt="Metric Scatter"/>
    <div class="fig-caption">Each point = one val sample. Diagonal dashed line = perfect fidelity. Colour = activity class.</div>
  </div>
</div>

<div class="section">
  <h2>📈 Per-activity RMSE</h2>
  <div class="fig-card">
    <img src="data:image/png;base64,{b64_act_rmse}" alt="Per-activity RMSE"/>
    <div class="fig-caption">Dashed horizontal line = overall pair RMSE. Lower is better.</div>
  </div>
</div>

<div class="section">
  <h2>🕸️ Metric Fidelity Radar (per activity)</h2>
  <div class="fig-grid">
    <div class="fig-card">
      <img src="data:image/png;base64,{b64_radar}" alt="Radar Chart"/>
      <div class="fig-caption">1 = lowest MAE (best), all values normalized across activities per metric.</div>
    </div>
    <div class="fig-card">
      <img src="data:image/png;base64,{b64_std}" alt="Std Ratio"/>
      <div class="fig-caption">Per-metric standard deviation ratio (σ_synth / σ_real). An ideal generator matches 1.0 per metric.</div>
    </div>
  </div>
</div>

<div class="section">
  <h2>📉 Nearest-Neighbour Distance Histogram</h2>
  <div class="fig-card">
    <img src="data:image/png;base64,{b64_nn}" alt="NN Distance"/>
    <div class="fig-caption">Lower distances = generated samples are closer to real val data in metric space.</div>
  </div>
</div>

<div class="section">
  <h2>🎨 Spectrogram Gallery</h2>
  <div class="fig-card">
    <img src="data:image/png;base64,{b64_gallery}" alt="Spectrogram Gallery"/>
    <div class="fig-caption">Randomly sampled val pairs: reference (real) vs generated. Metric annotations shown.</div>
  </div>
</div>

</main>
<footer>Generated by CGDAP run_eval_report.py &bull; {timestamp}</footer>
</body>
</html>
"""


def build_html_report(
    b64_figures: dict[str, str],
    agg: dict[str, Any],
    *,
    experiment: str,
    checkpoint: str,
    n_val: int,
    aug_mode: str,
    timestamp: str,
) -> str:
    return HTML_TEMPLATE.format(
        experiment=experiment,
        checkpoint=pathlib.Path(checkpoint).name,
        timestamp=timestamp,
        n_val=n_val,
        aug_mode=aug_mode,
        pair_rmse=f"{agg['pair_rmse']:.4f}",
        metric_mae=f"{agg['metric_mae']:.4f}",
        std_ratio=f"{agg['std_ratio_mean']:.4f}",
        nn_dist=f"{agg['nn_distance_mean']:.4f}",
        diversity=f"{agg['diversity_mean']:.4f}",
        worst_rmse=f"{agg['worst_activity_rmse']:.4f}",
        accent="#2b6cb0",
        **b64_figures,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Fix UnicodeEncodeError on Windows (CP1252 can't encode → U+2192)
    import sys
    for handler in logging.root.handlers:
        if hasattr(handler, "stream") and hasattr(handler.stream, "reconfigure"):
            try:
                handler.stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    processed_root = pathlib.Path(cfg.dataset.paths.processed)
    modalities = list(cfg.dataset.modalities)
    label_map = build_label_map(processed_root / "train", modality=modalities[0])

    val_dataset = PairedDataset(processed_root / "val", modalities, label_map)
    train_dataset = PairedDataset(processed_root / "train", modalities, label_map)
    if len(val_dataset) == 0:
        raise RuntimeError("Val dataset is empty. Re-run preprocessing first.")
    log.info("Val dataset: %d paired samples | Train dataset: %d paired samples",
             len(val_dataset), len(train_dataset))

    # Load model
    model, checkpoint_path = load_generator_model(cfg, device)
    model.eval()

    # Resolve num_steps
    num_steps_raw = (
        cfg.evaluation.product_eval.get("num_steps")
        or cfg.evaluation.augmentation.get("num_steps")
    )
    num_steps = int(num_steps_raw) if num_steps_raw is not None else 50
    log.info("Using num_steps=%d for generation", num_steps)

    # Output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = pathlib.Path("outputs") / "eval_report" / str(cfg.experiment_name) / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate for val samples (optionally capped) ─────────────────
    max_val_samples_raw = cfg.evaluation.get("max_val_samples")
    max_val_samples = int(max_val_samples_raw) if max_val_samples_raw is not None else None
    n_to_process = min(max_val_samples, len(val_dataset)) if max_val_samples is not None else len(val_dataset)
    log.info("Starting generation for %d / %d val samples...", n_to_process, len(val_dataset))
    records = collect_eval_data(
        cfg, val_dataset, label_map, model, device,
        num_steps=num_steps,
        seed=int(cfg.seed),
        max_samples=max_val_samples,
    )
    log.info("Done generating. Computing aggregate metrics...")

    # ── Arrays & aggregates ──────────────────────────────────────────
    arrays = build_metric_arrays(records, modalities)
    agg = compute_aggregate_metrics(arrays, train_dataset)

    log.info(
        "Summary | pair_rmse=%.4f | mae=%.4f | nn_dist=%.4f | std_ratio=%.4f | diversity=%.4f",
        agg["pair_rmse"], agg["metric_mae"], agg["nn_distance_mean"],
        agg["std_ratio_mean"], agg["diversity_mean"],
    )

    # ── Produce figures ──────────────────────────────────────────────
    log.info("Rendering figures...")
    b64: dict[str, str] = {}

    fig1 = fig_summary_table(agg, str(checkpoint_path), len(records))
    b64["b64_table"] = _save_figure(fig1, fig_dir / "01_summary_table.png")
    plt.close(fig1)

    fig2 = fig_metric_scatter(arrays, agg)
    b64["b64_scatter"] = _save_figure(fig2, fig_dir / "02_metric_scatter.png")
    plt.close(fig2)

    fig3 = fig_nn_histogram(agg)
    b64["b64_nn"] = _save_figure(fig3, fig_dir / "03_nn_distance_hist.png")
    plt.close(fig3)

    fig4 = fig_radar_chart(arrays)
    b64["b64_radar"] = _save_figure(fig4, fig_dir / "04_radar_chart.png")
    plt.close(fig4)

    fig5 = fig_std_ratio(agg, modalities)
    b64["b64_std"] = _save_figure(fig5, fig_dir / "05_std_ratio_bars.png")
    plt.close(fig5)

    fig6 = fig_spectrogram_gallery(records, modalities, n_show=4, seed=int(cfg.seed))
    b64["b64_gallery"] = _save_figure(fig6, fig_dir / "06_spectrogram_gallery.png")
    plt.close(fig6)

    fig7 = fig_per_activity_rmse(agg)
    b64["b64_act_rmse"] = _save_figure(fig7, fig_dir / "07_per_activity_rmse.png")
    plt.close(fig7)

    # ── Write HTML report ────────────────────────────────────────────
    html = build_html_report(
        b64,
        agg,
        experiment=str(cfg.experiment_name),
        checkpoint=str(checkpoint_path),
        n_val=len(records),
        aug_mode=str(cfg.augmentation.mode),
        timestamp=timestamp,
    )
    report_path = out_root / "report.html"
    report_path.write_text(html, encoding="utf-8")
    log.info("Report written -> %s", report_path)

    # ── Print final summary ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CGDAP EVALUATION REPORT — COMPLETE")
    print("=" * 65)
    print(f"  Experiment  : {cfg.experiment_name}")
    print(f"  Checkpoint  : {checkpoint_path.name}")
    print(f"  Val samples : {len(records)}")
    print(f"  Num steps   : {num_steps}")
    print(f"  Mode        : {cfg.augmentation.mode}")
    print("-" * 65)
    print(f"  Pair RMSE           : {agg['pair_rmse']:.4f}")
    print(f"  Metric MAE          : {agg['metric_mae']:.4f}")
    print(f"  NN Distance (val)   : {agg['nn_distance_mean']:.4f}")
    print(f"  Std-ratio mean      : {agg['std_ratio_mean']:.4f}")
    print(f"  Std-ratio drift     : {agg['std_ratio_drift_mean']:.4f}")
    print(f"  Diversity           : {agg['diversity_mean']:.4f}")
    print(f"  Worst-activity RMSE : {agg['worst_activity_rmse']:.4f}")
    print("-" * 65)
    for act, rmse in sorted(agg["per_activity_rmse"].items()):
        print(f"  {act:<20} RMSE={rmse:.4f}")
    print("=" * 65)
    print(f"\n  📄 HTML report  → {report_path.resolve()}")
    print(f"  🖼  Figures dir  → {fig_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()
