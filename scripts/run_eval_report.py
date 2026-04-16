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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0;}}
  :root{{
    --blue-dark:#1a365d;--blue-mid:#2b6cb0;--blue-light:#4299e1;--blue-pale:#ebf8ff;
    --green:#38a169;--green-pale:#f0fff4;--orange:#dd6b20;--orange-pale:#fffaf0;
    --red:#e53e3e;--red-pale:#fff5f5;--purple:#805ad5;--purple-pale:#faf5ff;
    --yellow:#d69e2e;--yellow-pale:#fffff0;--gray-50:#f7fafc;--gray-100:#edf2f7;
    --gray-200:#e2e8f0;--gray-400:#a0aec0;--gray-600:#718096;--gray-800:#2d3748;
    --shadow-sm:0 1px 3px rgba(0,0,0,0.08);--shadow-md:0 4px 12px rgba(0,0,0,0.1);
    --radius:12px;
  }}
  body{{font-family:'Inter',sans-serif;background:#f0f4f8;color:var(--gray-800);line-height:1.6;}}

  /* ── Header ── */
  header{{
    background:linear-gradient(135deg,var(--blue-dark) 0%,var(--blue-mid) 60%,var(--blue-light) 100%);
    color:white;padding:2.5rem 3rem 2rem;
  }}
  header h1{{font-size:2rem;font-weight:800;letter-spacing:-0.5px;}}
  header .subtitle{{opacity:0.85;margin-top:0.4rem;font-size:0.95rem;}}
  .badge{{display:inline-block;background:rgba(255,255,255,0.18);border-radius:6px;
          padding:0.25rem 0.75rem;font-size:0.82rem;margin-top:0.6rem;margin-right:0.5rem;
          backdrop-filter:blur(4px);border:1px solid rgba(255,255,255,0.25);}}

  /* ── Layout ── */
  main{{max-width:1260px;margin:2rem auto;padding:0 1.5rem;}}

  /* ── Intro / How-to-read box ── */
  .intro-box{{
    background:linear-gradient(135deg,var(--blue-pale),#fff);
    border:1.5px solid #bee3f8;border-radius:var(--radius);
    padding:1.6rem 2rem;margin-bottom:2rem;
    box-shadow:var(--shadow-sm);
  }}
  .intro-box h2{{font-size:1.1rem;font-weight:700;color:var(--blue-mid);margin-bottom:0.8rem;}}
  .intro-box p{{font-size:0.9rem;color:var(--gray-800);margin-bottom:0.6rem;}}
  .intro-box ul{{margin:0.4rem 0 0.6rem 1.4rem;font-size:0.88rem;}}
  .intro-box li{{margin-bottom:0.25rem;}}

  /* ── KPI grid ── */
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:2rem;}}
  .kpi{{
    background:white;border-radius:var(--radius);padding:1.4rem 1.6rem;
    box-shadow:var(--shadow-sm);border-left:4px solid var(--blue-mid);
    transition:box-shadow .2s;
  }}
  .kpi:hover{{box-shadow:var(--shadow-md);}}
  .kpi .k-label{{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--gray-600);font-weight:600;}}
  .kpi .k-value{{font-size:1.85rem;font-weight:800;color:var(--blue-dark);margin-top:0.2rem;}}
  .kpi .k-sub{{font-size:0.73rem;color:var(--gray-400);margin-top:0.2rem;}}
  .kpi .k-status{{
    display:inline-block;font-size:0.72rem;font-weight:600;border-radius:999px;
    padding:0.15rem 0.6rem;margin-top:0.5rem;
  }}
  .status-good{{background:#c6f6d5;color:#22543d;}}
  .status-ok{{background:#feebc8;color:#7b341e;}}
  .status-bad{{background:#fed7d7;color:#742a2a;}}

  /* ── Generic section card ── */
  .section{{
    background:white;border-radius:var(--radius);padding:1.8rem;
    box-shadow:var(--shadow-sm);margin-bottom:1.8rem;
  }}
  .section h2{{
    font-size:1.15rem;font-weight:700;color:var(--blue-mid);
    border-bottom:2px solid #bee3f8;padding-bottom:0.5rem;margin-bottom:1.2rem;
  }}

  /* ── Figure grid / cards ── */
  .fig-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:1.2rem;}}
  .fig-card{{border-radius:10px;overflow:hidden;border:1px solid var(--gray-200);}}
  .fig-card img{{width:100%;display:block;}}
  .fig-caption{{
    padding:0.6rem 0.9rem;font-size:0.8rem;color:var(--gray-600);
    background:var(--gray-50);border-top:1px solid var(--gray-200);
  }}

  /* ── Collapsible details ── */
  .explainer{{
    border:1px solid var(--gray-200);border-radius:10px;
    margin-top:1rem;overflow:hidden;
  }}
  .explainer summary{{
    cursor:pointer;padding:0.85rem 1.1rem;font-weight:600;font-size:0.9rem;
    background:var(--gray-50);color:var(--blue-mid);
    display:flex;align-items:center;gap:0.5rem;list-style:none;
    user-select:none;
  }}
  .explainer summary::-webkit-details-marker{{display:none;}}
  .explainer summary::before{{
    content:"▶";font-size:0.65rem;color:var(--gray-400);
    transition:transform .2s;display:inline-block;
  }}
  .explainer[open] summary::before{{transform:rotate(90deg);}}
  .explainer-body{{
    padding:1.1rem 1.3rem;font-size:0.88rem;line-height:1.7;
    background:white;border-top:1px solid var(--gray-200);
  }}
  .explainer-body p{{margin-bottom:0.7rem;}}
  .explainer-body ul{{margin:0.3rem 0 0.7rem 1.4rem;}}
  .explainer-body li{{margin-bottom:0.25rem;}}
  .explainer-body strong{{color:var(--blue-dark);}}

  /* ── Reference range table (blood-test style) ── */
  .ref-table{{
    width:100%;border-collapse:collapse;margin:0.8rem 0;font-size:0.83rem;
    border-radius:8px;overflow:hidden;
  }}
  .ref-table th{{
    background:var(--blue-dark);color:white;padding:0.5rem 0.75rem;
    text-align:left;font-weight:600;font-size:0.8rem;
  }}
  .ref-table td{{padding:0.45rem 0.75rem;border-bottom:1px solid var(--gray-100);}}
  .ref-table tr:last-child td{{border-bottom:none;}}
  .ref-table tr:nth-child(even) td{{background:var(--gray-50);}}
  .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle;}}
  .dot-green{{background:#38a169;}}
  .dot-yellow{{background:#d69e2e;}}
  .dot-red{{background:#e53e3e;}}

  /* ── Metric grid for explainers ── */
  .metric-cards{{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem;
    margin-top:1rem;
  }}
  .metric-card{{
    border:1px solid var(--gray-200);border-radius:10px;overflow:hidden;
  }}
  .metric-card-header{{
    padding:0.75rem 1rem;font-weight:700;font-size:0.9rem;
    background:var(--blue-dark);color:white;
  }}
  .metric-card-body{{padding:0.9rem 1rem;font-size:0.85rem;}}
  .metric-card-body p{{margin-bottom:0.5rem;}}

  /* ── Divider ── */
  hr.section-hr{{border:none;border-top:2px solid var(--gray-200);margin:1.5rem 0;}}

  /* ── Footer ── */
  footer{{text-align:center;color:var(--gray-400);font-size:0.78rem;padding:2rem 0 3rem;}}
</style>
</head>
<body>
<header>
  <h1>CGDAP Evaluation Report</h1>
  <p class="subtitle">Generative model quality assessment — how well does the model learn to synthesise sensor data?</p>
  <span class="badge">Experiment: {experiment}</span>
  <span class="badge">Checkpoint: {checkpoint}</span>
  <span class="badge">Generated: {timestamp}</span>
  <span class="badge">Val samples: {n_val}</span>
  <span class="badge">Augmentation mode: {aug_mode}</span>
</header>

<main>

<!-- ═══════════════════════════════════════════════════════════════
     HOW TO READ THIS REPORT
════════════════════════════════════════════════════════════════ -->
<div class="intro-box">
  <h2>How to Read This Report — A Non-Technical Guide</h2>
  <p>
    This report evaluates a machine-learning model called <strong>CGDAP</strong> (Conditioned Generative
    Data Augmentation Pipeline). Its job is to <em>synthesise</em> (invent) realistic IoT sensor
    recordings that look like real ones. Think of it like a very capable photocopier for sensor data —
    the better the model, the harder it is to tell the copy from the original.
  </p>
  <p>
    Every metric below is a different way to measure how good that "photocopier" is.
    Like a blood-test report, each metric has a <strong>healthy range</strong> — we explain
    exactly what that range is and what it means for each one.
  </p>
  <ul>
    <li><strong>Green</strong> results are within the healthy/expected range — great news.</li>
    <li><strong>Orange/Yellow</strong> results are borderline and worth monitoring.</li>
    <li><strong>Red</strong> results indicate something the model is struggling with.</li>
  </ul>
  <p style="margin-top:0.4rem;font-size:0.83rem;color:#718096;">
    Each section also has a <em>"What this measures &amp; how to interpret it"</em> expandable panel
    — click the ▶ arrows to read the detailed explanation for any metric.
  </p>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     KPI CARDS
════════════════════════════════════════════════════════════════ -->
<div class="kpi-grid">

  <div class="kpi" style="border-left-color:var(--red);">
    <div class="k-label">Pair RMSE</div>
    <div class="k-value">{pair_rmse}</div>
    <div class="k-sub">Target vs extracted metrics</div>
    <span class="k-status {pair_rmse_status}">{pair_rmse_label}</span>
  </div>

  <div class="kpi" style="border-left-color:var(--orange);">
    <div class="k-label">Metric MAE</div>
    <div class="k-value">{metric_mae}</div>
    <div class="k-sub">Mean absolute error</div>
    <span class="k-status {metric_mae_status}">{metric_mae_label}</span>
  </div>

  <div class="kpi" style="border-left-color:var(--green);">
    <div class="k-label">Std-ratio</div>
    <div class="k-value">{std_ratio}</div>
    <div class="k-sub">σ_synth / σ_real &nbsp;(1.0 = perfect)</div>
    <span class="k-status {std_ratio_status}">{std_ratio_label}</span>
  </div>

  <div class="kpi" style="border-left-color:var(--blue-mid);">
    <div class="k-label">NN Distance</div>
    <div class="k-value">{nn_dist}</div>
    <div class="k-sub">Synth → val real bank (metric space)</div>
    <span class="k-status {nn_dist_status}">{nn_dist_label}</span>
  </div>

  <div class="kpi" style="border-left-color:var(--purple);">
    <div class="k-label">Diversity</div>
    <div class="k-value">{diversity}</div>
    <div class="k-sub">Mean pairwise distance (synth)</div>
    <span class="k-status {diversity_status}">{diversity_label}</span>
  </div>

  <div class="kpi" style="border-left-color:var(--yellow);">
    <div class="k-label">Worst-activity RMSE</div>
    <div class="k-value">{worst_rmse}</div>
    <div class="k-sub">Hardest activity for the model</div>
    <span class="k-status {worst_rmse_status}">{worst_rmse_label}</span>
  </div>

</div>

<!-- ═══════════════════════════════════════════════════════════════
     METRIC EXPLAINERS (collapsible)
════════════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>Metric Reference Guide — Understanding Your Results</h2>
  <p style="font-size:0.88rem;color:var(--gray-600);margin-bottom:1rem;">
    Click any metric below to expand its full explanation, analogy, and healthy reference range.
  </p>

  <!-- 1. Pair RMSE -->
  <details class="explainer">
    <summary>Pair RMSE — How far off is each generated sample from its target?</summary>
    <div class="explainer-body">
      <p>
        <strong>What it measures:</strong> For every validation sample, the model is given a target
        (what values the generated sensor data should have) and then asked to generate a spectrogram.
        The pipeline then <em>extracts</em> the actual values from the generated image and compares
        them to those targets. <strong>RMSE</strong> (Root Mean Squared Error) is the average gap
        between what was asked for and what was actually produced — penalising large misses more
        heavily than small ones.
      </p>
      <p>
        <strong>Analogy:</strong> Imagine you ask someone to bake a cake at exactly 180 °C. RMSE
        measures how far the actual oven temperature was from 180 °C — averaged over every cake baked.
        A smaller RMSE means the oven (model) is more reliably accurate.
      </p>
      <table class="ref-table">
        <tr><th>Range</th><th>Interpretation</th><th>Action</th></tr>
        <tr><td><span class="dot dot-green"></span>0.00 – 0.05</td><td>Excellent — model follows targets very closely</td><td>No action needed</td></tr>
        <tr><td><span class="dot dot-yellow"></span>0.05 – 0.15</td><td>Good — minor deviations, acceptable in practice</td><td>Monitor; check per-metric MAE for culprits</td></tr>
        <tr><td><span class="dot dot-red"></span>&gt; 0.15</td><td>Poor — model regularly misses targets significantly</td><td>Retrain with more steps or larger model</td></tr>
      </table>
      <p><strong>Tip:</strong> The scatter plot in the next section shows <em>which</em> metrics and activities drive the RMSE up.</p>
    </div>
  </details>

  <!-- 2. Metric MAE -->
  <details class="explainer">
    <summary>Metric MAE — On average, how wrong is the model?</summary>
    <div class="explainer-body">
      <p>
        <strong>What it measures:</strong> <strong>MAE</strong> (Mean Absolute Error) is similar to
        RMSE but treats all errors equally (no squaring). It is the plain average of
        |target − generated| across every metric and every sample.
      </p>
      <p>
        <strong>Analogy:</strong> If you shoot 10 arrows at a bullseye, MAE is the average distance
        of each arrow from the centre. RMSE would give extra weight to the arrows that went really
        wide. MAE gives you a simple, intuitive "how wrong on average" number.
      </p>
      <table class="ref-table">
        <tr><th>Range</th><th>Interpretation</th><th>Action</th></tr>
        <tr><td><span class="dot dot-green"></span>0.00 – 0.04</td><td>Excellent fidelity</td><td>—</td></tr>
        <tr><td><span class="dot dot-yellow"></span>0.04 – 0.12</td><td>Acceptable — check per-metric breakdown</td><td>Focus training on high-MAE metrics</td></tr>
        <tr><td><span class="dot dot-red"></span>&gt; 0.12</td><td>High error — model may be ill-conditioned</td><td>Investigate metric extractor and conditioning signal</td></tr>
      </table>
    </div>
  </details>

  <!-- 3. Std Ratio -->
  <details class="explainer">
    <summary>Std-Ratio — Does the model produce enough variety?</summary>
    <div class="explainer-body">
      <p>
        <strong>What it measures:</strong> Standard deviation (σ) measures <em>spread</em> — how much
        values vary across samples. The std-ratio compares the spread of <em>generated</em> data to the
        spread of <em>real</em> data. A ratio of 1.0 means the model produces exactly as much variety
        as real recordings. Below 1 means the model is "playing it safe" (mode collapse); above 1
        means it is over-generating noisy variety.
      </p>
      <p>
        <strong>Analogy:</strong> Think of writing — if a ghostwriter is asked to mimic an author's
        style, a std-ratio of 1.0 means their sentence lengths vary just as much as the original.
        A ratio of 0.3 means they always write medium-length sentences (too uniform). A ratio of 2.0
        means they write wildly inconsistent sentences (too random).
      </p>
      <table class="ref-table">
        <tr><th>Range</th><th>Interpretation</th><th>Action</th></tr>
        <tr><td><span class="dot dot-green"></span>0.80 – 1.20</td><td>Excellent variance matching</td><td>—</td></tr>
        <tr><td><span class="dot dot-yellow"></span>0.50 – 0.80 or 1.20 – 1.60</td><td>Moderate mode collapse or over-dispersion</td><td>Review diversity / temperature settings</td></tr>
        <tr><td><span class="dot dot-red"></span>&lt; 0.50 or &gt; 1.60</td><td>Severe collapse or instability</td><td>Retrain; check diffusion scheduler settings</td></tr>
      </table>
      <p><strong>Std-ratio drift</strong> (|σ_synth/σ_real − 1|) is just the absolute deviation from perfect (1.0) — closer to 0 is better.</p>
    </div>
  </details>

  <!-- 4. NN Distance -->
  <details class="explainer">
    <summary>NN Distance — Are generated samples realistic? Can they "fit in" with real data?</summary>
    <div class="explainer-body">
      <p>
        <strong>What it measures:</strong> For each generated sample, we find its <em>nearest
        neighbour</em> (most similar sample) in the real validation bank — but in the
        multi-dimensional <em>metric space</em> (not pixel space). A low mean distance means
        generated samples are close to real ones; a high distance means they live in a different
        region of the distribution.
      </p>
      <p>
        <strong>Analogy:</strong> Imagine you generate a fake painting and show it alongside real
        paintings in a gallery. Art critics measure how similar the fake is to the nearest real one
        (style, colour palette, brushstroke texture). If the fake is indistinguishable, the distance
        is near 0. If it looks alien, the distance is large.
      </p>
      <p>
        <strong>Important caveat:</strong> distances are computed in a <em>standardised</em> metric
        space (z-scored), so the scale is in standard-deviation units, not raw sensor units.
      </p>
      <table class="ref-table">
        <tr><th>Range (z-score units)</th><th>Interpretation</th><th>Action</th></tr>
        <tr><td><span class="dot dot-green"></span>0.0 – 1.0</td><td>Generated samples sit inside the real distribution</td><td>—</td></tr>
        <tr><td><span class="dot dot-yellow"></span>1.0 – 2.0</td><td>Slight distributional gap — plausible but not identical</td><td>Check histogram tail; consider more training data</td></tr>
        <tr><td><span class="dot dot-red"></span>&gt; 2.0</td><td>Generated samples are OOD — unrealistic</td><td>Revisit conditioning and normalisation</td></tr>
      </table>
    </div>
  </details>

  <!-- 5. Diversity -->
  <details class="explainer">
    <summary>Diversity — Is the model creative, or does it always produce the same thing?</summary>
    <div class="explainer-body">
      <p>
        <strong>What it measures:</strong> Diversity is the mean pairwise distance between all
        generated samples in metric space. If the model always produces nearly identical outputs
        regardless of the input (called <em>mode collapse</em>), diversity will be very low.
        Healthy diversity means the model explored the full range of valid outputs.
      </p>
      <p>
        <strong>Analogy:</strong> Ask 100 students to draw a tree. Low diversity = they all draw the
        same cartoon tree. High diversity = every tree looks different (some are pine, some are oak,
        some are bare). For an augmentation model, you want high diversity — otherwise all your
        generated training data looks the same and provides no extra benefit.
      </p>
      <table class="ref-table">
        <tr><th>Range</th><th>Interpretation</th><th>Action</th></tr>
        <tr><td><span class="dot dot-green"></span>&gt; 2.0</td><td>Good variety — model explores the space</td><td>—</td></tr>
        <tr><td><span class="dot dot-yellow"></span>0.8 – 2.0</td><td>Limited variety — partial mode collapse</td><td>Increase diffusion steps or classifier-free guidance weight</td></tr>
        <tr><td><span class="dot dot-red"></span>&lt; 0.8</td><td>Severe mode collapse — almost no variety</td><td>Retrain; check conditioning and loss weighting</td></tr>
      </table>
      <p><strong>Note:</strong> Diversity and NN Distance are complementary — you want high diversity AND low NN distance. A model that is diverse but unrealistic scores badly on NN Distance. A model that is realistic but repetitive scores badly on Diversity.</p>
    </div>
  </details>

  <!-- 6. Per-activity RMSE -->
  <details class="explainer">
    <summary>Per-activity RMSE — Which human activities confuse the model most?</summary>
    <div class="explainer-body">
      <p>
        <strong>What it measures:</strong> The same RMSE as above, but broken down by
        <em>activity</em> (e.g., walking, running, sitting). This tells us if the model performs
        uniformly or if certain activities are harder to synthesise accurately.
      </p>
      <p>
        <strong>Why it matters:</strong> In a real deployment, a model that fails on one activity
        class could cause that class to be under-represented in augmented training data, ultimately
        hurting downstream classifier performance for that activity.
      </p>
      <table class="ref-table">
        <tr><th>Scenario</th><th>Interpretation</th></tr>
        <tr><td>All activities cluster tightly near the overall RMSE</td><td>Model is equitable across activities — ideal</td></tr>
        <tr><td>One activity is much higher than the rest</td><td>That activity has unusual sensor patterns the model hasn't learned well</td></tr>
        <tr><td>Overall RMSE is low but one bar is very tall</td><td>Consider activity-specific fine-tuning or more data for that class</td></tr>
      </table>
    </div>
  </details>

  <!-- 7. Sensor metrics -->
  <details class="explainer">
    <summary>The 5 Sensor Metrics — What the model tries to control</summary>
    <div class="explainer-body">
      <p>
        The model does not generate raw sensor values directly — it operates in
        <em>spectrogram space</em> (a 2-D frequency-vs-time image of the sensor signal). To
        condition the model, five compact numerical features are extracted from each spectrogram
        using a learned extractor network. These five features are what the MAE and RMSE computations
        operate on:
      </p>
      <div class="metric-cards">
        <div class="metric-card">
          <div class="metric-card-header">Mean Energy</div>
          <div class="metric-card-body">
            <p>The overall "loudness" or intensity of the sensor signal. A running person has higher accelerometer energy than a sitting person.</p>
            <p><strong>Good MAE:</strong> &lt; 0.05 | <strong>Concerning:</strong> &gt; 0.15</p>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-card-header">Spectral Centroid</div>
          <div class="metric-card-body">
            <p>The "centre of gravity" of the frequency content — where most of the signal energy sits on the frequency axis. High centroid = fast oscillations dominate.</p>
            <p><strong>Good MAE:</strong> &lt; 0.05 | <strong>Concerning:</strong> &gt; 0.15</p>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-card-header">Spectral Flux</div>
          <div class="metric-card-body">
            <p>How rapidly the frequency content changes over time. Walking has periodic flux; random noise has high irregular flux.</p>
            <p><strong>Good MAE:</strong> &lt; 0.05 | <strong>Concerning:</strong> &gt; 0.15</p>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-card-header">Temporal Entropy</div>
          <div class="metric-card-body">
            <p>A measure of signal unpredictability across time. Low entropy = repetitive, regular motion (e.g., cycling at constant speed). High entropy = irregular, noisy signal.</p>
            <p><strong>Good MAE:</strong> &lt; 0.05 | <strong>Concerning:</strong> &gt; 0.15</p>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-card-header">Crest Factor</div>
          <div class="metric-card-body">
            <p>The ratio of peak signal value to its RMS value. Reveals whether the signal has occasional sharp spikes (high crest) or a sustained level (low crest). Useful for detecting impact events.</p>
            <p><strong>Good MAE:</strong> &lt; 0.05 | <strong>Concerning:</strong> &gt; 0.15</p>
          </div>
        </div>
      </div>
      <p style="margin-top:1rem;font-size:0.83rem;color:var(--gray-600);">
        All five metrics are normalised before training, so a MAE of 0.05 corresponds to 5 % of the
        typical value range in the training data — not the raw physical unit.
      </p>
    </div>
  </details>

  <!-- 8. Std-ratio drift -->
  <details class="explainer">
    <summary>Std-Ratio Drift — How stable is the model's spread control?</summary>
    <div class="explainer-body">
      <p>
        <strong>What it measures:</strong> This is simply |std-ratio − 1.0|. While std-ratio tells you
        <em>direction</em> (over- or under-dispersed), drift tells you the <em>magnitude</em> of the
        problem without caring about direction. A drift of 0.0 is perfect; 0.5 means the model's spread
        is 50 % off from reality.
      </p>
      <table class="ref-table">
        <tr><th>Drift Range</th><th>Interpretation</th></tr>
        <tr><td><span class="dot dot-green"></span>0.00 – 0.15</td><td>Excellent variance control</td></tr>
        <tr><td><span class="dot dot-yellow"></span>0.15 – 0.40</td><td>Moderate — check which metrics have high drift in the bar chart</td></tr>
        <tr><td><span class="dot dot-red"></span>&gt; 0.40</td><td>High drift — model's spread is significantly wrong</td></tr>
      </table>
    </div>
  </details>

  <!-- 9. Spectrograms -->
  <details class="explainer">
    <summary>Spectrograms — What does "good" look like visually?</summary>
    <div class="explainer-body">
      <p>
        A <strong>spectrogram</strong> is a 2-D image where:
      </p>
      <ul>
        <li><strong>X-axis</strong> = time (left to right)</li>
        <li><strong>Y-axis</strong> = frequency (low at bottom, high at top)</li>
        <li><strong>Colour</strong> = intensity (brighter / more yellow = stronger signal at that frequency and time)</li>
      </ul>
      <p>
        In the gallery, each row shows a <em>reference</em> (real sensor recording) next to the
        model's <em>generated</em> counterpart for the same activity.
      </p>
      <p><strong>What to look for:</strong></p>
      <ul>
        <li>Do the two images have similar brightness patterns and colour distribution?</li>
        <li>Does the generated one have the same rough structure (e.g., periodic stripes for walking)?</li>
        <li>Are the frequency bands in similar regions?</li>
      </ul>
      <p>
        Small differences are normal and even desirable (augmentation = adding variety), but large
        structural differences (very different frequency bands or random noise) would indicate the model
        is not capturing the underlying activity pattern.
      </p>
    </div>
  </details>

</div>


<!-- ═══════════════════════════════════════════════════════════════
     SUMMARY TABLE
════════════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>Summary Metrics Table</h2>
  <p style="font-size:0.85rem;color:var(--gray-600);margin-bottom:1rem;">
    A consolidated view of all computed diagnostic values. Section headers in blue group
    related metrics together.
  </p>
  <div class="fig-grid">
    <div class="fig-card">
      <img src="data:image/png;base64,{b64_table}" alt="Summary Table"/>
      <div class="fig-caption">All aggregate diagnostic metrics from the evaluation pass. Blue section headers separate core quality, distribution, per-metric, and per-activity sub-groups.</div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     SCATTER PLOT
════════════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>Target vs Extracted Metrics — Scatter Plots</h2>
  <p style="font-size:0.85rem;color:var(--gray-600);margin-bottom:1rem;">
    Each scatter plot shows one metric (column) for one sensor modality (row). Each dot is one validation
    sample. The target value (what we asked for) is on the X-axis; the extracted value (what the model
    actually produced) is on the Y-axis. <strong>Perfect model = all dots on the dashed diagonal line.</strong>
    Colour encodes activity class. Dots scattered widely off the diagonal indicate the model struggles
    to hit targets for that metric/modality combination.
  </p>
  <div class="fig-card">
    <img src="data:image/png;base64,{b64_scatter}" alt="Metric Scatter"/>
    <div class="fig-caption">
      Each point = one val sample. Diagonal dashed line = perfect fidelity (extracted = target).
      Colour = activity class. MAE per sub-plot shown in title.
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     PER-ACTIVITY RMSE
════════════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>Per-Activity Pair RMSE</h2>
  <p style="font-size:0.85rem;color:var(--gray-600);margin-bottom:1rem;">
    RMSE broken down by physical activity class. The dashed line is the overall RMSE across all
    activities (your reference baseline). Bars <strong>above</strong> the line represent activities
    the model handles worse than average; bars <strong>below</strong> represent ones it handles
    better. Aim for all bars to be at roughly the same height — that means the model is equitable
    and no single activity is being neglected.
  </p>
  <div class="fig-card">
    <img src="data:image/png;base64,{b64_act_rmse}" alt="Per-activity RMSE"/>
    <div class="fig-caption">Bar height = pair RMSE for that activity. Dashed horizontal line = overall pair RMSE. Lower bars = better performance on that activity.</div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     RADAR + STD RATIO
════════════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>Metric Fidelity Radar &amp; Variance Matching</h2>
  <p style="font-size:0.85rem;color:var(--gray-600);margin-bottom:1rem;">
    <strong>Left — Radar chart:</strong> Each spoke is one sensor metric. Each coloured ring is one
    activity class. A value of <strong>1 (outer edge)</strong> means that activity had the
    <em>lowest</em> MAE (best) for that metric; <strong>0 (centre)</strong> means highest MAE (worst).
    All values are normalized across activities, so this is a relative comparison — not absolute error.
    Aim for all rings to be large and spread evenly out to the edges.
    <br/><br/>
    <strong>Right — Std-ratio bars:</strong> Each bar is one sensor metric. The dashed line at 1.0 is
    the ideal. A bar at 0.6 means the generated data is 40 % too uniform; a bar at 1.4 means it is
    40 % too noisy. The closer each bar is to 1.0, the better the model replicates the real world's
    variety for that metric.
  </p>
  <div class="fig-grid">
    <div class="fig-card">
      <img src="data:image/png;base64,{b64_radar}" alt="Radar Chart"/>
      <div class="fig-caption">Per-activity metric fidelity (1 = lowest MAE / best, normalised across activities). Outer edge = best; centre = worst.</div>
    </div>
    <div class="fig-card">
      <img src="data:image/png;base64,{b64_std}" alt="Std Ratio"/>
      <div class="fig-caption">Per-metric standard deviation ratio (σ_synth / σ_real). Dashed line at 1.0 = perfect variance match. &lt;1 means model is too uniform; &gt;1 means too noisy.</div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     NN DISTANCE HISTOGRAM
════════════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>Nearest-Neighbour Distance — Are Generated Samples "Real Enough"?</h2>
  <p style="font-size:0.85rem;color:var(--gray-600);margin-bottom:1rem;">
    This histogram answers: <em>"How close is each generated sample to the real data it is
    supposed to resemble?"</em> Each generated sample is projected into a multi-dimensional
    metric space (the 5 sensor features × number of modalities, z-scored). We then find its
    nearest real validation neighbour with the same activity label, and measure the Euclidean
    distance between them.
    <br><br>
    <strong>Left spike = very realistic samples.</strong> A long right tail = some generated
    samples are "out in the wilderness" far from any real example. The red dashed line is the
    mean — ideally below 1.0 in z-score units.
  </p>
  <div class="fig-card">
    <img src="data:image/png;base64,{b64_nn}" alt="NN Distance Histogram"/>
    <div class="fig-caption">Distribution of nearest-neighbour distances (synthetic → real val bank) in standardised metric space. Red dashed line = mean. Bars concentrated near 0 = realistic generation.</div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     SPECTROGRAM GALLERY
════════════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>Spectrogram Gallery — Real vs Generated, Side by Side</h2>
  <p style="font-size:0.85rem;color:var(--gray-600);margin-bottom:1rem;">
    A random selection of validation pairs. Each row shows one sample. For each row:
    the <strong>left image</strong> is the real sensor recording (spectrogram); the
    <strong>right image</strong> is what the model generated for the same activity and
    metric targets. The small text annotations list the 5 sensor metric values.
    <br><br>
    Different sensor modalities are shown as separate column pairs (e.g., accelerometer X/Y/Z,
    gyroscope, etc.). The colour scale (viridis) is the same for both images in each pair —
    so you can directly compare brightness and structure. Good generation will show similar
    texture, brightness distribution, and frequency band structure to the reference.
  </p>
  <div class="fig-card">
    <img src="data:image/png;base64,{b64_gallery}" alt="Spectrogram Gallery"/>
    <div class="fig-caption">
      Randomly sampled val pairs: reference (real) spectrogram vs model-generated spectrogram.
      Activity label shown in reference title. Metric annotations (target &amp; extracted values) annotated beside each image.
    </div>
  </div>
</div>

</main>
<footer>
  Generated by CGDAP <code>run_eval_report.py</code> &bull; {timestamp}
  &bull; <em>CGDAP — Conditioned Generative Data Augmentation Pipeline</em>
</footer>
</body>
</html>
"""


def _kpi_status(value: float, thresholds_good_ok: tuple[float, float], low_is_good: bool = True) -> tuple[str, str]:
    """Return (css_class, label) for a KPI value.

    thresholds_good_ok: (good_boundary, ok_boundary)
      - low_is_good=True  → value < good_boundary = good, < ok_boundary = ok, else bad
      - low_is_good=False → value > good_boundary = good, > ok_boundary = ok, else bad
    """
    good_t, ok_t = thresholds_good_ok
    if low_is_good:
        if value <= good_t:
            return "status-good", "Excellent"
        elif value <= ok_t:
            return "status-ok", "Acceptable"
        else:
            return "status-bad", "Needs work"
    else:
        if value >= good_t:
            return "status-good", "Excellent"
        elif value >= ok_t:
            return "status-ok", "Borderline"
        else:
            return "status-bad", "Needs work"


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
    # ── Compute per-KPI status badges ──────────────────────────────
    pair_rmse_val  = agg["pair_rmse"]
    mae_val        = agg["metric_mae"]
    std_ratio_val  = agg["std_ratio_mean"]
    nn_dist_val    = agg["nn_distance_mean"]
    diversity_val  = agg["diversity_mean"]
    worst_rmse_val = agg["worst_activity_rmse"]

    pr_cls, pr_lbl   = _kpi_status(pair_rmse_val,  (0.05, 0.15),  low_is_good=True)
    mae_cls, mae_lbl = _kpi_status(mae_val,         (0.04, 0.12),  low_is_good=True)
    nn_cls, nn_lbl   = _kpi_status(nn_dist_val,     (1.0, 2.0),    low_is_good=True)
    wr_cls, wr_lbl   = _kpi_status(worst_rmse_val,  (0.05, 0.20),  low_is_good=True)
    div_cls, div_lbl = _kpi_status(diversity_val,   (2.0, 0.8),    low_is_good=False)

    # Std-ratio: closest to 1.0 is best → convert to drift distance
    std_drift = abs(std_ratio_val - 1.0)
    sr_cls, sr_lbl = _kpi_status(std_drift, (0.20, 0.40), low_is_good=True)

    return HTML_TEMPLATE.format(
        experiment=experiment,
        checkpoint=pathlib.Path(checkpoint).name,
        timestamp=timestamp,
        n_val=n_val,
        aug_mode=aug_mode,
        pair_rmse=f"{pair_rmse_val:.4f}",
        metric_mae=f"{mae_val:.4f}",
        std_ratio=f"{std_ratio_val:.4f}",
        nn_dist=f"{nn_dist_val:.4f}",
        diversity=f"{diversity_val:.4f}",
        worst_rmse=f"{worst_rmse_val:.4f}",
        pair_rmse_status=pr_cls,  pair_rmse_label=pr_lbl,
        metric_mae_status=mae_cls, metric_mae_label=mae_lbl,
        std_ratio_status=sr_cls,  std_ratio_label=sr_lbl,
        nn_dist_status=nn_cls,    nn_dist_label=nn_lbl,
        diversity_status=div_cls, diversity_label=div_lbl,
        worst_rmse_status=wr_cls, worst_rmse_label=wr_lbl,
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
    print(f"\n  HTML report  -> {report_path.resolve()}")
    print(f"  Figures dir  -> {fig_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()
