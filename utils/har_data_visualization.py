"""har_data_visualization.py
Quick visualization of a few STFT samples from the processed HAR dataset.

Shows the magnitude spectrogram (|STFT|) for both acc and gyr, one sample
per activity class. Channels x, y, z are shown as separate rows.
"""

import pathlib
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf

# ------------------------------------------------------------------ #
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CFG = OmegaConf.load(_REPO_ROOT / "configs" / "dataset" / "har_dataset.yaml")

PROCESSED_ROOT = pathlib.Path(_CFG.dataset.paths.processed)
SPLIT = "train"          # "train" or "val"
N_SAMPLES_PER_CLASS = 1  # how many samples to show per activity
# ------------------------------------------------------------------ #

meta_path = PROCESSED_ROOT / "metadata.json"
with meta_path.open() as f:
    meta = json.load(f)

stft_meta = meta["stft"]
hop_length = stft_meta["hop_length"]
win_length = stft_meta["win_length"]
freq_bins  = stft_meta["freq_bins"]     # n_fft // 2 + 1

split_dir = PROCESSED_ROOT / SPLIT
activities = sorted(d.name for d in split_dir.iterdir() if d.is_dir())

CHANNELS = ["x", "y", "z"]
MODALITIES = ["acc", "gyr"]

# Collect one acc+gyr pair per activity
samples: list[dict] = []
for activity in activities:
    act_dir = split_dir / activity
    pt_files = sorted(act_dir.glob("*_acc.pt"))
    for pt in pt_files[:N_SAMPLES_PER_CLASS]:
        stem = pt.stem[:-4]  # strip "_acc"
        gyr_path = act_dir / f"{stem}_gyr.pt"
        if gyr_path.exists():
            samples.append({"activity": activity, "acc": pt, "gyr": gyr_path, "stem": stem})
            break

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
n_activities = len(samples)
n_modalities = len(MODALITIES)
n_channels   = len(CHANNELS)

# Pre-load all tensors
tensors_list: list[dict] = []
for s in samples:
    tensors_list.append({
        "acc": torch.load(s["acc"], weights_only=True),   # (3, freq_bins, time_frames)
        "gyr": torch.load(s["gyr"], weights_only=True),
    })


# ── Figure layout ────────────────────────────────────────────────── #
# Two vertical blocks (ACC / GYR), each containing n_channels rows.
# Within each block: n_activities columns.
# A visible gap separates the two modality blocks.
COL_W  = 3.0   # inches per activity column
ROW_H  = 1.6   # inches per channel row

fig = plt.figure(
    figsize=(COL_W * n_activities + 1.2, ROW_H * n_channels * n_modalities + 1.4),
    constrained_layout=False,
)
fig.suptitle(
    f"HAR – STFT Magnitude Spectrograms  |  split = {SPLIT}",
    fontsize=12, fontweight="bold", y=0.98,
)

# Outer grid: two rows (ACC block, GYR block) with a gap
outer = gridspec.GridSpec(
    n_modalities, 1,
    figure=fig,
    top=0.92, bottom=0.06,
    left=0.10, right=0.93,
    hspace=0.35,
)

axes_grid: dict[str, list[list]] = {}   # axes_grid[mod][ch_idx][act_idx]

for m_idx, mod in enumerate(MODALITIES):
    inner = gridspec.GridSpecFromSubplotSpec(
        n_channels, n_activities,
        subplot_spec=outer[m_idx],
        hspace=0.06,
        wspace=0.06,
    )

    axes_grid[mod] = []

    for c_idx, ch in enumerate(CHANNELS):
        row_axes: list = []
        for col, (s, tensors) in enumerate(zip(samples, tensors_list)):
            ax = fig.add_subplot(inner[c_idx, col])
            mag = tensors[mod].abs().numpy()   # (3, freq, time)
            S_db = 20 * np.log10(mag[c_idx] + 1e-8)
            vmin, vmax = np.percentile(S_db, [5, 95])

            im = ax.imshow(
                S_db,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            # Activity titles – top row of the ACC block only
            if m_idx == 0 and c_idx == 0:
                ax.set_title(
                    s["activity"].replace("_", "\n"),
                    fontsize=9, fontweight="bold", pad=4,
                )

            # Channel label – left column only
            if col == 0:
                ax.set_ylabel(ch, fontsize=9, rotation=0,
                              labelpad=10, va="center")

            # Time-frame label – bottom row of each modality block
            if c_idx == n_channels - 1:
                ax.set_xlabel("time", fontsize=7)

            row_axes.append(ax)

        axes_grid[mod].append(row_axes)

    # One shared vertical colorbar per modality block (rightmost column)
    last_col_axes = [axes_grid[mod][c][n_activities - 1] for c in range(n_channels)]
    cb = fig.colorbar(
        im, ax=last_col_axes,
        location="right",
        pad=0.03, fraction=0.06, shrink=0.9,
    )
    cb.ax.tick_params(labelsize=7)
    cb.set_label("|STFT| (dB)", fontsize=8)

    # Modality label flush-left of the block
    mid_row = axes_grid[mod][n_channels // 2][0]
    mid_row.annotate(
        mod.upper(),
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-42, 0), textcoords="offset points",
        fontsize=11, fontweight="bold", va="center", ha="center",
        rotation=90,
    )

plt.show()
