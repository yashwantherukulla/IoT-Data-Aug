# IoT Data Aug (CGDAP v2.1)

Conditional generative data augmentation pipeline for Human Activity Recognition (HAR) from wearable inertial sensors.

This repository contains:
- data preprocessing from RealWorld HAR raw sensor archives
- differentiable metric extraction from spectrograms
- a multimodal conditional diffusion model (`MultimodalCGDAP`)
- downstream HAR classifier evaluation (DeepSense + Transformer)
- **full validation evaluation report** with figures and a self-contained HTML report (`run_eval_report.py`)
- tests for datasets, metrics, and core model blocks

All details below were verified against the current codebase and current workspace artifacts on 2026-04-10.

## Verified Project Snapshot

- Package: `iot-data-aug` (`version = 2.1.0`)
- Python requirement: `>=3.13`
- Build backend: `hatchling`
- Config system: Hydra
- Default modalities: `acc`, `gyr`
- Default activities (canonical labels):
  - `climbing_down`
  - `climbing_up`
  - `jumping`
  - `running`
  - `walking`

Current workspace verification:
- `data/processed/HAR/metadata.json` exists and declares `contract_version: 2.1`
- `data/processed/HAR/train/{acc,gyr}` and `data/processed/HAR/val/{acc,gyr}` exist
- Test suite status: `17 passed`
- Checkpoints present: `outputs/checkpoints/test_run/ckpt_epoch0000.pt` … `ckpt_epoch0039.pt` (40 epochs)

## Repository Structure

```text
.
|-- cgdap/
|   |-- augmentation/      # metric-target generation for synthetic sampling
|   |-- data/              # preprocessing, dataset loaders, raw-loader shim
|   |-- evaluation/        # DeepSense and Transformer classifiers
|   |-- metrics/           # differentiable metric extraction
|   |-- models/            # base interfaces, condition embedder, DDPM, U-Net, CGDAP wrapper
|   `-- training/          # trainer loop
|-- configs/
|   |-- augmentation/
|   |-- dataset/
|   |-- evaluation/
|   |-- logging/
|   |-- model/
|   |-- training/
|   `-- config.yaml
|-- scripts/
|   |-- prepare_dataset.py
|   |-- train.py
|   |-- generate.py
|   |-- evaluate.py
|   `-- run_eval_report.py
|-- tests/
|-- data/
|   |-- raw/
|   `-- processed/
|-- outputs/
|-- main.py
`-- pyproject.toml
```

## Pipeline Overview

1. Preprocess raw RealWorld HAR subject archives from `data/raw/HAR` into spectrogram `.pt` samples.
2. Train `MultimodalCGDAP` on paired modalities (`acc` + `gyr`) with conditional diffusion + metric consistency.
3. Evaluate HAR classifiers on real-only vs real+synthetic training data.
4. **Run the evaluation report** (`scripts/run_eval_report.py`) to generate a full diagnostic report over all val samples.

The synthetic path in evaluation uses:
- `AugmentationEngine` (target metrics)
- `MultimodalCGDAP.sample(...)` (generate spectrograms)
- concatenation with real training data (`ConcatDataset`)

## Data Contracts

### Raw Data Layout

Expected root:

```text
data/raw/HAR/
  proband1/
  proband2/
  ...
  proband15/
```

Preprocessing expects exactly 15 subject directories containing activity sensor archives with names like:
- `acc_<activity>_csv.zip`
- `gyr_<activity>_csv.zip`

The active pipeline reads zipped CSV payloads directly.

Note on `dataset.pipeline.run_clean`:
- The legacy raw cleaning pipeline is now a deprecated no-op shim.
- Setting `dataset.pipeline.run_clean=true` does not transform raw data; it logs a warning and touches the sentinel.

### Processed Data Layout (v2.1)

```text
data/processed/HAR/
  metadata.json
  train/
    acc/<activity>/*.pt
    gyr/<activity>/*.pt
  val/
    acc/<activity>/*.pt
    gyr/<activity>/*.pt
```

Each `.pt` file contains:

```python
{
    "spectrogram": Tensor[3, F, T],
    "metrics": Tensor[5],
    "label": int,
    "activity": str,
    "subject": str,
    "window_index": int,
    "sample_rate_hz": float,
    "freq_axis_hz": Tensor[F],
    "time_axis_s": Tensor[T],
}
```

`PairedDataset` aligns `acc` and `gyr` by matching activity folder + filename stem.

## Spectrogram and Metric Details

Default dataset config (`configs/dataset/har_dataset.yaml`):
- sample rate: `100.0 Hz`
- window length: `2.5 s` (`250` samples)
- STFT window: `1500 ms` (`win_length=150`)
- STFT hop: `20 ms` (`hop_length=2`)
- `n_fft`: next power of two (`256`)
- spectrogram transform: magnitude (`power=1.0`) with `log1p=true`

Resulting default spectrogram dimensions from metadata:
- `F = 129`
- `T = 126`

Metrics (order is fixed):
1. `temporal_range`
2. `f0_amplitude`
3. `contrast`
4. `flatness`
5. `entropy`

Implemented in `cgdap/metrics/extractor.py` with both:
- functional API for preprocessing (`compute_metrics_fn`)
- batched differentiable module for training (`MetricExtractor`)

## Model Architecture

Top-level model: `MultimodalCGDAP` (`cgdap/models/cgdap.py`)

Composition:
- one denoiser per modality (independent weights)
- shared condition embedder
- shared DDPM schedule
- shared metric extractor

Default selected components:
- denoiser: `ConditionalUNet`
- schedule: `DDPMSchedule`
- embedder: `CrossAttentionConditionEmbedder`

Loss terms:
- `L_G`: diffusion noise-prediction MSE
- `L_metric`: weighted metric-consistency loss from reconstructed `x0`
- `L_total = L_G + L_metric`

Adaptive metric weights:
- updated only during training (`self.training` guard is present)
- EMA-smoothed per-metric losses
- clamped weight range from config

Sampling:
- reproducible optional base seed
- per-modality seed offsets (`seed + modality_index`) to decorrelate streams

## Augmentation Modes

Configured in `configs/augmentation/default.yaml`:
- `interpolation`
- `disturbance` (default)
- `domain_instruction`

Outputs target dictionary shape:
- `{ "acc": Tensor[5], "gyr": Tensor[5], "label": int }`

`domain_instruction` supports modality-specific metric ranges:
- `domain_instruction.<activity>.<modality>.<metric>`

## Evaluation Behavior

Entry point: `scripts/evaluate.py`

By default (`configs/evaluation/default.yaml`):
- classifiers: `deepsense`, `transformer`
- augmentation: `enabled: true`
- `samples_per_real: 1`

Important:
- With augmentation enabled, evaluation requires a diffusion checkpoint.
- Checkpoint resolution order:
  1. `evaluation.augmentation.checkpoint_path` (if set)
  2. latest `ckpt_epoch*.pt` under `training.checkpoint_dir/experiment_name`
- If no checkpoint is found, evaluation raises `FileNotFoundError`.

Given current workspace state (no checkpoint under `outputs/checkpoints/`), run evaluation either:
- after training, or
- with augmentation disabled.

## Configuration Map

Root config: `configs/config.yaml`

Default composition:
- `dataset: har_dataset`
- `model: cgdap`
- `training: default`
- `augmentation: default`
- `evaluation: default`
- `logging: console`

Useful override examples:

```bash
uv run python scripts/train.py training.max_epochs=10
uv run python scripts/train.py model.unet.base_channels=32 training.batch_size=4
uv run python scripts/train.py logging=wandb
uv run python scripts/evaluate.py evaluation.augmentation.enabled=false
uv run python scripts/evaluate.py evaluation.classifiers=[transformer]
```

## Setup

### Option A: uv (recommended)

```bash
uv sync
```

### Option B: existing virtual environment

```bash
.\.venv\Scripts\python.exe -m pip install -e .
```

## End-to-End Commands

### 1. Preprocess Dataset

```bash
uv run python scripts/prepare_dataset.py
```

Optional flags:

```bash
uv run python scripts/prepare_dataset.py dataset.pipeline.force_regenerate=false
uv run python scripts/prepare_dataset.py dataset.pipeline.run_clean=true
```

### 2. Train Diffusion Model

```bash
uv run python scripts/train.py
```

Smoke-style reduced run:

```bash
uv run python scripts/train.py model.unet.base_channels=32 training.batch_size=4 training.max_epochs=2
```

Resume from the latest checkpoint for the same experiment:

```bash
uv run python scripts/train.py experiment_name=test_run training.resume=true
```

Resume from an explicit checkpoint file:

```bash
uv run python scripts/train.py training.resume=true training.resume_checkpoint="outputs/checkpoints/test_run/ckpt_epoch0012.pt"
```

Outputs:
- Hydra run directories: `outputs/<date>/<time>/`
- Checkpoints: `outputs/checkpoints/<experiment_name>/ckpt_epochXXXX.pt`

### 3. Evaluate Classifiers

Real-only evaluation (no checkpoint needed):

```bash
uv run python scripts/evaluate.py evaluation.augmentation.enabled=false
```

Real + synthetic evaluation (checkpoint required):

```bash
uv run python scripts/evaluate.py
```

or explicit checkpoint:

```bash
uv run python scripts/evaluate.py evaluation.augmentation.checkpoint_path="outputs/checkpoints/test_run/ckpt_epoch0000.pt"
```

### 4. Run Evaluation Report

Generates a full diagnostic report over **all val samples** (2,513 paired samples). Produces a self-contained HTML report and 7 PNG figures covering metric fidelity, NN distances, spectrogram quality, and per-activity breakdowns.

```bash
# Default — auto-picks latest checkpoint for the configured experiment_name
uv run python scripts/run_eval_report.py

# or using the venv Python directly
.venv\Scripts\python.exe scripts/run_eval_report.py
```

With an explicit checkpoint:

```bash
.venv\Scripts\python.exe scripts/run_eval_report.py \
  evaluation.augmentation.checkpoint_path="outputs/checkpoints/test_run/ckpt_epoch0039.pt"
```

Fewer diffusion steps for faster runs (trades some quality):

```bash
.venv\Scripts\python.exe scripts/run_eval_report.py evaluation.product_eval.num_steps=25
```

Outputs written to `outputs/eval_report/<experiment_name>/<timestamp>/`:

| File | Content |
|---|---|
| `report.html` | Self-contained report with all figures embedded |
| `figures/01_summary_table.png` | All aggregate diagnostic metrics |
| `figures/02_metric_scatter.png` | Target vs extracted scatter (per metric × modality) |
| `figures/03_nn_distance_hist.png` | NN distance histogram (synth → val real bank) |
| `figures/04_radar_chart.png` | Per-activity metric fidelity radar chart |
| `figures/05_std_ratio_bars.png` | σ_synth / σ_real per metric (ideal = 1.0) |
| `figures/06_spectrogram_gallery.png` | Reference vs generated spectrogram pairs |
| `figures/07_per_activity_rmse.png` | Per-activity pair RMSE bar chart |

Key metrics reported:
- **Pair RMSE** — target vs extracted metric MSE (lower = better fidelity)
- **NN Distance** — synthetic samples' distance to real val bank in metric space
- **Std-ratio** — σ_synth / σ_val per metric (1.0 = perfect variance matching)
- **Diversity** — mean pairwise distance among generated samples
- **Per-activity RMSE** — identifies which activities the model struggles with most

### 6. Generate Standalone Synthetic Samples

Demo generation from one processed sample:

```bash
uv run python scripts/generate.py generation.reference_pt="data/processed/HAR/train/acc/walking/<sample>.pt" generation.checkpoint_path="outputs/checkpoints/test_run/ckpt_epoch0000.pt"
```

Generate multiple variants from the same reference:

```bash
uv run python scripts/generate.py generation.reference_pt="data/processed/HAR/train/acc/walking/<sample>.pt" generation.checkpoint_path="outputs/checkpoints/test_run/ckpt_epoch0000.pt" generation.num_samples=8
```

Outputs:
- modality-wise `.pt` files under `outputs/generated/<experiment_name>/{acc,gyr}/<activity>/`
- paired demo bundles under `outputs/generated/<experiment_name>/paired/<activity>/`
- `.png` preview plots saved next to each generated `.pt` plus a paired comparison image

### 7. Run Tests

```bash
uv run pytest tests -v
```

Fallback:

```bash
.\.venv\Scripts\python.exe -m pytest tests -v
```

## Logging

Default logging backend: console (`configs/logging/console.yaml`)

W&B mode:

```bash
uv run python scripts/train.py logging=wandb
```

Offline W&B:

```bash
uv run python scripts/train.py logging=wandb logging.mode=offline
```

Resume the same W&B run after an interrupted training session:

```bash
uv run python scripts/train.py logging=wandb training.resume=true
```

Notes:
- New checkpoints persist `wandb_run_id`, and resume uses that ID to reconnect.
- Older checkpoints without `wandb_run_id` will start a new W&B run.
- You can force a run ID manually with `logging.id=<run_id>` and control policy with `logging.resume=allow|must|never`.

## What Is Implemented vs Not

Implemented:
- full raw-to-processed dataset conversion
- paired multimodal training loop
- adaptive metric-weighted CGDAP objective
- standalone synthetic sample generation CLI
- synthetic generation pathway inside evaluation script
- classifier comparison on real-only vs real+augmented loaders

## Troubleshooting

`FileNotFoundError` in evaluation checkpoint resolution:
- cause: augmentation enabled but no checkpoint found
- fix: train first, or set `evaluation.augmentation.enabled=false`

`ValueError: Expected 15 subjects, found ...` during preprocessing:
- cause: raw subject folder count mismatch under `data/raw/HAR`
- fix: verify subject directories and path configuration

`KeyError` about missing sample keys in dataset loaders:
- cause: stale/older processed artifacts
- fix: regenerate processed dataset with `scripts/prepare_dataset.py`

Missing W&B package with `logging=wandb`:
- fix: run dependency install (`uv sync` or pip editable install)

## Development Notes

Main entry points:
- `scripts/prepare_dataset.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_eval_report.py`
- `scripts/generate.py`

Quick project summary command docs are also in `main.py`.
