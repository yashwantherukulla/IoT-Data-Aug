# IoT Data Aug

Conditional Generative Data Augmentation Pipeline for human activity recognition (HAR) from wearable inertial sensors.

This repository focuses on a multimodal diffusion-based pipeline built around accelerometer (`acc`) and gyroscope (`gyr`) spectrograms derived from the RealWorld HAR dataset. The codebase covers:

- raw HAR cleanup and preprocessing
- per-window spectrogram and metric extraction
- a multimodal conditional diffusion model (`CGDAP`)
- downstream HAR evaluation baselines (`DeepSense` and a Transformer classifier)
- tests for dataset, metrics, and core model components

The project uses Hydra for configuration management and `uv` for environment and command execution.

## What The Pipeline Does

At a high level, the pipeline is:

1. Start from raw RealWorld HAR sensor archives in `data/raw/HAR`.
2. Optionally clean and normalize the raw folder structure.
3. Segment each subject/activity recording into fixed windows.
4. Convert each window into 3-channel spectrograms for each modality.
5. Compute differentiable conditioning metrics from each spectrogram.
6. Train a multimodal diffusion model conditioned on metrics + activity labels.
7. Evaluate downstream HAR classifiers on the processed dataset.

The current implementation already contains:

- preprocessing for `acc` and `gyr`
- a cross-attention condition embedder
- a conditional U-Net denoiser
- a DDPM noise schedule
- a multimodal wrapper with adaptive metric-consistency weighting
- classifier evaluation on processed real data

The repository also contains augmentation and sampling primitives, but there is not yet a complete end-to-end CLI that saves generated synthetic samples back into a new training split and re-runs classifier evaluation on `real + augmented` data automatically.

## Repository Structure

```text
.
|-- cgdap/
|   |-- augmentation/      # metric-space augmentation target generation
|   |-- data/              # raw loading, preprocessing, dataset classes
|   |-- evaluation/        # HAR baselines: DeepSense and Transformer
|   |-- metrics/           # differentiable metric extraction
|   |-- models/            # base interfaces, condition embedder, DDPM, U-Net, CGDAP
|   `-- training/          # Hydra-driven trainer
|-- configs/
|   |-- augmentation/      # augmentation modes and ranges
|   |-- dataset/           # dataset paths, modalities, activities, STFT, split
|   |-- logging/           # console and W&B logging config
|   |-- model/             # CGDAP architecture config
|   |-- training/          # optimizer, scheduler, epochs, checkpointing
|   `-- config.yaml        # root Hydra config
|-- scripts/
|   |-- prepare_dataset.py # primary preprocessing entry point
|   |-- train.py           # training entry point
|   `-- evaluate.py        # downstream classifier evaluation
|-- tests/                 # dataset, metric, and model tests
|-- utils/                 # older utility scripts and visualization helpers
|-- data/
|   |-- raw/
|   `-- processed/
|-- outputs/               # Hydra run directories and checkpoints
|-- main.py                # quick-start command docstring
`-- pyproject.toml
```

## Core Concepts

### Dataset

The configured dataset is `realworld_har`, with:

- modalities: `acc`, `gyr`
- channels per modality: 3 (`x`, `y`, `z`)
- activities:
  - `climbingdown`
  - `climbingup`
  - `jumping`
  - `running`
  - `walking`

Activity folder names are normalized through `dataset.activity_map`, for example:

- `climbingdown -> climbing_down`
- `climbingup -> climbing_up`

### Windowing And Spectrograms

Each recording is segmented into non-overlapping windows using:

- `window_seconds: 2.5`
- `sample_rate_hz: 100.0`

Each window becomes a spectrogram using STFT settings from `configs/dataset/har_dataset.yaml`, including:

- `fft_window_ms`
- `hop_ms`
- `window`
- `power`
- `log1p`

For each modality, the saved spectrogram shape is:

- `Tensor[3, F, T]`

where the 3 channels correspond to the `x`, `y`, and `z` axes.

### Conditioning Metrics

The model conditions on five differentiable metrics extracted from the average spectrogram:

1. `temporal_range`
2. `f0_amplitude`
3. `contrast`
4. `flatness`
5. `entropy`

These are implemented in `cgdap/metrics/extractor.py` and are used both:

- offline during preprocessing
- online during training for metric-consistency loss

### Model

`MultimodalCGDAP` is the top-level model wrapper. It composes:

- one denoiser per modality
- one shared condition embedder
- one shared diffusion schedule
- one shared metric extractor

Current active architecture:

- denoiser: `ConditionalUNet`
- schedule: `DDPMSchedule`
- embedder: `CrossAttentionConditionEmbedder`

Training objective:

- diffusion reconstruction loss `L_G`
- metric-consistency loss `L_metric`
- adaptive per-metric weighting during training

### Evaluation Baselines

The repository includes two downstream HAR classifiers:

- `DeepSenseClassifier`
- `HATransformerClassifier`

These are used in `scripts/evaluate.py`.

## Data Layout

### Raw Data

Expected raw path:

```text
data/raw/HAR/
  proband1/
  proband2/
  ...
  proband15/
```

The code expects 15 subject folders and uses a fixed train/validation split count from config:

- train subjects: 10
- val subjects: 5

If you are starting from a fresh raw download, the cleaning pipeline can:

- remove images/videos
- retain only sensor zip files
- keep only the configured activities
- keep only the `upperarm` placement
- create a sentinel file so cleaning is not repeated accidentally

### Processed Data Contract

The current preprocessing code in `cgdap/data/preprocessing.py` writes modality-specific files in this format:

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

Important: the currently checked-in processed dataset in this repository appears to be from an older layout where files are grouped directly under activity folders instead of `train/acc/...` and `train/gyr/...`. The current training and dataset loaders expect the newer per-modality layout, so you should re-run preprocessing before training or evaluation.

## Configuration

The Hydra root config is `configs/config.yaml`. It composes:

- `dataset: har_dataset`
- `model: cgdap`
- `training: default`
- `augmentation: default`
- `logging: console`

Useful config files:

- `configs/dataset/har_dataset.yaml`
- `configs/model/cgdap.yaml`
- `configs/training/default.yaml`
- `configs/augmentation/default.yaml`
- `configs/logging/console.yaml`
- `configs/logging/wandb.yaml`

Hydra overrides work from the command line, for example:

```bash
uv run python scripts/train.py training.max_epochs=10
uv run python scripts/train.py model.unet.base_channels=32 training.batch_size=4
uv run python scripts/prepare_dataset.py dataset.pipeline.run_clean=true
```

## Setup

### Requirements

- Python `>= 3.13`
- `uv`

The main dependencies are declared in `pyproject.toml`:

- `hydra-core`
- `omegaconf`
- `torch`
- `tqdm`
- `wandb`
- `numpy`
- `matplotlib`
- `pandas`
- `pytest`

### Install Dependencies

```bash
uv sync
```

If you prefer using the existing local virtual environment directly:

```bash
.\.venv\Scripts\python.exe -m pip install -e .
```

## How To Run The Pipeline

### 1. Put The Raw Dataset In Place

Make sure your raw dataset is available under:

```text
data/raw/HAR/
```

You should have subject folders such as `proband1`, `proband2`, ..., `proband15`.

### 2. Prepare The Dataset

First-time run on a fresh raw download:

```bash
uv run python scripts/prepare_dataset.py dataset.pipeline.run_clean=true
```

Re-run preprocessing without raw cleanup:

```bash
uv run python scripts/prepare_dataset.py
```

Keep old processed artifacts instead of forcing regeneration:

```bash
uv run python scripts/prepare_dataset.py dataset.pipeline.force_regenerate=false
```

What this step produces:

- `data/processed/HAR/metadata.json`
- processed `train/acc`, `train/gyr`, `val/acc`, `val/gyr` trees
- one `.pt` file per window per modality

### 3. Train The CGDAP Model

Default training:

```bash
uv run python scripts/train.py
```

Example small smoke run:

```bash
uv run python scripts/train.py model.unet.base_channels=32 training.batch_size=4 training.max_epochs=2
```

Training outputs:

- Hydra run logs under `outputs/<date>/<time>/`
- checkpoints under `outputs/checkpoints/<experiment_name>/`
- live `tqdm` progress bars for epochs, training, and validation

Default experiment name:

- `cgdap_v2_har`

Enable Weights & Biases logging:

```bash
uv run python scripts/train.py logging=wandb
```

Run W&B in offline mode:

```bash
uv run python scripts/train.py logging=wandb logging.mode=offline
```

### 4. Evaluate Downstream HAR Classifiers

Run classifier evaluation on the processed dataset:

```bash
uv run python scripts/evaluate.py
```

This script currently trains and validates:

- DeepSense baseline
- Transformer baseline

Note: despite the docstring mentioning `evaluation.classifier=transformer`, the current script instantiates and evaluates both classifiers in sequence.

### 5. Run Tests

Preferred command:

```bash
uv run pytest tests -v
```

If `uv` has a local cache or permission issue on Windows, use the virtualenv Python directly:

```bash
.\.venv\Scripts\python.exe -m pytest tests -v
```

## Recommended End-To-End Command Sequence

If you want the shortest reliable path from raw data to a trained model:

```bash
uv sync
uv run python scripts/prepare_dataset.py dataset.pipeline.run_clean=true
uv run python scripts/train.py
uv run python scripts/evaluate.py
.\.venv\Scripts\python.exe -m pytest tests -v
```

If your raw data has already been cleaned before:

```bash
uv sync
uv run python scripts/prepare_dataset.py
uv run python scripts/train.py
uv run python scripts/evaluate.py
```

## Outputs And Artifacts

### Hydra Run Directories

Hydra creates run-specific directories under:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
```

These typically include:

- `.hydra/config.yaml`
- `.hydra/hydra.yaml`
- `.hydra/overrides.yaml`
- script log file such as `prepare_dataset.log`

### Checkpoints

Training checkpoints are saved under:

```text
outputs/checkpoints/<experiment_name>/
```

Each checkpoint contains:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- logged training metrics

## Important Caveats

### 1. Rebuild Processed Data Before Training

The current code expects the v2.1-style processed layout with separate modality directories. If your processed dataset was generated by the older utility scripts in `utils/`, training and dataset tests will fail until you regenerate the processed dataset with:

```bash
uv run python scripts/prepare_dataset.py
```

### 2. Augmentation Is Partially Wired

`cgdap/augmentation/engine.py` and `MultimodalCGDAP.sample(...)` provide the pieces needed for synthetic generation, but the repository does not yet include a complete production CLI that:

- generates synthetic samples
- writes them back to disk in dataset format
- trains downstream classifiers on `real + augmented`
- compares against `real-only` automatically

### 3. Some `utils/` Scripts Are Legacy Helpers

The `utils/` directory contains older or standalone helper scripts for:

- visualization
- data inspection
- earlier preprocessing approaches

For the main pipeline, prefer the entry points in `scripts/`.

## Code Walkthrough

### `cgdap/data/`

- `raw_loader.py`: raw dataset cleanup and organization
- `preprocessing.py`: main preprocessing pipeline
- `dataset.py`: single-modality and paired dataset loaders

### `cgdap/metrics/`

- `extractor.py`: differentiable metric implementations and batch extractor

### `cgdap/models/`

- `base.py`: abstract interfaces and model registry
- `condition.py`: cross-attention conditioning blocks
- `ddpm.py`: diffusion schedule
- `unet.py`: conditional U-Net denoiser
- `cgdap.py`: multimodal wrapper and loss computation

### `cgdap/training/`

- `trainer.py`: training loop, optimizer, scheduler, validation, checkpointing

### `cgdap/evaluation/`

- `deepsense.py`: DeepSense-style HAR baseline
- `transformer.py`: Transformer-based HAR baseline

## Development Notes

### Quick Smoke Tests

Use a smaller model and fewer epochs during iteration:

```bash
uv run python scripts/train.py model.unet.base_channels=32 training.batch_size=4 training.max_epochs=1
```

### Common Override Examples

Change training length:

```bash
uv run python scripts/train.py training.max_epochs=50
```

Change diffusion U-Net width:

```bash
uv run python scripts/train.py model.unet.base_channels=64
```

Switch augmentation mode in config:

```bash
uv run python scripts/train.py augmentation.mode=interpolation
```

## Current Verification Status

I inspected the codebase and verified the current behavior from the repository itself.

Observed during verification:

- the repo contains raw and processed HAR data locally
- the current processed snapshot is not in the layout expected by the active dataset loader
- running `.\.venv\Scripts\python.exe -m pytest tests -q` currently gives `10 passed, 3 failed`
- the failing tests are the dataset tests, and they fail because `data/processed/HAR/train/acc` does not exist yet in the current checked-in processed snapshot

Once you regenerate processed data with the current preprocessing pipeline, those dataset-path failures should be the first thing to re-check.
