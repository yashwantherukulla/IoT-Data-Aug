# Metric-Consistency Loss (L_metric)

> **Relevant source files**
> * [cgdap/metrics/extractor.py](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/metrics/extractor.py)
> * [cgdap/models/cgdap.py](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py)
> * [configs/training/default.yaml](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/configs/training/default.yaml)

The Metric-Consistency Loss ($L_{metric}$) is a secondary training objective in the CGDAP framework designed to ensure that the generated spectrograms strictly adhere to the physical properties defined by the input condition metrics. While the standard diffusion loss ($L_G$) focuses on overall structural distribution, $L_{metric}$ enforces fine-grained control over specific signal characteristics such as temporal range and spectral entropy.

### Purpose and Overview

During training, the model is tasked with reconstructing the original clean signal $\hat{x}*0$ from a noisy version $x_t$. $L*{metric}$ computes the mean squared error (MSE) between the metrics extracted from this reconstruction and the ground-truth target metrics. To balance this against the much larger diffusion loss, the system employs an **Adaptive Metric Weighting** mechanism that uses an Exponential Moving Average (EMA) and a target ratio controller to dynamically scale the importance of each metric.

### Implementation Flow

The computation of $L_{metric}$ occurs within the `forward` pass of the `MultimodalCGDAP` wrapper.

1. **Reconstruction**: The model uses the predicted noise $\epsilon_\theta$ to estimate the clean spectrogram $\hat{x}_0$ using the diffusion schedule's `predict_x0` method [cgdap/models/cgdap.py L15-L18](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L15-L18)
2. **Extraction**: The `MetricExtractor` module performs a differentiable forward pass on $\hat{x}_0$ to obtain five scalar metrics [cgdap/metrics/extractor.py L183-L200](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/metrics/extractor.py#L183-L200)
3. **Loss Calculation**: Per-metric MSE is calculated between the extracted values and the original targets [cgdap/models/cgdap.py L18-L19](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L18-L19)
4. **Adaptive Weighting**: The total loss is composed as $L_{total} = L_G + \sum_{i=1}^{5} w_i \cdot L_{metric, i}$ [cgdap/models/cgdap.py L19](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L19-L19)

#### Metric Extraction and Reconstruction Pipeline

The following diagram illustrates the data flow from the predicted noise back to the metric consistency evaluation.

**Training Metric Flow**

```

```

**Sources:** [cgdap/models/cgdap.py L9-L19](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L9-L19)

 [cgdap/metrics/extractor.py L148-L200](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/metrics/extractor.py#L148-L200)

### Adaptive Metric Weighting

Because different metrics (e.g., entropy vs. temporal range) operate on different numerical scales, static weighting is insufficient. The `MultimodalCGDAP` class manages dynamic weights stored in a `metric_weights` buffer [cgdap/models/cgdap.py L119-L122](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L119-L122)

#### Weight Update Logic

The weights are updated at the end of every training step using the following logic:

1. **EMA Tracking**: A running EMA of the per-metric loss is maintained in `metric_loss_ema` to smooth out batch-to-batch stochasticity [cgdap/models/cgdap.py L123-L126](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L123-L126)
2. **Target Ratio Controller**: The system attempts to maintain a specific ratio (default 10:1) between the diffusion loss $L_G$ and the total metric loss $L_{metric}$ [configs/training/default.yaml L28-L29](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/configs/training/default.yaml#L28-L29)
3. **Rescaling**: For each metric $i$, the weight $w_i$ is updated: $$w_i = \frac{L_G}{L_{metric, i} \cdot \text{target_ratio}}$$
4. **Clamping**: Weights are clamped between `weight_min` (0.01) and `weight_max` (10.0) to prevent instability [configs/training/default.yaml L35-L36](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/configs/training/default.yaml#L35-L36)

**Metric Weight Update Loop**

```

```

**Sources:** [cgdap/models/cgdap.py L84-L100](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L84-L100)

 [configs/training/default.yaml L25-L36](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/configs/training/default.yaml#L25-L36)

### Configuration Parameters

The behavior of $L_{metric}$ is controlled via the `training.loss` section of the configuration [configs/training/default.yaml L24-L37](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/configs/training/default.yaml#L24-L37)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `metric_weight_init` | float | 0.1 | Initial weight for all metric loss terms. |
| `target_ratio` | float | 10.0 | Desired ratio of $L_G / L_{metric}$. |
| `adaptive_start_epoch` | int | 1 | Epoch to begin dynamic weight updates. |
| `metric_weight_ema_decay` | float | 0.9 | Smoothing factor for the metric loss EMA. |
| `weight_min` | float | 0.01 | Minimum allowed value for any $w_i$. |
| `weight_max` | float | 10.0 | Maximum allowed value for any $w_i$. |

**Sources:** [configs/training/default.yaml L25-L36](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/configs/training/default.yaml#L25-L36)

 [cgdap/models/cgdap.py L75-L91](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L75-L91)

### Data Flow in Training Step

The `MultimodalCGDAP` module coordinates the interaction between the `DDPMSchedule`, `MetricExtractor`, and the adaptive buffers.

| Component | Code Entity | Role in $L_{metric}$ |
| --- | --- | --- |
| **Denoiser** | `self.denoisers` | Predicts noise $\epsilon_\theta$ from $x_t$. |
| **Schedule** | `self.schedule` | Reconstructs $\hat{x}*0$ from $\epsilon*\theta$. |
| **Extractor** | `self.metric_extractor` | Differentiably computes metrics from $\hat{x}_0$. |
| **Buffers** | `self.metric_weights` | Stores the current $w_i$ scalars. |

**Sources:** [cgdap/models/cgdap.py L102-L127](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/models/cgdap.py#L102-L127)

 [cgdap/metrics/extractor.py L148-L160](https://github.com/yashwantherukulla/IoT-Data-Aug/blob/3c2a9426/cgdap/metrics/extractor.py#L148-L160)