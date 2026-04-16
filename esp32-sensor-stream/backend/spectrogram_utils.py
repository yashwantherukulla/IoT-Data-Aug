import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

EPS = 1e-10

METRIC_NAMES = ["temporal_range", "f0_amplitude", "contrast", "flatness", "entropy"]

# ---------------------------------------------------------------------------
# Pure-NumPy metric implementations  (mirror cgdap/metrics/extractor.py)
# ---------------------------------------------------------------------------

def _metric_temporal_range(spec_2d: np.ndarray) -> float:
    """Max - min of mean amplitude over frequency bins, across time. [F,T]"""
    mean_over_freq = spec_2d.mean(axis=0)   # [T]
    return float(mean_over_freq.max() - mean_over_freq.min())


def _metric_f0_amplitude(
    spec_2d: np.ndarray,
    harmonics=(1.0, 0.5, 0.25),
    softmax_temp: float = 0.1,
) -> float:
    """F0 amplitude via Harmonic Product Spectrum. [F,T]"""
    mean_over_time = spec_2d.mean(axis=-1)   # [F]
    hps = mean_over_time.copy()
    F_orig = len(mean_over_time)

    for ratio in harmonics[1:]:
        target_len = max(1, int(round(F_orig * ratio)))
        # Simple linear interpolation (no torch required)
        x_old = np.linspace(0, 1, F_orig)
        x_new = np.linspace(0, 1, target_len)
        down = np.interp(x_new, x_old, mean_over_time)
        L = min(len(hps), len(down))
        hps = hps[:L] * down[:L]

    # Numerically stable softmax: subtract max before exp to prevent overflow
    scaled = hps / (softmax_temp + EPS)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    weights = weights / (weights.sum() + EPS)
    L = len(weights)
    return float((spec_2d[:L] * weights[:, None]).sum(axis=0).mean())


def _metric_contrast(spec_2d: np.ndarray, tail_ratio: float = 0.05) -> float:
    """Mean(top-k%) - mean(bottom-k%) over all spectrogram bins. [F,T]"""
    flat = spec_2d.ravel()
    k = max(1, int(len(flat) * tail_ratio))
    sorted_vals = np.sort(flat)
    return float(sorted_vals[-k:].mean() - sorted_vals[:k].mean())


def _metric_flatness(spec_2d: np.ndarray) -> float:
    """Spectral flatness: geometric_mean / arithmetic_mean. [F,T]"""
    x = np.clip(spec_2d, EPS, None)
    g_mean = np.exp(np.log(x).mean())
    a_mean = x.mean()
    return float(g_mean / (a_mean + EPS))


def _metric_entropy(spec_2d: np.ndarray) -> float:
    """Shannon entropy (base-2) over normalised spectrogram bins. [F,T]"""
    x = np.clip(spec_2d, EPS, None)
    p = x / x.sum()
    return float(-(p * np.log2(p + EPS)).sum())


def compute_window_metrics(data_xyz, cfg) -> dict:
    """
    Compute the 5 HAR metrics from a raw XYZ window.
    data_xyz: list or np.array of shape (N, 3)
    Returns dict with keys: temporal_range, f0_amplitude, contrast, flatness, entropy
    """
    sr = cfg.dataset.sample_rate_hz
    spec_cfg = cfg.dataset.spectrogram
    metric_cfg = cfg.dataset.metrics

    fft_ms = spec_cfg.fft_window_ms
    hop_ms = spec_cfg.hop_ms
    win_length = round(sr * fft_ms / 1000)
    n_fft = 1 << (win_length - 1).bit_length()
    hop_length = max(1, round(sr * hop_ms / 1000))

    signal_arr = np.array(data_xyz, dtype=np.float32).T   # [3, N]
    spec = compute_spectrogram(
        signal_arr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_type=spec_cfg.window,
        power=spec_cfg.power,
        use_log1p=spec_cfg.log1p,
    )
    spec_2d = spec.mean(axis=0)   # [F, T]  — channel-mean

    harmonics = tuple(metric_cfg.hps_harmonics)
    temp = float(metric_cfg.hps_softmax_temp)
    tail = float(metric_cfg.contrast_tail_ratio)

    raw = {
        "temporal_range": _metric_temporal_range(spec_2d),
        "f0_amplitude":   _metric_f0_amplitude(spec_2d, harmonics=harmonics, softmax_temp=temp),
        "contrast":       _metric_contrast(spec_2d, tail_ratio=tail),
        "flatness":       _metric_flatness(spec_2d),
        "entropy":        _metric_entropy(spec_2d),
    }
    # Replace NaN / Inf with None so the JSON encoder never chokes
    return {k: (None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v)
            for k, v in raw.items()}

def load_config():
    # Absolute path to the configs directory
    root = pathlib.Path(__file__).parent.parent.parent
    config_dir = str((root / "configs").resolve())
    
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=config_dir, version_base=None)
    
    return compose(config_name="config")

def compute_spectrogram(
    signal_arr: np.ndarray,      # [3, N]
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_type: str,
    power: float,
    use_log1p: bool,
) -> np.ndarray:
    """Return [3, F, T] spectrogram numpy array using only NumPy."""
    n_channels, n_samples = signal_arr.shape
    
    # Create window
    if window_type == "hann":
        window = np.hanning(win_length)
    elif window_type == "hamming":
        window = np.hamming(win_length)
    else:
        window = np.ones(win_length)
        
    # Standard STFT centering/padding (similar to torch default)
    pad_len = n_fft // 2
    padded_signal = np.pad(signal_arr, ((0, 0), (pad_len, pad_len)), mode='reflect')
    _, padded_samples = padded_signal.shape
    
    # Calculate windows
    n_windows = (padded_samples - win_length) // hop_length + 1
    
    # Compute per channel
    all_specs = []
    for c in range(n_channels):
        specs_list = []
        for i in range(n_windows):
            start = i * hop_length
            end = start + win_length
            if end > padded_samples:
                break
            
            segment = padded_signal[c, start:end] * window
            # Compute RFFT (Real Fast Fourier Transform)
            spectrum = np.fft.rfft(segment, n=n_fft)
            specs_list.append(np.abs(spectrum))
        
        # Stack to [F, T]
        spec = np.stack(specs_list, axis=1)
        
        if power != 1.0:
            spec = spec ** power
        if use_log1p:
            spec = np.log1p(spec)
        all_specs.append(spec)
        
    return np.array(all_specs)

def get_window_samples(cfg):
    sr = cfg.dataset.sample_rate_hz
    win_sec = cfg.dataset.window_seconds
    return round(sr * win_sec)

def generate_spectrogram_base64(data_xyz, cfg):
    """
    data_xyz: list or np.array of shape (N, 3)
    returns: base64 encoded png image
    """
    # Use config values
    sr = cfg.dataset.sample_rate_hz
    spec_cfg = cfg.dataset.spectrogram
    
    fft_ms = spec_cfg.fft_window_ms
    hop_ms = spec_cfg.hop_ms
    
    win_length = round(sr * fft_ms / 1000)
    n_fft = 1 << (win_length - 1).bit_length()
    hop_length = max(1, round(sr * hop_ms / 1000))
    
    window_type = spec_cfg.window
    signal_arr = np.array(data_xyz, dtype=np.float32).T
    
    spec = compute_spectrogram(
        signal_arr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_type=window_type,
        power=spec_cfg.power,
        use_log1p=spec_cfg.log1p
    )
    
    # Collapse to 2D for plotting (mean across channels)
    spec_2d = spec.mean(axis=0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(4, 2)) # Shorter height
    vmin, vmax = np.percentile(spec_2d, [2, 98])
    im = ax.imshow(
        spec_2d,
        origin="lower",
        aspect="auto",
        cmap="magma", # Modern look
        vmin=vmin,
        vmax=vmax
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')
