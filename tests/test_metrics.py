"""Tests for MetricExtractor."""
import torch, pytest
from cgdap.metrics.extractor import MetricExtractor, compute_metrics_fn


def make_spec(B=2, C=3, F=129, T=126):
    return torch.rand(B, C, F, T)


def test_metric_extractor_output_shape():
    extractor = MetricExtractor()
    spec = make_spec()
    out = extractor(spec)
    assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"


def test_metric_extractor_no_nan_low_energy():
    extractor = MetricExtractor()
    spec = torch.zeros(2, 3, 129, 126) + 1e-12
    out = extractor(spec)
    assert not torch.isnan(out).any(), "NaN in metrics for low-energy input"
    assert not torch.isinf(out).any(), "Inf in metrics for low-energy input"


def test_metric_extractor_gradients():
    extractor = MetricExtractor()
    spec = make_spec().requires_grad_(True)
    out = extractor(spec)
    out.sum().backward()
    assert spec.grad is not None
    assert not torch.isnan(spec.grad).any()


def test_compute_metrics_fn():
    from types import SimpleNamespace
    cfg = SimpleNamespace(hps_harmonics=[1.0, 0.5, 0.25], hps_softmax_temp=0.1, contrast_tail_ratio=0.05, eps=1e-10)
    spec_2d = torch.rand(129, 126)
    out = compute_metrics_fn(spec_2d, cfg)
    assert out.shape == (5,)
    assert not torch.isnan(out).any()
