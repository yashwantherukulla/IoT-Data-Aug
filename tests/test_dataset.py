"""Tests for dataset classes."""
import pathlib, torch, pytest
from cgdap.data.dataset import ModalityDataset, PairedDataset, build_label_map


PROCESSED = pathlib.Path("data/processed/HAR")


def _has_data():
    return (PROCESSED / "train").exists()


@pytest.mark.skipif(not _has_data(), reason="processed data not available")
def test_modality_dataset_shapes():
    train_dir = PROCESSED / "train"
    lmap = build_label_map(train_dir, "acc")
    ds = ModalityDataset(train_dir, "acc", lmap)
    assert len(ds) > 0
    item = ds[0]
    assert item["spectrogram"].shape[0] == 3
    assert item["metrics"].shape == (5,)
    assert isinstance(item["label"], int)


@pytest.mark.skipif(not _has_data(), reason="processed data not available")
def test_paired_dataset_alignment():
    train_dir = PROCESSED / "train"
    lmap = build_label_map(train_dir, "acc")
    ds = PairedDataset(train_dir, ["acc", "gyr"], lmap)
    assert len(ds) > 0
    item = ds[0]
    assert "acc" in item and "gyr" in item
    assert item["acc"]["spectrogram"].shape == item["gyr"]["spectrogram"].shape
    assert item["acc"]["spectrogram"].shape[0] == 3


@pytest.mark.skipif(not _has_data(), reason="processed data not available")
def test_dataset_no_nan():
    train_dir = PROCESSED / "train"
    lmap = build_label_map(train_dir, "acc")
    ds = ModalityDataset(train_dir, "acc", lmap)
    for i in range(min(10, len(ds))):
        item = ds[i]
        assert not torch.isnan(item["spectrogram"]).any()
        assert not torch.isnan(item["metrics"]).any()
