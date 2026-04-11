"""CGDAP v2.1 — IoT Data Augmentation Pipeline for HAR.

Quick start:
    # Prepare processed dataset (first time: add dataset.pipeline.run_clean=true)
    uv run python scripts/prepare_dataset.py

    # Train
    uv run python scripts/train.py

    # Generate demo samples from one processed .pt sample
    uv run python scripts/generate.py generation.reference_pt="data/processed/HAR/train/acc/walking/example.pt" generation.checkpoint_path="outputs/checkpoints/test_run/ckpt_epoch0000.pt"

    # Evaluate classifiers
    uv run python scripts/evaluate.py

    # Run tests
    uv run pytest tests/ -v
"""
