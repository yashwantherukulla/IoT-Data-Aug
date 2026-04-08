"""CGDAP v2.1 — IoT Data Augmentation Pipeline for HAR.

Quick start:
    # Prepare processed dataset (first time: add dataset.pipeline.run_clean=true)
    uv run python scripts/prepare_dataset.py

    # Train
    uv run python scripts/train.py

    # Evaluate classifiers
    uv run python scripts/evaluate.py

    # Run tests
    uv run pytest tests/ -v
"""
