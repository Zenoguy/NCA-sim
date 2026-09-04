"""
Unit tests for KdV dataset generation and leakage-free split verification.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from src.kdv_solver import KdVSolver
from src.dataset import (
    KdVTrajectoryDataset,
    build_experiment_datasets,
    save_datasets,
    load_datasets,
)


def test_dataset_generation_and_shapes(tmp_path):
    """Verify that dataset generation creates valid partitions with correct shapes."""
    solver = KdVSolver(N=64, Lx=30.0, alpha=6.0, beta=1.0, dt=0.01)
    datasets = build_experiment_datasets(
        solver, delta_T=0.1, train_horizon=4, long_horizon=8, seed=123
    )

    expected_splits = [
        "train",
        "val",
        "test_interp",
        "test_extrap",
        "test_unseen_params",
        "test_perturbed_pulses",
        "test_two_pulses",
        "test_long_horizon",
    ]

    for split in expected_splits:
        assert split in datasets, f"Missing split: {split}"
        data = datasets[split]["data"]
        # shape: (n_trajectories, num_macro_steps + 1, N)
        assert data.ndim == 3
        assert data.shape[2] == 64
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))

    # Test extrapolation range: all extrap amplitudes must be > max(train amplitudes)
    train_A = [m["A"] for m in datasets["train"]["metadata"]]
    extrap_A = [m["A"] for m in datasets["test_extrap"]["metadata"]]
    assert min(extrap_A) > max(train_A), "Data leakage: Extrapolation amplitude overlap with train!"

    # Save and load roundtrip test
    save_dir = tmp_path / "data"
    save_datasets(datasets, save_dir)
    loaded_datasets = load_datasets(save_dir)

    for split in expected_splits:
        assert split in loaded_datasets
        pt_ds = loaded_datasets[split]
        assert isinstance(pt_ds, KdVTrajectoryDataset)
        assert len(pt_ds) == len(datasets[split]["data"])
        x, meta = pt_ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 3  # (num_macro_steps + 1, 1, N)
