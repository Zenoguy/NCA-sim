"""
Unit tests for Coupled Non-Markovian KdV Solver and Sparse Probe Dataset.
"""

import numpy as np
import pytest
import torch
from src.non_markovian_solver import (
    CoupledNonMarkovianKdVSolver,
    build_non_markovian_datasets,
)
from src.sparse_dataset import SparseProbeDataset


def test_coupled_non_markovian_solver_stability():
    """Verify coupled (u, w) solver runs stably without NaNs or divergence."""
    solver = CoupledNonMarkovianKdVSolver(
        N=128,
        Lx=50.0,
        dt=0.005,
        lambda_rel=1.0,
        kappa=1.5,
    )

    # Initial condition: solitary pulse
    A = 1.0
    x0 = 0.0
    L = np.sqrt(12.0 * solver.beta / (solver.alpha * A))
    xi = (solver.x - x0 + solver.Lx / 2.0) % solver.Lx - solver.Lx / 2.0
    u0 = A / (np.cosh(xi / L) ** 2)

    t_eval = np.linspace(0.0, 1.0, 11)  # 10 macro steps of delta_T = 0.1
    u_traj, w_traj = solver.solve(u0, w0=None, t_eval=t_eval)

    # Check shapes
    assert u_traj.shape == (11, 128)
    assert w_traj.shape == (11, 128)

    # Check finiteness
    assert np.all(np.isfinite(u_traj))
    assert np.all(np.isfinite(w_traj))

    # At t=0, w should be zero
    assert np.allclose(w_traj[0], 0.0)

    # At t>0, w should grow in response to u
    assert np.max(w_traj[-1]) > 0.1


def test_coupled_dataset_generation():
    """Verify build_non_markovian_datasets creates clean, isolated splits."""
    solver = CoupledNonMarkovianKdVSolver(N=128, Lx=50.0, dt=0.005)
    ds = build_non_markovian_datasets(
        solver, delta_T=0.1, train_horizon=8, n_train=4, n_val=2, seed=42
    )

    assert "train" in ds and "val" in ds
    assert ds["train"]["data"].shape == (4, 9, 128)
    assert ds["val"]["data"].shape == (2, 9, 128)
    assert len(ds["train"]["metadata"]) == 4
    assert len(ds["val"]["metadata"]) == 2


def test_sparse_probe_dataset_masking():
    """Verify sparse probe dataset properly masks unobserved grid cells."""
    data = np.ones((5, 10, 128), dtype=np.float32)
    dataset = SparseProbeDataset(data, num_probes=16)

    sparse_traj, full_traj = dataset[0]

    assert sparse_traj.shape == (10, 1, 128)
    assert full_traj.shape == (10, 1, 128)

    # Check that exactly 16 points per time step are non-zero in sparse_traj
    for t in range(10):
        non_zero_count = torch.count_nonzero(sparse_traj[t, 0]).item()
        assert non_zero_count == 16

    # Verify full_traj remains completely unmasked
    assert torch.all(full_traj == 1.0)
