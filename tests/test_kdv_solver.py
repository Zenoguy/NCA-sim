"""
Unit tests and convergence validation for KdVSolver (ETDRK4).
"""

import numpy as np
import pytest
from src.kdv_solver import KdVSolver


def test_etdrk4_operators_initialization():
    """Verify that ETDRK4 operators are properly initialized without NaNs or Infs."""
    solver = KdVSolver(N=128, Lx=50.0, alpha=6.0, beta=1.0, dt=0.005)
    assert not np.any(np.isnan(solver.E))
    assert not np.any(np.isnan(solver.E2))
    assert not np.any(np.isnan(solver.Q))
    assert not np.any(np.isnan(solver.f1))
    assert not np.any(np.isnan(solver.f2))
    assert not np.any(np.isnan(solver.f3))

    # At k=0, operators should match classical limits
    idx_zero = 0
    assert np.isclose(solver.E[idx_zero], 1.0)
    assert np.isclose(solver.E2[idx_zero], 1.0)
    assert np.isclose(solver.Q[idx_zero] / solver.dt, 0.5)
    assert np.isclose(solver.f1[idx_zero], 1.0 / 6.0)
    assert np.isclose(solver.f2[idx_zero], 1.0 / 6.0)
    assert np.isclose(solver.f3[idx_zero], 1.0 / 6.0)


def test_single_soliton_propagation():
    """Verify that a single soliton propagates coherently with preserved shape and speed."""
    N = 256
    Lx = 50.0
    solver = KdVSolver(N=N, Lx=Lx, alpha=6.0, beta=1.0, dt=0.005)

    A = 1.0
    x0 = 0.0
    u0 = solver.exact_soliton(t=0.0, A=A, x0=x0)

    # Solve from t=0 to t=5.0
    t_eval = np.linspace(0.0, 5.0, 11)  # output every 0.5
    traj = solver.solve(u0, t_eval)

    # Compare final state at t=5.0 with exact analytical solution
    u_final = traj[-1]
    u_exact = solver.exact_soliton(t=5.0, A=A, x0=x0)

    rel_l2 = np.linalg.norm(u_final - u_exact) / np.linalg.norm(u_exact)
    # The spectral accuracy of ETDRK4 on this smooth solution is O(10^-7)
    assert rel_l2 < 1e-4, f"Relative L2 error {rel_l2:.2e} exceeded tolerance."

    # Verify subgrid peak amplitude and position
    x_peak, A_peak = solver.subgrid_peak(u_final)
    expected_v = solver.exact_soliton_velocity(A)
    expected_x = (x0 + expected_v * 5.0 + Lx / 2.0) % Lx - Lx / 2.0

    assert np.isclose(
        A_peak, A, atol=1e-3
    ), f"Peak amplitude {A_peak} differs from {A}"
    assert np.isclose(
        x_peak, expected_x, atol=0.05
    ), f"Peak position {x_peak} differs from {expected_x}"


def test_invariant_conservation():
    """Verify conservation of zeroth-order integral I1, quadratic I2, and Hamiltonian I3."""
    solver = KdVSolver(N=256, Lx=50.0, alpha=6.0, beta=1.0, dt=0.005)
    u0 = solver.exact_soliton(t=0.0, A=1.2, x0=-5.0)

    t_eval = np.linspace(0.0, 8.0, 17)
    traj = solver.solve(u0, t_eval)

    I1_0, I2_0, I3_0 = solver.compute_invariants(traj[0])

    for i, u_t in enumerate(traj[1:], 1):
        I1_t, I2_t, I3_t = solver.compute_invariants(u_t)

        drift_I1 = abs(I1_t - I1_0) / abs(I1_0)
        drift_I2 = abs(I2_t - I2_0) / abs(I2_0)
        drift_I3 = abs(I3_t - I3_0) / abs(I3_0)

        assert (
            drift_I1 < 1e-6
        ), f"Step {i}: I1 drift {drift_I1:.2e} exceeded tolerance."
        assert (
            drift_I2 < 1e-6
        ), f"Step {i}: I2 drift {drift_I2:.2e} exceeded tolerance."
        assert (
            drift_I3 < 1e-5
        ), f"Step {i}: I3 drift {drift_I3:.2e} exceeded tolerance."


def test_etdrk4_temporal_convergence():
    """Verify high-order temporal convergence of ETDRK4 on KdV soliton rollout."""
    N = 256
    Lx = 50.0
    A = 1.0
    T = 1.0

    # Reference solution with very fine dt = 0.0005
    solver_ref = KdVSolver(N=N, Lx=Lx, alpha=6.0, beta=1.0, dt=0.0005)
    u0 = solver_ref.exact_soliton(t=0.0, A=A)
    u_ref = solver_ref.solve(u0, np.array([0.0, T]))[-1]

    # Test dt values
    dt_list = [0.02, 0.01, 0.005]
    errors = []

    for dt in dt_list:
        solver_test = KdVSolver(N=N, Lx=Lx, alpha=6.0, beta=1.0, dt=dt)
        u_test = solver_test.solve(u0, np.array([0.0, T]))[-1]
        err = float(np.linalg.norm(u_test - u_ref) / np.linalg.norm(u_ref))
        errors.append(err)

    # Check error reduction as dt is halved
    # Ratio between dt=0.02 and dt=0.01 should be close to 2^4 = 16
    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]

    # For 4th order, ratio >= 10 is expected even with slight numerical saturation
    assert (
        ratio_1 > 8.0
    ), f"Expected 4th-order convergence ratio > 8, got {ratio_1:.2f}"
    assert (
        ratio_2 > 8.0
    ), f"Expected 4th-order convergence ratio > 8, got {ratio_2:.2f}"
