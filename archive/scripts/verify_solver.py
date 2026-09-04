"""
Stage 2 Script: Verify Numerical Solver Behavior and Generate Figure 1.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from src.kdv_solver import KdVSolver
from src.visualization import plot_solver_validation


def main():
    print("=== Stage 2: Numerical KdV Solver Verification ===")
    N = 256
    Lx = 50.0
    alpha = 6.0
    beta = 1.0
    dt_internal = 0.005

    solver = KdVSolver(N=N, Lx=Lx, alpha=alpha, beta=beta, dt=dt_internal)

    # Single canonical soliton
    A = 1.0
    x0 = -10.0
    u0 = solver.exact_soliton(t=0.0, A=A, x0=x0)

    # Roll out over physical time horizon t in [0, 10]
    t_eval = np.linspace(0.0, 10.0, 101)  # Delta T = 0.1
    print(f"Solving KdV on grid N={N}, Lx={Lx}, from t=0.0 to t=10.0 (101 macro steps)...")
    traj_num = solver.solve(u0, t_eval)

    # Compute analytical trajectory on same grid for comparison
    traj_exact = np.zeros_like(traj_num)
    for i, t in enumerate(t_eval):
        traj_exact[i] = solver.exact_soliton(t=t, A=A, x0=x0)

    # Invariants tracking
    I1_list, I2_list, I3_list = [], [], []
    for u in traj_num:
        i1, i2, i3 = solver.compute_invariants(u)
        I1_list.append(i1)
        I2_list.append(i2)
        I3_list.append(i3)

    I1_arr = np.array(I1_list)
    I2_arr = np.array(I2_list)
    I3_arr = np.array(I3_list)

    drift_I1 = np.abs(I1_arr - I1_arr[0]) / np.abs(I1_arr[0])
    drift_I2 = np.abs(I2_arr - I2_arr[0]) / np.abs(I2_arr[0])
    drift_I3 = np.abs(I3_arr - I3_arr[0]) / np.abs(I3_arr[0])

    invariants = {
        "drift_I1": drift_I1,
        "drift_I2": drift_I2,
        "drift_I3": drift_I3,
    }

    # Compute final metrics
    final_rel_l2 = np.linalg.norm(traj_num[-1] - traj_exact[-1]) / np.linalg.norm(traj_exact[-1])
    x_peak_num, A_peak_num = solver.subgrid_peak(traj_num[-1])
    x_peak_exact, A_peak_exact = solver.subgrid_peak(traj_exact[-1])

    amp_error = abs(A_peak_num - A)
    center_error = abs(x_peak_num - x_peak_exact)

    print(f"Final t=10.0 Relative L2 error: {final_rel_l2:.3e}")
    print(f"Final Peak Amplitude Error:     {amp_error:.3e} (Numerical: {A_peak_num:.5f}, Target: {A:.5f})")
    print(f"Final Peak Center Error:        {center_error:.3e}")
    print(f"Max I1 drift:                   {np.max(drift_I1):.3e}")
    print(f"Max I2 drift:                   {np.max(drift_I2):.3e}")
    print(f"Max I3 drift:                   {np.max(drift_I3):.3e}")

    # Generate Figure 1
    plot_path = project_root / "outputs" / "plots" / "fig1_solver_validation.png"
    plot_solver_validation(t_eval, solver.x, traj_num, traj_exact, invariants, str(plot_path))
    print(f"Generated validation plot: {plot_path}")

    # Verify sanity checks
    assert final_rel_l2 < 1e-4, f"Final relative L2 error {final_rel_l2:.2e} too high!"
    assert amp_error < 1e-3, f"Peak amplitude error {amp_error:.2e} too high!"
    assert np.max(drift_I1) < 1e-6, "I1 conservation drift too high!"
    assert np.max(drift_I2) < 1e-6, "I2 conservation drift too high!"
    print("ALL NUMERICAL SOLVER SANITY CHECKS PASSED!")


if __name__ == "__main__":
    main()
