"""
Experiment 9: Causal Memory Swapping Diagnostic with Controls.

Tests whether persistent memory exerts specific, structured causal control over dynamics:
1. Past warm-up sequence in Regime A (alpha=4.0) -> extracts memory mA(t*)
2. Past warm-up sequence in Regime B (alpha=8.0) -> extracts memory mB(t*)
3. Hold instantaneous physical state u(t*) identical.
4. Roll out 4 conditions:
   - u(t*) + mA (Regime A memory)
   - u(t*) + mB (Regime B memory)
   - u(t*) + m_rand (Random Gaussian control)
   - u(t*) + m_zero (Zero baseline control)

Generates Figure 8: fig8_memory_swapping_causal.png
"""

import argparse
import json
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import torch
from src.kdv_solver import KdVSolver
from src.memory_nca import MemoryNCA
from src.evaluate import evaluate_causal_memory_swap
from src.visualization import plot_memory_swapping


def main():
    parser = argparse.ArgumentParser(description="Run Causal Memory Swapping Diagnostic")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["physics"]
    t_cfg = cfg["time_discretization"]
    m_cfg = cfg["models"]
    out_dir = Path(cfg["paths"]["output_dir"])
    plots_dir = out_dir / "plots"

    N = p_cfg["N"]
    Lx = p_cfg["Lx"]
    dx = Lx / N
    x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)
    K = t_cfg["K"]
    delta_T = t_cfg["delta_T"]

    print("=== Running Contextual Causal Memory Swapping Diagnostic ===")

    # 1. Solvers for two physical regimes:
    # Regime A: alpha=4.0, beta=1.0 (slower propagation v = 4*A/3)
    # Regime B: alpha=8.0, beta=1.0 (faster propagation v = 8*A/3)
    solver_A = KdVSolver(N=N, Lx=Lx, alpha=4.0, beta=1.0, dt=p_cfg["dt_internal"])
    solver_B = KdVSolver(N=N, Lx=Lx, alpha=8.0, beta=1.0, dt=p_cfg["dt_internal"])

    # Common initial state at t=0
    A = 1.0
    x0 = -10.0
    u0 = solver_A.exact_soliton(t=0.0, A=A, x0=x0)

    # Past warm-up sequence over 5 macro steps (t in [0, 0.5])
    warmup_steps = 5
    t_warmup = np.linspace(0.0, warmup_steps * delta_T, warmup_steps + 1)
    traj_warmup_A = solver_A.solve(u0, t_warmup)
    traj_warmup_B = solver_B.solve(u0, t_warmup)

    # Identical physical state held fixed at t* = warmup_steps * delta_T
    # We choose the exact soliton state at x=0 for both to isolate memory
    u_shared_np = solver_A.exact_soliton(t=0.0, A=A, x0=0.0)
    u_shared = torch.from_numpy(u_shared_np).float().unsqueeze(0).unsqueeze(0)  # (1, 1, N)

    warmup_tensor_A = torch.from_numpy(traj_warmup_A).float().unsqueeze(1)  # (steps+1, 1, N)
    warmup_tensor_B = torch.from_numpy(traj_warmup_B).float().unsqueeze(1)

    # Load or instantiate trained Memory-NCA
    ckpt_path = out_dir / "checkpoints" / "memory_persistent_seed42.pt"
    model = MemoryNCA(
        hidden_dim=m_cfg["hidden_dim"],
        memory_dim=m_cfg["memory_dim"],
        kernel_size=m_cfg["kernel_size"],
        mlp_hidden=m_cfg["mlp_hidden"],
    )
    if ckpt_path.exists():
        loaded = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(loaded["model_state"])
        print(f"Loaded trained Memory-NCA checkpoint: {ckpt_path.name}")
    else:
        print("Warning: Trained checkpoint not found. Evaluating on initialized model.")

    # 2. Run causal memory swap
    rollout_steps = 10
    swap_res = evaluate_causal_memory_swap(
        model=model,
        u_shared=u_shared,
        warmup_A=warmup_tensor_A,
        warmup_B=warmup_tensor_B,
        K=K,
        num_rollout_steps=rollout_steps,
    )

    # Measure final peak positions under the 4 conditions
    x_peak_A = float(x[np.argmax(swap_res["traj_mA"][-1])])
    x_peak_B = float(x[np.argmax(swap_res["traj_mB"][-1])])
    x_peak_rand = float(x[np.argmax(swap_res["traj_m_rand"][-1])])
    x_peak_zero = float(x[np.argmax(swap_res["traj_m_zero"][-1])])

    amp_A = float(np.max(swap_res["traj_mA"][-1]))
    amp_B = float(np.max(swap_res["traj_mB"][-1]))
    amp_rand = float(np.max(swap_res["traj_m_rand"][-1]))
    amp_zero = float(np.max(swap_res["traj_m_zero"][-1]))

    print(f"\nCondition 1 (Regime A Memory): Final Peak x = {x_peak_A:.2f}, Amp = {amp_A:.3f}")
    print(f"Condition 2 (Regime B Memory): Final Peak x = {x_peak_B:.2f}, Amp = {amp_B:.3f}")
    print(f"Condition 3 (Random Memory):   Final Peak x = {x_peak_rand:.2f}, Amp = {amp_rand:.3f}")
    print(f"Condition 4 (Zero Memory):     Final Peak x = {x_peak_zero:.2f}, Amp = {amp_zero:.3f}")

    swap_metrics = {
        "x_peak_A": x_peak_A,
        "x_peak_B": x_peak_B,
        "x_peak_rand": x_peak_rand,
        "x_peak_zero": x_peak_zero,
        "amp_A": amp_A,
        "amp_B": amp_B,
        "amp_rand": amp_rand,
        "amp_zero": amp_zero,
    }

    json_path = out_dir / "memory_swapping_results.json"
    with open(json_path, "w") as f:
        json.dump(swap_metrics, f, indent=2)
    print(f"Saved memory swap metrics to {json_path}")

    # Generate Figure 8
    t_rollout = np.linspace(0.0, rollout_steps * delta_T, rollout_steps + 1)
    fig8_path = plots_dir / "fig8_memory_swapping_causal.png"
    plot_memory_swapping(t_rollout, x, swap_res, str(fig8_path))
    print(f"Generated Figure 8: {fig8_path}")


if __name__ == "__main__":
    main()
