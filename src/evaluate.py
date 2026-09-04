"""
Evaluation Engine for Autonomous Multi-Step Rollout and Physics Diagnostics.

Enforces strict research protocols:
- Primary evaluation is 100% autonomous rollout (zero teacher forcing).
- One-step oracle test evaluates instantaneous transition with matched K micro-steps.
- Comprehensive physics-aware metric tracking across horizons.
- Contextual memory swapping with random and zero controls.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from src.dataset import KdVTrajectoryDataset
from src.metrics import (
    evaluate_trajectory_metrics,
    relative_l2_error,
    peak_amplitude_error,
    center_error,
    fwhm_error,
)


def evaluate_one_step_oracle(
    model: nn.Module,
    dataset: KdVTrajectoryDataset,
    K: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    One-step oracle diagnostic:
    Tests instantaneous transition operator u(t) -> u(t + Delta T) with matched K micro-steps.
    Decouples whether the network can learn the local transition from compounding autoregressive drift.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    errors = []
    amp_errors = []

    with torch.no_grad():
        for i in range(len(dataset)):
            traj, _ = dataset[i]  # (steps + 1, 1, N)
            traj = traj.to(device)
            num_steps = traj.shape[0] - 1

            for t in range(num_steps):
                u_curr = traj[t : t + 1]  # (1, 1, N)
                u_target = traj[t + 1 : t + 2]

                if hasattr(model, "forward"):
                    u_pred, _ = (
                        model.forward(u_curr, K=K)[:2]
                        if hasattr(model, "memory_dim")
                        else model.forward(u_curr, K=K)
                    )
                else:
                    u_pred = model(u_curr)

                err = relative_l2_error(u_pred.squeeze(), u_target.squeeze())
                amp_err = peak_amplitude_error(u_pred.squeeze(), u_target.squeeze())
                errors.append(err)
                amp_errors.append(amp_err)

    return {
        "one_step_mean_rel_l2": float(np.mean(errors)),
        "one_step_std_rel_l2": float(np.std(errors)),
        "one_step_mean_amp_err": float(np.mean(amp_errors)),
    }


def evaluate_autonomous_rollout(
    model: nn.Module,
    dataset: KdVTrajectoryDataset,
    K: int = 2,
    num_macro_steps: Optional[int] = None,
    x: Optional[np.ndarray] = None,
    Lx: float = 50.0,
    dx: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Autonomous multi-step rollout evaluation on full trajectory dataset.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    if x is None:
        N = dataset[0][0].shape[-1]
        dx = Lx / N
        x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)
    else:
        dx = dx or float(x[1] - x[0])

    all_rel_l2 = []
    all_amp_err = []
    all_center_err = []
    all_fwhm_err = []
    all_shape_err = []
    sample_rollouts = []

    with torch.no_grad():
        for i in range(len(dataset)):
            traj, meta = dataset[i]
            total_steps = traj.shape[0] - 1
            steps = num_macro_steps or total_steps
            steps = min(steps, total_steps)

            u0 = traj[0:1].to(device)  # (1, 1, N)

            if hasattr(model, "rollout"):
                res = model.rollout(u0, num_macro_steps=steps, K=K)
                pred_traj = res[0] if isinstance(res, tuple) else res
            else:
                pred_traj = model.rollout(u0, num_macro_steps=steps, K=K)

            pred_np = pred_traj.squeeze(0).squeeze(1).cpu().numpy()  # (steps+1, N)
            true_np = traj[: steps + 1].squeeze(1).cpu().numpy()  # (steps+1, N)

            metrics = evaluate_trajectory_metrics(pred_np, true_np, x, Lx, dx)
            all_rel_l2.append(metrics["rel_l2"])
            all_amp_err.append(metrics["amp_err"])
            all_center_err.append(metrics["center_err"])
            all_fwhm_err.append(metrics["fwhm_err"])
            all_shape_err.append(metrics["shape_err"])

            if i == 0:
                sample_rollouts = {
                    "pred": pred_np,
                    "true": true_np,
                }

    mean_rel_l2_time = np.mean(all_rel_l2, axis=0)
    mean_amp_err_time = np.mean(all_amp_err, axis=0)
    mean_center_err_time = np.mean(all_center_err, axis=0)
    mean_fwhm_err_time = np.mean(all_fwhm_err, axis=0)
    mean_shape_err_time = np.mean(all_shape_err, axis=0)

    return {
        "mean_rel_l2_overall": float(np.mean(mean_rel_l2_time)),
        "final_rel_l2": float(mean_rel_l2_time[-1]),
        "mean_amp_err_overall": float(np.mean(mean_amp_err_time)),
        "final_amp_err": float(mean_amp_err_time[-1]),
        "mean_center_err_overall": float(np.mean(mean_center_err_time)),
        "final_center_err": float(mean_center_err_time[-1]),
        "mean_fwhm_err_overall": float(np.mean(mean_fwhm_err_time)),
        "rel_l2_vs_time": mean_rel_l2_time,
        "amp_err_vs_time": mean_amp_err_time,
        "center_err_vs_time": mean_center_err_time,
        "fwhm_err_vs_time": mean_fwhm_err_time,
        "shape_err_vs_time": mean_shape_err_time,
        "sample": sample_rollouts,
    }


def evaluate_causal_memory_swap(
    model: nn.Module,
    u_shared: torch.Tensor,
    warmup_A: torch.Tensor,
    warmup_B: torch.Tensor,
    K: int = 2,
    num_rollout_steps: int = 15,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """
    Rigorous causal memory swap experiment:
    1. Memory mA extracted from past warm-up window for Regime A.
    2. Memory mB extracted from past warm-up window for Regime B.
    3. Hold instantaneous state u_shared identical.
    4. Roll out 4 conditions:
       - (u_shared, mA)
       - (u_shared, mB)
       - (u_shared, m_random)
       - (u_shared, m_zero)
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    u_shared = u_shared.to(device)

    with torch.no_grad():
        # Warmup sequence A (past only)
        _, mA = model.rollout(warmup_A[0:1].to(device), num_macro_steps=warmup_A.shape[0] - 1, K=K)
        # Warmup sequence B (past only)
        _, mB = model.rollout(warmup_B[0:1].to(device), num_macro_steps=warmup_B.shape[0] - 1, K=K)

        # Control 1: Random memory N(0, 1)
        m_rand = torch.randn_like(mA)
        # Control 2: Zero memory
        m_zero = torch.zeros_like(mA)

        # Rollout under all 4 conditions
        traj_A, _ = model.rollout(u_shared, num_macro_steps=num_rollout_steps, K=K, m0=mA)
        traj_B, _ = model.rollout(u_shared, num_macro_steps=num_rollout_steps, K=K, m0=mB)
        traj_rand, _ = model.rollout(u_shared, num_macro_steps=num_rollout_steps, K=K, m0=m_rand)
        traj_zero, _ = model.rollout(u_shared, num_macro_steps=num_rollout_steps, K=K, m0=m_zero)

    return {
        "traj_mA": traj_A.squeeze().cpu().numpy(),
        "traj_mB": traj_B.squeeze().cpu().numpy(),
        "traj_m_rand": traj_rand.squeeze().cpu().numpy(),
        "traj_m_zero": traj_zero.squeeze().cpu().numpy(),
    }
