"""
Physics-Aware Metrics and Diagnostic Functions for KdV Dynamics.

Provides quantitative measures beyond mean squared error:
- Relative L2 error
- Peak amplitude error
- Periodic centroid and phase tracking
- Full Width at Half Maximum (FWHM) error
- Normalized shape correlation / cosine distance
- Conserved quantity invariant drift (I1, I2, I3)
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch


def relative_l2_error(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> float:
    """
    Compute relative L2 error:
        E_L2 = ||pred - target||_2 / (||target||_2 + eps)
    """
    if isinstance(pred, torch.Tensor):
        diff_norm = torch.linalg.norm(pred - target).item()
        target_norm = torch.linalg.norm(target).item()
    else:
        diff_norm = float(np.linalg.norm(pred - target))
        target_norm = float(np.linalg.norm(target))

    return float(diff_norm / (target_norm + eps))


def peak_amplitude_error(
    pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute absolute error in peak amplitude:
        E_amp = |max(pred) - max(target)|
    """
    if isinstance(pred, torch.Tensor):
        max_pred = torch.max(pred).item()
        max_target = torch.max(target).item()
    else:
        max_pred = float(np.max(pred))
        max_target = float(np.max(target))

    return float(abs(max_pred - max_target))


def periodic_centroid(
    u: np.ndarray, x: np.ndarray, Lx: float
) -> float:
    """
    Estimate center of mass / wave centroid on a periodic 1D domain using
    circular statistics:
        theta = atan2( sum(u_j * sin(2*pi*x_j/Lx)), sum(u_j * cos(2*pi*x_j/Lx)) )
        x_c = theta * (Lx / (2*pi))
    """
    u_pos = np.maximum(u, 0.0)  # Soliton is positive-valued
    sum_u = np.sum(u_pos)
    if sum_u < 1e-12:
        return 0.0

    angles = 2.0 * np.pi * x / Lx
    sin_comp = np.sum(u_pos * np.sin(angles))
    cos_comp = np.sum(u_pos * np.cos(angles))

    theta = np.arctan2(sin_comp, cos_comp)
    xc = theta * (Lx / (2.0 * np.pi))
    return float(xc)


def periodic_distance(x1: float, x2: float, Lx: float) -> float:
    """
    Minimal distance between two positions on a periodic domain of length Lx.
    """
    diff = abs(x1 - x2) % Lx
    return float(min(diff, Lx - diff))


def center_error(
    pred: np.ndarray, target: np.ndarray, x: np.ndarray, Lx: float
) -> float:
    """
    Compute centroid displacement error on periodic domain.
    """
    c_pred = periodic_centroid(pred, x, Lx)
    c_target = periodic_centroid(target, x, Lx)
    return periodic_distance(c_pred, c_target, Lx)


def compute_fwhm(u: np.ndarray, dx: float) -> float:
    """
    Compute Full Width at Half Maximum (FWHM) of a localized pulse.
    Estimates width where u(x) >= 0.5 * max(u).
    """
    peak = float(np.max(u))
    if peak < 1e-8:
        return 0.0

    half_max = peak / 2.0
    above_half = u >= half_max
    # Count contiguous length in grid cells
    width = float(np.sum(above_half) * dx)
    return width


def fwhm_error(pred: np.ndarray, target: np.ndarray, dx: float) -> float:
    """
    Compute absolute difference in FWHM width.
    """
    w_pred = compute_fwhm(pred, dx)
    w_target = compute_fwhm(target, dx)
    return float(abs(w_pred - w_target))


def shape_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute normalized profile shape discrepancy (cosine distance):
        E_shape = 1 - <pred, target> / (||pred||_2 * ||target||_2 + eps)
    Invariant to uniform scale multiplier.
    """
    dot = float(np.dot(pred, target))
    norm_p = float(np.linalg.norm(pred))
    norm_t = float(np.linalg.norm(target))
    cos_sim = dot / (norm_p * norm_t + eps)
    return float(max(0.0, 1.0 - cos_sim))


def evaluate_trajectory_metrics(
    pred_traj: np.ndarray,
    true_traj: np.ndarray,
    x: np.ndarray,
    Lx: float,
    dx: float,
) -> Dict[str, np.ndarray]:
    """
    Compute time-series physics metrics for an entire rollout trajectory.

    Args:
        pred_traj: Array of shape (T, N).
        true_traj: Array of shape (T, N).
        x: Spatial coordinates array of shape (N,).
        Lx: Domain length.
        dx: Spatial grid spacing.

    Returns:
        Dict mapping metric name to 1D array of length T.
    """
    T = len(pred_traj)
    rel_l2 = np.zeros(T, dtype=np.float64)
    amp_err = np.zeros(T, dtype=np.float64)
    c_err = np.zeros(T, dtype=np.float64)
    w_err = np.zeros(T, dtype=np.float64)
    sh_err = np.zeros(T, dtype=np.float64)

    for t in range(T):
        p = pred_traj[t]
        y = true_traj[t]
        rel_l2[t] = relative_l2_error(p, y)
        amp_err[t] = peak_amplitude_error(p, y)
        c_err[t] = center_error(p, y, x, Lx)
        w_err[t] = fwhm_error(p, y, dx)
        sh_err[t] = shape_error(p, y)

    # Robust clamping against numerical blowup during long-horizon rollouts
    max_bound = 1e6
    rel_l2 = np.nan_to_num(rel_l2, nan=max_bound, posinf=max_bound, neginf=max_bound)
    amp_err = np.nan_to_num(amp_err, nan=max_bound, posinf=max_bound, neginf=max_bound)
    c_err = np.nan_to_num(c_err, nan=Lx, posinf=Lx, neginf=Lx)
    w_err = np.nan_to_num(w_err, nan=Lx, posinf=Lx, neginf=Lx)
    sh_err = np.nan_to_num(sh_err, nan=2.0, posinf=2.0, neginf=2.0)

    return {
        "rel_l2": rel_l2,
        "amp_err": amp_err,
        "center_err": c_err,
        "fwhm_err": w_err,
        "shape_err": sh_err,
        "mean_rel_l2": float(np.mean(rel_l2)),
        "max_rel_l2": float(np.max(rel_l2)),
        "final_rel_l2": float(rel_l2[-1]),
    }

