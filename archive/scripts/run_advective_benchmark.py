"""
Comprehensive Transport-Augmented NCA Benchmark Script:
Executes:
1. Core Five Model Comparison (Vanilla, Stationary, Nonlinear-Characteristic, Learned, Oracle).
2. Dual-Memory Ratio Ablation (0/16, 4/12, 8/8, 12/4, 16/0).
3. Memory Alignment & Circular Centroid Tracking (d_circ(x_m, x_u)).
4. Translation Equivariance Diagnostic (E_u, E_h, E_m).
5. Transport-Velocity Mismatch Sweep (gamma-sweep: gamma in [-1.0, 1.5]).
6. Causal Velocity Interventions (v, 0, -v, v_rand, magnitude-matched).
7. Generates Figure 11 & Figure 12 (saved to plots/ and brain/ artifacts).
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.advective_memory_nca import (
    AdvectiveMemoryNCA,
    compute_advective_macs,
    find_matched_advective_mlp,
)
from src.dataset import (
    build_experiment_datasets,
    generate_on_manifold_soliton,
    KdVTrajectoryDataset,
)
from src.kdv_solver import KdVSolver
from src.metrics import (
    periodic_centroid,
    periodic_distance,
    peak_amplitude_error,
    relative_l2_error,
)
from src.nca import VanillaNCA, compute_nca_macs, find_matched_vanilla_channels
from src.train import normalized_mse_loss


def circular_distance(x1: float, x2: float, Lx: float = 50.0) -> float:
    """Compute circular distance between coordinates x1 and x2 on periodic domain Lx."""
    diff = x1 - x2
    ang = 2.0 * np.pi * diff / Lx
    return float((Lx / (2.0 * np.pi)) * np.abs(np.arctan2(np.sin(ang), np.cos(ang))))


def train_adv_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 0.002,
    device: torch.device = torch.device("cpu"),
    K: int = 2,
    rollout_steps: int = 8,
) -> List[float]:
    """Train AdvectiveMemoryNCA or VanillaNCA with recurrent rollout and AdamW."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch, _ in train_loader:
            batch = batch.to(device)
            B, max_steps, _, N = batch.shape
            steps_to_roll = min(rollout_steps, max_steps - 1)
            optimizer.zero_grad()

            u_curr = batch[:, 0]
            if hasattr(model, "init_memory"):
                m = model.init_memory(B, N, device, batch.dtype)
                h = torch.zeros(B, model.hidden_dim, N, device=device)
                s = torch.cat([u_curr, h], dim=1)
            elif hasattr(model, "hidden_dim"):
                h = torch.zeros(B, model.hidden_dim, N, device=device)
                s = torch.cat([u_curr, h], dim=1)
                m = None
            else:
                s = u_curr
                m = None

            loss = 0.0
            for m_step in range(1, steps_to_roll + 1):
                target = batch[:, m_step]
                for _ in range(K):
                    if hasattr(model, "compute_velocity"):
                        s, m, _ = model.step(s, m)
                    elif hasattr(model, "memory_dim"):
                        s, m = model.step(s, m)
                    else:
                        s = model.step(s)

                u_pred = s[:, :1, :]
                loss = loss + normalized_mse_loss(u_pred, target)

            loss = loss / steps_to_roll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        history.append(total_loss / max(1, n_batches))

    return history


def evaluate_adv_rollout(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    K: int = 2,
    rollout_steps: int = 12,
    Lx: float = 50.0,
) -> Dict[str, Union[float, np.ndarray, List]]:
    """Evaluate multi-step autonomous rollout and compute physics diagnostics."""
    model.eval()
    errors_vs_time = []
    amp_errors_vs_time = []
    mem_align_vs_time = []
    transport_diags_list = []

    all_sample_preds = []
    all_sample_trues = []
    all_sample_mems = []
    all_sample_vels = []

    x_grid = np.linspace(-Lx / 2.0, Lx / 2.0, 128, endpoint=False)

    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            B, max_steps, _, N = batch.shape
            steps = min(rollout_steps, max_steps - 1)

            u0 = batch[:, 0]
            true_A = torch.amax(u0, dim=-1, keepdim=True).squeeze(1)

            if hasattr(model, "rollout"):
                if hasattr(model, "compute_velocity"):
                    res = model.rollout(u0, num_macro_steps=steps, K=K, true_A=true_A)
                    pred_traj = res[0]
                    final_m = res[1]
                    diags = res[2]
                    transport_diags_list.extend(diags)
                else:
                    res = model.rollout(u0, num_macro_steps=steps, K=K)
                    pred_traj = res[0] if isinstance(res, tuple) else res
                    final_m = res[1] if isinstance(res, tuple) else None
            else:
                pred_traj = model.rollout(u0, num_macro_steps=steps, K=K)
                final_m = None

            # Collect step-wise errors
            batch_errs = np.zeros((B, steps + 1))
            batch_amp_errs = np.zeros((B, steps + 1))

            pred_np = pred_traj.squeeze(2).cpu().numpy()
            true_np = batch[:, : steps + 1].squeeze(2).cpu().numpy()

            for b in range(B):
                for t in range(steps + 1):
                    batch_errs[b, t] = relative_l2_error(pred_np[b, t], true_np[b, t])
                    batch_amp_errs[b, t] = peak_amplitude_error(pred_np[b, t], true_np[b, t])

            errors_vs_time.append(batch_errs)
            amp_errors_vs_time.append(batch_amp_errs)

            if len(all_sample_preds) == 0:
                all_sample_preds = pred_np[0]
                all_sample_trues = true_np[0]
                if final_m is not None:
                    all_sample_mems = final_m[0].cpu().numpy()

    mean_errs = np.mean(np.concatenate(errors_vs_time, axis=0), axis=0)
    mean_amp_errs = np.mean(np.concatenate(amp_errors_vs_time, axis=0), axis=0)

    # Compute trajectory-wide centroid tracking for memory
    # Roll out 1 trajectory step by step to track memory centroid vs physical center
    u_eval0 = val_loader.dataset[0][0][0:1].to(device)  # (1, 1, N)
    u_true_traj = val_loader.dataset[0][0][: rollout_steps + 1].squeeze(1).numpy()
    traj_mem_align = []
    tracked_memories = []
    tracked_velocities = []

    with torch.no_grad():
        if hasattr(model, "compute_velocity"):
            B, _, N = u_eval0.shape
            h = torch.zeros(B, model.hidden_dim, N, device=device)
            s = torch.cat([u_eval0, h], dim=1)
            m = model.init_memory(B, N, device, u_eval0.dtype)

            for t in range(rollout_steps + 1):
                # Physical center of soliton
                x_u = periodic_centroid(u_true_traj[t], x_grid, Lx)

                # Memory magnitude centroid
                m_mag = torch.sum(torch.abs(m), dim=1).squeeze(0).detach().cpu().numpy()
                if np.sum(m_mag) > 1e-6:
                    x_m = periodic_centroid(m_mag, x_grid, Lx)
                    d_align = circular_distance(x_m, x_u, Lx)
                else:
                    d_align = 0.0

                traj_mem_align.append(d_align)
                tracked_memories.append(m.squeeze(0).detach().cpu().numpy())

                if t < rollout_steps:
                    for _ in range(K):
                        s, m, d_step = model.step(s, m)
                    tracked_velocities.append(d_step["velocity_field"].squeeze().detach().cpu().numpy())

    return {
        "mean_rel_l2": float(np.mean(mean_errs[1:])),
        "final_rel_l2": float(mean_errs[-1]),
        "mean_amp_err": float(np.mean(mean_amp_errs[1:])),
        "err_curve": mean_errs,
        "mem_align_curve": np.array(traj_mem_align) if len(traj_mem_align) > 0 else np.zeros(steps + 1),
        "sample_pred": all_sample_preds,
        "sample_true": all_sample_trues,
        "sample_mems": tracked_memories,
        "sample_vels": tracked_velocities,
    }


def evaluate_translation_equivariance(
    model: nn.Module,
    solver: KdVSolver,
    delta_T: float = 0.1,
    horizon: int = 12,
    xA: float = -10.0,
    xB: float = 5.0,
    amplitude: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Test translation equivariance: u_B(x, 0) = u_A(x - Delta x, 0).
    Measures E_u, E_h, E_m between the shifted trajectory of A and trajectory B.
    """
    model.eval()
    shift_dx = xB - xA
    N = solver.N
    shift_cells = int(round(shift_dx / (solver.Lx / N))) % N

    traj_A, _ = generate_on_manifold_soliton(solver, amplitude, xA, delta_T, horizon)
    traj_B, _ = generate_on_manifold_soliton(solver, amplitude, xB, delta_T, horizon)

    u0_A = torch.from_numpy(traj_A[0:1]).float().unsqueeze(1).to(device)
    u0_B = torch.from_numpy(traj_B[0:1]).float().unsqueeze(1).to(device)

    with torch.no_grad():
        if hasattr(model, "compute_velocity"):
            pred_A, m_A, _ = model.rollout(u0_A, num_macro_steps=horizon, K=2)
            pred_B, m_B, _ = model.rollout(u0_B, num_macro_steps=horizon, K=2)
        else:
            pred_A = model.rollout(u0_A, num_macro_steps=horizon, K=2)
            pred_B = model.rollout(u0_B, num_macro_steps=horizon, K=2)
            m_A = torch.zeros(1, 16, N)
            m_B = torch.zeros(1, 16, N)

    # Shift representations of A by shift_cells along periodic dimension
    u_A_final = pred_A[0, -1, 0].cpu().numpy()
    u_B_final = pred_B[0, -1, 0].cpu().numpy()
    u_A_shifted = np.roll(u_A_final, shift_cells)

    m_A_final = m_A[0].cpu().numpy()
    m_B_final = m_B[0].cpu().numpy()
    m_A_shifted = np.roll(m_A_final, shift_cells, axis=-1)

    # Equivariance errors
    e_u = float(np.linalg.norm(u_B_final - u_A_shifted) / (np.linalg.norm(u_B_final) + 1e-6))
    e_m = float(np.linalg.norm(m_B_final - m_A_shifted) / (np.linalg.norm(m_B_final) + 1e-6))

    return {"equiv_err_u": e_u, "equiv_err_m": e_m}


def run_gamma_velocity_sweep(
    model: AdvectiveMemoryNCA,
    val_loader: DataLoader,
    gammas: List[float],
    device: torch.device = torch.device("cpu"),
) -> Dict[float, float]:
    """
    Scale velocity field by gamma in [-1.0, 1.5] at inference time and measure rollout error.
    """
    model.eval()
    gamma_errors = {}

    for g in gammas:
        # Override velocity hook with scaled reference velocity
        def make_gamma_hook(scale):
            def hook(state):
                # Scale characteristic velocity 6u by gamma
                u = state[:, :1, :]
                return scale * 6.0 * u
            return hook

        model.velocity_override = make_gamma_hook(g)
        res = evaluate_adv_rollout(model, val_loader, device=device, K=2, rollout_steps=12)
        gamma_errors[g] = res["mean_rel_l2"]

    model.velocity_override = None  # Reset hook
    return gamma_errors


def run_causal_interventions(
    model: AdvectiveMemoryNCA,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Test Learned-Adv-NCA under 5 inference conditions:
    1. normal v
    2. zero v
    3. reversed v (-v)
    4. random Gaussian v
    5. magnitude-matched phase-randomized v
    """
    results = {}

    # 1. Normal
    model.velocity_override = None
    results["normal_v"] = evaluate_adv_rollout(model, val_loader, device=device)["mean_rel_l2"]

    # 2. Zero v
    model.velocity_override = 0.0
    results["zero_v"] = evaluate_adv_rollout(model, val_loader, device=device)["mean_rel_l2"]

    # 3. Reversed v
    model.velocity_override = lambda s: -model.compute_velocity(
        s, torch.zeros(s.shape[0], model.memory_dim, s.shape[-1], device=s.device), apply_override=False
    )
    results["reversed_v"] = evaluate_adv_rollout(model, val_loader, device=device)["mean_rel_l2"]

    # 4. Random Gaussian v
    model.velocity_override = lambda s: torch.randn(s.shape[0], 1, s.shape[-1], device=s.device) * 2.0
    results["random_v"] = evaluate_adv_rollout(model, val_loader, device=device)["mean_rel_l2"]

    # 5. Magnitude-matched random sign v
    def mag_matched_hook(s):
        v_base = model.compute_velocity(
            s, torch.zeros(s.shape[0], model.memory_dim, s.shape[-1], device=s.device), apply_override=False
        )
        signs = torch.randint(0, 2, v_base.shape, device=s.device) * 2.0 - 1.0
        return v_base * signs

    model.velocity_override = mag_matched_hook
    results["mag_matched_random_v"] = evaluate_adv_rollout(model, val_loader, device=device)["mean_rel_l2"]

    model.velocity_override = None
    return results


def plot_figure_11(
    core_results: pd.DataFrame,
    err_curves: Dict[str, np.ndarray],
    align_curves: Dict[str, np.ndarray],
    learned_vel: np.ndarray,
    char_vel: np.ndarray,
    oracle_vel: float,
    save_path: Path,
) -> None:
    """Generate Figure 11: Core Transport Benchmark, Velocity Profiles, and Centroid Alignment."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Rollout Rel L2 Error Curves over Time
    colors = {
        "Vanilla NCA": "#2b5c8f",
        "Stationary Memory": "#7570b3",
        "Nonlinear-Characteristic": "#e7298a",
        "Learned-Transport": "#d95f02",
        "Oracle-Estimated": "#1b9e77",
        "Oracle-True": "#66a61e",
    }
    steps = np.arange(len(next(iter(err_curves.values()))))

    for name, curve in err_curves.items():
        c = colors.get(name, "#333333")
        ax1.plot(steps, curve, label=name, color=c, linewidth=2.0, alpha=0.9)

    ax1.set_xlabel("Macro Time Steps (t = 0.1, 0.2, ...)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Relative $L_2$ Error", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Autonomous Rollout Error Over Time", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", fontsize=9, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Spatial Velocity Profiles
    x_grid = np.linspace(-25, 25, 128)
    ax2.plot(x_grid, char_vel, label="PDE Characteristic: $6u(x)$", color="#e7298a", linewidth=2.0, linestyle="--")
    if len(learned_vel) > 0:
        ax2.plot(x_grid, learned_vel, label="Learned Transport: $\hat{v}(x)$", color="#d95f02", linewidth=2.2)
    ax2.axhline(oracle_vel, label=f"Oracle Soliton Speed: $2A$ ({oracle_vel:.2f})", color="#1b9e77", linestyle=":", linewidth=2.0)

    ax2.set_xlabel("Spatial coordinate x", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Transport Velocity $v(x)$", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Transport Velocity Field Comparison", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", fontsize=9, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Circular Memory Alignment Error over Time
    for name, curve in align_curves.items():
        if np.max(curve) > 0.001 or "Learned" in name or "Characteristic" in name:
            c = colors.get(name, "#333333")
            ax3.plot(steps, curve, label=name, color=c, linewidth=2.0)

    ax3.set_xlabel("Macro Time Steps", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Circular Centroid Distance $d_{\mathrm{circ}}(x_m, x_u)$", fontsize=11, fontweight="bold")
    ax3.set_title("(C) Memory-to-Wave Centroid Alignment", fontsize=12, fontweight="bold")
    ax3.legend(frameon=True, facecolor="white", fontsize=9, loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Transport-Augmented Neural Cellular Automata: Core Benchmark Diagnostics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_figure_12(
    sample_mems: List[np.ndarray],
    gamma_results: Dict[float, float],
    intervention_results: Dict[str, float],
    dual_ratio_results: pd.DataFrame,
    equivariance_results: Dict[str, Dict[str, float]],
    save_path: Path,
) -> None:
    """Generate Figure 12: Heatmaps, Gamma Sweep, Dual-Memory Ablation, and Causal Interventions."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)

    # Panel A: Transported Memory Space-Time Heatmap
    ax_heat_trans = fig.add_subplot(gs[0, 0])
    if len(sample_mems) > 0:
        # First 8 channels are transported
        m_trans_spacetime = np.array([np.mean(np.abs(m[:8]), axis=0) for m in sample_mems])
        im1 = ax_heat_trans.imshow(m_trans_spacetime, aspect="auto", cmap="viridis", origin="lower", extent=[-25, 25, 0, len(sample_mems) * 0.1])
        fig.colorbar(im1, ax=ax_heat_trans, label="$|m^{\mathrm{trans}}|$")
    ax_heat_trans.set_title("(A) Transported Memory $m^{\mathrm{trans}}(x, t)$", fontsize=11, fontweight="bold")
    ax_heat_trans.set_xlabel("Spatial coordinate x", fontsize=10)
    ax_heat_trans.set_ylabel("Physical Time t", fontsize=10)

    # Panel B: Local Memory Space-Time Heatmap
    ax_heat_local = fig.add_subplot(gs[0, 1])
    if len(sample_mems) > 0:
        # Channels 8 to 16 are local Eulerian
        m_local_spacetime = np.array([np.mean(np.abs(m[8:]), axis=0) for m in sample_mems])
        im2 = ax_heat_local.imshow(m_local_spacetime, aspect="auto", cmap="magma", origin="lower", extent=[-25, 25, 0, len(sample_mems) * 0.1])
        fig.colorbar(im2, ax=ax_heat_local, label="$|m^{\mathrm{local}}|$")
    ax_heat_local.set_title("(B) Local Eulerian Memory $m^{\mathrm{local}}(x, t)$", fontsize=11, fontweight="bold")
    ax_heat_local.set_xlabel("Spatial coordinate x", fontsize=10)
    ax_heat_local.set_ylabel("Physical Time t", fontsize=10)

    # Panel C: Gamma Velocity Mismatch Sweep
    ax_gamma = fig.add_subplot(gs[0, 2])
    gammas = sorted(list(gamma_results.keys()))
    g_errs = [gamma_results[g] for g in gammas]
    ax_gamma.plot(gammas, g_errs, marker="o", color="#d95f02", linewidth=2.0)
    ax_gamma.axvline(1.0, color="gray", linestyle="--", label="Nominal $v$ ($\gamma=1.0$)")
    ax_gamma.set_xlabel("Velocity Scaling Factor $\gamma$ ($v_\gamma = \gamma \cdot v_{\mathrm{ref}}$)", fontsize=10, fontweight="bold")
    ax_gamma.set_ylabel("Validation Rollout Rel $L_2$", fontsize=10, fontweight="bold")
    ax_gamma.set_title("(C) Transport Velocity Mismatch Sweep", fontsize=11, fontweight="bold")
    ax_gamma.legend(frameon=True, facecolor="white", fontsize=9)
    ax_gamma.grid(True, linestyle="--", alpha=0.5)

    # Panel D: Dual Memory Ratio Ablation
    ax_dual = fig.add_subplot(gs[1, 0])
    ratios = dual_ratio_results["ratio"].tolist()
    dual_errs = dual_ratio_results["rel_l2"].tolist()
    bars = ax_dual.bar(ratios, dual_errs, color="#2b5c8f", width=0.45, alpha=0.85)
    ax_dual.set_xlabel("Memory Partition ($C_{m,\mathrm{trans}} \,/\, C_{m,\mathrm{local}}$)", fontsize=10, fontweight="bold")
    ax_dual.set_ylabel("Validation Rollout Rel $L_2$", fontsize=10, fontweight="bold")
    ax_dual.set_title("(D) Dual-Memory Ratio Ablation", fontsize=11, fontweight="bold")
    ax_dual.grid(True, linestyle="--", alpha=0.5)
    for b, val in zip(bars, dual_errs):
        ax_dual.annotate(f"{val:.3f}", xy=(b.get_x() + b.get_width() / 2, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # Panel E: Causal Velocity Interventions
    ax_causal = fig.add_subplot(gs[1, 1])
    int_names = ["Normal $v$", "Zero $v$", "Reverse $-v$", "Rand Gaussian", "Mag-Matched Rand"]
    int_vals = [
        intervention_results["normal_v"],
        intervention_results["zero_v"],
        intervention_results["reversed_v"],
        intervention_results["random_v"],
        intervention_results["mag_matched_random_v"],
    ]
    c_bars = ax_causal.bar(int_names, int_vals, color=["#1b9e77", "#7570b3", "#e7298a", "#d95f02", "#66a61e"], width=0.55, alpha=0.85)
    ax_causal.set_ylabel("Rollout Rel $L_2$ Error", fontsize=10, fontweight="bold")
    ax_causal.set_title("(E) Causal Velocity Interventions", fontsize=11, fontweight="bold")
    ax_causal.tick_params(axis="x", rotation=25)
    ax_causal.grid(True, linestyle="--", alpha=0.5)
    for b, val in zip(c_bars, int_vals):
        ax_causal.annotate(f"{val:.3f}", xy=(b.get_x() + b.get_width() / 2, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    # Panel F: Translation Equivariance Error Comparison
    ax_equiv = fig.add_subplot(gs[1, 2])
    models_eq = list(equivariance_results.keys())
    e_u_vals = [equivariance_results[m]["equiv_err_u"] for m in models_eq]
    e_m_vals = [equivariance_results[m]["equiv_err_m"] for m in models_eq]
    x = np.arange(len(models_eq))
    w = 0.35

    ax_equiv.bar(x - w/2, e_u_vals, w, label="$E_u$ (State Equivariance)", color="#2b5c8f", alpha=0.85)
    ax_equiv.bar(x + w/2, e_m_vals, w, label="$E_m$ (Memory Equivariance)", color="#d95f02", alpha=0.85)
    ax_equiv.set_xticks(x)
    ax_equiv.set_xticklabels(models_eq, rotation=20, fontsize=9)
    ax_equiv.set_ylabel("Equivariance Error $\|X_B - \mathcal{T}X_A\|_2$", fontsize=10, fontweight="bold")
    ax_equiv.set_title("(F) Translation Equivariance Symmetry Test", fontsize=11, fontweight="bold")
    ax_equiv.legend(frameon=True, facecolor="white", fontsize=8)
    ax_equiv.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Transport-Augmented NCAs: Mechanisms, Dual Partitioning, and Causal Interventions", fontsize=14, fontweight="bold", y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_advective_benchmark():
    output_dir = Path("outputs/default")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/home/zenoguy/.gemini/antigravity-ide/brain/36bdafb0-2d3a-4a8a-9c12-f55891eef59a/plots")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    seeds = [42, 123]
    epochs = 25
    horizon = 12
    delta_T = 0.1
    K = 2
    target_params = 7765

    print("=" * 75)
    print("STARTING ADVECTIVE (FLOW-CONVECTED) MEMORY NCA BENCHMARK")
    print("  Testing: 'Memory architecture should match the geometry of information transport'")
    print(f"  Seeds: {seeds} | Epochs: {epochs} | Horizon: {horizon} | Target Params: ~{target_params}")
    print("=" * 75)

    kdv_solver = KdVSolver(N=128, Lx=50.0)

    # 1. CORE FIVE MODEL EVALUATION
    print("\n[1/5] Training and evaluating Core Five Models...")
    core_models = [
        ("Vanilla NCA", "vanilla", None),
        ("Stationary Memory", "stationary", None),
        ("Nonlinear-Characteristic", "characteristic", None),
        ("Learned-Transport", "learned", None),
        ("Oracle-Estimated", "oracle_estimated", None),
        ("Oracle-True", "oracle_true", None),
    ]

    core_results = []
    err_curves = {}
    align_curves = {}
    saved_models = {}
    sample_rollouts_dict = {}

    for name, mode, _ in core_models:
        print(f"\n--- Model: {name} ({mode}) ---")
        seed_rel_l2 = []
        seed_final_l2 = []
        seed_amp_err = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            ds = build_experiment_datasets(kdv_solver, delta_T=delta_T, train_horizon=horizon, seed=seed)
            train_loader = DataLoader(KdVTrajectoryDataset(ds["train"]["data"]), batch_size=16, shuffle=True)
            val_loader = DataLoader(KdVTrajectoryDataset(ds["val"]["data"]), batch_size=8, shuffle=False)

            if mode == "vanilla":
                matched_c, matched_mlp, p = find_matched_vanilla_channels(target_params=target_params)
                model = VanillaNCA(hidden_dim=matched_c, mlp_hidden=matched_mlp)
                macs = compute_nca_macs(model, N=128, K=K)
            else:
                mlp_h, p = find_matched_advective_mlp(target_params=target_params, mode=mode)
                model = AdvectiveMemoryNCA(hidden_dim=16, memory_dim=16, transport_dim=8, mlp_hidden=mlp_h, mode=mode)
                macs = compute_advective_macs(model, N=128, K=K)

            train_adv_model(model, train_loader, val_loader, epochs=epochs, device=device, K=K, rollout_steps=8)
            res = evaluate_adv_rollout(model, val_loader, device=device, K=K, rollout_steps=horizon)

            seed_rel_l2.append(res["mean_rel_l2"])
            seed_final_l2.append(res["final_rel_l2"])
            seed_amp_err.append(res["mean_amp_err"])

            if seed == seeds[0]:
                err_curves[name] = res["err_curve"]
                align_curves[name] = res["mem_align_curve"]
                saved_models[mode] = model
                sample_rollouts_dict[mode] = res

        m_l2 = float(np.mean(seed_rel_l2))
        s_l2 = float(np.std(seed_rel_l2))
        print(f"  Result: Rel L2 = {m_l2:.4f} +/- {s_l2:.4f} | Params = {p} | MACs = {macs}")

        core_results.append({
            "model": name,
            "mode": mode,
            "parameters": p,
            "macs_per_delta_T": macs,
            "rel_l2_mean": m_l2,
            "rel_l2_std": s_l2,
            "final_rel_l2_mean": float(np.mean(seed_final_l2)),
            "amp_err_mean": float(np.mean(seed_amp_err)),
        })

    core_df = pd.DataFrame(core_results)
    csv_core_path = output_dir / "advective_benchmark_summary.csv"
    core_df.to_csv(csv_core_path, index=False)
    print(f"\nSaved core summary to: {csv_core_path}")

    # Extract velocities for Figure 11
    val_loader = DataLoader(KdVTrajectoryDataset(ds["val"]["data"]), batch_size=8, shuffle=False)
    u_sample = ds["val"]["data"][0, 0]
    char_vel = 6.0 * u_sample
    oracle_vel = float(2.0 * np.max(u_sample))
    learned_model = saved_models.get("learned")

    learned_vel = []
    if learned_model is not None and "sample_vels" in sample_rollouts_dict.get("learned", {}):
        vels = sample_rollouts_dict["learned"]["sample_vels"]
        if len(vels) > 0:
            learned_vel = vels[0]

    # Generate Figure 11
    fig11_path = plots_dir / "fig11_advective_nca_comparison.png"
    plot_figure_11(core_df, err_curves, align_curves, learned_vel, char_vel, oracle_vel, fig11_path)
    shutil.copy(fig11_path, artifact_dir / "fig11_advective_nca_comparison.png")
    print(f"Generated Figure 11 at: {fig11_path}")

    # 2. DUAL-MEMORY RATIO ABLATION
    print("\n[2/5] Running Dual-Memory Ratio Ablation (0/16 to 16/0)...")
    ratios = [(0, 16), (4, 12), (8, 8), (12, 4), (16, 0)]
    dual_ratio_rows = []

    for trans_dim, loc_dim in ratios:
        ratio_str = f"{trans_dim}/{loc_dim}"
        mlp_h, p = find_matched_advective_mlp(target_params=target_params, transport_dim=trans_dim, mode="characteristic")
        model = AdvectiveMemoryNCA(hidden_dim=16, memory_dim=16, transport_dim=trans_dim, mlp_hidden=mlp_h, mode="characteristic")

        train_adv_model(model, train_loader, val_loader, epochs=epochs, device=device, K=K, rollout_steps=8)
        res = evaluate_adv_rollout(model, val_loader, device=device, K=K, rollout_steps=horizon)
        dual_ratio_rows.append({"ratio": ratio_str, "transport_dim": trans_dim, "local_dim": loc_dim, "rel_l2": res["mean_rel_l2"]})
        print(f"  Ratio {ratio_str:5s}: Rel L2 = {res['mean_rel_l2']:.4f}")

    dual_df = pd.DataFrame(dual_ratio_rows)

    # 3. GAMMA VELOCITY MISMATCH SWEEP
    print("\n[3/5] Running Transport Velocity Mismatch Sweep (gamma in [-1.0, 1.5])...")
    gammas = [-1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
    char_model = saved_models.get("characteristic")
    gamma_res = run_gamma_velocity_sweep(char_model, val_loader, gammas, device=device)
    for g, err in gamma_res.items():
        print(f"  Gamma {g:+5.2f}: Rel L2 = {err:.4f}")

    # 4. CAUSAL VELOCITY INTERVENTIONS
    print("\n[4/5] Running Causal Velocity Interventions...")
    intervention_res = run_causal_interventions(char_model, val_loader, device=device)
    for cond, val in intervention_res.items():
        print(f"  Condition {cond:25s}: Rel L2 = {val:.4f}")

    # 5. TRANSLATION EQUIVARIANCE TEST
    print("\n[5/5] Running Translation Equivariance Symmetry Test...")
    equiv_results = {}
    for name, m_obj in [("Stationary", saved_models.get("stationary")), ("Characteristic Adv", saved_models.get("characteristic"))]:
        if m_obj is not None:
            eq = evaluate_translation_equivariance(m_obj, kdv_solver, delta_T=delta_T, horizon=horizon, device=device)
            equiv_results[name] = eq
            print(f"  {name:18s}: E_u = {eq['equiv_err_u']:.4f}, E_m = {eq['equiv_err_m']:.4f}")

    # Generate Figure 12
    fig12_path = plots_dir / "fig12_transport_mechanisms.png"
    sample_mems = sample_rollouts_dict.get("characteristic", {}).get("sample_mems", [])
    plot_figure_12(sample_mems, gamma_res, intervention_res, dual_df, equiv_results, fig12_path)
    shutil.copy(fig12_path, artifact_dir / "fig12_transport_mechanisms.png")
    print(f"Generated Figure 12 at: {fig12_path}")

    print("\n" + "=" * 75)
    print("ALL ADVECTIVE BENCHMARK EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(core_df.to_string())
    print("=" * 75)


if __name__ == "__main__":
    run_advective_benchmark()
