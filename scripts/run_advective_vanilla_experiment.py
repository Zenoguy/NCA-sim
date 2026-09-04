"""
Comprehensive Advective Vanilla NCA Experiment & Protocol Freezing Script (Refined).

Fulfills all 16 Scientific Protocol Requirements:
1. Canonical Velocity Family:
   v_gamma(x, t) = gamma * 6u(x, t)
   - Eulerian Vanilla NCA (gamma = 0, stationary control)
   - Scaled Characteristic (gamma = 0.2)
   - Peak-Matched Characteristic Scaling (gamma = 1/3, v = 2u)
   - Intermediate Scaled Characteristic (gamma = 0.5)
   - Full Characteristic (gamma = 1.0, v = 6u)
   - Coherent-Structure Oracle (v = 2 * A_true)
   - Learned Velocity (v = v_hat_theta)
   - Stationary Memory-NCA (Cm = 16)
   - Advective Memory-NCA (Cm = 16)
2. Provenance Reproduction of 0.3287 vs 0.3649.
3. Explicit Stage 1 (Fixed-theta Intervention across 3 seeds: mu_gamma +/- sigma_gamma)
   vs Stage 2 (Transport-Conditioned Training: theta_gamma^* across seeds).
4. Full Parameter / Neural MAC / Transport Operations Accounting.
5. Velocity Diagnostics Block (max |v|, mean |v|, max displacement, CFL frac > 1).
6. Integer-Shift Exact Translation Equivariance (shifts in {1, 4, 16, 32} cells).
7. Long-Horizon Log-Scale Trajectory at T in {1, 5, 10, 25, 50, 100}.
8. Publication-Grade Figure 13 (6 panels).
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
from src.advective_vanilla_nca import (
    AdvectiveVanillaNCA,
    compute_advective_vanilla_macs,
)
from src.dataset import (
    build_experiment_datasets,
    generate_on_manifold_soliton,
    KdVTrajectoryDataset,
)
from src.kdv_solver import KdVSolver
from src.metrics import (
    peak_amplitude_error,
    periodic_centroid,
    periodic_distance,
    relative_l2_error,
)
from src.nca import VanillaNCA, compute_nca_macs, count_parameters
from src.train import normalized_mse_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 25,
    lr: float = 0.002,
    device: torch.device = torch.device("cpu"),
    K: int = 2,
    rollout_steps: int = 8,
) -> List[float]:
    """Train model using recurrent rollouts and AdamW."""
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
            true_A = torch.amax(u_curr, dim=-1, keepdim=True).squeeze(1)

            if isinstance(model, AdvectiveMemoryNCA):
                m = model.init_memory(B, N, device, batch.dtype)
                h = torch.zeros(B, model.hidden_dim, N, device=device)
                s = torch.cat([u_curr, h], dim=1)
            else:
                h = torch.zeros(B, model.hidden_dim, N, device=device)
                s = torch.cat([u_curr, h], dim=1)
                m = None

            loss = 0.0
            for m_step in range(1, steps_to_roll + 1):
                target = batch[:, m_step]
                for _ in range(K):
                    if isinstance(model, AdvectiveMemoryNCA):
                        s, m, _ = model.step(s, m, true_A=true_A)
                    elif isinstance(model, AdvectiveVanillaNCA):
                        s, _ = model.step(s, true_A=true_A)
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


def evaluate_rollout(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    K: int = 2,
    rollout_steps: int = 12,
    Lx: float = 50.0,
) -> Dict[str, Union[float, np.ndarray, List, Dict]]:
    """Evaluate multi-step rollout, velocity diagnostics, and physical errors."""
    model.eval()
    errors_vs_time = []
    amp_errors_vs_time = []

    all_sample_preds = []
    all_sample_trues = []
    all_sample_h = []
    all_diags = []

    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            B, max_steps, _, N = batch.shape
            steps = min(rollout_steps, max_steps - 1)

            u0 = batch[:, 0]
            true_A = torch.amax(u0, dim=-1, keepdim=True).squeeze(1)

            if isinstance(model, AdvectiveMemoryNCA):
                pred_traj, final_state, diags = model.rollout(u0, num_macro_steps=steps, K=K, true_A=true_A)
                all_diags.extend(diags)
            elif isinstance(model, AdvectiveVanillaNCA):
                pred_traj, final_state, diags = model.rollout(u0, num_macro_steps=steps, K=K, true_A=true_A)
                all_diags.extend(diags)
            else:
                pred_traj = model.rollout(u0, num_macro_steps=steps, K=K)
                final_state = None

            target_traj = batch[:, : steps + 1]

            for t in range(steps + 1):
                u_p = pred_traj[:, t, 0, :].cpu().numpy()
                u_t = target_traj[:, t, 0, :].cpu().numpy()

                step_rel_l2 = [relative_l2_error(u_p[b], u_t[b]) for b in range(B)]
                step_amp_err = [peak_amplitude_error(u_p[b], u_t[b]) for b in range(B)]

                if len(errors_vs_time) <= t:
                    errors_vs_time.append([])
                    amp_errors_vs_time.append([])
                errors_vs_time[t].extend(step_rel_l2)
                amp_errors_vs_time[t].extend(step_amp_err)

            if len(all_sample_preds) == 0:
                all_sample_preds = [pred_traj[0, t, 0, :].cpu().numpy() for t in range(steps + 1)]
                all_sample_trues = [target_traj[0, t, 0, :].cpu().numpy() for t in range(steps + 1)]
                if final_state is not None:
                    all_sample_h = final_state[0].cpu().numpy()

    mean_err_curve = [float(np.mean(e)) for e in errors_vs_time]
    mean_amp_curve = [float(np.mean(a)) for a in amp_errors_vs_time]

    # Velocity diagnostics computation
    v_stats = {
        "max_abs_v": 0.0,
        "mean_abs_v": 0.0,
        "max_disp": 0.0,
        "mean_disp": 0.0,
        "frac_disp_gt_1": 0.0,
    }
    if len(all_diags) > 0:
        v_stats["max_abs_v"] = float(np.max([d.get("max_abs_v", 0.0) for d in all_diags]))
        v_stats["mean_abs_v"] = float(np.mean([d.get("mean_abs_v", 0.0) for d in all_diags]))
        v_stats["max_disp"] = float(np.max([d.get("max_disp", 0.0) for d in all_diags]))
        v_stats["mean_disp"] = float(np.mean([d.get("mean_disp", 0.0) for d in all_diags]))
        v_stats["frac_disp_gt_1"] = float(np.mean([d.get("frac_disp_gt_1", 0.0) for d in all_diags]))

    return {
        "mean_rel_l2": float(np.mean(mean_err_curve[1:])),
        "final_rel_l2": float(mean_err_curve[-1]),
        "mean_amp_err": float(np.mean(mean_amp_curve[1:])),
        "err_curve": mean_err_curve,
        "sample_preds": all_sample_preds,
        "sample_trues": all_sample_trues,
        "sample_h": all_sample_h,
        "velocity_stats": v_stats,
    }


def evaluate_integer_shift_equivariance(
    model: nn.Module,
    val_loader: DataLoader,
    shifts: List[int] = [1, 4, 16, 32],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluate exact integer-cell translation equivariance:
        E_u(l) = ||F(T_l u) - T_l F(u)||_2 / ||F(T_l u)||_2
    """
    model.eval()
    shift_errors = []

    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            u0 = batch[:4, 0]  # First 4 samples
            true_A = torch.amax(u0, dim=-1, keepdim=True).squeeze(1)

            for l in shifts:
                u0_shifted = torch.roll(u0, shifts=l, dims=-1)

                if isinstance(model, (AdvectiveMemoryNCA, AdvectiveVanillaNCA)):
                    out_base = model(u0, K=2, true_A=true_A)[0]
                    out_shifted = model(u0_shifted, K=2, true_A=true_A)[0]
                else:
                    out_base = model(u0, K=2)[0]
                    out_shifted = model(u0_shifted, K=2)[0]

                expected = torch.roll(out_base, shifts=l, dims=-1)
                err = torch.norm(out_shifted - expected) / (torch.norm(out_shifted) + 1e-7)
                shift_errors.append(err.item())
            break

    return {
        "equiv_err_mean": float(np.mean(shift_errors)),
        "equiv_err_max": float(np.max(shift_errors)),
    }


def plot_figure_13(
    trained_results_df: pd.DataFrame,
    err_curves: Dict[str, List[float]],
    stage1_intervention_df: pd.DataFrame,
    long_horizon_dict: Dict[str, Dict[int, float]],
    heatmap_data: Dict[str, np.ndarray],
    equiv_results: Dict[str, float],
    save_path: Path,
) -> None:
    """Generate Figure 13: 6-panel publication diagnostic."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.25)

    colors = {
        "Eulerian Vanilla NCA (gamma=0)": "#2b5c8f",
        "Advective Vanilla NCA (gamma=0.2)": "#33a02c",
        "Advective Vanilla NCA (gamma=1/3, Peak-Matched)": "#e31a1c",
        "Advective Vanilla NCA (gamma=0.5)": "#ff7f00",
        "Advective Vanilla NCA (gamma=1.0, Char)": "#6a3d9a",
        "Oracle Coherent (v=2A_true)": "#b15928",
        "Learned Velocity (v=v_hat_theta)": "#1f78b4",
        "Advective Memory-NCA (Cm=16)": "#a6cee3",
        "Stationary Memory-NCA (Cm=16)": "#b2df8a",
    }

    # -------------------------------------------------------------
    # Panel A: Rollout Rel L2 Error vs Time
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    steps = np.arange(len(next(iter(err_curves.values()))))
    for name, curve in err_curves.items():
        c = colors.get(name, "#333333")
        ls = "--" if "Memory" in name else "-"
        lw = 2.4 if ("Char" in name or "Vanilla NCA" in name or "Learned" in name) else 1.6
        ax1.plot(steps, curve, label=name, color=c, linestyle=ls, linewidth=lw)

    ax1.set_xlabel("Macro Time Steps ($\Delta T = 0.1$)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Validation Relative $L_2$ Error", fontsize=10, fontweight="bold")
    ax1.set_title("(A) Autonomous Rollout Error Over Time", fontsize=11, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", fontsize=7.0, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel B: Stage 1 (Fixed-theta) vs Stage 2 (Trained)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    # Stage 1: Fixed-theta intervention (mean +/- std across 3 checkpoints)
    s1_g = stage1_intervention_df["gamma"].values
    s1_mean = stage1_intervention_df["mean_rel_l2"].values
    s1_std = stage1_intervention_df["std_rel_l2"].values
    ax2.plot(s1_g, s1_mean, label="Stage 1: Fixed-$\\theta^\\star_{\\gamma=1}$ Intervention", color="#6a3d9a", linestyle="--", marker="s", markersize=4)
    ax2.fill_between(s1_g, s1_mean - s1_std, s1_mean + s1_std, color="#6a3d9a", alpha=0.2)

    # Stage 2: Transport-conditioned training
    stage2_subset = trained_results_df[trained_results_df["model"].str.contains("Advective Vanilla|Eulerian Vanilla")].copy()
    stage2_subset = stage2_subset.sort_values("gamma")
    s2_g = stage2_subset["gamma"].values
    s2_mean = stage2_subset["rel_l2_mean"].values
    s2_std = stage2_subset["rel_l2_std"].values
    ax2.errorbar(s2_g, s2_mean, yerr=s2_std, label="Stage 2: Trained $\\theta_\\gamma^\\star = \\arg\\min_\\theta \\mathcal{L}(\\theta;\\gamma)$",
                 color="#e31a1c", marker="o", markersize=7, linewidth=2.2, capsize=4)

    ax2.axvline(1.0/3.0, color="#e31a1c", linestyle=":", alpha=0.7, label="Peak-matched scaling ($\gamma=1/3$)")
    ax2.axvline(1.0, color="#6a3d9a", linestyle=":", alpha=0.7, label="Characteristic ($\gamma=1.0$)")
    ax2.set_xlabel("Velocity Scaling Factor $\gamma$ in $v_\\gamma = 6\\gamma u$", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Validation Rollout Rel $L_2$", fontsize=10, fontweight="bold")
    ax2.set_title("(B) Transport Scale: Intervention vs. Optimization", fontsize=11, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", fontsize=7.5, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel C: Long-Horizon Error Growth (log E_L2 vs T)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    for name, t_dict in long_horizon_dict.items():
        t_vals = sorted(list(t_dict.keys()))
        err_vals = [t_dict[t] for t in t_vals]
        c = colors.get(name, "#333333")
        ls = "--" if "Memory" in name else "-"
        ax3.plot(t_vals, err_vals, label=name, color=c, linestyle=ls, marker="o", markersize=4, linewidth=1.8)

    ax3.set_yscale("log")
    ax3.set_xlabel("Horizon $T \in \{1, 5, 10, 25, 50, 100\}$", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Rollout Rel $L_2$ (Log Scale)", fontsize=10, fontweight="bold")
    ax3.set_title("(C) Long-Horizon Stability ($\log E_{L_2}$)", fontsize=11, fontweight="bold")
    ax3.legend(frameon=True, facecolor="white", fontsize=7.0, loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel D: Representation Transport Heatmaps
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    if "u_traj" in heatmap_data and "h_spatial" in heatmap_data:
        u_mat = heatmap_data["u_traj"]
        im4 = ax4.imshow(u_mat, aspect="auto", cmap="viridis", origin="lower", extent=[-25, 25, 0, u_mat.shape[0] * 0.1])
        fig.colorbar(im4, ax=ax4, label="$u(x, t)$")
    ax4.set_title("(D) Physical Wave Propagation $u(x, t)$", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Spatial Coordinate x", fontsize=10)
    ax4.set_ylabel("Physical Time t", fontsize=10)

    # -------------------------------------------------------------
    # Panel E: Translation Equivariance Diagnostic
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    eq_models = list(equiv_results.keys())
    eq_vals = [equiv_results[m] for m in eq_models]
    b_colors = [colors.get(m, "#2b5c8f") for m in eq_models]
    bars = ax5.bar(eq_models, eq_vals, color=b_colors, width=0.55, alpha=0.85)
    ax5.set_ylabel("Integer Shift Equivariance Error $E_u$", fontsize=10, fontweight="bold")
    ax5.set_title("(E) Spatial Translation Equivariance", fontsize=11, fontweight="bold")
    ax5.tick_params(axis="x", rotation=30)
    ax5.grid(True, linestyle="--", alpha=0.5)
    for b, v in zip(bars, eq_vals):
        ax5.annotate(f"{v:.4f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    # -------------------------------------------------------------
    # Panel F: Parameters / Neural MACs / Transport Ops Frontier
    # -------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    for _, row in trained_results_df.iterrows():
        m_name = row["model"]
        macs_m = row["neural_macs_per_delta_T"] / 1e6
        err = row["rel_l2_mean"]
        p_count = row["parameters"]
        t_ops = row["transport_ops_per_delta_T"]
        c = colors.get(m_name, "#333333")

        size = (p_count / 7765.0) * 120.0
        marker = "s" if "Memory" in m_name else "o"
        ax6.scatter(macs_m, err, s=size, color=c, alpha=0.85, edgecolors="black", linewidth=1.2, marker=marker, zorder=5)
        # Offset label with transport ops
        lbl = m_name.split("(")[0].strip()
        ax6.annotate(f"{lbl}\n(+{t_ops} ops)", (macs_m, err), textcoords="offset points", xytext=(5, 4), fontsize=7.0)

    ax6.set_xlabel("Neural Compute (Million MACs per $\Delta T$)", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Rollout Rel $L_2$ Error", fontsize=10, fontweight="bold")
    ax6.set_title("(F) Parameter / Neural MACs / Accuracy Frontier", fontsize=11, fontweight="bold")
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Advective Vanilla NCA: Decoupling Representation Geometry from Gating Overhead", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_advective_vanilla_experiment():
    output_dir = Path("outputs/default")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/home/zenoguy/.gemini/antigravity-ide/brain/36bdafb0-2d3a-4a8a-9c12-f55891eef59a/plots")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    seeds = [42, 123, 999]
    epochs = 25
    horizon = 12
    long_horizons = [1, 5, 10, 25, 50, 100]
    delta_T = 0.1
    K = 2

    print("=" * 80)
    print("STARTING ADVECTIVE VANILLA NCA EXPERIMENT & PROTOCOL FREEZING")
    print("  Testing Hypothesis: 'Memory architecture should match the geometry of information transport'")
    print("  Adv-Vanilla: Cm = 0, full 115 MLP width, strictly 7,765 parameters (Zero Gating Tax)")
    print(f"  Seeds: {seeds} | Epochs: {epochs} | Micro-step dt = {delta_T/K} | Long Horizons: {long_horizons}")
    print("=" * 80)

    kdv_solver = KdVSolver(N=128, Lx=50.0)

    # Pre-generate datasets by seed to prevent repeated expensive solver calls
    print("\n[Pre-generation] Generating train & val datasets for seeds [42, 123, 999]...")
    cached_datasets = {}
    cached_long_val = {}
    for s in seeds:
        cached_datasets[s] = build_experiment_datasets(kdv_solver, delta_T=delta_T, train_horizon=horizon, seed=s)
        long_ds = build_experiment_datasets(kdv_solver, delta_T=delta_T, train_horizon=100, seed=s)
        cached_long_val[s] = DataLoader(KdVTrajectoryDataset(long_ds["val"]["data"]), batch_size=4, shuffle=False)

    # -------------------------------------------------------------
    # 1. TRAINED ARCHITECTURE MATRIX (STAGE 2)
    # -------------------------------------------------------------
    print("\n[1/5] Training and evaluating Stage 2 Architectural Matrix across seeds...")
    model_configs = [
        ("Eulerian Vanilla NCA (gamma=0)", "vanilla", 0.0),
        ("Advective Vanilla NCA (gamma=0.2)", "scaled_characteristic", 0.2),
        ("Advective Vanilla NCA (gamma=1/3, Peak-Matched)", "peak_matched", 1.0 / 3.0),
        ("Advective Vanilla NCA (gamma=0.5)", "scaled_characteristic", 0.5),
        ("Advective Vanilla NCA (gamma=1.0, Char)", "characteristic", 1.0),
        ("Oracle Coherent (v=2A_true)", "oracle_true", 1.0),
        ("Learned Velocity (v=v_hat_theta)", "learned", 1.0),
        ("Advective Memory-NCA (Cm=16)", "adv_memory", 1.0),
        ("Stationary Memory-NCA (Cm=16)", "stat_memory", 0.0),
    ]

    trained_records = []
    err_curves = {}
    long_horizon_dict = {}
    saved_models_by_seed = {s: {} for s in seeds}
    sample_rollouts = {}

    for name, mode, gamma_val in model_configs:
        print(f"\n--- Model Configuration: {name} ---")
        seed_rel_l2 = []
        seed_final_l2 = []
        seed_amp_err = []
        seed_v_stats = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            ds = cached_datasets[seed]
            train_loader = DataLoader(KdVTrajectoryDataset(ds["train"]["data"]), batch_size=16, shuffle=True)
            val_loader = DataLoader(KdVTrajectoryDataset(ds["val"]["data"]), batch_size=8, shuffle=False)

            if mode == "vanilla":
                model = VanillaNCA(hidden_dim=16, mlp_hidden=115)
                p_total = count_parameters(model)
                p_trans = 0
                macs_dict = {"total_macs_per_delta_T": compute_nca_macs(model, N=128, K=K), "transport_arithmetic_ops_per_micro": 0}
            elif mode in ("scaled_characteristic", "peak_matched", "characteristic", "oracle_true", "learned"):
                model = AdvectiveVanillaNCA(hidden_dim=16, mlp_hidden=115, mode=mode, gamma=gamma_val)
                p_total = count_parameters(model)
                p_trans = count_parameters(model.velocity_net) if model.velocity_net is not None else 0
                macs_dict = compute_advective_vanilla_macs(model, N=128, K=K)
            elif mode == "adv_memory":
                mlp_h, p_total = find_matched_advective_mlp(target_params=7765, transport_dim=16, mode="characteristic")
                model = AdvectiveMemoryNCA(hidden_dim=16, memory_dim=16, transport_dim=16, mlp_hidden=mlp_h, mode="characteristic")
                p_trans = 0
                macs_dict = {"total_macs_per_delta_T": compute_advective_macs(model, N=128, K=K), "transport_arithmetic_ops_per_micro": 16 * 5 * 128}
            elif mode == "stat_memory":
                mlp_h, p_total = find_matched_advective_mlp(target_params=7765, transport_dim=0, mode="stationary")
                model = AdvectiveMemoryNCA(hidden_dim=16, memory_dim=16, transport_dim=0, mlp_hidden=mlp_h, mode="stationary")
                p_trans = 0
                macs_dict = {"total_macs_per_delta_T": compute_advective_macs(model, N=128, K=K), "transport_arithmetic_ops_per_micro": 0}

            train_model(model, train_loader, epochs=epochs, device=device, K=K, rollout_steps=8)
            res = evaluate_rollout(model, val_loader, device=device, K=K, rollout_steps=horizon)

            seed_rel_l2.append(res["mean_rel_l2"])
            seed_final_l2.append(res["final_rel_l2"])
            seed_amp_err.append(res["mean_amp_err"])
            seed_v_stats.append(res["velocity_stats"])
            saved_models_by_seed[seed][name] = model

            if seed == seeds[0]:
                err_curves[name] = res["err_curve"]
                sample_rollouts[name] = res

                # Evaluate long-horizon checkpoints: T in {1, 5, 10, 25, 50, 100}
                long_val_loader = cached_long_val[seed]
                res_lh = evaluate_rollout(model, long_val_loader, device=device, K=K, rollout_steps=100)
                long_horizon_dict[name] = {t: res_lh["err_curve"][t] for t in long_horizons if t < len(res_lh["err_curve"])}

        m_l2 = float(np.mean(seed_rel_l2))
        s_l2 = float(np.std(seed_rel_l2))
        f_l2 = float(np.mean(seed_final_l2))
        a_err = float(np.mean(seed_amp_err))
        mean_v = float(np.mean([vs["mean_abs_v"] for vs in seed_v_stats]))
        max_v = float(np.max([vs["max_abs_v"] for vs in seed_v_stats]))
        cfl_frac = float(np.mean([vs["frac_disp_gt_1"] for vs in seed_v_stats]))

        print(f"  Result: Rel L2 = {m_l2:.4f} +/- {s_l2:.4f} | Final = {f_l2:.4f} | Mean |v| = {mean_v:.2f} | CFL>1 = {cfl_frac:.1%}")

        trained_records.append({
            "model": name,
            "mode": mode,
            "gamma": gamma_val,
            "parameters": p_total,
            "transport_params": p_trans,
            "neural_macs_per_delta_T": macs_dict["total_macs_per_delta_T"],
            "transport_ops_per_delta_T": K * macs_dict.get("transport_arithmetic_ops_per_micro", 0),
            "rel_l2_mean": m_l2,
            "rel_l2_std": s_l2,
            "final_rel_l2_mean": f_l2,
            "amp_err_mean": a_err,
            "mean_abs_v": mean_v,
            "max_abs_v": max_v,
            "cfl_frac_gt_1": cfl_frac,
        })

    trained_df = pd.DataFrame(trained_records)
    csv_path = output_dir / "advective_vanilla_summary.csv"
    trained_df.to_csv(csv_path, index=False)
    print(f"\nSaved Stage 2 model summary to: {csv_path}")

    # -------------------------------------------------------------
    # 2. STAGE 1: MULTI-SEED FIXED-WEIGHT INTERVENTION SWEEP
    # -------------------------------------------------------------
    print("\n[2/5] Running Stage 1 Multi-Seed Fixed-Weight Intervention Sweep across gamma in [-1.0, 2.0]...")
    gammas_to_test = [-1.0, -0.5, 0.0, 0.1, 0.2, 1.0/3.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    stage1_rows = []

    for g in gammas_to_test:
        g_seed_errs = []
        for seed in seeds:
            char_model = saved_models_by_seed[seed].get("Advective Vanilla NCA (gamma=1.0, Char)")
            char_model.gamma = g
            val_loader = DataLoader(KdVTrajectoryDataset(cached_datasets[seed]["val"]["data"]), batch_size=8, shuffle=False)
            res_g = evaluate_rollout(char_model, val_loader, device=device, K=K, rollout_steps=horizon)
            g_seed_errs.append(res_g["mean_rel_l2"])
            char_model.gamma = 1.0  # Restore nominal gamma

        m_g = float(np.mean(g_seed_errs))
        s_g = float(np.std(g_seed_errs))
        stage1_rows.append({"gamma": g, "mean_rel_l2": m_g, "std_rel_l2": s_g})
        print(f"  gamma = {g:+6.3f} -> Rel L2 = {m_g:.4f} +/- {s_g:.4f}")

    stage1_df = pd.DataFrame(stage1_rows)
    gamma_csv_path = output_dir / "gamma_sweep_results.csv"
    stage1_df.to_csv(gamma_csv_path, index=False)
    print(f"Saved Stage 1 sweep results to: {gamma_csv_path}")

    # -------------------------------------------------------------
    # 3. TRANSLATION EQUIVARIANCE TEST (INTEGER SHIFT EXACT)
    # -------------------------------------------------------------
    print("\n[3/5] Evaluating Integer-Shift Translation Equivariance Diagnostics...")
    equiv_results = {}
    val_loader_eq = DataLoader(KdVTrajectoryDataset(cached_datasets[seeds[0]]["val"]["data"]), batch_size=8, shuffle=False)

    for m_name in [
        "Eulerian Vanilla NCA (gamma=0)",
        "Advective Vanilla NCA (gamma=1/3, Peak-Matched)",
        "Advective Vanilla NCA (gamma=1.0, Char)",
        "Oracle Coherent (v=2A_true)",
        "Learned Velocity (v=v_hat_theta)",
        "Advective Memory-NCA (Cm=16)",
    ]:
        m_obj = saved_models_by_seed[seeds[0]].get(m_name)
        if m_obj is not None:
            eq = evaluate_integer_shift_equivariance(m_obj, val_loader_eq, shifts=[1, 4, 16, 32], device=device)
            equiv_results[m_name] = eq["equiv_err_mean"]
            print(f"  {m_name:45s}: Mean Equiv Error = {eq['equiv_err_mean']:.5f} (Max = {eq['equiv_err_max']:.5f})")

    # -------------------------------------------------------------
    # 4. PROVENANCE REPRODUCTION AUDIT: 0.3287 vs 0.3649
    # -------------------------------------------------------------
    print("\n[4/5] Executing Provenance Audit for 0.3287 vs 0.3649...")
    # Config A: 100% Transport Characteristic Memory (trans_dim=16, loc_dim=0)
    mlp_h_16, _ = find_matched_advective_mlp(target_params=7765, transport_dim=16, mode="characteristic")
    model_16 = AdvectiveMemoryNCA(hidden_dim=16, memory_dim=16, transport_dim=16, mlp_hidden=mlp_h_16, mode="characteristic")
    train_model(model_16, DataLoader(KdVTrajectoryDataset(cached_datasets[42]["train"]["data"]), batch_size=16, shuffle=True), epochs=25, device=device)
    res_16 = evaluate_rollout(model_16, DataLoader(KdVTrajectoryDataset(cached_datasets[42]["val"]["data"]), batch_size=8, shuffle=False), device=device)

    # Config B: 50% Transport Learned Memory (trans_dim=8, loc_dim=8)
    mlp_h_8, _ = find_matched_advective_mlp(target_params=7765, transport_dim=8, mode="learned")
    model_8 = AdvectiveMemoryNCA(hidden_dim=16, memory_dim=16, transport_dim=8, mlp_hidden=mlp_h_8, mode="learned")
    train_model(model_8, DataLoader(KdVTrajectoryDataset(cached_datasets[42]["train"]["data"]), batch_size=16, shuffle=True), epochs=25, device=device)
    res_8 = evaluate_rollout(model_8, DataLoader(KdVTrajectoryDataset(cached_datasets[42]["val"]["data"]), batch_size=8, shuffle=False), device=device)

    print(f"  Audit Result Config A (100% Transport Characteristic): Rel L2 = {res_16['mean_rel_l2']:.4f}")
    print(f"  Audit Result Config B (50% Transport Learned):          Rel L2 = {res_8['mean_rel_l2']:.4f}")
    provenance_doc = {
        "config_A_100pct_transport_characteristic": {
            "description": "AdvectiveMemoryNCA with Cm=16, transport_dim=16 (all-transported), mode='characteristic'",
            "reproduced_rel_l2": res_16["mean_rel_l2"],
            "original_reported_value": 0.3287,
        },
        "config_B_50pct_transport_learned": {
            "description": "AdvectiveMemoryNCA with Cm=16, transport_dim=8 (dual 50/50 partition), mode='learned'",
            "reproduced_rel_l2": res_8["mean_rel_l2"],
            "original_reported_value": 0.3649,
        },
    }
    with open(output_dir / "provenance_audit.json", "w") as f:
        json.dump(provenance_doc, f, indent=2)

    # -------------------------------------------------------------
    # 5. GENERATE FIGURE 13 & SAVE ARTIFACTS
    # -------------------------------------------------------------
    print("\n[5/5] Generating Figure 13 (Publication-Grade Multi-Panel Diagnostic)...")
    fig13_path = plots_dir / "fig13_advective_vanilla_gamma_sweep.png"

    heatmap_data = {}
    if "Advective Vanilla NCA (gamma=1.0, Char)" in sample_rollouts:
        heatmap_data["u_traj"] = np.array(sample_rollouts["Advective Vanilla NCA (gamma=1.0, Char)"]["sample_trues"])
        heatmap_data["h_spatial"] = sample_rollouts["Advective Vanilla NCA (gamma=1.0, Char)"]["sample_h"]

    plot_figure_13(
        trained_results_df=trained_df,
        err_curves=err_curves,
        stage1_intervention_df=stage1_df,
        long_horizon_dict=long_horizon_dict,
        heatmap_data=heatmap_data,
        equiv_results=equiv_results,
        save_path=fig13_path,
    )
    shutil.copy(fig13_path, artifact_dir / "fig13_advective_vanilla_gamma_sweep.png")
    print(f"Generated Figure 13 at: {fig13_path}")
    print(f"Copied to artifact dir at: {artifact_dir / 'fig13_advective_vanilla_gamma_sweep.png'}")

    print("\n" + "=" * 80)
    print("ALL ADVECTIVE VANILLA EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(trained_df[["model", "gamma", "parameters", "neural_macs_per_delta_T", "transport_ops_per_delta_T", "rel_l2_mean", "rel_l2_std", "final_rel_l2_mean"]].to_string())
    print("=" * 80)


if __name__ == "__main__":
    run_advective_vanilla_experiment()
