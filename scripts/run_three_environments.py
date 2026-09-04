"""
Three Controlled Environments Benchmark:
Evaluating Persistent Memory vs. Parameter-Matched Vanilla NCA across:
  Environment A: Fully Observed KdV (Markovian Baseline)
  Environment B: Partially Observed KdV (Sparse Probes, P=16/128)
  Environment C: Non-Markovian Coupled KdV (Mori-Zwanzig Latent Field)

Quantifies:
  Memory Advantage (%) = (E_Vanilla - E_Memory) / E_Vanilla * 100%
as a function of the Degree of Non-Markovianity / Latent Information.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_experiment_datasets, KdVTrajectoryDataset
from src.kdv_solver import KdVSolver
from src.memory_nca import MemoryNCA
from src.metrics import relative_l2_error, peak_amplitude_error
from src.nca import VanillaNCA
from src.non_markovian_solver import CoupledNonMarkovianKdVSolver, build_non_markovian_datasets
from src.sparse_dataset import SparseProbeDataset
from src.train import normalized_mse_loss, train_model, validate


def run_training_env_a(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train on Environment A (Fully Observed Markovian KdV)."""
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.002,
        K=2,
        rollout_steps=12,
        device=device,
        verbose=False,
    )
    val_loss = validate(model, val_loader, device=device, K=2, rollout_steps=12)
    return float(val_loss)


def run_training_env_b(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    device: torch.device = torch.device("cpu"),
    is_memory: bool = False,
) -> float:
    """Train on Environment B (Sparse Probes Assimilation)."""
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    probe_idx = train_loader.dataset.probe_indices

    model.to(device)
    for epoch in range(epochs):
        model.train()
        for sparse_traj, full_traj in train_loader:
            sparse_traj, full_traj = sparse_traj.to(device), full_traj.to(device)
            B, T, _, N = full_traj.shape
            opt.zero_grad()
            loss = 0.0

            s = torch.zeros(B, model.total_channels, N, device=device)
            s[:, 0, probe_idx] = sparse_traj[:, 0, 0, probe_idx]

            if is_memory:
                m = model.init_memory(B, N, device, sparse_traj.dtype)

            for t in range(1, 13):
                target = full_traj[:, t]
                s[:, 0, probe_idx] = sparse_traj[:, t, 0, probe_idx]

                for _ in range(2):  # K=2
                    if is_memory:
                        s, m = model.step(s, m)
                    else:
                        s = model.step(s)

                u_pred = s[:, :1]
                loss = loss + normalized_mse_loss(u_pred, target)

            loss = loss / 12
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    # Validation evaluation
    model.eval()
    val_errors = []
    with torch.no_grad():
        for sparse_traj, full_traj in val_loader:
            sparse_traj, full_traj = sparse_traj.to(device), full_traj.to(device)
            B, T, _, N = full_traj.shape
            s = torch.zeros(B, model.total_channels, N, device=device)
            s[:, 0, probe_idx] = sparse_traj[:, 0, 0, probe_idx]

            if is_memory:
                m = model.init_memory(B, N, device, sparse_traj.dtype)

            for t in range(1, 13):
                target = full_traj[:, t]
                s[:, 0, probe_idx] = sparse_traj[:, t, 0, probe_idx]
                for _ in range(2):
                    if is_memory:
                        s, m = model.step(s, m)
                    else:
                        s = model.step(s)
                u_pred = s[:, :1]
                for b in range(B):
                    err = relative_l2_error(
                        u_pred[b, 0].cpu().numpy(), target[b, 0].cpu().numpy()
                    )
                    val_errors.append(err)

    return float(np.mean(val_errors))


def run_training_env_c(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train on Environment C (Coupled Non-Markovian KdV)."""
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.002,
        K=2,
        rollout_steps=12,
        device=device,
        verbose=False,
    )
    val_loss = validate(model, val_loader, device=device, K=2, rollout_steps=12)
    return float(val_loss)


def plot_figure_10(
    results_df: pd.DataFrame,
    sample_rollouts: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
) -> None:
    """Generate Figure 10 showing 3-Environment Comparison and Memory Advantage."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.25)

    # 1. Bar Chart: Relative L2 Error by Environment
    ax_bar = fig.add_subplot(gs[0, 0:2])
    envs = ["Env A\n(Fully Observed)", "Env B\n(Sparse Probes)", "Env C\n(Coupled Memory)"]
    x = np.arange(len(envs))
    width = 0.32

    v_means = results_df[results_df["model"] == "Vanilla NCA"]["rel_l2_mean"].values
    v_stds = results_df[results_df["model"] == "Vanilla NCA"]["rel_l2_std"].values
    m_means = results_df[results_df["model"] == "Memory-NCA"]["rel_l2_mean"].values
    m_stds = results_df[results_df["model"] == "Memory-NCA"]["rel_l2_std"].values

    rects1 = ax_bar.bar(
        x - width / 2, v_means, width, yerr=v_stds, label="Vanilla NCA (7,765 p)", color="#2b5c8f", capsize=4, alpha=0.9
    )
    rects2 = ax_bar.bar(
        x + width / 2, m_means, width, yerr=m_stds, label="Memory-NCA (7,769 p)", color="#d95f02", capsize=4, alpha=0.9
    )

    ax_bar.set_ylabel("Validation Rollout Rel $L_2$ Error", fontsize=11, fontweight="bold")
    ax_bar.set_title("(A) Validation Error Across Three Controlled Physical Environments", fontsize=12, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(envs, fontsize=10, fontweight="bold")
    ax_bar.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax_bar.grid(True, linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax_bar.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax_bar.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # 2. Memory Advantage Curve
    ax_adv = fig.add_subplot(gs[0, 2])
    adv_values = []
    for vm, mm in zip(v_means, m_means):
        adv = (vm - mm) / vm * 100.0
        adv_values.append(adv)

    colors = ["#d95f02" if a > 0 else "#7570b3" for a in adv_values]
    bars = ax_adv.bar(envs, adv_values, color=colors, width=0.45, alpha=0.85)
    ax_adv.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax_adv.set_ylabel("Memory Advantage (%)", fontsize=11, fontweight="bold")
    ax_adv.set_title("(B) Memory Advantage vs. Non-Markovianity", fontsize=12, fontweight="bold")
    ax_adv.grid(True, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, adv_values):
        offset = 2 if val >= 0 else -6
        ax_adv.annotate(f"{val:+.1f}%", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, offset),
                        textcoords="offset points", ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=10, fontweight="bold")

    # 3. Sample Rollout Panels for each Environment
    panel_titles = [
        "(C) Env A: Fully Observed KdV (t=1.2)",
        "(D) Env B: Sparse Probes Reconstruction (t=1.2)",
        "(E) Env C: Non-Markovian Mori-Zwanzig (t=1.2)",
    ]
    env_keys = ["env_a", "env_b", "env_c"]

    for idx, (key, title) in enumerate(zip(env_keys, panel_titles)):
        ax_sample = fig.add_subplot(gs[1, idx])
        data = sample_rollouts[key]
        x_grid = np.linspace(-25, 25, 128)

        ax_sample.plot(x_grid, data["true"], label="Ground Truth", color="black", linewidth=2.0)
        ax_sample.plot(x_grid, data["vanilla"], label="Vanilla NCA", color="#2b5c8f", linestyle="--", linewidth=1.8)
        ax_sample.plot(x_grid, data["memory"], label="Memory-NCA", color="#d95f02", linestyle=":", linewidth=2.0)

        if key == "env_b" and "probes" in data:
            ax_sample.scatter(x_grid[data["probes"]], data["true"][data["probes"]], color="red", s=25, zorder=5, label="Sparse Probes (16)")

        ax_sample.set_title(title, fontsize=11, fontweight="bold")
        ax_sample.set_xlabel("Spatial coordinate x", fontsize=10)
        if idx == 0:
            ax_sample.set_ylabel("Wave amplitude u(x, t)", fontsize=10)
        ax_sample.legend(frameon=True, facecolor="white", fontsize=8, loc="upper right")
        ax_sample.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Benchmark across Three Controlled Environments: Testing Non-Markovian Memory Advantage", fontsize=14, fontweight="bold", y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_benchmark():
    output_dir = Path("outputs/default")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    seeds = [42, 123]
    epochs = 25
    horizon = 12
    delta_T = 0.1

    print("=" * 75)
    print("RUNNING THREE CONTROLLED ENVIRONMENTS BENCHMARK")
    print("  Env A: Fully Observed KdV (Markovian Baseline)")
    print("  Env B: Partially Observed KdV (16 Sparse Probes)")
    print("  Env C: Coupled Non-Markovian KdV (Mori-Zwanzig Latent Field)")
    print(f"  Seeds: {seeds} | Epochs: {epochs} | Horizon: {horizon} | Delta T: {delta_T}")
    print("=" * 75)

    results = []
    sample_rollouts = {}

    # Initialize Solvers and Datasets
    print("\n[1/3] Generating datasets for all environments...")
    kdv_solver = KdVSolver(N=128, Lx=50.0)
    nm_solver = CoupledNonMarkovianKdVSolver(N=128, Lx=50.0, dt=0.005, lambda_rel=1.0, kappa=1.5)

    # --- ENVIRONMENT A ---
    print("\n[2/3] Benchmarking Environment A (Fully Observed KdV)...")
    env_a_v_errs, env_a_m_errs = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds_a = build_experiment_datasets(kdv_solver, delta_T=delta_T, train_horizon=horizon, seed=seed)
        train_loader = DataLoader(KdVTrajectoryDataset(ds_a["train"]["data"]), batch_size=16, shuffle=True)
        val_loader = DataLoader(KdVTrajectoryDataset(ds_a["val"]["data"]), batch_size=8, shuffle=False)

        v_model = VanillaNCA(hidden_dim=24, mlp_hidden=88)
        m_model = MemoryNCA(hidden_dim=16, memory_dim=16, mlp_hidden=64)

        err_v = run_training_env_a(v_model, train_loader, val_loader, epochs=epochs, device=device)
        err_m = run_training_env_a(m_model, train_loader, val_loader, epochs=epochs, device=device)
        env_a_v_errs.append(err_v)
        env_a_m_errs.append(err_m)

        if seed == seeds[0]:
            # Save sample rollouts
            u0 = torch.tensor(ds_a["val"]["data"][0:1, 0]).float().unsqueeze(1)
            pred_v = v_model.rollout(u0, num_macro_steps=horizon, K=2)
            pred_m, _ = m_model.rollout(u0, num_macro_steps=horizon, K=2)
            sample_rollouts["env_a"] = {
                "true": ds_a["val"]["data"][0, horizon],
                "vanilla": pred_v[0, horizon, 0].detach().cpu().numpy(),
                "memory": pred_m[0, horizon, 0].detach().cpu().numpy(),
            }

    results.append({"env": "Env A (Fully Observed)", "model": "Vanilla NCA", "rel_l2_mean": float(np.mean(env_a_v_errs)), "rel_l2_std": float(np.std(env_a_v_errs))})
    results.append({"env": "Env A (Fully Observed)", "model": "Memory-NCA", "rel_l2_mean": float(np.mean(env_a_m_errs)), "rel_l2_std": float(np.std(env_a_m_errs))})

    # --- ENVIRONMENT B ---
    print("\n[3/3] Benchmarking Environment B (Partially Observed KdV - Sparse Probes)...")
    env_b_v_errs, env_b_m_errs = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds_b = build_experiment_datasets(kdv_solver, delta_T=delta_T, train_horizon=horizon, seed=seed)
        sp_train = SparseProbeDataset(ds_b["train"]["data"], num_probes=16)
        sp_val = SparseProbeDataset(ds_b["val"]["data"], num_probes=16)
        train_loader = DataLoader(sp_train, batch_size=16, shuffle=True)
        val_loader = DataLoader(sp_val, batch_size=8, shuffle=False)

        v_model = VanillaNCA(hidden_dim=24, mlp_hidden=88)
        m_model = MemoryNCA(hidden_dim=16, memory_dim=16, mlp_hidden=64)

        err_v = run_training_env_b(v_model, train_loader, val_loader, epochs=epochs, device=device, is_memory=False)
        err_m = run_training_env_b(m_model, train_loader, val_loader, epochs=epochs, device=device, is_memory=True)
        env_b_v_errs.append(err_v)
        env_b_m_errs.append(err_m)

        if seed == seeds[0]:
            # Sample reconstruction
            sparse_sample, full_sample = sp_val[0]
            probe_idx = sp_val.probe_indices
            B = 1
            s_v = torch.zeros(B, v_model.total_channels, 128)
            s_m = torch.zeros(B, m_model.total_channels, 128)
            s_v[:, 0, probe_idx] = sparse_sample[0, 0, probe_idx]
            s_m[:, 0, probe_idx] = sparse_sample[0, 0, probe_idx]
            m_state = m_model.init_memory(B, 128, device, torch.float32)

            for t in range(1, horizon + 1):
                s_v[:, 0, probe_idx] = sparse_sample[t, 0, probe_idx]
                s_m[:, 0, probe_idx] = sparse_sample[t, 0, probe_idx]
                for _ in range(2):
                    s_v = v_model.step(s_v)
                    s_m, m_state = m_model.step(s_m, m_state)

            sample_rollouts["env_b"] = {
                "true": full_sample[horizon, 0].numpy(),
                "vanilla": s_v[0, 0].detach().cpu().numpy(),
                "memory": s_m[0, 0].detach().cpu().numpy(),
                "probes": probe_idx,
            }

    results.append({"env": "Env B (Sparse Probes)", "model": "Vanilla NCA", "rel_l2_mean": float(np.mean(env_b_v_errs)), "rel_l2_std": float(np.std(env_b_v_errs))})
    results.append({"env": "Env B (Sparse Probes)", "model": "Memory-NCA", "rel_l2_mean": float(np.mean(env_b_m_errs)), "rel_l2_std": float(np.std(env_b_m_errs))})

    # --- ENVIRONMENT C ---
    print("\n[4/4] Benchmarking Environment C (Non-Markovian Coupled KdV)...")
    env_c_v_errs, env_c_m_errs = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds_c = build_non_markovian_datasets(nm_solver, delta_T=delta_T, train_horizon=horizon, seed=seed)
        train_loader = DataLoader(KdVTrajectoryDataset(ds_c["train"]["data"]), batch_size=16, shuffle=True)
        val_loader = DataLoader(KdVTrajectoryDataset(ds_c["val"]["data"]), batch_size=8, shuffle=False)

        v_model = VanillaNCA(hidden_dim=24, mlp_hidden=88)
        m_model = MemoryNCA(hidden_dim=16, memory_dim=16, mlp_hidden=64)

        err_v = run_training_env_c(v_model, train_loader, val_loader, epochs=epochs, device=device)
        err_m = run_training_env_c(m_model, train_loader, val_loader, epochs=epochs, device=device)
        env_c_v_errs.append(err_v)
        env_c_m_errs.append(err_m)

        if seed == seeds[0]:
            u0 = torch.tensor(ds_c["val"]["data"][0:1, 0]).float().unsqueeze(1)
            pred_v = v_model.rollout(u0, num_macro_steps=horizon, K=2)
            pred_m, _ = m_model.rollout(u0, num_macro_steps=horizon, K=2)
            sample_rollouts["env_c"] = {
                "true": ds_c["val"]["data"][0, horizon],
                "vanilla": pred_v[0, horizon, 0].detach().cpu().numpy(),
                "memory": pred_m[0, horizon, 0].detach().cpu().numpy(),
            }

    results.append({"env": "Env C (Coupled Memory)", "model": "Vanilla NCA", "rel_l2_mean": float(np.mean(env_c_v_errs)), "rel_l2_std": float(np.std(env_c_v_errs))})
    results.append({"env": "Env C (Coupled Memory)", "model": "Memory-NCA", "rel_l2_mean": float(np.mean(env_c_m_errs)), "rel_l2_std": float(np.std(env_c_m_errs))})

    results_df = pd.DataFrame(results)
    csv_path = output_dir / "three_environments_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV to: {csv_path}")
    print(results_df.to_string())

    # Generate Figure 10
    fig10_path = plots_dir / "fig10_three_environments.png"
    plot_figure_10(results_df, sample_rollouts, fig10_path)
    print(f"Generated Figure 10 at: {fig10_path}")

    # Copy to artifacts directory
    artifact_dir = Path("/home/zenoguy/.gemini/antigravity-ide/brain/36bdafb0-2d3a-4a8a-9c12-f55891eef59a/plots")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig10_path, artifact_dir / "fig10_three_environments.png")
    print(f"Copied Figure 10 to artifact directory: {artifact_dir / 'fig10_three_environments.png'}")

    print("\nBENCHMARK COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_benchmark()
