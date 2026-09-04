"""
Publication-Quality Scientific Visualizations for KdV NCA Experiments.

Styles follow academic standards: clean layouts, clear legends,
labeled axes, and robust color schemes (viridis / inferno / coolwarm).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_plot_style():
    """Apply clean, publication-ready plot settings."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "lines.linewidth": 1.8,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.grid": True,
        }
    )


def plot_solver_validation(
    t_eval: np.ndarray,
    x: np.ndarray,
    traj_num: np.ndarray,
    traj_exact: np.ndarray,
    invariants: Dict[str, np.ndarray],
    save_path: str,
):
    """
    Figure 1: Comprehensive Numerical Solver Validation.
    Includes:
    1. Spatial profiles at multiple times (initial, mid, final).
    2. Space-time heatmap u(x, t).
    3. Peak amplitude vs time (numerical vs exact).
    4. Soliton center position vs time.
    5. Invariant drifts (I1, I2, I3).
    6. Relative L2 error over time.
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Figure 1: Numerical Ground Truth (ETDRK4) KdV Soliton Validation", y=0.98)

    # 1. Profiles at sample timesteps
    ax = axes[0, 0]
    sample_indices = np.linspace(0, len(t_eval) - 1, 5, dtype=int)
    for idx in sample_indices:
        t_val = t_eval[idx]
        ax.plot(x, traj_num[idx], label=f"t = {t_val:.1f}")
    ax.set_title("Soliton Waveform Evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend(loc="upper right", framealpha=0.9)

    # 2. Space-time heatmap
    ax = axes[0, 1]
    extent = [x[0], x[-1], t_eval[-1], t_eval[0]]
    im = ax.imshow(
        traj_num, aspect="auto", extent=extent, cmap="viridis", interpolation="nearest"
    )
    fig.colorbar(im, ax=ax, label="u(x, t)", pad=0.02)
    ax.set_title("Space-Time Heatmap u(x, t)")
    ax.set_xlabel("x")
    ax.set_ylabel("Physical Time t")

    # 3. Peak amplitude vs time
    ax = axes[0, 2]
    num_peaks = np.max(traj_num, axis=1)
    exact_peaks = np.max(traj_exact, axis=1)
    ax.plot(t_eval, exact_peaks, "k--", label="Exact Analytical")
    ax.plot(t_eval, num_peaks, "b-", alpha=0.8, label="Numerical ETDRK4")
    ax.set_title("Soliton Peak Amplitude vs Time")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Peak Amplitude")
    ax.legend()

    # 4. Soliton center vs time
    ax = axes[1, 0]
    num_centers = [float(x[np.argmax(u)]) for u in traj_num]
    exact_centers = [float(x[np.argmax(u)]) for u in traj_exact]
    ax.plot(t_eval, exact_centers, "k--", label="Analytical Center")
    ax.plot(t_eval, num_centers, "ro", markersize=3, label="Numerical Peak")
    ax.set_title("Soliton Trajectory (Center vs Time)")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Spatial Position x")
    ax.legend()

    # 5. Invariant drifts
    ax = axes[1, 1]
    ax.plot(t_eval, invariants["drift_I1"], label=r"Zeroth-order $I_1$", color="navy")
    ax.plot(t_eval, invariants["drift_I2"], label=r"Quadratic $I_2$", color="teal")
    ax.plot(t_eval, invariants["drift_I3"], label=r"Hamiltonian $I_3$", color="crimson")
    ax.set_yscale("log")
    ax.set_title("Conserved Invariant Relative Drift")
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"$|\Delta I_k(t)| / |I_k(0)|$")
    ax.legend()

    # 6. Relative L2 error vs time
    ax = axes[1, 2]
    rel_l2 = [
        np.linalg.norm(traj_num[i] - traj_exact[i]) / np.linalg.norm(traj_exact[i])
        for i in range(len(t_eval))
    ]
    ax.plot(t_eval, rel_l2, color="purple", label=r"Relative $L_2$ Error")
    ax.set_yscale("log")
    ax.set_title(r"Global $L_2$ Discretization Error")
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"$E_{L2}(t)$")
    ax.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_rollout_comparison(
    t_eval: np.ndarray,
    x: np.ndarray,
    models_rollouts: Dict[str, np.ndarray],
    save_path: str,
):
    """
    Figure 2: Multi-Model Autonomous Rollout Comparison.
    Rows correspond to models (Ground Truth, Vanilla NCA, Matched NCA, Memory-NCA, etc.).
    Columns correspond to advancing time snapshots.
    """
    set_plot_style()
    n_models = len(models_rollouts)
    sample_indices = np.linspace(0, len(t_eval) - 1, 5, dtype=int)
    n_times = len(sample_indices)

    fig, axes = plt.subplots(
        n_models, n_times, figsize=(4 * n_times, 2.5 * n_models), sharex=True, sharey=True
    )
    fig.suptitle("Figure 2: Autonomous Multi-Step Rollout Snapshots", y=0.99)

    model_names = list(models_rollouts.keys())
    for r, name in enumerate(model_names):
        traj = models_rollouts[name]
        for c, t_idx in enumerate(sample_indices):
            ax = axes[r, c] if n_models > 1 else axes[c]
            ax.plot(x, traj[t_idx], color="tab:blue" if r == 0 else "tab:red")
            if r == 0:
                ax.set_title(f"t = {t_eval[t_idx]:.1f}")
            if c == 0:
                ax.set_ylabel(name)
            if r == n_models - 1:
                ax.set_xlabel("x")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_error_vs_time(
    t_eval: np.ndarray,
    model_errors: Dict[str, np.ndarray],
    save_path: str,
):
    """
    Figure 3: Relative L2 Error vs Time across models.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Figure 3: Autonomous Rollout Error Accumulation Over Time", y=0.96)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for i, (name, err) in enumerate(model_errors.items()):
        color = colors[i % len(colors)]
        ax.plot(t_eval, err, label=name, color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Physical Time t")
    ax.set_ylabel(r"Relative $L_2$ Error $E_{L2}(t)$")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_memory_ablation(
    memory_dims: List[int],
    rollout_errors: List[float],
    long_horizon_stabilities: List[float],
    param_counts: List[int],
    save_path: str,
):
    """
    Figure 4: Memory Size Ablation Study.
    Memory dimension vs Rollout Error and Long-Horizon Stability.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Figure 4: Impact of Memory Dimension $C_m$ on NCA Dynamics", y=0.98)

    # 1. Memory dim vs Error
    ax1.plot(memory_dims, rollout_errors, "o-", color="crimson", markersize=6)
    ax1.set_xlabel(r"Memory Channels $C_m$")
    ax1.set_ylabel(r"Mean Rollout Relative $L_2$ Error")
    ax1.set_title("Accuracy vs Memory Dimension")
    ax1.set_xticks(memory_dims)

    # 2. Memory dim vs Stability
    ax2.plot(memory_dims, long_horizon_stabilities, "s-", color="teal", markersize=6)
    ax2.set_xlabel(r"Memory Channels $C_m$")
    ax2.set_ylabel("Long-Horizon Stability Metric")
    ax2.set_title("Long-Horizon Coherence vs Memory Dimension")
    ax2.set_xticks(memory_dims)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_soliton_diagnostics(
    t_eval: np.ndarray,
    diagnostics: Dict[str, Dict[str, np.ndarray]],
    save_path: str,
):
    """
    Figure 6: Soliton Physical Diagnostics Over Time.
    Compares Amplitude, Center Position, and Width across models against Ground Truth.
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 6: Soliton Physical Waveform Diagnostics", y=0.98)

    for name, diag in diagnostics.items():
        ls = "-" if name != "Ground Truth" else "--"
        lw = 2.0 if name == "Ground Truth" else 1.6
        axes[0].plot(t_eval, diag["amplitude"], label=name, linestyle=ls, linewidth=lw)
        axes[1].plot(t_eval, diag["center"], label=name, linestyle=ls, linewidth=lw)
        axes[2].plot(t_eval, diag["width"], label=name, linestyle=ls, linewidth=lw)

    axes[0].set_title("Peak Amplitude vs Time")
    axes[0].set_xlabel("Time t")
    axes[0].set_ylabel("Amplitude A(t)")
    axes[0].legend()

    axes[1].set_title("Centroid Position vs Time")
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Position x(t)")

    axes[2].set_title("Pulse Width (FWHM) vs Time")
    axes[2].set_xlabel("Time t")
    axes[2].set_ylabel("Width L(t)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_cpu_scaling(
    threads_list: List[int],
    grid_sizes: List[int],
    inference_throughput: Dict[int, List[float]],
    whole_sim_throughput: Dict[int, List[float]],
    efficiency: Dict[int, List[float]],
    save_path: str,
):
    """
    Figure 7: Multicore CPU Scaling and Parallel Efficiency on Ryzen 5 5600H.
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 7: Multicore CPU Scaling and Efficiency Analysis", y=0.98)

    # 1. Inference throughput
    for N in grid_sizes:
        axes[0].plot(threads_list, inference_throughput[N], "o-", label=f"N={N}")
    axes[0].set_title("Model Inference Throughput")
    axes[0].set_xlabel("CPU Threads (p)")
    axes[0].set_ylabel("Macro-Steps / Sec")
    axes[0].legend()

    # 2. Whole simulation throughput
    for N in grid_sizes:
        axes[1].plot(threads_list, whole_sim_throughput[N], "s-", label=f"N={N}")
    axes[1].set_title("Whole Simulation Throughput")
    axes[1].set_xlabel("CPU Threads (p)")
    axes[1].set_ylabel("Rollouts / Sec")
    axes[1].legend()

    # 3. Parallel efficiency
    for N in grid_sizes:
        axes[2].plot(threads_list, efficiency[N], "^-", label=f"N={N}")
    axes[2].axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal (100%)")
    axes[2].set_title(r"Parallel Scaling Efficiency $T_1 / (p \cdot T_p)$")
    axes[2].set_xlabel("CPU Threads (p)")
    axes[2].set_ylabel("Efficiency")
    axes[2].set_ylim(0, 1.2)
    axes[2].legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_generalization_tests(
    test_names: List[str],
    results_by_model: Dict[str, List[float]],
    save_path: str,
):
    """
    Figure 5: Generalization Breakdown across distinct test regimes:
    Interpolation, Extrapolation, Test A (unseen params), Test B (pulses), Test C (two-pulse collision).
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Figure 5: Physical Generalization Breakdown Across Regimes", y=0.98)

    n_tests = len(test_names)
    n_models = len(results_by_model)
    x = np.arange(n_tests)
    width = 0.8 / max(1, n_models)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for i, (m_name, errs) in enumerate(results_by_model.items()):
        offset = (i - n_models / 2.0 + 0.5) * width
        ax.bar(x + offset, errs, width, label=m_name, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(test_names)
    ax.set_ylabel(r"Autonomous Rollout Relative $L_2$ Error")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_memory_swapping(
    t_eval: np.ndarray,
    x: np.ndarray,
    swap_results: Dict[str, np.ndarray],
    save_path: str,
):
    """
    Figure 8: Causal Memory Swapping Diagnostic.
    Compares wave evolution under:
    1. u + mA (Regime A)
    2. u + mB (Regime B)
    3. u + m_rand (Random memory control)
    4. u + m_zero (Zero memory control)
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle("Figure 8: Causal Memory Swapping Test with Controls", y=0.98)

    conditions = [
        ("traj_mA", r"Regime A Memory $u(t^*) + m_A$", axes[0, 0]),
        ("traj_mB", r"Regime B Memory $u(t^*) + m_B$", axes[0, 1]),
        ("traj_m_rand", r"Random Control $u(t^*) + m_{\mathrm{rand}}$", axes[1, 0]),
        ("traj_m_zero", r"Zero Control $u(t^*) + m_{\mathrm{zero}}$", axes[1, 1]),
    ]

    sample_indices = np.linspace(0, len(t_eval) - 1, 4, dtype=int)

    for key, title, ax in conditions:
        traj = swap_results[key]
        for idx in sample_indices:
            ax.plot(x, traj[idx], label=f"t = {t_eval[idx]:.1f}")
        ax.set_title(title)
        ax.set_ylabel("u(x, t)")
        ax.legend(loc="upper right")

    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

