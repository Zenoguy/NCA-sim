"""
Experiment 8: Micro-Steps (K) Sensitivity Study at Fixed Physical Delta T.

Enforces:
    Delta T = constant (0.1 physical time) while K in {1, 2, 4, 8} varies.
Measures:
- Error per Delta T
- Compute (MACs and latency) per Delta T
- Identifies Pareto-optimal computational balance.
"""

import argparse
import time
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.dataset import load_datasets
from src.nca import VanillaNCA, find_matched_vanilla_channels, compute_nca_macs, count_parameters
from src.memory_nca import MemoryNCA
from src.train import train_model
from src.evaluate import evaluate_autonomous_rollout


def main():
    parser = argparse.ArgumentParser(description="K-Sensitivity Study")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["physics"]
    t_cfg = cfg["time_discretization"]
    d_cfg = cfg["dataset"]
    m_cfg = cfg["models"]
    tr_cfg = cfg["training"]
    out_dir = Path(cfg["paths"]["output_dir"])
    plots_dir = out_dir / "plots"

    datasets = load_datasets(d_cfg["data_dir"])
    train_loader = DataLoader(datasets["train"], batch_size=tr_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=tr_cfg["batch_size"], shuffle=False)

    N = p_cfg["N"]
    Lx = p_cfg["Lx"]
    dx = Lx / N
    x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)

    k_list = args.k_values
    records = []

    print(f"=== Running K-Sensitivity at Constant Delta T = {t_cfg['delta_T']} ===")

    for K_val in k_list:
        print(f"\nEvaluating K = {K_val} micro-steps per Delta T...")

        # Memory-NCA
        torch.manual_seed(42)
        np.random.seed(42)
        mem_model = MemoryNCA(
            hidden_dim=m_cfg["hidden_dim"],
            memory_dim=m_cfg["memory_dim"],
            kernel_size=m_cfg["kernel_size"],
            mlp_hidden=m_cfg["mlp_hidden"],
        )
        mem_macs = compute_nca_macs(mem_model, N=N, K=K_val)

        t0 = time.perf_counter()
        train_model(
            mem_model, train_loader, val_loader, epochs=tr_cfg["epochs"], lr=tr_cfg["learning_rate"], K=K_val, rollout_steps=tr_cfg["rollout_steps"], verbose=False
        )
        train_time = time.perf_counter() - t0

        # Benchmark inference latency per step
        u0_test = datasets["val"][0][0][0:1]
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                mem_model.forward(u0_test, K=K_val)
        latency_ms = (time.perf_counter() - t_infer_start) / 100.0 * 1000.0

        eval_res = evaluate_autonomous_rollout(mem_model, datasets["val"], K=K_val, x=x, Lx=Lx, dx=dx)
        l2_err = eval_res["mean_rel_l2_overall"]

        print(f"  K={K_val}: Rel L2 = {l2_err:.4e} | MACs/Delta T = {mem_macs:,} | Latency = {latency_ms:.2f} ms")

        records.append(
            {
                "K": K_val,
                "delta_T": t_cfg["delta_T"],
                "macs_per_delta_T": mem_macs,
                "latency_ms": latency_ms,
                "mean_rel_l2": l2_err,
                "train_time_sec": train_time,
            }
        )

    df = pd.DataFrame(records)
    csv_path = out_dir / "k_sensitivity_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved K-sensitivity summary to {csv_path}")

    # Plot Pareto curve: Error vs Compute per Delta T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("K-Sensitivity & Compute vs Accuracy Trade-off (Fixed $\Delta T = 0.1$)", y=0.98)

    ax1.plot(df["K"], df["mean_rel_l2"], "o-", color="crimson", linewidth=2, markersize=7)
    ax1.set_xlabel("NCA Micro-Updates per Macro Step (K)")
    ax1.set_ylabel("Autonomous Rollout Relative $L_2$ Error")
    ax1.set_title("Accuracy vs Recurrent Depth K")
    ax1.set_xticks(df["K"])
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["macs_per_delta_T"], df["mean_rel_l2"], "s-", color="purple", linewidth=2, markersize=7)
    for _, row in df.iterrows():
        ax2.annotate(f"K={int(row['K'])}", (row["macs_per_delta_T"], row["mean_rel_l2"]), textcoords="offset points", xytext=(0, 8), ha="center")
    ax2.set_xlabel("Compute per $\Delta T$ (MACs)")
    ax2.set_ylabel("Autonomous Rollout Relative $L_2$ Error")
    ax2.set_title("Pareto Frontier: Accuracy vs Compute")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    pareto_plot_path = plots_dir / "fig9_pareto_cost_accuracy.png"
    plt.savefig(pareto_plot_path, dpi=300)
    plt.close(fig)
    print(f"Generated Pareto Frontier plot: {pareto_plot_path}")


if __name__ == "__main__":
    main()
