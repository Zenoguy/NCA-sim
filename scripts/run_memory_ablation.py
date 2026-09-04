"""
Experiment 7: Memory-Size Ablation Study.

Sweeps memory dimensions C_m in {0, 4, 8, 16, 32, 64} with fixed C_h=16.
Measures:
- Parameter count
- MACs per Delta T
- Autonomous rollout relative L2 error
- Long-horizon stability metric
Generates Figure 4: fig4_memory_ablation.png
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
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import load_datasets
from src.memory_nca import MemoryNCA
from src.nca import count_parameters, compute_nca_macs
from src.train import train_model
from src.evaluate import evaluate_autonomous_rollout
from src.visualization import plot_memory_ablation


def main():
    parser = argparse.ArgumentParser(description="Run Memory Size Ablation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dims", nargs="+", type=int, default=[0, 4, 8, 16, 32, 64])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["physics"]
    t_cfg = cfg["time_discretization"]
    d_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    out_dir = Path(cfg["paths"]["output_dir"]) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(cfg["paths"]["output_dir"]) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(d_cfg["data_dir"])
    datasets = load_datasets(data_dir)
    train_loader = DataLoader(datasets["train"], batch_size=tr_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=tr_cfg["batch_size"], shuffle=False)

    N = p_cfg["N"]
    Lx = p_cfg["Lx"]
    dx = Lx / N
    x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)
    K = t_cfg["K"]

    memory_dims = args.dims
    rollout_errors = []
    long_horizon_stabilities = []
    param_counts = []
    macs_list = []
    ablation_records = []

    print(f"=== Running Memory Dimension Sweep: C_m in {memory_dims} ===")

    for m_dim in memory_dims:
        print(f"\nEvaluating C_m = {m_dim}...")
        torch.manual_seed(42)
        np.random.seed(42)

        model = MemoryNCA(
            hidden_dim=cfg["models"]["hidden_dim"],
            memory_dim=m_dim,
            kernel_size=cfg["models"]["kernel_size"],
            mlp_hidden=cfg["models"]["mlp_hidden"],
            mode="persistent",
        )
        n_params = count_parameters(model)
        n_macs = compute_nca_macs(model, N=N, K=K)
        param_counts.append(n_params)
        macs_list.append(n_macs)

        # Train model
        train_model(
            model,
            train_loader,
            val_loader,
            epochs=tr_cfg["epochs"],
            lr=tr_cfg["learning_rate"],
            weight_decay=tr_cfg["weight_decay"],
            K=K,
            rollout_steps=tr_cfg["rollout_steps"],
            verbose=False,
        )

        # Evaluate on val set (training horizon)
        val_eval = evaluate_autonomous_rollout(model, datasets["val"], K=K, x=x, Lx=Lx, dx=dx)
        val_l2 = val_eval["mean_rel_l2_overall"]
        rollout_errors.append(val_l2)

        # Evaluate on long-horizon set
        long_eval = evaluate_autonomous_rollout(
            model, datasets["test_long_horizon"], K=K, x=x, Lx=Lx, dx=dx
        )
        long_l2 = long_eval["mean_rel_l2_overall"]
        long_final = long_eval["final_rel_l2"]

        # Stability metric: inverse of final error or 1 / (1 + long_final)
        stability = float(1.0 / (1.0 + long_final))
        long_horizon_stabilities.append(stability)

        print(f"  C_m={m_dim:2d} | Params: {n_params:6,d} | Val L2: {val_l2:.4e} | Long L2: {long_l2:.4e} | Stability: {stability:.4f}")

        ablation_records.append(
            {
                "memory_dim": m_dim,
                "parameters": n_params,
                "macs_per_delta_T": n_macs,
                "val_rel_l2": val_l2,
                "long_horizon_rel_l2": long_l2,
                "stability_metric": stability,
            }
        )

    # Save ablation summary
    df = pd.DataFrame(ablation_records)
    csv_path = out_dir / "memory_ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved ablation summary to {csv_path}")

    # Plot Figure 4
    fig_path = plots_dir / "fig4_memory_ablation.png"
    plot_memory_ablation(
        memory_dims, rollout_errors, long_horizon_stabilities, param_counts, str(fig_path)
    )
    print(f"Generated Figure 4: {fig_path}")


if __name__ == "__main__":
    main()
