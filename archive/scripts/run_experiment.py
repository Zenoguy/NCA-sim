"""
Master Experiment Runner: Main Benchmark and Early Falsification Gate.

Evaluates:
- Ground Truth ETDRK4
- Vanilla NCA (equal-hidden)
- Vanilla NCA (parameter-matched)
- Memory-NCA (no-persistence control)
- Memory-NCA (persistent random memory control)
- Memory-NCA (persistent learned - primary model)
- CNN Baseline

Across 3 independent seeds with mean +/- std error reporting.
Generates Figures 2, 3, 6, summary.csv, and metrics.json.
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
from src.kdv_solver import KdVSolver
from src.dataset import build_experiment_datasets, save_datasets, load_datasets
from src.nca import VanillaNCA, count_parameters, compute_nca_macs, find_matched_vanilla_channels
from src.memory_nca import MemoryNCA
from src.cnn_baseline import CNNBaseline
from src.train import train_model
from src.evaluate import evaluate_autonomous_rollout, evaluate_one_step_oracle
from src.visualization import (
    plot_rollout_comparison,
    plot_error_vs_time,
    plot_soliton_diagnostics,
)


def instantiate_model(model_key: str, cfg: dict, matched_channels: tuple) -> torch.nn.Module:
    """Instantiate model based on key."""
    m_cfg = cfg["models"]
    c_matched, mlp_matched, _ = matched_channels

    if model_key == "vanilla_equal":
        return VanillaNCA(
            hidden_dim=m_cfg["hidden_dim"],
            kernel_size=m_cfg["kernel_size"],
            mlp_hidden=m_cfg["mlp_hidden"],
        )
    elif model_key == "vanilla_matched":
        return VanillaNCA(
            hidden_dim=c_matched,
            kernel_size=m_cfg["kernel_size"],
            mlp_hidden=mlp_matched,
        )
    elif model_key == "memory_no_persistence":
        return MemoryNCA(
            hidden_dim=m_cfg["hidden_dim"],
            memory_dim=m_cfg["memory_dim"],
            kernel_size=m_cfg["kernel_size"],
            mlp_hidden=m_cfg["mlp_hidden"],
            mode="no_persistence",
        )
    elif model_key == "memory_random_persistence":
        return MemoryNCA(
            hidden_dim=m_cfg["hidden_dim"],
            memory_dim=m_cfg["memory_dim"],
            kernel_size=m_cfg["kernel_size"],
            mlp_hidden=m_cfg["mlp_hidden"],
            mode="random_persistence",
        )
    elif model_key == "memory_persistent":
        return MemoryNCA(
            hidden_dim=m_cfg["hidden_dim"],
            memory_dim=m_cfg["memory_dim"],
            kernel_size=m_cfg["kernel_size"],
            mlp_hidden=m_cfg["mlp_hidden"],
            mode="persistent",
        )
    elif model_key == "cnn_baseline":
        return CNNBaseline(hidden_dim=32, kernel_size=5, num_layers=4)
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def main():
    parser = argparse.ArgumentParser(description="Run Core NCA KdV Experiment Suite")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["physics"]
    t_cfg = cfg["time_discretization"]
    d_cfg = cfg["dataset"]
    m_cfg = cfg["models"]
    tr_cfg = cfg["training"]
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    ckpt_dir = out_dir / "checkpoints"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset verification or generation
    data_dir = Path(d_cfg["data_dir"])
    if not (data_dir / "train.npz").exists():
        print(f"Generating primary datasets in {data_dir}...")
        solver = KdVSolver(
            N=p_cfg["N"],
            Lx=p_cfg["Lx"],
            alpha=p_cfg["alpha"],
            beta=p_cfg["beta"],
            dt=p_cfg["dt_internal"],
        )
        datasets = build_experiment_datasets(
            solver,
            delta_T=t_cfg["delta_T"],
            train_horizon=d_cfg["train_horizon"],
            long_horizon=d_cfg["long_horizon"],
            seed=d_cfg["seed"],
        )
        save_datasets(datasets, data_dir)

    print(f"Loading datasets from {data_dir}...")
    datasets = load_datasets(data_dir)
    train_loader = DataLoader(datasets["train"], batch_size=tr_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=tr_cfg["batch_size"], shuffle=False)

    N = p_cfg["N"]
    Lx = p_cfg["Lx"]
    dx = Lx / N
    x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)
    K = t_cfg["K"]
    delta_T = t_cfg["delta_T"]

    # 2. Automated parameter matching solver
    target_mem = MemoryNCA(
        hidden_dim=m_cfg["hidden_dim"],
        memory_dim=m_cfg["memory_dim"],
        kernel_size=m_cfg["kernel_size"],
        mlp_hidden=m_cfg["mlp_hidden"],
    )
    target_mem_params = count_parameters(target_mem)
    matched_c, matched_mlp, matched_params = find_matched_vanilla_channels(
        target_params=target_mem_params, kernel_size=m_cfg["kernel_size"]
    )
    matched_info = (matched_c, matched_mlp, matched_params)

    model_keys = [
        "vanilla_equal",
        "vanilla_matched",
        "memory_no_persistence",
        "memory_random_persistence",
        "memory_persistent",
        "cnn_baseline",
    ]

    model_labels = {
        "vanilla_equal": "Vanilla NCA (equal-hidden)",
        "vanilla_matched": "Vanilla NCA (param-matched)",
        "memory_no_persistence": "Memory-NCA (no-persistence)",
        "memory_random_persistence": "Memory-NCA (random-persistence)",
        "memory_persistent": "Memory-NCA (persistent)",
        "cnn_baseline": "CNN Baseline",
    }

    # Measure params and MACs for each architecture
    model_stats = {}
    for k in model_keys:
        m = instantiate_model(k, cfg, matched_info)
        params = count_parameters(m)
        macs = compute_nca_macs(m, N=N, K=K)
        model_stats[k] = {"params": params, "macs": macs}

    print("\n================ ARCHITECTURAL COST COMPARISON ================")
    for k in model_keys:
        p = model_stats[k]["params"]
        m = model_stats[k]["macs"]
        print(f"{model_labels[k]:32s} | Params: {p:6,d} | MACs/Delta T: {m:8,d}")
    print("===============================================================\n")

    # 3. Multi-seed training and evaluation
    seeds = args.seeds
    results_by_model = {k: [] for k in model_keys}
    sample_rollouts = {}
    time_series_errors = {}
    diagnostics_by_model = {}

    for k in model_keys:
        print(f"\n--- Benchmarking: {model_labels[k]} across seeds {seeds} ---")
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = instantiate_model(k, cfg, matched_info)
            ckpt_file = ckpt_dir / f"{k}_seed{seed}.pt"

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
                grad_clip=tr_cfg["grad_clip"],
                save_path=ckpt_file,
                verbose=False,
            )

            # 1-step oracle evaluation
            oracle_res = evaluate_one_step_oracle(model, datasets["val"], K=K)

            # Multi-step autonomous rollout evaluation on val
            val_eval = evaluate_autonomous_rollout(
                model, datasets["val"], K=K, x=x, Lx=Lx, dx=dx
            )

            # Long-horizon rollout evaluation on test_long_horizon (100 macro steps = 10.0 physical time)
            long_eval = evaluate_autonomous_rollout(
                model, datasets["test_long_horizon"], K=K, x=x, Lx=Lx, dx=dx
            )

            # Interpolation test evaluation
            interp_eval = evaluate_autonomous_rollout(
                model, datasets["test_interp"], K=K, x=x, Lx=Lx, dx=dx
            )

            # Extrapolation test evaluation
            extrap_eval = evaluate_autonomous_rollout(
                model, datasets["test_extrap"], K=K, x=x, Lx=Lx, dx=dx
            )

            seed_summary = {
                "seed": seed,
                "one_step_rel_l2": oracle_res["one_step_mean_rel_l2"],
                "val_rollout_rel_l2": val_eval["mean_rel_l2_overall"],
                "val_final_rel_l2": val_eval["final_rel_l2"],
                "val_mean_amp_err": val_eval["mean_amp_err_overall"],
                "val_mean_center_err": val_eval["mean_center_err_overall"],
                "long_horizon_rel_l2": long_eval["mean_rel_l2_overall"],
                "long_horizon_final_rel_l2": long_eval["final_rel_l2"],
                "interp_rel_l2": interp_eval["mean_rel_l2_overall"],
                "extrap_rel_l2": extrap_eval["mean_rel_l2_overall"],
            }
            results_by_model[k].append(seed_summary)

            # Keep representative sample from first seed for plots
            if seed == seeds[0]:
                sample_rollouts[model_labels[k]] = val_eval["sample"]["pred"]
                time_series_errors[model_labels[k]] = val_eval["rel_l2_vs_time"]
                diagnostics_by_model[model_labels[k]] = {
                    "amplitude": np.max(val_eval["sample"]["pred"], axis=1),
                    "center": [float(x[np.argmax(u)]) for u in val_eval["sample"]["pred"]],
                    "width": [
                        float(np.sum(u >= 0.5 * np.max(u)) * dx) for u in val_eval["sample"]["pred"]
                    ],
                }

        # Print seed-averaged performance
        mean_val_l2 = np.mean([r["val_rollout_rel_l2"] for r in results_by_model[k]])
        std_val_l2 = np.std([r["val_rollout_rel_l2"] for r in results_by_model[k]])
        mean_long_l2 = np.mean([r["long_horizon_rel_l2"] for r in results_by_model[k]])
        std_long_l2 = np.std([r["long_horizon_rel_l2"] for r in results_by_model[k]])
        print(
            f"  {model_labels[k]}: Val L2 = {mean_val_l2:.4e} +/- {std_val_l2:.2e} | Long Horizon L2 = {mean_long_l2:.4e} +/- {std_long_l2:.2e}"
        )

    # Add Ground Truth to sample rollouts and diagnostics for plotting
    sample_rollouts["Ground Truth"] = val_eval["sample"]["true"]
    diagnostics_by_model["Ground Truth"] = {
        "amplitude": np.max(val_eval["sample"]["true"], axis=1),
        "center": [float(x[np.argmax(u)]) for u in val_eval["sample"]["true"]],
        "width": [
            float(np.sum(u >= 0.5 * np.max(u)) * dx) for u in val_eval["sample"]["true"]
        ],
    }

    # 4. Generate Figures 2, 3, 6
    print("\n--- Generating Publication Visualizations ---")
    t_eval = np.linspace(0.0, d_cfg["train_horizon"] * delta_T, d_cfg["train_horizon"] + 1)

    # Figure 2: Rollout comparison
    plot_rollout_comparison(
        t_eval, x, sample_rollouts, str(plots_dir / "fig2_rollout_comparison.png")
    )
    print(f"Generated Figure 2: {plots_dir / 'fig2_rollout_comparison.png'}")

    # Figure 3: Error over time
    plot_error_vs_time(
        t_eval, time_series_errors, str(plots_dir / "fig3_error_vs_time.png")
    )
    print(f"Generated Figure 3: {plots_dir / 'fig3_error_vs_time.png'}")

    # Figure 6: Physical soliton diagnostics
    plot_soliton_diagnostics(
        t_eval, diagnostics_by_model, str(plots_dir / "fig6_soliton_diagnostics.png")
    )
    print(f"Generated Figure 6: {plots_dir / 'fig6_soliton_diagnostics.png'}")

    # 5. Compile Summary Table and JSON
    summary_rows = []
    metrics_export = {}

    for k in model_keys:
        res_list = results_by_model[k]
        params = model_stats[k]["params"]
        macs = model_stats[k]["macs"]

        def clean_val(v):
            if isinstance(v, (int, str, bool)):
                return v
            v_f = float(v)
            if np.isnan(v_f):
                return 1e6
            if np.isinf(v_f):
                return 1e6 if v_f > 0 else -1e6
            return v_f

        row = {
            "model_key": k,
            "model_name": model_labels[k],
            "parameters": params,
            "macs_per_delta_T": macs,
            "one_step_rel_l2_mean": clean_val(np.mean([r["one_step_rel_l2"] for r in res_list])),
            "one_step_rel_l2_std": clean_val(np.std([r["one_step_rel_l2"] for r in res_list])),
            "val_rollout_rel_l2_mean": clean_val(np.mean([r["val_rollout_rel_l2"] for r in res_list])),
            "val_rollout_rel_l2_std": clean_val(np.std([r["val_rollout_rel_l2"] for r in res_list])),
            "val_final_rel_l2_mean": clean_val(np.mean([r["val_final_rel_l2"] for r in res_list])),
            "val_amp_err_mean": clean_val(np.mean([r["val_mean_amp_err"] for r in res_list])),
            "val_center_err_mean": clean_val(np.mean([r["val_mean_center_err"] for r in res_list])),
            "long_horizon_rel_l2_mean": clean_val(np.mean([r["long_horizon_rel_l2"] for r in res_list])),
            "long_horizon_rel_l2_std": clean_val(np.std([r["long_horizon_rel_l2"] for r in res_list])),
            "interp_rel_l2_mean": clean_val(np.mean([r["interp_rel_l2"] for r in res_list])),
            "extrap_rel_l2_mean": clean_val(np.mean([r["extrap_rel_l2"] for r in res_list])),
        }
        summary_rows.append(row)
        metrics_export[k] = {
            "stats": model_stats[k],
            "summary": row,
            "seed_details": [
                {key: clean_val(val) for key, val in r.items()} for r in res_list
            ],
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved experiment summary to: {summary_csv_path}")

    metrics_json_path = out_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"Saved full metrics JSON to: {metrics_json_path}")


    # 6. Early Falsification Gate Assessment
    print("\n=======================================================")
    print("============= EARLY FALSIFICATION GATE ================")
    print("=======================================================")
    matched_l2 = summary_df.loc[summary_df["model_key"] == "vanilla_matched", "val_rollout_rel_l2_mean"].values[0]
    mem_l2 = summary_df.loc[summary_df["model_key"] == "memory_persistent", "val_rollout_rel_l2_mean"].values[0]
    matched_long = summary_df.loc[summary_df["model_key"] == "vanilla_matched", "long_horizon_rel_l2_mean"].values[0]
    mem_long = summary_df.loc[summary_df["model_key"] == "memory_persistent", "long_horizon_rel_l2_mean"].values[0]

    print(f"Vanilla NCA (Param-Matched) Rollout L2: {matched_l2:.4e} (Long Horizon: {matched_long:.4e})")
    print(f"Memory-NCA (Persistent)    Rollout L2: {mem_l2:.4e} (Long Horizon: {mem_long:.4e})")

    diff_val = (mem_l2 - matched_l2) / matched_l2
    diff_long = (mem_long - matched_long) / matched_long

    if mem_l2 < matched_l2:
        print(f"Finding: Memory-NCA achieved {abs(diff_val)*100:.2f}% LOWER rollout error than parameter-matched Vanilla NCA.")
    else:
        print(f"Finding: Memory-NCA did NOT improve rollout error over parameter-matched Vanilla NCA ({abs(diff_val)*100:.2f}% higher).")

    if mem_long < matched_long:
        print(f"Long-Horizon Finding: Memory-NCA achieved {abs(diff_long)*100:.2f}% LOWER long-horizon error.")
    else:
        print(f"Long-Horizon Finding: Memory-NCA did NOT improve long-horizon stability ({abs(diff_long)*100:.2f}% higher).")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
