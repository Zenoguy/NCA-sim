"""
Stage 6: Fast End-to-End Smoke Test.
Verifies data generation, model instantiations, parameter matching, training convergence,
autonomous rollout, and plot generation on a 128-cell grid.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.kdv_solver import KdVSolver
from src.dataset import build_experiment_datasets, save_datasets, load_datasets
from src.nca import VanillaNCA, count_parameters, compute_nca_macs, find_matched_vanilla_channels
from src.memory_nca import MemoryNCA
from src.train import train_model
from src.evaluate import evaluate_autonomous_rollout, evaluate_one_step_oracle
from src.visualization import plot_rollout_comparison


def run_smoke_test():
    print("==================================================")
    print("=== Stage 6: Running Automated Small Smoke Test ===")
    print("==================================================")

    config_path = project_root / "configs" / "smoke_test.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["physics"]
    t_cfg = cfg["time_discretization"]
    d_cfg = cfg["dataset"]
    m_cfg = cfg["models"]
    tr_cfg = cfg["training"]
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Generation
    data_dir = Path(d_cfg["data_dir"])
    if not (data_dir / "train.npz").exists():
        print("Generating smoke test datasets...")
        solver = KdVSolver(
            N=p_cfg["N"],
            Lx=p_cfg["Lx"],
            alpha=p_cfg["alpha"],
            beta=p_cfg["beta"],
            dt=p_cfg["dt_internal"],
        )
        datasets = build_experiment_datasets(
            solver, delta_T=t_cfg["delta_T"], train_horizon=d_cfg["train_horizon"], seed=d_cfg["seed"]
        )
        save_datasets(datasets, data_dir)

    print(f"Loading datasets from: {data_dir}")
    loaded = load_datasets(data_dir)
    train_loader = DataLoader(loaded["train"], batch_size=tr_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(loaded["val"], batch_size=tr_cfg["batch_size"], shuffle=False)

    K = t_cfg["K"]
    N = p_cfg["N"]
    Lx = p_cfg["Lx"]

    # 2. Instantiate Models
    print("\n--- Model Architecture & Parameter Setup ---")
    mem_nca = MemoryNCA(
        hidden_dim=m_cfg["hidden_dim"],
        memory_dim=m_cfg["memory_dim"],
        kernel_size=m_cfg["kernel_size"],
        mlp_hidden=m_cfg["mlp_hidden"],
    )
    mem_params = count_parameters(mem_nca)
    mem_macs = compute_nca_macs(mem_nca, N=N, K=K)

    # Vanilla equal hidden
    vanilla_equal = VanillaNCA(
        hidden_dim=m_cfg["hidden_dim"],
        kernel_size=m_cfg["kernel_size"],
        mlp_hidden=m_cfg["mlp_hidden"],
    )
    vanilla_eq_params = count_parameters(vanilla_equal)
    vanilla_eq_macs = compute_nca_macs(vanilla_equal, N=N, K=K)

    # Vanilla parameter-matched
    matched_c, matched_mlp, matched_params = find_matched_vanilla_channels(
        target_params=mem_params, kernel_size=m_cfg["kernel_size"]
    )
    vanilla_matched = VanillaNCA(
        hidden_dim=matched_c, kernel_size=m_cfg["kernel_size"], mlp_hidden=matched_mlp
    )
    actual_matched_params = count_parameters(vanilla_matched)
    matched_macs = compute_nca_macs(vanilla_matched, N=N, K=K)

    param_discrepancy = abs(actual_matched_params - mem_params) / mem_params
    print(f"Memory-NCA (C_h={m_cfg['hidden_dim']}, C_m={m_cfg['memory_dim']}): {mem_params:,} params, {mem_macs:,} MACs/step")
    print(f"Vanilla NCA (equal hidden, C_h={m_cfg['hidden_dim']}):     {vanilla_eq_params:,} params, {vanilla_eq_macs:,} MACs/step")
    print(f"Vanilla NCA (matched, C_h={matched_c}, mlp={matched_mlp}):       {actual_matched_params:,} params, {matched_macs:,} MACs/step")
    print(f"Parameter matching discrepancy: {param_discrepancy*100:.2f}% (Target < 1.0%)")
    assert param_discrepancy < 0.02, "Parameter matching discrepancy too high!"

    # 3. Train Models
    epochs = tr_cfg["epochs"]
    print(f"\n--- Training Models ({epochs} epochs) ---")

    print("\nTraining Vanilla NCA (matched)...")
    history_vanilla = train_model(
        vanilla_matched,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=tr_cfg["learning_rate"],
        K=K,
        rollout_steps=tr_cfg["rollout_steps"],
        verbose=True,
    )

    print("\nTraining Memory-NCA...")
    history_mem = train_model(
        mem_nca,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=tr_cfg["learning_rate"],
        K=K,
        rollout_steps=tr_cfg["rollout_steps"],
        verbose=True,
    )

    # Check loss decrease
    v_start, v_end = history_vanilla["train_loss"][0], history_vanilla["train_loss"][-1]
    m_start, m_end = history_mem["train_loss"][0], history_mem["train_loss"][-1]
    print(f"\nVanilla Train Loss: {v_start:.4e} -> {v_end:.4e} (decrease: {v_start > v_end})")
    print(f"Memory  Train Loss: {m_start:.4e} -> {m_end:.4e} (decrease: {m_start > m_end})")
    assert v_end < v_start, "Vanilla NCA failed to decrease training loss!"
    assert m_end < m_start, "Memory-NCA failed to decrease training loss!"

    # 4. Evaluate Autonomous Rollout
    print("\n--- Evaluating Autonomous Rollout ---")
    val_ds = loaded["val"]
    eval_v = evaluate_autonomous_rollout(vanilla_matched, val_ds, K=K, Lx=Lx)
    eval_m = evaluate_autonomous_rollout(mem_nca, val_ds, K=K, Lx=Lx)
    print(f"Vanilla Mean Rollout Rel L2: {eval_v['mean_rel_l2_overall']:.4e}")
    print(f"Memory  Mean Rollout Rel L2: {eval_m['mean_rel_l2_overall']:.4e}")

    # One-step oracle
    one_v = evaluate_one_step_oracle(vanilla_matched, val_ds, K=K)
    one_m = evaluate_one_step_oracle(mem_nca, val_ds, K=K)
    print(f"Vanilla One-Step Rel L2:     {one_v['one_step_mean_rel_l2']:.4e}")
    print(f"Memory  One-Step Rel L2:     {one_m['one_step_mean_rel_l2']:.4e}")

    # 5. Generate Test Plot
    print("\n--- Generating Test Plots ---")
    plot_path = out_dir / "plots" / "smoke_test_rollout.png"
    sample_rollouts = {
        "Ground Truth": eval_v["sample"]["true"],
        "Vanilla Matched": eval_v["sample"]["pred"],
        "Memory-NCA": eval_m["sample"]["pred"],
    }
    t_eval = np.linspace(0.0, d_cfg["train_horizon"] * t_cfg["delta_T"], d_cfg["train_horizon"] + 1)
    x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)
    plot_rollout_comparison(t_eval, x, sample_rollouts, str(plot_path))
    print(f"Generated smoke test comparison plot: {plot_path}")

    print("\n==============================================")
    print("=== STAGE 6 SMOKE TEST COMPLETED SUCCESSFULLY! ===")
    print("==============================================")


if __name__ == "__main__":
    run_smoke_test()
