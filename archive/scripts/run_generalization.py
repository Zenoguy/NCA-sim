"""
Experiment Suite: Generalization Tests (Interpolation, Extrapolation, Tests A, B, C).

Evaluates models on:
- Interpolation: Unseen parameter combinations inside training range
- Extrapolation: Amplitudes outside training range (A in [1.3, 1.8])
- Test A: Unseen single-soliton parameters
- Test B: Off-manifold perturbed single pulses (L != L(A))
- Test C: Two-pulse collision initial condition

Generates Figure 5: fig5_generalization_tests.png
"""

import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import pandas as pd
import torch
from src.dataset import load_datasets
from src.nca import VanillaNCA, find_matched_vanilla_channels
from src.memory_nca import MemoryNCA
from src.cnn_baseline import CNNBaseline
from src.evaluate import evaluate_autonomous_rollout
from src.visualization import plot_generalization_tests


def main():
    parser = argparse.ArgumentParser(description="Run Generalization Benchmark")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["physics"]
    t_cfg = cfg["time_discretization"]
    d_cfg = cfg["dataset"]
    m_cfg = cfg["models"]
    out_dir = Path(cfg["paths"]["output_dir"])
    plots_dir = out_dir / "plots"
    ckpt_dir = out_dir / "checkpoints"

    data_dir = Path(d_cfg["data_dir"])
    datasets = load_datasets(data_dir)

    N = p_cfg["N"]
    Lx = p_cfg["Lx"]
    dx = Lx / N
    x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)
    K = t_cfg["K"]

    test_keys = [
        ("test_interp", "Interpolation"),
        ("test_extrap", "Extrapolation"),
        ("test_unseen_params", "Test A: Unseen Params"),
        ("test_perturbed_pulses", "Test B: Off-Manifold Pulses"),
        ("test_two_pulses", "Test C: 2-Pulse Collision"),
    ]

    target_mem = MemoryNCA(
        hidden_dim=m_cfg["hidden_dim"],
        memory_dim=m_cfg["memory_dim"],
        kernel_size=m_cfg["kernel_size"],
        mlp_hidden=m_cfg["mlp_hidden"],
    )
    from src.nca import count_parameters
    mem_params = count_parameters(target_mem)
    matched_c, matched_mlp, _ = find_matched_vanilla_channels(mem_params, kernel_size=m_cfg["kernel_size"])

    model_factories = {
        "Vanilla (equal)": lambda: VanillaNCA(
            hidden_dim=m_cfg["hidden_dim"], kernel_size=m_cfg["kernel_size"], mlp_hidden=m_cfg["mlp_hidden"]
        ),
        "Vanilla (matched)": lambda: VanillaNCA(
            hidden_dim=matched_c, kernel_size=m_cfg["kernel_size"], mlp_hidden=matched_mlp
        ),
        "Memory-NCA (no-pers)": lambda: MemoryNCA(
            hidden_dim=m_cfg["hidden_dim"], memory_dim=m_cfg["memory_dim"], kernel_size=m_cfg["kernel_size"], mlp_hidden=m_cfg["mlp_hidden"], mode="no_persistence"
        ),
        "Memory-NCA (rand-pers)": lambda: MemoryNCA(
            hidden_dim=m_cfg["hidden_dim"], memory_dim=m_cfg["memory_dim"], kernel_size=m_cfg["kernel_size"], mlp_hidden=m_cfg["mlp_hidden"], mode="random_persistence"
        ),
        "Memory-NCA (pers)": lambda: MemoryNCA(
            hidden_dim=m_cfg["hidden_dim"], memory_dim=m_cfg["memory_dim"], kernel_size=m_cfg["kernel_size"], mlp_hidden=m_cfg["mlp_hidden"], mode="persistent"
        ),
        "CNN Baseline": lambda: CNNBaseline(hidden_dim=32, kernel_size=5, num_layers=4),
    }

    key_map = {
        "Vanilla (equal)": "vanilla_equal",
        "Vanilla (matched)": "vanilla_matched",
        "Memory-NCA (no-pers)": "memory_no_persistence",
        "Memory-NCA (rand-pers)": "memory_random_persistence",
        "Memory-NCA (pers)": "memory_persistent",
        "CNN Baseline": "cnn_baseline",
    }

    results_table = []
    plot_data = {m_name: [] for m_name in model_factories}

    print("=== Evaluating Generalization Tests across Regimes ===")

    for m_name, factory in model_factories.items():
        k = key_map[m_name]
        ckpt_path = ckpt_dir / f"{k}_seed42.pt"
        model = factory()
        if ckpt_path.exists():
            loaded = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(loaded["model_state"])
            print(f"Loaded checkpoint for {m_name} from {ckpt_path.name}")
        else:
            print(f"Warning: Checkpoint {ckpt_path.name} not found, using untrained model.")

        for test_key, display_name in test_keys:
            ds = datasets[test_key]
            eval_res = evaluate_autonomous_rollout(model, ds, K=K, x=x, Lx=Lx, dx=dx)
            l2_err = eval_res["mean_rel_l2_overall"]
            final_err = eval_res["final_rel_l2"]
            amp_err = eval_res["mean_amp_err_overall"]

            results_table.append(
                {
                    "model": m_name,
                    "test_regime": display_name,
                    "mean_rel_l2": l2_err,
                    "final_rel_l2": final_err,
                    "mean_amp_err": amp_err,
                }
            )
            plot_data[m_name].append(l2_err)
            print(f"  {m_name:24s} on {display_name:28s}: Rel L2 = {l2_err:.4e}")

    # Save summary
    df = pd.DataFrame(results_table)
    summary_path = out_dir / "generalization_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved generalization summary to: {summary_path}")

    # Generate Figure 5
    test_labels = [name for _, name in test_keys]
    fig5_path = plots_dir / "fig5_generalization_tests.png"
    plot_generalization_tests(test_labels, plot_data, str(fig5_path))
    print(f"Generated Figure 5: {fig5_path}")


if __name__ == "__main__":
    main()
