"""
Dataset generation CLI script.
Generates and saves all partition-isolated datasets specified in config.
"""

import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
from src.kdv_solver import KdVSolver
from src.dataset import build_experiment_datasets, save_datasets


def main():
    parser = argparse.ArgumentParser(description="Generate KdV Trajectory Datasets")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["physics"]
    t_cfg = cfg["time_discretization"]
    d_cfg = cfg["dataset"]

    print("=== Generating Datasets ===")
    print(f"Grid N: {p_cfg['N']}, Lx: {p_cfg['Lx']}, alpha: {p_cfg['alpha']}, beta: {p_cfg['beta']}")
    print(f"delta_T: {t_cfg['delta_T']}, K updates: {t_cfg['K']}")
    print(f"Train horizon: {d_cfg['train_horizon']}, Long horizon: {d_cfg['long_horizon']}")

    solver = KdVSolver(
        N=p_cfg["N"],
        Lx=p_cfg["Lx"],
        alpha=p_cfg["alpha"],
        beta=p_cfg["beta"],
        dt=p_cfg["dt_internal"],
    )

    datasets = build_experiment_datasets(
        solver=solver,
        delta_T=t_cfg["delta_T"],
        train_horizon=d_cfg["train_horizon"],
        long_horizon=d_cfg["long_horizon"],
        seed=d_cfg["seed"],
    )

    out_dir = Path(d_cfg["data_dir"])
    save_datasets(datasets, out_dir)
    print(f"Datasets successfully generated and saved to: {out_dir}")
    for name, ds in datasets.items():
        print(f"  - {name:22s}: shape {ds['data'].shape}")


if __name__ == "__main__":
    main()
