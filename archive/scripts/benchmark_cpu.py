"""
Experiment 10: Multicore CPU Scaling and Parallel Efficiency Benchmark.

Hardware Target: AMD Ryzen 5 5600H (6 cores, 12 threads).
Evaluates:
- Single-simulation latency (Batch Size = 1)
- Batch throughput (Batch Size = 16)
- Model inference throughput vs whole-simulation throughput
- Grid sizes: N in {128, 256, 512, 1024, 2048}
- Threads: p in {1, 2, 4, 8, 12}
- Parallel scaling efficiency: T1 / (p * Tp)

Generates Figure 7: fig7_cpu_multicore_scaling.png
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
import torch
from src.memory_nca import MemoryNCA
from src.visualization import plot_cpu_scaling


def benchmark_single_config(
    model: torch.nn.Module,
    N: int,
    threads: int,
    K: int = 2,
    warmup: int = 10,
    steps: int = 50,
) -> dict:
    """Benchmark inference and simulation throughput for fixed N and thread count."""
    torch.set_num_threads(threads)

    # 1. Single-simulation latency (Batch size 1)
    u_single = torch.randn(1, 1, N)
    with torch.no_grad():
        for _ in range(warmup):
            model.forward(u_single, K=K)

        t0 = time.perf_counter()
        for _ in range(steps):
            model.forward(u_single, K=K)
        t_single = time.perf_counter() - t0

    single_latency_ms = (t_single / steps) * 1000.0
    inference_throughput = steps / t_single  # macro-steps / sec

    # 2. Batch throughput (Batch size 16)
    B = 16
    u_batch = torch.randn(B, 1, N)
    with torch.no_grad():
        for _ in range(warmup):
            model.forward(u_batch, K=K)

        t0 = time.perf_counter()
        for _ in range(steps):
            model.forward(u_batch, K=K)
        t_batch = time.perf_counter() - t0

    batch_throughput = (steps * B) / t_batch  # states / sec

    # 3. Whole simulation rollout throughput (50 macro steps)
    rollout_len = 20
    with torch.no_grad():
        t0 = time.perf_counter()
        model.rollout(u_single, num_macro_steps=rollout_len, K=K)
        t_rollout = time.perf_counter() - t0

    sim_throughput = (rollout_len) / max(1e-6, t_rollout)  # rollouts / sec

    return {
        "N": N,
        "threads": threads,
        "single_latency_ms": single_latency_ms,
        "inference_throughput": inference_throughput,
        "batch_throughput": batch_throughput,
        "whole_sim_throughput": sim_throughput,
        "t_single": t_single,
    }


def main():
    parser = argparse.ArgumentParser(description="Multicore CPU Scaling Benchmark")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_dir"])
    plots_dir = out_dir / "plots"

    grid_sizes = [128, 256, 512, 1024, 2048]
    threads_list = [1, 2, 4, 8, 12]
    K = cfg["time_discretization"]["K"]

    model = MemoryNCA(hidden_dim=16, memory_dim=16, kernel_size=3)
    model.eval()

    records = []
    inf_dict = {N: [] for N in grid_sizes}
    sim_dict = {N: [] for N in grid_sizes}
    eff_dict = {N: [] for N in grid_sizes}

    print("=== Running Multicore CPU Scaling Benchmark (Ryzen 5 5600H) ===")

    for N in grid_sizes:
        print(f"\n--- Grid Size N = {N} ---")
        t1_baseline = None

        for p in threads_list:
            res = benchmark_single_config(model, N, threads=p, K=K)
            if p == 1:
                t1_baseline = res["t_single"]
                eff = 1.0
            else:
                # Scaling efficiency = T1 / (p * Tp)
                eff = float(t1_baseline / (p * res["t_single"]))

            res["parallel_efficiency"] = eff
            records.append(res)

            inf_dict[N].append(res["inference_throughput"])
            sim_dict[N].append(res["whole_sim_throughput"])
            eff_dict[N].append(eff)

            print(
                f"  Threads={p:2d} | Latency: {res['single_latency_ms']:6.2f} ms | Inf Throughput: {res['inference_throughput']:6.1f} steps/s | Efficiency: {eff*100:5.1f}%"
            )

    df = pd.DataFrame(records)
    csv_path = out_dir / "cpu_scaling_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CPU benchmark summary to: {csv_path}")

    # Generate Figure 7
    fig7_path = plots_dir / "fig7_cpu_multicore_scaling.png"
    plot_cpu_scaling(threads_list, grid_sizes, inf_dict, sim_dict, eff_dict, str(fig7_path))
    print(f"Generated Figure 7: {fig7_path}")


if __name__ == "__main__":
    main()
