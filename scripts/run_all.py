"""
Master Pipeline Orchestrator for Memory-Augmented NCA Research Prototype.

Executes complete experimental suite:
- Phase 0: ETDRK4 Solver Validation & Figure 1
- Phase 1-4: Core Experiment Suite (3 seeds, 6 models) & Figures 2, 3, 6
- Phase 5: Memory Size Ablation & Figure 4
- Phase 6: Physical Generalization Suite (Tests A, B, C) & Figure 5
- Phase 7: K-Sensitivity Study & Figure 9
- Phase 8: Causal Memory Swapping with Controls & Figure 8
- Phase 9: Multicore CPU Scaling & Parallel Efficiency & Figure 7
- Phase 10: Automated Comprehensive Research Report (outputs/report.md)
"""

import argparse
import subprocess
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import json


def run_command(cmd_list, desc):
    print(f"\n=======================================================")
    print(f"=== Running: {desc} ===")
    print(f"Command: {' '.join(cmd_list)}")
    print(f"=======================================================")
    res = subprocess.run(cmd_list, check=True)
    return res


def generate_automated_report(out_dir: Path):
    print("\n=== Generating Automated Research Report (report.md) ===")
    summary_csv = out_dir / "summary.csv"
    report_md = out_dir / "report.md"

    df = pd.read_csv(summary_csv) if summary_csv.exists() else None

    # Load ablation if exists
    abl_csv = out_dir / "ablation" / "memory_ablation_summary.csv"
    abl_df = pd.read_csv(abl_csv) if abl_csv.exists() else None

    # Load generalization if exists
    gen_csv = out_dir / "generalization_summary.csv"
    gen_df = pd.read_csv(gen_csv) if gen_csv.exists() else None

    # Load CPU scaling if exists
    cpu_csv = out_dir / "cpu_scaling_summary.csv"
    cpu_df = pd.read_csv(cpu_csv) if cpu_csv.exists() else None

    # Load memory swap if exists
    swap_json = out_dir / "memory_swapping_results.json"
    swap_data = {}
    if swap_json.exists():
        with open(swap_json, "r") as f:
            swap_data = json.load(f)

    with open(report_md, "w") as f:
        f.write("# Automated Research Report: Memory-Augmented Neural Cellular Automata for KdV Dynamics\n\n")
        f.write("## 1. Scientific Hypothesis\n\n")
        f.write("> **\"Does adding persistent memory to a Neural Cellular Automaton (NCA) improve its ability to learn, generalize, and stably roll out nonlinear physical dynamics compared with an otherwise comparable vanilla NCA?\"**\n\n")
        f.write("Target dynamical system: 1D Korteweg–de Vries (KdV) equation: $u_t + 6 u u_x + u_{xxx} = 0$.\n\n")

        f.write("## 2. Executive Summary of Findings\n\n")
        if df is not None:
            v_matched = df.loc[df["model_key"] == "vanilla_matched"].iloc[0]
            mem_pers = df.loc[df["model_key"] == "memory_persistent"].iloc[0]
            v_equal = df.loc[df["model_key"] == "vanilla_equal"].iloc[0]
            cnn_row = df.loc[df["model_key"] == "cnn_baseline"].iloc[0] if (df["model_key"] == "cnn_baseline").any() else None

            f.write(f"- **Parameter-Matched Comparison**: Parameter-matched Vanilla NCA ({v_matched['parameters']:,} params) vs. Persistent Memory-NCA ({mem_pers['parameters']:,} params, {mem_pers['parameters'] - v_matched['parameters']:+d} params diff).\n")
            f.write(f"- **Validation Rollout Error**: Vanilla NCA (matched) mean $E_{{L2}} = {v_matched['val_rollout_rel_l2_mean']:.4e} \pm {v_matched['val_rollout_rel_l2_std']:.2e}$ vs. Memory-NCA $E_{{L2}} = {mem_pers['val_rollout_rel_l2_mean']:.4e} \pm {mem_pers['val_rollout_rel_l2_std']:.2e}$.\n")
            f.write(f"- **One-Step Transition Oracle**: Vanilla NCA $E_{{L2}} = {v_matched['one_step_rel_l2_mean']:.4e}$ vs. Memory-NCA $E_{{L2}} = {mem_pers['one_step_rel_l2_mean']:.4e}$.\n")

            if mem_pers["val_rollout_rel_l2_mean"] < v_matched["val_rollout_rel_l2_mean"]:
                diff = (v_matched["val_rollout_rel_l2_mean"] - mem_pers["val_rollout_rel_l2_mean"]) / v_matched["val_rollout_rel_l2_mean"]
                f.write(f"- **Primary Hypothesis Verdict**: The persistent Memory-NCA achieved {diff*100:.2f}% lower autonomous rollout error than the parameter-matched Vanilla NCA in this benchmark.\n\n")
            else:
                diff = (mem_pers["val_rollout_rel_l2_mean"] - v_matched["val_rollout_rel_l2_mean"]) / v_matched["val_rollout_rel_l2_mean"]
                f.write(f"- **Primary Hypothesis Verdict (Neutral / Negative Result)**: Persistent Memory-NCA did not yield a statistically significant advantage over parameter-matched Vanilla NCA on autonomous KdV rollouts ({diff*100:.2f}% higher relative error). For single-soliton KdV flows, local state capacity appears equally or more effective than explicit recurrent memory gating.\n\n")

        f.write("## 3. Main Benchmark Performance Table (Mean $\pm$ Std across 3 Seeds)\n\n")
        if df is not None:
            f.write("| Model | Parameters | MACs/$\Delta T$ | One-Step Rel $L_2$ | Val Rollout Rel $L_2$ | Long Horizon Rel $L_2$ |\n")
            f.write("|---|---|---|---|---|---|\n")
            for _, r in df.iterrows():
                f.write(f"| **{r['model_name']}** | {int(r['parameters']):,d} | {int(r['macs_per_delta_T']):,d} | {r['one_step_rel_l2_mean']:.3e} $\pm$ {r['one_step_rel_l2_std']:.1e} | {r['val_rollout_rel_l2_mean']:.3e} $\pm$ {r['val_rollout_rel_l2_std']:.1e} | {r['long_horizon_rel_l2_mean']:.3e} |\n")
            f.write("\n")

        if gen_df is not None:
            f.write("## 4. Generalization Breakdown Across Physical Regimes\n\n")
            f.write("| Model | Regime | Mean Rel $L_2$ | Final Rel $L_2$ | Peak Amplitude Error |\n")
            f.write("|---|---|---|---|---|\n")
            for _, r in gen_df.iterrows():
                f.write(f"| {r['model']} | {r['test_regime']} | {r['mean_rel_l2']:.4e} | {r['final_rel_l2']:.4e} | {r['mean_amp_err']:.4e} |\n")
            f.write("\n")

        if abl_df is not None:
            f.write("## 5. Memory-Size Ablation Study\n\n")
            f.write("| Memory Channels ($C_m$) | Trainable Parameters | Validation Rel $L_2$ | Long-Horizon Stability Metric |\n")
            f.write("|---|---|---|---|\n")
            for _, r in abl_df.iterrows():
                f.write(f"| {int(r['memory_dim'])} | {int(r['parameters']):,d} | {r['val_rel_l2']:.4e} | {r['stability_metric']:.4f} |\n")
            f.write("\n")

        if swap_data:
            f.write("## 6. Causal Memory Swapping Diagnostic\n\n")
            f.write("Evaluating whether memory causally governs regime dynamics:\n")
            f.write(f"- Regime A Memory ($u + m_A$): Final Peak Position = {swap_data.get('x_peak_A', 0.0):.2f}, Amplitude = {swap_data.get('amp_A', 0.0):.3f}\n")
            f.write(f"- Regime B Memory ($u + m_B$): Final Peak Position = {swap_data.get('x_peak_B', 0.0):.2f}, Amplitude = {swap_data.get('amp_B', 0.0):.3f}\n")
            f.write(f"- Random Memory Control ($u + m_{{rand}}$): Final Peak Position = {swap_data.get('x_peak_rand', 0.0):.2f}, Amplitude = {swap_data.get('amp_rand', 0.0):.3f}\n")
            f.write(f"- Zero Memory Control ($u + m_{{zero}}$): Final Peak Position = {swap_data.get('x_peak_zero', 0.0):.2f}, Amplitude = {swap_data.get('amp_zero', 0.0):.3f}\n\n")

        if cpu_df is not None:
            f.write("## 7. Multicore CPU Parallel Efficiency (Ryzen 5 5600H)\n\n")
            f.write("Representative scaling at $N=256$ and $N=1024$:\n\n")
            sub_cpu = cpu_df[cpu_df["N"].isin([256, 1024])]
            f.write("| Grid Size $N$ | CPU Threads | Latency (ms) | Inference Steps/s | Parallel Efficiency |\n")
            f.write("|---|---|---|---|---|\n")
            for _, r in sub_cpu.iterrows():
                f.write(f"| {int(r['N'])} | {int(r['threads'])} | {r['single_latency_ms']:.2f} ms | {r['inference_throughput']:.1f} | {r['parallel_efficiency']*100:.1f}% |\n")
            f.write("\n")

        f.write("## 8. Generated Figures Catalog\n\n")
        f.write("- **Figure 1**: `outputs/plots/fig1_solver_validation.png` - ETDRK4 numerical ground-truth validation.\n")
        f.write("- **Figure 2**: `outputs/plots/fig2_rollout_comparison.png` - Multi-model autonomous rollout snapshots.\n")
        f.write("- **Figure 3**: `outputs/plots/fig3_error_vs_time.png` - Autonomous rollout error over time.\n")
        f.write("- **Figure 4**: `outputs/plots/fig4_memory_ablation.png` - Memory dimension $C_m$ ablation curve.\n")
        f.write("- **Figure 5**: `outputs/plots/fig5_generalization_tests.png` - Generalization breakdown across regimes.\n")
        f.write("- **Figure 6**: `outputs/plots/fig6_soliton_diagnostics.png` - Amplitude, center, and width diagnostics.\n")
        f.write("- **Figure 7**: `outputs/plots/fig7_cpu_multicore_scaling.png` - Multicore CPU throughput and parallel efficiency.\n")
        f.write("- **Figure 8**: `outputs/plots/fig8_memory_swapping_causal.png` - Causal memory swapping dynamics.\n")
        f.write("- **Figure 9**: `outputs/plots/fig9_pareto_cost_accuracy.png` - K-sensitivity Pareto frontier.\n\n")

    print(f"Research report written to: {report_md}")


def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    python_bin = sys.executable

    # 1. Phase 0: Solver Verification
    run_command([python_bin, "scripts/verify_solver.py"], "Phase 0: ETDRK4 Numerical Solver Verification")

    # 2. Phase 1-4: Core Benchmark across 3 seeds
    run_command([python_bin, "scripts/run_experiment.py", "--config", args.config], "Phase 1-4: Core Benchmark Suite")

    # 3. Phase 5: Memory Size Ablation
    run_command([python_bin, "scripts/run_memory_ablation.py", "--config", args.config], "Phase 5: Memory Size Ablation")

    # 4. Phase 6: Generalization Suite (Tests A, B, C)
    run_command([python_bin, "scripts/run_generalization.py", "--config", args.config], "Phase 6: Physical Generalization Suite")

    # 5. Phase 7: K-Sensitivity Sweep
    run_command([python_bin, "scripts/run_k_sensitivity.py", "--config", args.config], "Phase 7: K-Sensitivity Study")

    # 6. Phase 8: Causal Memory Swapping
    run_command([python_bin, "scripts/run_memory_swapping.py", "--config", args.config], "Phase 8: Causal Memory Swapping")

    # 7. Phase 9: Multicore CPU Scaling
    run_command([python_bin, "scripts/benchmark_cpu.py", "--config", args.config], "Phase 9: Multicore CPU Scaling")

    # 8. Phase 10: Generate Comprehensive Report
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["paths"]["output_dir"])
    generate_automated_report(out_dir)

    print("\n=======================================================")
    print("=== FULL RESEARCH PROTOTYPE PIPELINE COMPLETE! ===")
    print("=======================================================")


if __name__ == "__main__":
    main()
