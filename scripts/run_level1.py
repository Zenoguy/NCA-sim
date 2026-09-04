"""
Phase 1 Calibration Runner & Benchmark Table Generator.

Aggregates profiling results (params, FLOPs/tok, activation memory, throughput)
and training metrics (Val PPL, Test PPL, Test Loss) across the four Phase 1 baselines:
1. Primary Transformer (Full Causal Attention Reference)
2. Sliding Transformer (W=128 Control)
3. Mamba (Pure-PyTorch Selective SSM Reference)
4. GRU Baseline

Saves outputs to outputs/level1/calibration_table.json and prints markdown table for README.md.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


BASELINE_MODELS = [
    {
        "key": "gru",
        "name": "GRU Baseline",
        "config": "configs/level1_gru.yaml",
        "profile_key": "gru",
        "summary_dir": "outputs/level1/gru",
    },
    {
        "key": "primary_transformer",
        "name": "Primary Transformer",
        "config": "configs/level1_transformer.yaml",
        "profile_key": "primary_transformer",
        "summary_dir": "outputs/level1/transformer",
    },
    {
        "key": "sliding_transformer",
        "name": "Sliding Transformer (W=128)",
        "config": "configs/level1_transformer_sliding.yaml",
        "profile_key": "sliding_transformer",
        "summary_dir": "outputs/level1/transformer_sliding",
    },
    {
        "key": "mamba",
        "name": "Mamba (Selective SSM)",
        "config": "configs/level1_mamba.yaml",
        "profile_key": "mamba",
        "summary_dir": "outputs/level1/mamba",
    },
]


def build_calibration_table(output_file="outputs/level1/calibration_table.json"):
    profiles_path = Path("outputs/level1/model_profiles.json")
    profiles = {}
    if profiles_path.exists():
        with open(profiles_path, "r") as f:
            profiles = json.load(f)

    calibration_data = {
        "description": "Phase 1 Frozen Calibration Table (10M parameter budget, WikiText-2)",
        "ngram_floor": {
            "3_gram_test_ppl": 89.56,
            "5_gram_test_ppl": 99.40,
        },
        "models": {},
    }

    rows = []
    for model_info in BASELINE_MODELS:
        key = model_info["key"]
        name = model_info["name"]
        prof = profiles.get(model_info["profile_key"], {})

        summary_file = Path(model_info["summary_dir"]) / "training_summary.json"
        trained = summary_file.exists()
        summary = {}
        if trained:
            with open(summary_file, "r") as f:
                summary = json.load(f)

        params_total = prof.get("total_params", summary.get("total_parameters", 0))
        params_str = f"{params_total / 1e6:.2f}M" if params_total else "N/A"
        flops_fwd = prof.get("forward_flops_per_token", 0)
        flops_str = f"{flops_fwd / 1e6:.1f}M" if flops_fwd else "N/A"
        train_tok_s = prof.get("train_tokens_per_sec", 0.0)
        infer_tok_s = prof.get("infer_tokens_per_sec", 0.0)

        val_ppl = summary.get("best_val_perplexity", None)
        test_ppl = summary.get("test_perplexity", None)
        test_loss = summary.get("test_loss", None)

        val_ppl_str = f"{val_ppl:.2f}" if val_ppl is not None else "[Pending]"
        test_ppl_str = f"{test_ppl:.2f}" if test_ppl is not None else "[Pending]"
        test_loss_str = f"{test_loss:.4f}" if test_loss is not None else "[Pending]"

        model_entry = {
            "name": name,
            "config": model_info["config"],
            "total_params": params_total,
            "forward_flops_per_token": flops_fwd,
            "train_tokens_per_sec": train_tok_s,
            "infer_tokens_per_sec": infer_tok_s,
            "trained": trained,
            "val_ppl": val_ppl,
            "test_ppl": test_ppl,
            "test_loss": test_loss,
        }
        calibration_data["models"][key] = model_entry

        rows.append({
            "name": name,
            "params": params_str,
            "val_ppl": val_ppl_str,
            "test_ppl": test_ppl_str,
            "test_loss": test_loss_str,
            "train_throughput": f"{train_tok_s:.0f}" if train_tok_s else "N/A",
            "infer_throughput": f"{infer_tok_s:.0f}" if infer_tok_s else "N/A",
            "flops_tok": flops_str,
        })

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calibration_data, f, indent=2)

    # Print Markdown Table
    print("\n" + "=" * 105)
    print("PHASE 1 FROZEN CALIBRATION TABLE")
    print("=" * 105)
    header = f"| {'Model':<30} | {'Params':<8} | {'Val PPL':<10} | {'Test PPL':<10} | {'Test Loss':<10} | {'Train tok/s':<11} | {'FLOPs/tok':<10} |"
    sep = f"|{'-'*32}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*13}|{'-'*12}|"
    print(header)
    print(sep)
    for r in rows:
        print(f"| {r['name']:<30} | {r['params']:<8} | {r['val_ppl']:<10} | {r['test_ppl']:<10} | {r['test_loss']:<10} | {r['train_throughput']:<11} | {r['flops_tok']:<10} |")
    print(sep)
    print("Floor Reference: Kneser-Ney 3-Gram Test PPL = 89.56 | 5-Gram Test PPL = 99.40")
    print(f"Saved calibration data to: {out_path}\n")

    return calibration_data


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Benchmark & Calibration Driver")
    parser.add_argument("--action", choices=["table", "profile", "smoke"], default="table")
    args = parser.parse_args()

    if args.action == "smoke":
        from scripts.run_synthetic_smoke import run_smoke_test
        run_smoke_test()
    elif args.action == "profile":
        from scripts.profile_models import profile_all_models
        profile_all_models()
    elif args.action == "table":
        build_calibration_table()


if __name__ == "__main__":
    main()
