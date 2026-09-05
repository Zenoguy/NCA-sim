"""
Phase 2 Driver: 2x2 Factorial Matrix Evaluation & Comparison Table.

Evaluates the four Phase 2 models:
- Variant A: Shared NCA (d=288, K=6, ~3.6M)
- Variant B: Unshared CNN (d=160, K=6, ~3.6M)
- Variant C: Unshared CNN (d=288, K=6, ~9.8M)
- Variant D: Shared NCA (d=576, K=6, ~9.7M)

Computes:
1. Pure Weight-Sharing Penalty: Variant A vs Variant B (at ~3.6M budget)
2. Capacity Compensation Effect: Variant D vs Variant C (at ~10M budget)
3. Width Effect: Variant A vs Variant C (identical width d=288)
4. Evaluates Gate 1 (Autoregressive LM viability vs n-gram floor 89.56) and Gate 2.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


PHASE2_MODELS = [
    {
        "key": "variant_a_shared_3m",
        "name": "Variant A (Shared 3.6M, d=288)",
        "config": "configs/level2_nca_shared_3m.yaml",
        "summary_dir": "outputs/level2/nca_shared_3m",
        "role": "Baseline cellular rule",
    },
    {
        "key": "variant_b_unshared_3m",
        "name": "Variant B (Unshared 3.6M, d=160)",
        "config": "configs/level2_nca_unshared_3m.yaml",
        "summary_dir": "outputs/level2/nca_unshared_3m",
        "role": "Pure sharing control (equal params)",
    },
    {
        "key": "variant_c_unshared_10m",
        "name": "Variant C (Unshared 9.8M, d=288)",
        "config": "configs/level2_nca_unshared_10m.yaml",
        "summary_dir": "outputs/level2/nca_unshared_10m",
        "role": "Width-matched control (d=288)",
    },
    {
        "key": "variant_d_shared_10m",
        "name": "Variant D (Shared 9.7M, d=576)",
        "config": "configs/level2_nca_shared_10m.yaml",
        "summary_dir": "outputs/level2/nca_shared_10m",
        "role": "Capacity compensation control",
    },
]


def build_factorial_table(output_file="outputs/level2/factorial_matrix.json"):
    profiles_path = Path("outputs/level2/model_profiles.json")
    profiles = {}
    if profiles_path.exists():
        with open(profiles_path, "r") as f:
            profiles = json.load(f)

    # Also load Phase 1 sliding window baseline for reference
    phase1_path = Path("outputs/level1/calibration_table.json")
    sliding_ref = None
    if phase1_path.exists():
        with open(phase1_path, "r") as f:
            p1_data = json.load(f)
            sliding_ref = p1_data.get("models", {}).get("sliding_transformer", {})

    matrix_data = {
        "description": "Phase 2 2x2 Factorial Matrix: Weight-Sharing and Capacity Compensation",
        "reference_floors": {
            "3_gram_test_ppl": 89.56,
            "5_gram_test_ppl": 99.40,
            "sliding_transformer_10m_test_ppl": sliding_ref.get("test_ppl") if sliding_ref else None,
        },
        "models": {},
    }

    rows = []
    ppl_map = {}

    for model_info in PHASE2_MODELS:
        key = model_info["key"]
        name = model_info["name"]
        prof = profiles.get(key, {})

        summary_file = Path(model_info["summary_dir"]) / "training_summary.json"
        trained = summary_file.exists()
        summary = {}
        if trained:
            with open(summary_file, "r") as f:
                summary = json.load(f)
        elif Path(output_file).exists():
            try:
                with open(output_file, "r") as f:
                    cached = json.load(f).get("models", {}).get(key, {})
                    if cached.get("trained"):
                        trained = True
                        summary = {
                            "best_val_perplexity": cached.get("val_ppl"),
                            "test_perplexity": cached.get("test_ppl"),
                            "test_loss": cached.get("test_loss"),
                            "total_parameters": cached.get("total_params"),
                        }
            except Exception:
                pass

        params_total = prof.get("total_params", summary.get("total_parameters", 0))
        params_str = f"{params_total / 1e6:.2f}M" if params_total else "N/A"
        flops_fwd = prof.get("forward_flops_per_token", 0)
        flops_str = f"{flops_fwd / 1e6:.1f}M" if flops_fwd else "N/A"
        train_tok_s = prof.get("train_tokens_per_sec", 0.0)
        infer_tok_s = prof.get("infer_tokens_per_sec", 0.0)

        val_ppl = summary.get("best_val_perplexity", None)
        test_ppl = summary.get("test_perplexity", None)
        test_loss = summary.get("test_loss", None)

        ppl_map[key] = test_ppl

        val_ppl_str = f"{val_ppl:.2f}" if val_ppl is not None else "[Pending]"
        test_ppl_str = f"{test_ppl:.2f}" if test_ppl is not None else "[Pending]"
        test_loss_str = f"{test_loss:.4f}" if test_loss is not None else "[Pending]"

        matrix_data["models"][key] = {
            "name": name,
            "config": model_info["config"],
            "role": model_info["role"],
            "total_params": params_total,
            "forward_flops_per_token": flops_fwd,
            "train_tokens_per_sec": train_tok_s,
            "infer_tokens_per_sec": infer_tok_s,
            "trained": trained,
            "val_ppl": val_ppl,
            "test_ppl": test_ppl,
            "test_loss": test_loss,
        }

        rows.append({
            "name": name,
            "role": model_info["role"],
            "params": params_str,
            "val_ppl": val_ppl_str,
            "test_ppl": test_ppl_str,
            "train_throughput": f"{train_tok_s:.0f}" if train_tok_s else "N/A",
            "flops_tok": flops_str,
        })

    # Compute comparative deltas if trained
    deltas = {}
    p_a = ppl_map.get("variant_a_shared_3m")
    p_b = ppl_map.get("variant_b_unshared_3m")
    p_c = ppl_map.get("variant_c_unshared_10m")
    p_d = ppl_map.get("variant_d_shared_10m")

    if p_a is not None and p_b is not None:
        deltas["pure_weight_sharing_penalty_3m"] = round(p_a - p_b, 2)
    if p_d is not None and p_c is not None:
        deltas["capacity_compensation_delta_10m"] = round(p_d - p_c, 2)
    if p_a is not None and p_c is not None:
        deltas["width_effect_delta_w288"] = round(p_a - p_c, 2)

    matrix_data["comparative_deltas"] = deltas

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(matrix_data, f, indent=2)

    # Print Formatted Markdown Table
    print("\n" + "=" * 115)
    print("PHASE 2: 2x2 FACTORIAL MATRIX (WEIGHT SHARING & CAPACITY COMPENSATION)")
    print("=" * 115)
    header = f"| {'Variant':<34} | {'Role':<32} | {'Params':<8} | {'Val PPL':<10} | {'Test PPL':<10} | {'FLOPs/tok':<10} |"
    sep = f"|{'-'*36}|{'-'*34}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*12}|"
    print(header)
    print(sep)
    for r in rows:
        print(f"| {r['name']:<34} | {r['role']:<32} | {r['params']:<8} | {r['val_ppl']:<10} | {r['test_ppl']:<10} | {r['flops_tok']:<10} |")
    print(sep)
    print("Baseline References:")
    print("  - Kneser-Ney 3-Gram Test PPL: 89.56 (Empirical Floor)")
    print("  - Sliding Transformer (W=128, ~10M): [Context-matched Control]")
    if deltas:
        print("\nComparative Findings:")
        for k, v in deltas.items():
            print(f"  - {k}: {v:+.2f} PPL")
    print(f"\nSaved matrix data to: {out_path}\n")

    return matrix_data


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Factorial Matrix Driver")
    parser.add_argument("--action", choices=["table", "profile", "smoke"], default="table")
    args = parser.parse_args()

    if args.action == "smoke":
        from scripts.run_synthetic_smoke import run_all_smoke_tests
        run_all_smoke_tests()
    elif args.action == "profile":
        from scripts.profile_models import run_profiling
        run_profiling(phase="2")
    elif args.action == "table":
        build_factorial_table()


if __name__ == "__main__":
    main()
