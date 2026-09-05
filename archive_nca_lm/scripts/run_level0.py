"""
End-to-End Driver for Phase 0 / Level 0 Baseline.
1. Downloads WikiText-2.
2. Trains and serializes the 8k ByteLevel BPE Tokenizer.
3. Encodes and caches train, valid, and test token arrays.
4. Fits and evaluates 3-gram and 5-gram Kneser-Ney models.
5. Saves results to outputs/level0/metrics.json.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import yaml
from data.prepare_wikitext2 import prepare_wikitext2
from data.tokenizer import BPETokenizerWrapper
from models.ngram import NGramLanguageModel


def run_level0(config_path: str = "configs/level0_ngram.yaml"):
    start_time = time.time()
    print("=" * 70)
    print("STARTING PHASE 0: DATASET PREP, TOKENIZATION & N-GRAM FLOOR")
    print("=" * 70)

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Download WikiText-2
    raw_dir = cfg["dataset"]["raw_dir"]
    splits = prepare_wikitext2(raw_dir)

    # 2. Train or Load Tokenizer
    tok_cfg = cfg["tokenizer"]
    tok_path = Path(tok_cfg["save_path"])
    vocab_size = tok_cfg["vocab_size"]

    if tok_path.exists():
        print(f"Loading existing tokenizer from {tok_path}...")
        tokenizer = BPETokenizerWrapper(tok_path)
    else:
        print(f"Training new {vocab_size}-vocab tokenizer on {splits['train']}...")
        tokenizer = BPETokenizerWrapper.train_from_files(
            files=[splits["train"]],
            vocab_size=vocab_size,
            min_frequency=tok_cfg.get("min_frequency", 2),
            save_path=tok_path,
        )

    # 3. Encode & Cache Token Splits
    token_arrays = {}
    for split_name, file_path in splits.items():
        arr = tokenizer.encode_file(file_path)
        token_arrays[split_name] = arr
        print(f"Split [{split_name}]: {len(arr):,} tokens.")

    # 4. Train and Evaluate N-Gram Baselines
    ngram_cfg = cfg["ngram"]
    orders = ngram_cfg.get("orders", [3, 5])
    discount = ngram_cfg.get("discount", 0.75)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocab_size": tokenizer.vocab_size,
        "token_counts": {k: len(v) for k, v in token_arrays.items()},
        "models": {},
    }

    print("\n" + "=" * 70)
    print("TRAINING & EVALUATING N-GRAM BASELINES")
    print("=" * 70)

    for order in orders:
        print(f"\n--- Order {order}-Gram (Discount = {discount}) ---")
        model = NGramLanguageModel(n=order, discount=discount, vocab_size=tokenizer.vocab_size)
        
        t0 = time.time()
        model.fit(token_arrays["train"])
        fit_time = time.time() - t0

        t0 = time.time()
        val_metrics = model.evaluate(token_arrays["valid"])
        val_time = time.time() - t0

        t0 = time.time()
        test_metrics = model.evaluate(token_arrays["test"])
        test_time = time.time() - t0

        print(f"Order-{order} Val  Loss: {val_metrics['loss']:.4f} | Perplexity: {val_metrics['perplexity']:.2f} ({val_time:.1f}s)")
        print(f"Order-{order} Test Loss: {test_metrics['loss']:.4f} | Perplexity: {test_metrics['perplexity']:.2f} ({test_time:.1f}s)")

        results["models"][f"{order}_gram"] = {
            "order": order,
            "discount": discount,
            "fit_time_sec": round(fit_time, 2),
            "val": val_metrics,
            "test": test_metrics,
        }

    # 5. Save Results
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(cfg["output"]["metrics_file"])
    
    total_time = time.time() - start_time
    results["total_elapsed_sec"] = round(total_time, 2)

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PHASE 0 COMPLETED IN {total_time:.1f}s — METRICS SAVED TO {out_file}")
    print("=" * 70)
    print(f"{'Model':<15} | {'Val Loss':<10} | {'Val PPL':<10} | {'Test Loss':<10} | {'Test PPL':<10}")
    print("-" * 65)
    for name, res in results["models"].items():
        print(f"{name:<15} | {res['val']['loss']:<10.4f} | {res['val']['perplexity']:<10.2f} | {res['test']['loss']:<10.4f} | {res['test']['perplexity']:<10.2f}")
    print("-" * 65)

    return results


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/level0_ngram.yaml"
    run_level0(config)
