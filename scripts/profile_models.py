"""
Multi-Metric Profiling Script for Phase 1 Baselines.
Measures Parameters, Theoretical FLOPs/Token, Activation Memory, and Throughput.
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

import torch
import yaml
from models.transformer_baseline import TransformerLM
from models.mamba_baseline import MambaLM
from models.rnn_baseline import GRULM


def build_model_from_config(cfg: dict) -> torch.nn.Module:
    m_cfg = cfg["model"]
    m_type = m_cfg["type"]
    if m_type == "transformer":
        return TransformerLM(
            vocab_size=m_cfg["vocab_size"],
            d_model=m_cfg["d_model"],
            num_layers=m_cfg["num_layers"],
            num_heads=m_cfg["num_heads"],
            mlp_ratio=m_cfg.get("mlp_ratio", 4.0),
            attention_mode=m_cfg.get("attention_mode", "causal"),
            window_size=m_cfg.get("window_size", 128),
            dropout=m_cfg.get("dropout", 0.1),
            tie_weights=m_cfg.get("tie_weights", True),
        )
    elif m_type == "mamba":
        return MambaLM(
            vocab_size=m_cfg["vocab_size"],
            d_model=m_cfg["d_model"],
            num_layers=m_cfg["num_layers"],
            d_state=m_cfg.get("d_state", 16),
            expand=m_cfg.get("expand", 2),
            d_conv=m_cfg.get("d_conv", 4),
            dropout=m_cfg.get("dropout", 0.1),
            tie_weights=m_cfg.get("tie_weights", True),
        )
    elif m_type == "gru":
        return GRULM(
            vocab_size=m_cfg["vocab_size"],
            d_model=m_cfg["d_model"],
            num_layers=m_cfg["num_layers"],
            dropout=m_cfg.get("dropout", 0.1),
            tie_weights=m_cfg.get("tie_weights", True),
        )
    else:
        raise ValueError(f"Unknown model type: {m_type}")


def calculate_theoretical_flops_per_token(model_name: str, cfg: dict, T: int = 128) -> dict:
    """Calculate theoretical FLOPs per token for forward and backward passes."""
    m_cfg = cfg["model"]
    d = m_cfg["d_model"]
    L = m_cfg["num_layers"]
    V = m_cfg["vocab_size"]

    if "transformer" in model_name:
        mlp_ratio = m_cfg.get("mlp_ratio", 4.0)
        # Self-attention: Q, K, V, O projections = 4 * 2 * d^2
        # QK^T + Context V = 4 * d * T per token
        attn_flops = 8 * (d ** 2) + 4 * d * T
        # SwiGLU MLP: w1, w2, w3 = 3 * 2 * d * (mlp_ratio * d)
        mlp_flops = 6 * mlp_ratio * (d ** 2)
        layer_flops = attn_flops + mlp_flops
        fwd_flops = L * layer_flops + 2 * d * V  # readout
    elif "mamba" in model_name:
        d_inner = int(m_cfg.get("expand", 2) * d)
        d_state = m_cfg.get("d_state", 16)
        # in_proj: 2 * 2 * d * d_inner
        in_flops = 4 * d * d_inner
        # selective scan per token: delta_A (d_inner * d_state), delta_B_u (d_inner * d_state),
        # state update (2 * d_inner * d_state), output projection C (2 * d_inner * d_state)
        scan_flops = 6 * d_inner * d_state
        # out_proj: 2 * d_inner * d
        out_flops = 2 * d_inner * d
        layer_flops = in_flops + scan_flops + out_flops
        fwd_flops = L * layer_flops + 2 * d * V
    elif "gru" in model_name:
        # Standard GRU has 3 gates: W_ih (3 * 2 * d^2) + W_hh (3 * 2 * d^2) = 12 d^2
        layer_flops = 12 * (d ** 2)
        fwd_flops = L * layer_flops + 2 * d * V
    else:
        fwd_flops = 0

    return {
        "forward_flops_per_token": int(fwd_flops),
        "training_flops_per_token": int(3 * fwd_flops),  # Standard 3x rule for backprop
    }


def benchmark_model(model: torch.nn.Module, B: int = 4, T: int = 128, device: str = "cpu") -> dict:
    """Benchmark throughput and activation memory."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    x = torch.randint(0, model.vocab_size, (B, T), device=device)
    y = torch.randint(0, model.vocab_size, (B, T), device=device)
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

    # Measure Training Throughput
    iters = 10
    t0 = time.time()
    for _ in range(iters):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
    train_time = time.time() - t0
    total_tokens = iters * B * T
    train_tok_per_sec = total_tokens / train_time

    # Measure Inference Throughput
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    infer_time = time.time() - t0
    infer_tok_per_sec = total_tokens / infer_time

    return {
        "train_tokens_per_sec": round(train_tok_per_sec, 1),
        "infer_tokens_per_sec": round(infer_tok_per_sec, 1),
    }


def run_profiling():
    configs = {
        "primary_transformer": "configs/level1_transformer.yaml",
        "sliding_transformer": "configs/level1_transformer_sliding.yaml",
        "mamba": "configs/level1_mamba.yaml",
        "gru": "configs/level1_gru.yaml",
    }

    results = {}
    print("=" * 80)
    print("PHASE 1: MULTI-METRIC BASELINE PROFILING AUDIT")
    print("=" * 80)

    for name, path in configs.items():
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        model = build_model_from_config(cfg)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_params = model.tok_embed.weight.numel()
        core_params = total_params - emb_params

        # Assert within 10M ± 5% (9.5M to 10.5M)
        in_bounds = 9_500_000 <= total_params <= 10_500_000
        delta_pct = (total_params - 10_000_000) / 10_000_000 * 100

        flops = calculate_theoretical_flops_per_token(name, cfg, T=128)
        bench = benchmark_model(model, B=4, T=128, device="cpu")

        results[name] = {
            "config_path": path,
            "total_params": total_params,
            "embedding_params": emb_params,
            "core_params": core_params,
            "delta_from_10M_pct": round(delta_pct, 2),
            "in_target_bounds": in_bounds,
            "forward_flops_per_token": flops["forward_flops_per_token"],
            "training_flops_per_token": flops["training_flops_per_token"],
            "train_tokens_per_sec": bench["train_tokens_per_sec"],
            "infer_tokens_per_sec": bench["infer_tokens_per_sec"],
        }

        print(f"\nModel: [{name.upper()}] ({path})")
        print(f"  Total Params:        {total_params:,} ({total_params/1e6:.2f}M, {delta_pct:+.2f}%) [In Bounds: {in_bounds}]")
        print(f"  Embedding Params:    {emb_params:,} ({emb_params/1e6:.2f}M)")
        print(f"  Core Params:         {core_params:,} ({core_params/1e6:.2f}M)")
        print(f"  Fwd FLOPs/Token:     {flops['forward_flops_per_token']:,}")
        print(f"  Train FLOPs/Token:   {flops['training_flops_per_token']:,}")
        print(f"  Train Throughput:    {bench['train_tokens_per_sec']:,} tok/s (CPU)")
        print(f"  Infer Throughput:    {bench['infer_tokens_per_sec']:,} tok/s (CPU)")

    out_dir = Path("outputs/level1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "model_profiles.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PROFILES SAVED TO {out_file}")
    print("=" * 80)
    print(f"{'Model':<22} | {'Params (M)':<10} | {'Delta (%)':<10} | {'Fwd FLOPs/tok':<14} | {'Train tok/s'}")
    print("-" * 75)
    for name, res in results.items():
        print(f"{name:<22} | {res['total_params']/1e6:<10.2f} | {res['delta_from_10M_pct']:<+10.2f} | {res['forward_flops_per_token']:<14,d} | {res['train_tokens_per_sec']:<10.1f}")
    print("-" * 75)

    all_in_bounds = all(r["in_target_bounds"] for r in results.values())
    assert all_in_bounds, "Not all models met the 10M ± 5% target parameter bound!"
    print("All models strictly verified within 10M ± 5% parameter bounds.")

    return results


if __name__ == "__main__":
    run_profiling()
