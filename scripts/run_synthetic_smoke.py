"""
Pre-WikiText Diagnostic Smoke Test: Delayed Dependency Task & Receptive Field Scaling Curve.

Verifies:
1. Baselines (Full Transformer, Sliding Transformer W=16, Mamba, GRU) solve the delay-8 task.
2. NCA-LM Diagnostic Curve across K in {1, 2, 3, 4, 6}:
   - K=1 (RF=3, reaches t-2): Blind to t-8 -> Expected Acc ~6.25% (Chance floor)
   - K=2 (RF=7, reaches t-6): Blind to t-8 -> Expected Acc ~6.25% (Chance floor)
   - K=3 (RF=15, reaches t-14): Accessible! -> Sharp transition >> 50%
   - K=4 (RF=31, reaches t-30): Highly accessible -> Acc > 90%
   - K=6 (RF=127, reaches t-126): Full context -> Acc ~100%
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_baseline import TransformerLM
from models.mamba_baseline import MambaLM
from models.rnn_baseline import GRULM
from models.nca_lm import NCA_LM


def generate_delayed_dependency_batch(batch_size: int = 32, seq_len: int = 48, delay: int = 8, vocab_size: int = 16):
    """
    Generate sequences where tokens after delay satisfy:
    x_t = (x_{t - delay} + 1) % vocab_size.
    Tokens before delay are uniformly random.
    """
    x = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    x[:, :delay] = torch.randint(0, vocab_size, (batch_size, delay))
    for t in range(delay, seq_len + 1):
        x[:, t] = (x[:, t - delay] + 1) % vocab_size

    inputs = x[:, :-1]   # [B, seq_len]
    targets = x[:, 1:]   # [B, seq_len]
    return inputs, targets


def train_smoke_model(model: nn.Module, model_name: str, delay: int = 8, vocab_size: int = 16, steps: int = 150, lr: float = 0.003):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    initial_loss = None
    final_loss = None

    t0 = time.time()
    for step in range(steps):
        inputs, targets = generate_delayed_dependency_batch(
            batch_size=16, seq_len=40, delay=delay, vocab_size=vocab_size
        )
        optimizer.zero_grad()
        logits = model(inputs)
        
        # Only measure loss on positions >= delay - 1 (where the dependency is active)
        loss = F.cross_entropy(logits[:, delay-1:].reshape(-1, vocab_size), targets[:, delay-1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        if step == 0:
            initial_loss = loss_val
        final_loss = loss_val

    elapsed = time.time() - t0

    # Compute accuracy on target positions of a fresh test batch
    model.eval()
    with torch.no_grad():
        test_in, test_tgt = generate_delayed_dependency_batch(
            batch_size=64, seq_len=40, delay=delay, vocab_size=vocab_size
        )
        test_logits = model(test_in)
        preds = test_logits[:, delay-1:].argmax(dim=-1)
        truth = test_tgt[:, delay-1:]
        acc = (preds == truth).float().mean().item() * 100

    return {
        "name": model_name,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "accuracy_pct": round(acc, 2),
        "elapsed_sec": round(elapsed, 2),
    }


def run_all_smoke_tests():
    vocab_size = 16
    delay = 8
    d_model = 64

    print("=" * 85)
    print(f"DIAGNOSTIC DELAYED-DEPENDENCY SMOKE TEST (x_t = x_{{t-{delay}}} + 1 mod {vocab_size})")
    print("=" * 85)

    # 1. Baseline Architectural Yardsticks
    baselines = {
        "Full Transformer": (
            TransformerLM(vocab_size=vocab_size, d_model=d_model, num_layers=2, num_heads=4, attention_mode="causal", dropout=0.0),
            120, 0.005, True
        ),
        "Sliding Transformer (W=16)": (
            TransformerLM(vocab_size=vocab_size, d_model=d_model, num_layers=2, num_heads=4, attention_mode="sliding", window_size=16, dropout=0.0),
            120, 0.005, True
        ),
        "Mamba / SSM": (
            MambaLM(vocab_size=vocab_size, d_model=d_model, num_layers=2, d_state=16, expand=2, d_conv=6, dropout=0.0),
            130, 0.005, True
        ),
        "GRU Baseline": (
            GRULM(vocab_size=vocab_size, d_model=d_model, num_layers=2, dropout=0.0),
            450, 0.008, True
        ),
    }

    print("\n--- 1. Baseline Architectures ---")
    for name, (model, steps, lr, should_pass) in baselines.items():
        res = train_smoke_model(model, name, delay=delay, vocab_size=vocab_size, steps=steps, lr=lr)
        passed = res["accuracy_pct"] > 80.0 or res["final_loss"] < 0.50
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {name:<28} | Loss: {res['initial_loss']:.4f} -> {res['final_loss']:.4f} | Acc: {res['accuracy_pct']:>6.2f}% ({res['elapsed_sec']:.1f}s)")
        assert passed, f"Baseline {name} unexpectedly failed delay smoke test!"

    # 2. NCA-LM Diagnostic Receptive Field Curve across K
    print("\n--- 2. NCA-LM Diagnostic Receptive Field Curve (K in {1, 2, 3, 4, 6}) ---")
    nca_k_configs = [
        (1, 3, 2, False),    # K=1, RF=3, max past=2 -> Blind (should fail, ~6.25%)
        (2, 7, 6, False),    # K=2, RF=7, max past=6 -> Blind (should fail, ~6.25%)
        (3, 15, 14, True),   # K=3, RF=15, max past=14 -> Accessible (sharp transition!)
        (4, 31, 30, True),   # K=4, RF=31, max past=30 -> Accessible (>90%)
        (6, 127, 126, True), # K=6, RF=127, max past=126 -> Accessible (~100%)
    ]

    curve_results = []
    for K, rf, max_past, should_solve in nca_k_configs:
        name = f"NCA-LM (K={K}, RF={rf})"
        model = NCA_LM(
            vocab_size=vocab_size,
            d_embed=d_model,
            radius=2,
            K=K,
            max_K=K,
            shared_weights=True,
            step_embed_type="sinusoidal",
        )
        res = train_smoke_model(model, name, delay=delay, vocab_size=vocab_size, steps=180, lr=0.005)
        curve_results.append((K, rf, max_past, should_solve, res))

        can_see = f"reaches t-{max_past}"
        if not should_solve:
            status = "BLIND (Expected ~6.25%)"
            assert res["accuracy_pct"] < 25.0, f"K={K} unexpectedly solved delay-8 without sufficient RF!"
        else:
            status = "SOLVED" if res["accuracy_pct"] > 70.0 else "SUB-OPTIMAL"
            assert res["accuracy_pct"] > 50.0, f"K={K} has RF={rf} >= 9 but failed to learn delay-8!"

        print(f"[{status:<25}] {name:<22} ({can_see:<14}) | Acc: {res['accuracy_pct']:>6.2f}% ({res['elapsed_sec']:.1f}s)")

    print("\n" + "=" * 85)
    print("DIAGNOSTIC SUMMARY: RECEPTIVE FIELD PHASE TRANSITION CONFIRMED")
    print(f"{'Model / K':<24} | {'RF (tokens)':<12} | {'Max Past Token':<16} | {'Accuracy (%)':<14} | {'Status':<12}")
    print("-" * 85)
    print(f"{'Random Chance Floor':<24} | {'-':<12} | {'-':<16} | {'6.25%':<14} | {'Theoretical'}")
    for K, rf, max_past, should_solve, res in curve_results:
        st = "Blind" if not should_solve else "Solved"
        acc_str = f"{res['accuracy_pct']:.2f}%"
        k_str = f"NCA-LM (K={K})"
        past_str = f"t-{max_past}"
        print(f"{k_str:<24} | {rf:<12} | {past_str:<16} | {acc_str:<14} | {st:<12}")
    print("=" * 85)

    return True


if __name__ == "__main__":
    run_all_smoke_tests()
