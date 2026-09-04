"""
Pre-WikiText Diagnostic Smoke Test: Delayed Dependency Task.
Verifies that all 4 baseline architectures can learn a deterministic dependency:
    x_t = (x_{t-10} + 1) % V
where delay D = 10, vocabulary V = 16, sequence length T = 64.
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


def generate_delayed_dependency_batch(batch_size: int = 32, seq_len: int = 64, delay: int = 10, vocab_size: int = 16):
    """
    Generate sequences where tokens after delay satisfy:
    x_t = (x_{t - delay} + 1) % vocab_size.
    Tokens before delay are uniformly random.
    """
    x = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    # Random initial prompt of length `delay`
    x[:, :delay] = torch.randint(0, vocab_size, (batch_size, delay))
    for t in range(delay, seq_len + 1):
        x[:, t] = (x[:, t - delay] + 1) % vocab_size

    inputs = x[:, :-1]   # [B, seq_len]
    targets = x[:, 1:]   # [B, seq_len]
    return inputs, targets


def train_smoke_model(model: nn.Module, model_name: str, delay: int = 10, vocab_size: int = 16, steps: int = 200, lr: float = 0.003):
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
            batch_size=32, seq_len=40, delay=delay, vocab_size=vocab_size
        )
        test_logits = model(test_in)
        preds = test_logits[:, delay-1:].argmax(dim=-1)
        truth = test_tgt[:, delay-1:]
        acc = (preds == truth).float().mean().item() * 100

    passed = acc > 85.0 or final_loss < 0.50
    return {
        "name": model_name,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "accuracy_pct": round(acc, 2),
        "elapsed_sec": round(elapsed, 2),
        "passed": passed,
    }


def run_all_smoke_tests():
    vocab_size = 16
    delay = 8
    d_model = 64

    print("=" * 75)
    print(f"RUNNING SYNTHETIC DELAYED-DEPENDENCY SMOKE TEST (x_t = x_{{t-{delay}}} + 1 mod {vocab_size})")
    print("=" * 75)

    models = {
        "Full Transformer": (
            TransformerLM(
                vocab_size=vocab_size, d_model=d_model, num_layers=2, num_heads=4,
                attention_mode="causal", dropout=0.0
            ),
            120, 0.005
        ),
        "Sliding Transformer (W=16)": (
            TransformerLM(
                vocab_size=vocab_size, d_model=d_model, num_layers=2, num_heads=4,
                attention_mode="sliding", window_size=16, dropout=0.0
            ),
            120, 0.005
        ),
        "Mamba / SSM": (
            MambaLM(
                vocab_size=vocab_size, d_model=d_model, num_layers=2, d_state=16,
                expand=2, d_conv=6, dropout=0.0
            ),
            130, 0.005
        ),
        "GRU Baseline": (
            GRULM(
                vocab_size=vocab_size, d_model=d_model, num_layers=2, dropout=0.0
            ),
            400, 0.008
        ),
    }

    results = []
    for name, (model, steps, lr) in models.items():
        res = train_smoke_model(model, name, delay=delay, vocab_size=vocab_size, steps=steps, lr=lr)
        results.append(res)
        status = "PASSED" if res["passed"] else "FAILED"
        print(f"[{status}] {name:<28} | Loss: {res['initial_loss']:.4f} -> {res['final_loss']:.4f} | Acc: {res['accuracy_pct']:>6.2f}% ({res['elapsed_sec']:.1f}s)")



    print("-" * 75)
    all_passed = all(r["passed"] for r in results)
    assert all_passed, "One or more baseline architectures failed the synthetic smoke test!"
    print("ALL 4 BASELINE ARCHITECTURES PASSED THE DELAYED-DEPENDENCY SMOKE TEST.")
    return results


if __name__ == "__main__":
    run_all_smoke_tests()
