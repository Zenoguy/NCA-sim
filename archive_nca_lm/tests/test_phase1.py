"""
Phase 1 Unit Test Suite.
Tests causal masking numerical tolerance, sliding-window boundary conditions,
and same-seed determinism across all baseline architectures.
"""

import math
import pytest
import torch

from models.transformer_baseline import TransformerLM
from models.mamba_baseline import MambaLM
from models.rnn_baseline import GRULM


def test_transformer_causality():
    """Verify that future tokens do not affect past token logits in Full Transformer."""
    torch.manual_seed(42)
    vocab_size = 200
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        attention_mode="causal",
        dropout=0.0,
    )
    model.eval()

    T = 32
    x1 = torch.randint(0, vocab_size, (1, T))
    x2 = x1.clone()
    
    # Modify token at index 20
    changed_idx = 20
    x2[0, changed_idx] = (x1[0, changed_idx] + 7) % vocab_size

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Output before changed_idx must be identical within float tolerance
    max_diff_before = (out1[0, :changed_idx] - out2[0, :changed_idx]).abs().max().item()
    assert max_diff_before < 1e-5, f"Transformer causal leakage before index {changed_idx}: {max_diff_before}"

    # Output at and after changed_idx should differ
    max_diff_after = (out1[0, changed_idx:] - out2[0, changed_idx:]).abs().max().item()
    assert max_diff_after > 1e-3, "Perturbation had no effect downstream"


def test_sliding_window_boundaries():
    """
    Rigorously test the sliding window attention boundary across multiple offsets:
    Window W = 8.
    Target query position t = 20.
    Allowed keys: [t - W + 1, ..., t] = [20 - 8 + 1, 20] = [13, 20].
    Forbidden keys: <= 12 and > 20.
    """
    torch.manual_seed(42)
    vocab_size = 200
    W = 8
    t = 20
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=1,  # 1 layer isolates single attention receptive field W
        num_heads=4,
        attention_mode="sliding",
        window_size=W,
        dropout=0.0,
    )
    model.eval()

    T = 32
    base_x = torch.randint(0, vocab_size, (1, T))

    with torch.no_grad():
        base_logits = model(base_x)

    # 1. Test position (t - W + 1) = 13 (Inside window): MUST affect logits at t
    x_inside = base_x.clone()
    x_inside[0, t - W + 1] = (base_x[0, t - W + 1] + 5) % vocab_size
    with torch.no_grad():
        logits_inside = model(x_inside)
    diff_inside = (logits_inside[0, t] - base_logits[0, t]).abs().max().item()
    assert diff_inside > 1e-4, f"Token at inside boundary t-W+1 ({t - W + 1}) had no effect at t={t}!"

    # 2. Test position (t - W) = 12 (Outside window): MUST NOT affect logits at t
    x_boundary = base_x.clone()
    x_boundary[0, t - W] = (base_x[0, t - W] + 5) % vocab_size
    with torch.no_grad():
        logits_boundary = model(x_boundary)
    diff_boundary = (logits_boundary[0, t] - base_logits[0, t]).abs().max().item()
    assert diff_boundary < 1e-5, f"Token at boundary t-W ({t - W}) leaked into t={t}! Diff: {diff_boundary}"

    # 3. Test position (t - W - 1) = 11 (Outside window): MUST NOT affect logits at t
    x_out1 = base_x.clone()
    x_out1[0, t - W - 1] = (base_x[0, t - W - 1] + 5) % vocab_size
    with torch.no_grad():
        logits_out1 = model(x_out1)
    diff_out1 = (logits_out1[0, t] - base_logits[0, t]).abs().max().item()
    assert diff_out1 < 1e-5, f"Token at t-W-1 ({t - W - 1}) leaked into t={t}! Diff: {diff_out1}"

    # 4. Test position (t - W - 2) = 10 (Outside window): MUST NOT affect logits at t
    x_out2 = base_x.clone()
    x_out2[0, t - W - 2] = (base_x[0, t - W - 2] + 5) % vocab_size
    with torch.no_grad():
        logits_out2 = model(x_out2)
    diff_out2 = (logits_out2[0, t] - base_logits[0, t]).abs().max().item()
    assert diff_out2 < 1e-5, f"Token at t-W-2 ({t - W - 2}) leaked into t={t}! Diff: {diff_out2}"


def test_gru_causality():
    """Verify that future tokens do not affect past token logits in GRU."""
    torch.manual_seed(42)
    vocab_size = 200
    model = GRULM(vocab_size=vocab_size, d_model=64, num_layers=2, dropout=0.0)
    model.eval()

    T = 24
    x1 = torch.randint(0, vocab_size, (1, T))
    x2 = x1.clone()
    changed_idx = 16
    x2[0, changed_idx] = (x1[0, changed_idx] + 3) % vocab_size

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    max_diff_before = (out1[0, :changed_idx] - out2[0, :changed_idx]).abs().max().item()
    assert max_diff_before < 1e-5, f"GRU causal leakage before index {changed_idx}: {max_diff_before}"

    max_diff_after = (out1[0, changed_idx:] - out2[0, changed_idx:]).abs().max().item()
    assert max_diff_after > 1e-3, "GRU perturbation had no effect downstream"


def test_determinism_same_seed_same_result():
    """
    Critical Reproducibility Assertion:
    Running the exact same model with the exact same seed produces identical logits within 1e-6.
    """
    vocab_size = 100
    d_model = 64
    x = torch.randint(0, vocab_size, (2, 16))

    # Test Transformer
    torch.manual_seed(1234)
    m1 = TransformerLM(vocab_size, d_model=d_model, num_layers=2, num_heads=4)
    m1.eval()
    out1 = m1(x)

    torch.manual_seed(1234)
    m2 = TransformerLM(vocab_size, d_model=d_model, num_layers=2, num_heads=4)
    m2.eval()
    out2 = m2(x)

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"Transformer non-deterministic under same seed: diff={diff}"

    # Test Mamba
    torch.manual_seed(1234)
    mb1 = MambaLM(vocab_size, d_model=d_model, num_layers=2)
    mb1.eval()
    out_mb1 = mb1(x)

    torch.manual_seed(1234)
    mb2 = MambaLM(vocab_size, d_model=d_model, num_layers=2)
    mb2.eval()
    out_mb2 = mb2(x)

    diff_mb = (out_mb1 - out_mb2).abs().max().item()
    assert diff_mb < 1e-6, f"Mamba non-deterministic under same seed: diff={diff_mb}"
