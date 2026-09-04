"""
Verification Suite for Pure-PyTorch Mamba Selective State-Space Baseline.
Validates discrete recurrence, gradient propagation, and causal property.
"""

import math
import pytest
import torch

from models.mamba_baseline import MambaBlock, MambaLM, selective_scan_loop


def test_selective_scan_basic():
    torch.manual_seed(42)
    B, T, D_in, N = 2, 16, 8, 4
    u = torch.randn(B, T, D_in)
    delta = torch.rand(B, T, D_in) * 0.1
    A = -torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).repeat(D_in, 1)
    B_mat = torch.randn(B, T, N)
    C_mat = torch.randn(B, T, N)
    D = torch.ones(D_in)

    out = selective_scan_loop(u, delta, A, B_mat, C_mat, D)
    assert out.shape == (B, T, D_in)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_mamba_block_gradient_flow():
    torch.manual_seed(42)
    d_model = 64
    block = MambaBlock(d_model=d_model, d_state=8, expand=2, d_conv=4)
    x = torch.randn(2, 32, d_model, requires_grad=True)
    out = block(x)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad.abs() > 0).any()

    # Check key parameter gradients
    assert block.A_log.grad is not None and not torch.isnan(block.A_log.grad).any()
    assert block.D.grad is not None and not torch.isnan(block.D.grad).any()
    assert block.conv1d.weight.grad is not None
    assert block.in_proj.weight.grad is not None
    assert block.out_proj.weight.grad is not None


def test_mamba_causality():
    """
    Assert that altering token at index t+1 does not affect logits at index t.
    Numerical tolerance epsilon = 1e-5.
    """
    torch.manual_seed(42)
    vocab_size = 100
    d_model = 64
    model = MambaLM(vocab_size=vocab_size, d_model=d_model, num_layers=2, d_state=8, expand=2)
    model.eval()

    T = 20
    x1 = torch.randint(0, vocab_size, (1, T))
    x2 = x1.clone()
    
    # Change token at index 15
    target_idx = 10
    changed_idx = 15
    x2[0, changed_idx] = (x1[0, changed_idx] + 1) % vocab_size

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Output at target_idx (which is before changed_idx) must be strictly identical within float tolerance
    diff_before = (out1[0, :changed_idx] - out2[0, :changed_idx]).abs().max().item()
    assert diff_before < 1e-5, f"Mamba leaked future token backwards! Diff: {diff_before}"

    # Output at or after changed_idx SHOULD differ
    diff_after = (out1[0, changed_idx:] - out2[0, changed_idx:]).abs().max().item()
    assert diff_after > 1e-3, "Changing token had no effect downstream!"
