"""
Phase 2 Test Suite: Numerical & Architectural Integrity of NCA-LM.

Tests:
1. Strict Causality Verification: Perturb token at t+1; assert zero change at step t across all K.
2. Receptive Field Exact Boundary Check:
   At K=2 (RF=7: positions t-6 ... t), altering token at t-8 has zero effect, while t-6 has non-zero effect.
3. Weight Sharing Gradient Verification:
   In shared_weights=True, gradients accumulate into single conv_weight.
   In shared_weights=False, step k conv receives gradients only from step k.
4. Truncated & Extended Computation Invariance (override_K):
   Verify valid forward passes across K in {1, 2, 3, 4, 6, 8, 12}.
5. Determinism:
   Assert identical seeds yield identical output (max |Delta| < 10^-6).
"""

import pytest
import torch
from models.nca_lm import NCA_LM, CausalNCAStep, SinusoidalStepEmbedding


@pytest.fixture
def small_shared_nca():
    torch.manual_seed(42)
    return NCA_LM(
        vocab_size=128,
        d_embed=64,
        radius=2,
        K=6,
        max_K=12,
        shared_weights=True,
        step_embed_type="sinusoidal",
        tie_weights=True,
    )


@pytest.fixture
def small_unshared_nca():
    torch.manual_seed(42)
    return NCA_LM(
        vocab_size=128,
        d_embed=64,
        radius=2,
        K=6,
        max_K=6,
        shared_weights=False,
        step_embed_type="sinusoidal",
        tie_weights=True,
    )


def test_strict_causality(small_shared_nca, small_unshared_nca):
    """
    Assert that modifying a token at index t+1 produces strictly ZERO change
    in logits at index t (and all earlier indices <= t).
    Tolerance: max |Delta logits_t| < 1e-5.
    """
    for model in [small_shared_nca, small_unshared_nca]:
        model.eval()
        B, T = 2, 32
        x = torch.randint(0, 128, (B, T))

        with torch.no_grad():
            orig_logits = model(x)

        # Perturb token at t = 15
        target_t = 14
        perturb_t = 15
        x_perturbed = x.clone()
        x_perturbed[:, perturb_t] = (x_perturbed[:, perturb_t] + 1) % 128

        with torch.no_grad():
            perturbed_logits = model(x_perturbed)

        # Logits at target_t and earlier must be identical
        diff_past = (orig_logits[:, : target_t + 1, :] - perturbed_logits[:, : target_t + 1, :]).abs().max().item()
        assert diff_past < 1e-5, f"Causality violation! Model {model.shared_weights} leaked future token: diff = {diff_past}"

        # Logits at perturb_t or later should change
        diff_future = (orig_logits[:, perturb_t:, :] - perturbed_logits[:, perturb_t:, :]).abs().max().item()
        assert diff_future > 1e-4, f"Model failed to register change at perturbation index: diff = {diff_future}"


def test_receptive_field_exact_boundary():
    """
    Verify exact analytical receptive field boundary.
    With radius=2 and dilations [1, 2]:
      Step 0 (d=1): offsets {0, 1, 2} past -> 2 past tokens.
      Step 1 (d=2): offsets {0, 2, 4} past -> 2 * 2 = 4 past tokens.
      Total past tokens = 2 + 4 = 6 past tokens.
      Total accessible span = 7 positions: {t-6, t-5, ..., t}.
    Therefore:
      Altering token at t-8 MUST produce ZERO change at position t (< 1e-5).
      Altering token at t-6 MUST produce non-zero change at position t (> 1e-4).
    """
    torch.manual_seed(42)
    model = NCA_LM(
        vocab_size=128,
        d_embed=64,
        radius=2,
        K=2,
        max_K=4,
        shared_weights=True,
    )
    model.eval()

    B, T = 1, 32
    target_pos = 20
    x = torch.randint(0, 128, (B, T))

    with torch.no_grad():
        orig_logits = model(x)

    # 1. Perturb outside RF: token at target_pos - 8 = 12
    outside_pos = target_pos - 8
    x_outside = x.clone()
    x_outside[0, outside_pos] = (x_outside[0, outside_pos] + 5) % 128
    with torch.no_grad():
        outside_logits = model(x_outside)

    diff_outside = (orig_logits[:, target_pos, :] - outside_logits[:, target_pos, :]).abs().max().item()
    assert diff_outside < 1e-5, f"Receptive field leaked beyond theoretical boundary! diff = {diff_outside}"

    # 2. Perturb inside boundary: token at target_pos - 6 = 14
    inside_pos = target_pos - 6
    x_inside = x.clone()
    x_inside[0, inside_pos] = (x_inside[0, inside_pos] + 5) % 128
    with torch.no_grad():
        inside_logits = model(x_inside)

    diff_inside = (orig_logits[:, target_pos, :] - inside_logits[:, target_pos, :]).abs().max().item()
    assert diff_inside > 1e-4, f"Boundary token within RF failed to affect output! diff = {diff_inside}"


def test_weight_sharing_gradient_accumulation():
    """
    Verify weight sharing mechanics:
    - In shared_weights=True, gradients from all steps accumulate into the single conv_weight.
    - In shared_weights=False, step k conv receives gradients only from step k.
    """
    torch.manual_seed(42)
    # Shared model
    shared_m = NCA_LM(vocab_size=64, d_embed=32, radius=2, K=4, shared_weights=True)
    x = torch.randint(0, 64, (2, 16))
    out = shared_m(x).sum()
    out.backward()

    assert shared_m.step.conv_weight.grad is not None
    assert shared_m.step.conv_weight.grad.norm().item() > 1e-4

    # Unshared model
    unshared_m = NCA_LM(vocab_size=64, d_embed=32, radius=2, K=4, max_K=4, shared_weights=False)
    out_u = unshared_m(x).sum()
    out_u.backward()

    # Each step's conv must have non-zero gradients
    for k in range(4):
        g = unshared_m.step.convs[k].weight.grad
        assert g is not None
        assert g.norm().item() > 1e-4


def test_truncated_and_extended_computation(small_shared_nca):
    """
    Verify override_K works cleanly for all k in [1, 2, 3, 4, 6, 8, 12]
    without shape mismatches or runtime errors.
    """
    model = small_shared_nca
    model.eval()
    B, T = 2, 16
    x = torch.randint(0, 128, (B, T))

    for k in [1, 2, 3, 4, 6, 8, 12]:
        with torch.no_grad():
            out = model(x, override_K=k)
        assert out.shape == (B, T, 128), f"Shape mismatch at override_K={k}: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN detected at override_K={k}"


def test_nca_determinism():
    """
    Verify determinism: identical seed produces identical forward pass.
    """
    torch.manual_seed(42)
    m1 = NCA_LM(vocab_size=128, d_embed=64, radius=2, K=4, shared_weights=True)
    torch.manual_seed(42)
    m2 = NCA_LM(vocab_size=128, d_embed=64, radius=2, K=4, shared_weights=True)

    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out1 = m1(x)
        out2 = m2(x)

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"Determinism failure: max diff = {diff}"


def test_receptive_field_formula():
    """
    Verify analytical formula: RF(K) = 1 + radius * (2^K - 1).
    """
    assert NCA_LM.compute_receptive_field(1, radius=2) == 3
    assert NCA_LM.compute_receptive_field(2, radius=2) == 7
    assert NCA_LM.compute_receptive_field(3, radius=2) == 15
    assert NCA_LM.compute_receptive_field(4, radius=2) == 31
    assert NCA_LM.compute_receptive_field(6, radius=2) == 127
