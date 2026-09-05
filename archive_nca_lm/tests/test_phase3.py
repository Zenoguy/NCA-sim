"""
Unit Test Suite for Phase 3: Candidate Capability Probing.
Validates:
  1. Depth scaling evaluation, FLOPs calculation, and analytical receptive field
  2. Strict causality of perturbation hooks and non-canceling damage area D
  3. In-vocabulary noise generation and relative degradation ratio R(p)
  4. Streaming state complexity regression slopes (O(1) vs O(T))
  5. Gate 3 decision evaluator logic
"""

import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from eval.probing_depth import calculate_model_flops, calculate_step_flops, evaluate_depth_scaling
from eval.probing_perturbation import apply_perturbation, evaluate_perturbation_attenuation
from eval.probing_robustness import corrupt_token_tensor, evaluate_noise_robustness
from eval.probing_streaming import compute_streaming_state_memory, evaluate_streaming_state_complexity
from models.nca_lm import NCA_LM
from models.transformer_baseline import TransformerLM
from scripts.run_level3 import evaluate_gate_3


@pytest.fixture
def small_synthetic_dataloader():
    torch.manual_seed(42)
    tokens = torch.randint(4, 512, (8, 65))
    inputs = tokens[:, :-1]  # [8, 64]
    targets = tokens[:, 1:]  # [8, 64]
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_override_K_and_receptive_field(small_synthetic_dataloader):
    model = NCA_LM(
        vocab_size=512,
        d_embed=64,
        d_hidden_channels=0,
        radius=2,
        K=4,
        max_K=8,
        shared_weights=True,
    )
    res = evaluate_depth_scaling(
        model,
        small_synthetic_dataloader,
        k_values=[1, 2, 4, 6],
        device="cpu",
        is_shared=True,
    )
    curve = res["curve"]
    assert len(curve) == 4
    # Check RF formula: 1 + 2 * (2^K - 1)
    assert curve[0]["receptive_field"] == 1 + 2 * (2**1 - 1)  # 3
    assert curve[1]["receptive_field"] == 1 + 2 * (2**2 - 1)  # 7
    assert curve[2]["receptive_field"] == 1 + 2 * (2**4 - 1)  # 31
    assert curve[3]["receptive_field"] == 1 + 2 * (2**6 - 1)  # 127

    # Check FLOPs monotonicity
    for i in range(1, len(curve)):
        assert curve[i]["flops_per_token"] > curve[i - 1]["flops_per_token"]


def test_perturbation_attenuation_causality(small_synthetic_dataloader):
    model = NCA_LM(
        vocab_size=512,
        d_embed=64,
        d_hidden_channels=0,
        radius=2,
        K=4,
        max_K=6,
        shared_weights=True,
    )
    pos = 32
    res = evaluate_perturbation_attenuation(
        model,
        small_synthetic_dataloader,
        pos=pos,
        noise_type="gaussian",
        sigma=0.5,
        device="cpu",
    )
    # Strictly zero error prior to pos=32
    assert res["causality_check_prior_max_error"] < 1e-5
    # Damage area non-canceling D >= 0
    assert res["cumulative_damage_area"] >= 0.0
    # Recovery distance is positive integer
    assert res["recovery_distance_tokens"] >= 1


def test_in_vocab_corruption_and_slope(small_synthetic_dataloader):
    tokens = torch.randint(0, 512, (4, 32))
    p = 0.25
    corrupted = corrupt_token_tensor(tokens, p=p, vocab_size=512)
    assert corrupted.shape == tokens.shape
    # All corrupted tokens must be within valid vocab range
    assert (corrupted >= 0).all() and (corrupted < 512).all()

    model = TransformerLM(vocab_size=512, d_model=64, num_layers=2, num_heads=2)
    res = evaluate_noise_robustness(
        model,
        small_synthetic_dataloader,
        corruption_rates=[0.0, 0.05, 0.10],
        device="cpu",
        vocab_size=512,
    )
    curve = res["curve"]
    assert len(curve) == 3
    # At p=0, relative degradation ratio must be exactly 1.0
    assert curve[0]["relative_degradation_ratio"] == 1.0


def test_streaming_state_complexity_slopes():
    res = evaluate_streaming_state_complexity(seq_lengths=[128, 256, 512, 1024])
    tf = res["primary_transformer"]
    nca = res["nca_variant_d"]
    gru = res["gru_baseline"]

    # Transformer KV-cache grows linearly O(T), slope > 0
    assert tf["asymptotic_complexity"] == "O(T)"
    assert tf["marginal_slope_mb_per_128_tokens"] > 0.0

    # NCA with bounded receptive field buffer (RF=127) scales O(1) for T >= 128
    assert nca["asymptotic_complexity"] == "O(1)"
    assert abs(nca["marginal_slope_mb_per_128_tokens"]) < 1e-4

    # GRU recurrent state scales O(1)
    assert gru["asymptotic_complexity"] == "O(1)"
    assert abs(gru["marginal_slope_mb_per_128_tokens"]) < 1e-4


def test_gate3_evaluator_rubric(tmp_path):
    # Test evaluation when result files exist
    d_dir = tmp_path / "outputs" / "level3"
    d_dir.mkdir(parents=True, exist_ok=True)

    # When denoising advantage is established
    denoise_data = {
        "models": {
            "variant_d_shared_10m": {
                "metrics": {
                    "final_contraction_E_K": 0.25,
                    "is_statistically_contractive": True,
                }
            },
            "variant_c_unshared_10m": {
                "metrics": {
                    "final_contraction_E_K": 0.80,
                    "is_statistically_contractive": True,
                }
            },
        }
    }
    with open(d_dir / "latent_denoising_contraction.json", "w") as f:
        import json
        json.dump(denoise_data, f)

    verdict = evaluate_gate_3(d_dir)
    assert verdict["passed"] is True
    assert "PROCEED" in verdict["recommendation"]


def test_latent_error_contraction(small_synthetic_dataloader):
    from eval.probing_denoising import evaluate_latent_error_contraction
    model = NCA_LM(
        vocab_size=512,
        d_embed=64,
        d_hidden_channels=0,
        radius=2,
        K=4,
        max_K=6,
        shared_weights=True,
    )
    res = evaluate_latent_error_contraction(
        model,
        small_synthetic_dataloader,
        K=4,
        pos=32,
        noise_type="gaussian",
        sigma=0.5,
        device="cpu",
        num_bootstrap=100,
    )
    assert "trajectory_E_k" in res
    assert len(res["trajectory_E_k"]) == 5  # k=0, 1, 2, 3, 4
    # E_0 must be normalized to 1.0
    assert abs(res["trajectory_E_k"][0] - 1.0) < 1e-4
    # Bootstrap CI exists and is properly ordered
    ci = res["ci_95"]
    assert ci[0] <= res["final_contraction_E_K"] <= ci[1]

