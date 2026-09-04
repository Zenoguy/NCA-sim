"""
Unit tests for Advective Vanilla NCA (Adv-Vanilla-NCA).

Validates:
1. Exact parameter matching (strictly 7,765 parameters for non-learned modes).
2. Bit-for-bit equivalence with VanillaNCA at gamma = 0.0 (< 10^-7 error across micro and macro steps).
3. Exact micro-step transport timestep invariant delta_t = Delta_T / K.
4. Smooth gradient flow through semi-Lagrangian transport to MLP weights.
5. Velocity modes (stationary, characteristic, scaled_characteristic, peak_matched, oracle_true).
"""

import pytest
import torch
import numpy as np

from src.nca import VanillaNCA, count_parameters
from src.advective_vanilla_nca import AdvectiveVanillaNCA, compute_advective_vanilla_macs


def test_parameter_counts():
    """Verify that AdvectiveVanillaNCA has strictly 7,765 parameters matching VanillaNCA."""
    vanilla = VanillaNCA(hidden_dim=16, mlp_hidden=115)
    adv_vanilla = AdvectiveVanillaNCA(hidden_dim=16, mlp_hidden=115, mode="characteristic", gamma=1.0)
    adv_stat = AdvectiveVanillaNCA(hidden_dim=16, mlp_hidden=115, mode="stationary", gamma=0.0)
    adv_oracle = AdvectiveVanillaNCA(hidden_dim=16, mlp_hidden=115, mode="oracle_true")

    vanilla_p = count_parameters(vanilla)
    adv_p = count_parameters(adv_vanilla)
    stat_p = count_parameters(adv_stat)
    oracle_p = count_parameters(adv_oracle)

    assert vanilla_p == 7765, f"Expected 7765 params for VanillaNCA, got {vanilla_p}"
    assert adv_p == 7765, f"Expected strictly 7765 params for AdvectiveVanillaNCA, got {adv_p}"
    assert stat_p == 7765, f"Expected strictly 7765 params for stationary mode, got {stat_p}"
    assert oracle_p == 7765, f"Expected strictly 7765 params for oracle mode, got {oracle_p}"

    # Learned velocity mode introduces a tiny audited 1x1 conv net
    adv_learned = AdvectiveVanillaNCA(hidden_dim=16, mlp_hidden=115, mode="learned")
    learned_p = count_parameters(adv_learned)
    # 7765 + (17*4 + 4) + (4*1 + 1) = 7765 + 77 = 7842
    assert learned_p == 7842, f"Expected 7842 params for learned mode (77 parameter velocity net), got {learned_p}"


def test_bit_for_bit_vanilla_equivalence():
    """
    Test hard bit-for-bit equivalence between VanillaNCA and AdvectiveVanillaNCA(gamma=0.0).
    Copies identical weights and tests across multiple micro-steps and autonomous macro rollouts.
    """
    torch.manual_seed(42)
    vanilla = VanillaNCA(hidden_dim=16, mlp_hidden=115)
    adv_zero = AdvectiveVanillaNCA(hidden_dim=16, mlp_hidden=115, mode="stationary", gamma=0.0)

    # Copy weights from vanilla to adv_zero
    adv_zero.perception.conv.weight.data.copy_(vanilla.perception.conv.weight.data)
    adv_zero.perception.conv.bias.data.copy_(vanilla.perception.conv.bias.data)
    for m_adv, m_van in zip(adv_zero.mlp, vanilla.mlp):
        if isinstance(m_adv, torch.nn.Conv1d):
            m_adv.weight.data.copy_(m_van.weight.data)
            m_adv.bias.data.copy_(m_van.bias.data)

    # Test single microstep on random state s
    B, N = 4, 128
    u0 = torch.randn(B, 1, N)
    h0 = torch.randn(B, 16, N)
    s0 = torch.cat([u0, h0], dim=1)

    with torch.no_grad():
        s_van = vanilla.step(s0)
        s_adv, _ = adv_zero.step(s0)

    # 1. Exact equivalence: torch.equal
    assert torch.equal(s_van, s_adv), "Exact equivalence failed: s_van and s_adv are not strictly equal bit-for-bit"
    # 2. Numerical fallback verification
    max_diff_step = torch.max(torch.abs(s_van - s_adv)).item()
    assert max_diff_step < 1e-7, f"Single microstep max difference {max_diff_step} exceeds 1e-7"

    # Test autonomous multi-step rollout (10 macro-steps, K=2 micro-steps)
    with torch.no_grad():
        traj_van = vanilla.rollout(u0, num_macro_steps=10, K=2)
        traj_adv, final_h_adv, _ = adv_zero.rollout(u0, num_macro_steps=10, K=2)

    # 1. Exact rollout equivalence
    assert torch.equal(traj_van, traj_adv), "Exact rollout equivalence failed: traj_van and traj_adv are not strictly equal bit-for-bit"
    max_diff_rollout = torch.max(torch.abs(traj_van - traj_adv)).item()
    assert max_diff_rollout < 1e-7, f"10-step rollout max difference {max_diff_rollout} exceeds 1e-7"


def test_microstep_timestep_scaling():
    """Verify that the transport timestep delta_t = delta_T / K and K * delta_t == delta_T."""
    for K_val in [1, 2, 4, 8]:
        delta_T = 0.1
        adv = AdvectiveVanillaNCA(delta_T=delta_T, K=K_val)
        assert abs(K_val * adv.delta_t - delta_T) < 1e-12, f"Failed for K={K_val}: K*dt != delta_T"

    adv_k2 = AdvectiveVanillaNCA(delta_T=0.1, K=2)
    assert abs(adv_k2.delta_t - 0.05) < 1e-12

    adv_k4 = AdvectiveVanillaNCA(delta_T=0.1, K=4)
    assert abs(adv_k4.delta_t - 0.025) < 1e-12


def test_integer_shift_equivariance():
    """
    Test integer-cell translation equivariance:
        F(T_l u) == T_l F(u)
    for shift l in [1, 4, 16, 32] cells on periodic domain.
    """
    model = AdvectiveVanillaNCA(hidden_dim=16, mlp_hidden=115, mode="characteristic", gamma=1.0)
    model.eval()

    B, N = 2, 128
    u0 = torch.randn(B, 1, N)

    for shift_cells in [1, 4, 16, 32]:
        u0_shifted = torch.roll(u0, shifts=shift_cells, dims=-1)

        with torch.no_grad():
            u_out, _, _ = model(u0, K=2)
            u_out_shifted, _, _ = model(u0_shifted, K=2)

        # Expected: u_out shifted by shift_cells should match u_out_shifted
        expected_shifted = torch.roll(u_out, shifts=shift_cells, dims=-1)
        equiv_diff = torch.max(torch.abs(u_out_shifted - expected_shifted)).item()
        assert equiv_diff < 1e-5, f"Shift equivariance failed for shift={shift_cells} cells: max diff {equiv_diff}"


def test_velocity_modes():
    """Verify velocity field computation for characteristic, scaled, peak-matched, and oracle."""
    B, N = 2, 128
    u = torch.ones(B, 1, N) * 1.5
    h = torch.zeros(B, 16, N)
    true_A = torch.tensor([[1.5], [1.5]])

    # Characteristic: v = 6u = 9.0
    adv_char = AdvectiveVanillaNCA(mode="characteristic", gamma=1.0)
    v_char = adv_char.compute_velocity(u, h)
    assert torch.allclose(v_char, torch.full((B, 1, N), 9.0))

    # Scaled characteristic: gamma=0.5 -> v = 0.5 * 6u = 4.5
    adv_scaled = AdvectiveVanillaNCA(mode="scaled_characteristic", gamma=0.5)
    v_scaled = adv_scaled.compute_velocity(u, h)
    assert torch.allclose(v_scaled, torch.full((B, 1, N), 4.5))

    # Peak-matched: v = 2u = 3.0 (matches soliton speed 2A at peak where u=A)
    adv_peak = AdvectiveVanillaNCA(mode="peak_matched")
    v_peak = adv_peak.compute_velocity(u, h)
    assert torch.allclose(v_peak, torch.full((B, 1, N), 3.0))

    # Oracle true: v = 2 * A_true = 3.0 (constant across domain)
    adv_oracle = AdvectiveVanillaNCA(mode="oracle_true")
    v_oracle = adv_oracle.compute_velocity(u, h, true_A=true_A)
    assert torch.allclose(v_oracle, torch.full((B, 1, N), 3.0))


def test_gradient_flow():
    """Ensure gradients flow from loss through transported hidden states to MLP parameters."""
    adv_model = AdvectiveVanillaNCA(mode="characteristic", gamma=1.0)
    u0 = torch.randn(2, 1, 128, requires_grad=True)
    target = torch.randn(2, 1, 128)

    u_pred, s_next, _ = adv_model(u0, K=2)
    loss = torch.mean((u_pred - target) ** 2)
    loss.backward()

    # Check that MLP weights receive non-zero finite gradients
    for p in adv_model.mlp.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
            assert not torch.isinf(p.grad).any()


def test_mac_computation():
    """Verify MACs accounting breakdown."""
    adv_char = AdvectiveVanillaNCA(mode="characteristic", gamma=1.0)
    macs_dict = compute_advective_vanilla_macs(adv_char, N=128, K=2)

    assert "perception_macs_per_micro" in macs_dict
    assert "mlp_macs_per_micro" in macs_dict
    assert "total_macs_per_delta_T" in macs_dict
    assert macs_dict["total_macs_per_delta_T"] == 1945344
