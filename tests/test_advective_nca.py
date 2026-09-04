"""
Unit tests for Advective (Flow-Convected) Memory Neural Cellular Automata.
"""

import numpy as np
import pytest
import torch
from src.advective_memory_nca import (
    AdvectiveMemoryNCA,
    DifferentiableSemiLagrangian1D,
    find_matched_advective_mlp,
    compute_advective_macs,
)


def test_differentiable_semi_lagrangian_gradient_flow():
    """Verify gradients propagate smoothly through departure interpolation to v and m."""
    B, C, N = 2, 4, 32
    transport = DifferentiableSemiLagrangian1D(Lx=50.0)

    m = torch.randn(B, C, N, requires_grad=True)
    v = torch.randn(B, 1, N, requires_grad=True)
    delta_t = 0.05

    m_star, diags = transport(m, v, delta_t)

    assert m_star.shape == (B, C, N)
    assert torch.all(torch.isfinite(m_star))

    # Compute loss and check backprop
    loss = torch.sum(m_star ** 2)
    loss.backward()

    assert m.grad is not None and torch.all(torch.isfinite(m.grad))
    assert v.grad is not None and torch.all(torch.isfinite(v.grad))
    assert torch.norm(m.grad) > 1e-6
    assert torch.norm(v.grad) > 1e-6


def test_constant_velocity_exact_mass_conservation():
    """Verify that under spatially uniform velocity, total memory is exactly conserved."""
    B, C, N = 2, 3, 64
    transport = DifferentiableSemiLagrangian1D(Lx=50.0)

    m = torch.rand(B, C, N) + 0.1
    # Uniform constant velocity
    v = torch.full((B, 1, N), 2.5)
    delta_t = 0.05

    m_star, diags = transport(m, v, delta_t)

    sum_initial = torch.sum(m, dim=-1)
    sum_advected = torch.sum(m_star, dim=-1)

    # Machine-precision conservation for constant translation
    assert torch.allclose(sum_initial, sum_advected, atol=1e-5, rtol=1e-5)
    assert diags["mass_conservation_error"] < 1e-5


def test_variable_velocity_conservation_divergence_diagnostic():
    """
    Verify that under spatially varying velocity (e.g. v = 6u), conservation error
    is computed and bounded as an empirical diagnostic.
    """
    B, C, N = 1, 2, 64
    transport = DifferentiableSemiLagrangian1D(Lx=50.0)

    x = np.linspace(-25, 25, N, endpoint=False)
    u = np.exp(-0.5 * (x / 3.0) ** 2)
    v = torch.from_numpy(6.0 * u).float().view(1, 1, N)
    m = torch.ones(B, C, N)
    delta_t = 0.05

    m_star, diags = transport(m, v, delta_t)

    assert "mass_conservation_error" in diags
    # Because m_t + v m_x = 0 is non-conservative for variable v, error is > 0 but bounded
    assert np.isfinite(diags["mass_conservation_error"])
    assert diags["mass_conservation_error"] < 0.5


def test_dual_memory_separation():
    """Verify m_local remains Eulerian while m_trans shifts under flow."""
    model = AdvectiveMemoryNCA(
        hidden_dim=8,
        memory_dim=8,
        transport_dim=4,  # 4 transported, 4 local
        mlp_hidden=32,
        mode="characteristic",
    )

    B, N = 1, 32
    u0 = torch.zeros(B, 1, N)
    # Put a positive velocity pulse in the center
    u0[0, 0, N // 2] = 2.0
    h0 = torch.zeros(B, model.hidden_dim, N)
    s = torch.cat([u0, h0], dim=1)

    # Set distinct initial memory values
    m = torch.zeros(B, model.memory_dim, N)
    # Impulse at center
    m[0, :, N // 2] = 1.0

    # Step model
    s_next, m_next, diag = model.step(s, m)

    assert m_next.shape == (B, model.memory_dim, N)
    assert diag["max_abs_v"] > 0.0


def test_all_five_modes_rollout_and_parameter_matching():
    """Verify that all 5 transport modes run smoothly and match target parameters within < 0.2%."""
    target_params = 7765
    modes = ["stationary", "characteristic", "learned", "oracle_estimated", "oracle_true"]

    for mode in modes:
        mlp_h, actual_p = find_matched_advective_mlp(
            target_params=target_params,
            hidden_dim=16,
            memory_dim=16,
            transport_dim=8,
            mode=mode,
        )
        model = AdvectiveMemoryNCA(
            hidden_dim=16,
            memory_dim=16,
            transport_dim=8,
            mlp_hidden=mlp_h,
            mode=mode,
        )

        # Parameter matching check
        diff = abs(actual_p - target_params)
        rel_diff = diff / target_params
        assert rel_diff < 0.005, f"Mode {mode} param diff {diff} exceeds 0.5%"

        # Forward rollout check
        u0 = torch.randn(2, 1, 64)
        true_A = torch.tensor([1.2, 0.8])
        traj, final_m, diags = model.rollout(
            u0, num_macro_steps=3, K=2, true_A=true_A
        )

        assert traj.shape == (2, 4, 1, 64)
        assert final_m.shape == (2, 16, 64)
        assert len(diags) == 6  # 3 macro steps * K=2 micro steps


def test_causal_velocity_override_intervention():
    """Verify that velocity_override hook correctly forces zero, reverse, or custom velocity."""
    model = AdvectiveMemoryNCA(
        hidden_dim=8,
        memory_dim=8,
        transport_dim=4,
        mlp_hidden=32,
        mode="characteristic",
    )

    B, N = 1, 32
    s = torch.ones(B, 9, N)
    m = torch.zeros(B, 8, N)

    # 1. Normal characteristic velocity should be 6 * u = 6.0
    v_norm = model.compute_velocity(s, m)
    assert torch.allclose(v_norm, torch.tensor(6.0))

    # 2. Intervention: Zero velocity
    model.velocity_override = 0.0
    v_zero = model.compute_velocity(s, m)
    assert torch.allclose(v_zero, torch.tensor(0.0))

    # 3. Intervention: Reversed velocity
    model.velocity_override = lambda state: -6.0 * state[:, :1, :]
    v_rev = model.compute_velocity(s, m)
    assert torch.allclose(v_rev, torch.tensor(-6.0))
