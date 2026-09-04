"""
Unit tests for NCA architectures, strict locality, and parameter matching solver.
"""

import pytest
import torch
from src.nca import VanillaNCA, count_parameters, compute_nca_macs, find_matched_vanilla_channels
from src.memory_nca import MemoryNCA
from src.cnn_baseline import CNNBaseline


def test_nca_strict_locality():
    """
    Verify that NCA is strictly local:
    With radius r=1, perturbing cell i+3 must NOT affect cell i after 1 micro-step.
    """
    model = VanillaNCA(hidden_dim=8, kernel_size=3)
    model.eval()

    B, N = 1, 64
    u_base = torch.zeros(B, 1, N)
    u_pert = torch.zeros(B, 1, N)

    target_cell = 20
    pert_cell = target_cell + 3  # outside radius r=1

    u_pert[0, 0, pert_cell] = 10.0

    with torch.no_grad():
        h0 = torch.zeros(B, model.hidden_dim, N)
        s_base = torch.cat([u_base, h0], dim=1)
        s_pert = torch.cat([u_pert, h0], dim=1)

        s_base_next = model.step(s_base)
        s_pert_next = model.step(s_pert)

    # State at target_cell must be identical
    diff = torch.abs(s_base_next[0, :, target_cell] - s_pert_next[0, :, target_cell])
    assert torch.max(diff).item() == 0.0, "Locality violated: perturbation leaked beyond radius!"


def test_parameter_matching_solver():
    """Verify that automated parameter matching achieves < 1% parameter difference."""
    mem_model = MemoryNCA(hidden_dim=16, memory_dim=16, kernel_size=3)
    mem_params = count_parameters(mem_model)

    matched_c, matched_mlp, vanilla_params = find_matched_vanilla_channels(mem_params, kernel_size=3)
    vanilla_model = VanillaNCA(hidden_dim=matched_c, kernel_size=3, mlp_hidden=matched_mlp)
    actual_params = count_parameters(vanilla_model)

    rel_diff = abs(actual_params - mem_params) / mem_params
    assert rel_diff < 0.01, f"Parameter matching diff {rel_diff*100:.2f}% exceeded 1% target!"



def test_memory_nca_modes_and_rollout():
    """Verify forward and rollout execution across all Memory-NCA modes."""
    B, N = 2, 64
    u0 = torch.randn(B, 1, N)

    for mode in ["persistent", "no_persistence", "random_persistence"]:
        model = MemoryNCA(hidden_dim=8, memory_dim=8, kernel_size=3, mode=mode)
        traj, final_m = model.rollout(u0, num_macro_steps=4, K=2)

        assert traj.shape == (B, 5, 1, N)
        if mode != "random_persistence":
            assert final_m.shape == (B, 8, N)
        assert not torch.isnan(traj).any()


def test_macs_calculation():
    """Verify MACs computation returns positive integers proportional to K and N."""
    model = VanillaNCA(hidden_dim=8, kernel_size=3)
    macs_k1 = compute_nca_macs(model, N=128, K=1)
    macs_k2 = compute_nca_macs(model, N=128, K=2)
    macs_n256 = compute_nca_macs(model, N=256, K=1)

    assert macs_k2 == 2 * macs_k1
    assert macs_n256 == 2 * macs_k1
    assert macs_k1 > 0


def test_cnn_baseline_rollout():
    """Verify CNN baseline forward and rollout."""
    cnn = CNNBaseline(hidden_dim=16, kernel_size=5, num_layers=4)
    u0 = torch.randn(2, 1, 64)
    traj = cnn.rollout(u0, num_macro_steps=4)
    assert traj.shape == (2, 5, 1, 64)
    assert not torch.isnan(traj).any()
