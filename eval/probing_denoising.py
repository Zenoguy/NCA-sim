"""
Probe 3E: Iterative Latent Error Contraction & Denoising Dynamics.

Directly tests the core dynamical hypothesis:
"Does repeated application of the learned shared cellular rule contract the norm
of internal state perturbations over microsteps k = 0, 1, ..., K?"

Measures:
  - Error norm trajectory: E_k = ||s_k^{pert} - s_k^{clean}||_F / ||s_0^{pert} - s_0^{clean}||_F
  - Step contraction ratio: rho_k = E_k / E_{k-1} (rho < 1 indicates contractive dynamics)
  - Final contraction factor: E_K / E_0
  - 95% Bootstrap Confidence Intervals across sequences
  - Direct comparison: Shared NCA (Variant D, Variant A) vs Unshared CNN (Variant C)
"""

from typing import Dict, List, Optional, Tuple, Union, Iterable
import numpy as np
import torch


def compute_relative_error_norm(pert: torch.Tensor, clean: torch.Tensor, init_norm: float) -> float:
    """Computes Frobenius error norm normalized by the initial perturbation norm."""
    diff = (pert - clean).float()
    norm = float(torch.norm(diff).item())
    return (norm / init_norm) if init_norm > 1e-8 else 1.0


@torch.no_grad()
def evaluate_latent_error_contraction(
    model: torch.nn.Module,
    dataloader: Iterable,
    K: int = 6,
    pos: Optional[int] = 64,
    noise_type: str = "gaussian",
    sigma: float = 0.5,
    device: Union[str, torch.device] = "cpu",
    num_batches: Optional[int] = None,
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Union[Dict, List, float]]:
    """
    Evaluates latent error norm contraction E_k over microsteps k in [0..K].

    Args:
        model: NCA_LM (Shared or Unshared).
        dataloader: Test dataloader yielding (inputs, targets).
        K: Number of microsteps to evaluate.
        pos: If specified, injects perturbation at position `pos`. If None, global state.
        noise_type: 'gaussian' or 'dropout'.
        sigma: Noise standard deviation.
        device: Device to run evaluation on.
        num_batches: Max batches for fast evaluation.
        num_bootstrap: Bootstrap iterations for 95% CI.
        seed: Random seed.

    Returns:
        Structured dictionary with E_k trajectory, step contraction ratios rho_k,
        final contraction factor E_K, and 95% bootstrap confidence intervals.
    """
    model.eval()
    model.to(device)

    gen = torch.Generator(device=device if torch.device(device).type == "cuda" else "cpu")
    gen.manual_seed(seed)

    # Accumulate per-sequence error norms at each step k
    # list of length K+1, each containing error norms for every sequence in the dataset
    step_errors_per_sequence = [[] for _ in range(K + 1)]
    final_contraction_factors = []

    batch_idx = 0
    for inputs, _ in dataloader:
        if num_batches is not None and batch_idx >= num_batches:
            break
        batch_idx += 1

        inputs = inputs.to(device)
        B, T = inputs.shape

        # 1. Clean forward intermediates
        clean_intermediates = model.forward_intermediates(inputs, override_K=K)
        # clean_intermediates is list of [s_0, s_1, ..., s_K] where each is [B, d_model, T]

        s0_clean = clean_intermediates[0]
        s0_pert = s0_clean.clone()

        # 2. Inject perturbation into s_0
        if pos is not None and pos < T:
            target_slice = s0_pert[:, :, pos]
            if noise_type == "gaussian":
                noise = torch.randn(target_slice.shape, device=device, dtype=s0_pert.dtype, generator=gen) * sigma
                s0_pert[:, :, pos] = target_slice + noise
            elif noise_type == "dropout":
                mask = (torch.rand(target_slice.shape, device=device, generator=gen) > 0.5).to(s0_pert.dtype)
                s0_pert[:, :, pos] = target_slice * mask
        else:
            if noise_type == "gaussian":
                noise = torch.randn(s0_pert.shape, device=device, dtype=s0_pert.dtype, generator=gen) * sigma
                s0_pert = s0_pert + noise
            elif noise_type == "dropout":
                mask = (torch.rand(s0_pert.shape, device=device, generator=gen) > 0.5).to(s0_pert.dtype)
                s0_pert = s0_pert * mask

        # 3. Step forward through the model's microsteps
        pert_intermediates = [s0_pert]
        curr_pert = s0_pert
        for k in range(K):
            curr_pert = model.step(curr_pert, step_idx=k)
            pert_intermediates.append(curr_pert)

        # 4. Measure per-sequence error norms
        for b in range(B):
            diff0 = (s0_pert[b] - s0_clean[b]).float()
            init_norm = float(torch.norm(diff0).item())
            if init_norm < 1e-8:
                continue

            seq_errors = []
            for k in range(K + 1):
                diff_k = (pert_intermediates[k][b] - clean_intermediates[k][b]).float()
                norm_k = float(torch.norm(diff_k).item())
                e_k = norm_k / init_norm
                seq_errors.append(e_k)
                step_errors_per_sequence[k].append(e_k)

            final_contraction_factors.append(seq_errors[-1])

    if not final_contraction_factors:
        return {"error": "No sequences evaluated"}

    # Compute mean trajectory E_k
    mean_trajectory = [float(np.mean(step_errors_per_sequence[k])) for k in range(K + 1)]

    # Compute step-by-step contraction ratios rho_k = E_k / E_{k-1}
    step_ratios = []
    for k in range(1, K + 1):
        prev = mean_trajectory[k - 1]
        curr = mean_trajectory[k]
        ratio = (curr / prev) if prev > 0 else 1.0
        step_ratios.append(round(ratio, 4))

    # Bootstrap 95% Confidence Interval for final contraction factor E_K
    np.random.seed(seed)
    n_samples = len(final_contraction_factors)
    factors_arr = np.array(final_contraction_factors)
    bootstrap_means = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(factors_arr, size=n_samples, replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    mean_final = float(np.mean(factors_arr))

    # Determine contractive status
    is_contractive = (mean_final < 1.0) and (ci_upper < 1.0)

    return {
        "K": K,
        "pos": pos,
        "noise_type": noise_type,
        "sigma": sigma,
        "total_sequences_evaluated": n_samples,
        "trajectory_E_k": [round(val, 4) for val in mean_trajectory],
        "step_contraction_ratios_rho": step_ratios,
        "final_contraction_E_K": round(mean_final, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "is_statistically_contractive": is_contractive,
    }
