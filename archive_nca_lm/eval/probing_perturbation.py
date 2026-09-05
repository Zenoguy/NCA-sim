"""
Probe 3B: Causal Perturbation Attenuation & Recovery Dynamics Module.

Evaluates how internal-state perturbations at position t_perturb = 64 propagate
forward through causal computation and how rapidly prediction errors attenuate
across subsequent tokens t in [65, 128].

Tracks:
  - Token-by-token error trajectory: Delta L_t = L_t^{perturbed} - L_t^{clean}
  - Cumulative damage area: D = sum_{t=65}^{T-1} max(0, Delta L_t)
  - Attenuation half-life t_{1/2} and recovery distance t_{rec}
  - Cross-factorial comparison: Variant D vs Variant C vs Variant A vs Transformer vs GRU
"""

from typing import Callable, Dict, List, Optional, Tuple, Union, Iterable
import numpy as np
import torch
import torch.nn.functional as F


def apply_perturbation(
    tensor: torch.Tensor,
    pos: int,
    noise_type: str = "gaussian",
    sigma: float = 0.5,
    dropout_prob: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Applies an impulse perturbation at position `pos`.
    tensor shape: [B, T, D] or [B, D, T]
    """
    out = tensor.clone()
    is_channel_first = (tensor.dim() == 3 and tensor.shape[1] < tensor.shape[2] and pos < tensor.shape[2])

    if is_channel_first:
        # [B, D, T]
        target = out[:, :, pos]
        if noise_type == "gaussian":
            noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator) * sigma
            out[:, :, pos] = target + noise
        elif noise_type == "dropout":
            mask = (torch.rand(target.shape, device=target.device, generator=generator) > dropout_prob).to(target.dtype)
            out[:, :, pos] = target * mask
        elif noise_type == "zero":
            out[:, :, pos] = 0.0
    else:
        # [B, T, D]
        target = out[:, pos, :]
        if noise_type == "gaussian":
            noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator) * sigma
            out[:, pos, :] = target + noise
        elif noise_type == "dropout":
            mask = (torch.rand(target.shape, device=target.device, generator=generator) > dropout_prob).to(target.dtype)
            out[:, pos, :] = target * mask
        elif noise_type == "zero":
            out[:, pos, :] = 0.0

    return out


@torch.no_grad()
def evaluate_perturbation_attenuation(
    model: torch.nn.Module,
    dataloader: Iterable,
    pos: int = 64,
    noise_type: str = "gaussian",
    sigma: float = 0.5,
    dropout_prob: float = 0.5,
    device: Union[str, torch.device] = "cpu",
    num_batches: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Union[float, List[float], Dict]]:
    """
    Runs paired clean vs. perturbed forward passes to measure causal error attenuation.

    Args:
        model: Any causal LM (NCA_LM, TransformerLM, GRULM).
        dataloader: Test dataloader yielding (inputs, targets) of length T >= pos + 1.
        pos: Index where perturbation is injected (default: 64).
        noise_type: "gaussian", "dropout", or "zero".
        sigma: Standard deviation for Gaussian noise.
        dropout_prob: Dropout probability for channel zeroing.
        device: Device to run on.
        num_batches: Max batches to evaluate for smoke testing.
        seed: Random seed for reproducible noise.

    Returns:
        Dictionary with per-token delta_L_t, damage area D, t_{1/2}, and t_{rec}.
    """
    model.eval()
    model.to(device)

    # Accumulators for clean and perturbed cross-entropy losses per position t
    clean_loss_per_t = None
    pert_loss_per_t = None
    total_sequences = 0

    gen = torch.Generator(device=device if torch.device(device).type == "cuda" else "cpu")
    gen.manual_seed(seed)

    batch_idx = 0
    for inputs, targets in dataloader:
        if num_batches is not None and batch_idx >= num_batches:
            break
        batch_idx += 1

        inputs = inputs.to(device)
        targets = targets.to(device)
        B, T = inputs.shape
        if pos >= T:
            raise ValueError(f"Perturbation pos {pos} exceeds sequence length {T}")

        # 1. Clean forward pass
        clean_logits = model(inputs)  # [B, T, V]
        V = clean_logits.shape[-1]
        loss_clean = F.cross_entropy(
            clean_logits.view(-1, V), targets.view(-1), reduction="none"
        ).view(B, T)

        # 2. Perturbed forward pass via embedding hook
        # Intercept token embeddings at the input layer
        perturbed_logits = None
        hook_handle = None

        # Determine embedding module
        embed_module = None
        if hasattr(model, "embed") and isinstance(model.embed, torch.nn.Embedding):
            embed_module = model.embed
        elif hasattr(model, "tok_embed") and isinstance(model.tok_embed, torch.nn.Embedding):
            embed_module = model.tok_embed

        if embed_module is not None:
            def hook_fn(module, input_args, output):
                return apply_perturbation(
                    output,
                    pos=pos,
                    noise_type=noise_type,
                    sigma=sigma,
                    dropout_prob=dropout_prob,
                    generator=gen,
                )
            hook_handle = embed_module.register_forward_hook(hook_fn)
            try:
                perturbed_logits = model(inputs)
            finally:
                hook_handle.remove()
        else:
            # Fallback: if no embedding module found, raise informative error
            raise AttributeError("Model has neither .embed nor .tok_embed attribute.")

        loss_pert = F.cross_entropy(
            perturbed_logits.view(-1, V), targets.view(-1), reduction="none"
        ).view(B, T)

        # Accumulate sums
        batch_clean_sum = loss_clean.sum(dim=0).cpu().numpy()
        batch_pert_sum = loss_pert.sum(dim=0).cpu().numpy()

        if clean_loss_per_t is None:
            clean_loss_per_t = batch_clean_sum
            pert_loss_per_t = batch_pert_sum
        else:
            clean_loss_per_t += batch_clean_sum
            pert_loss_per_t += batch_pert_sum

        total_sequences += B

    if total_sequences == 0:
        return {"error": "No sequences evaluated"}

    mean_clean = clean_loss_per_t / total_sequences
    mean_pert = pert_loss_per_t / total_sequences
    delta_L = mean_pert - mean_clean  # array of length T

    # Strict causality assertion: delta_L for t < pos must be virtually zero
    prior_delta_max = float(np.max(np.abs(delta_L[:pos])))

    # Focus on trajectory after perturbation: t in [pos + 1, T - 1]
    subsequent_delta = delta_L[pos + 1 :]
    initial_shock = float(delta_L[pos])
    shock_t_plus_1 = float(delta_L[pos + 1]) if len(delta_L) > pos + 1 else 0.0
    max_window = len(subsequent_delta)

    # Non-canceling cumulative damage area: D = sum max(0, Delta L_t)
    cumulative_damage_area = float(np.sum(np.maximum(0, subsequent_delta)))

    # Attenuation half-life t_{1/2}: first token offset where delta_L <= 0.5 * shock_t_plus_1
    half_life_threshold = 0.5 * max(shock_t_plus_1, 1e-6)
    t_half = None
    for offset, d in enumerate(subsequent_delta):
        if d <= half_life_threshold:
            t_half = offset + 1  # 1-indexed token steps after pos
            break

    # Recovery distance t_{rec}: first token offset where delta_L <= 0.05 * shock_t_plus_1
    recovery_threshold = 0.05 * max(shock_t_plus_1, 1e-6)
    t_rec = None
    for offset in range(len(subsequent_delta)):
        window = subsequent_delta[offset : min(offset + 3, len(subsequent_delta))]
        if np.all(window <= recovery_threshold):
            t_rec = offset + 1
            break

    # Explicit right-censoring indicators
    half_life_censored = (t_half is None)
    recovery_censored = (t_rec is None)

    half_life_str = f"{t_half} tok" if not half_life_censored else f">{max_window} tok (censored)"
    recovery_str = f"{t_rec} tok" if not recovery_censored else f">{max_window} tok (censored)"

    return {
        "pos": pos,
        "noise_type": noise_type,
        "sigma": sigma,
        "total_sequences": total_sequences,
        "causality_check_prior_max_error": round(prior_delta_max, 6),
        "initial_shock_delta": round(initial_shock, 4),
        "t_plus_1_shock_delta": round(shock_t_plus_1, 4),
        "cumulative_damage_area": round(cumulative_damage_area, 4),
        "half_life_tokens": t_half if not half_life_censored else None,
        "half_life_censored": half_life_censored,
        "half_life_display": half_life_str,
        "recovery_distance_tokens": t_rec if not recovery_censored else None,
        "recovery_censored": recovery_censored,
        "recovery_display": recovery_str,
        "trajectory_subsequent_delta": [round(float(x), 4) for x in subsequent_delta],
    }

