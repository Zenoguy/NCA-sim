"""
Probe 3A: Test-Time Compute-Depth Scaling (K-Scaling) Module.

Evaluates models under varying iterations of the learned shared rule (K in [1, 12]).
Measures:
  - PPL(K) and NLL(K)
  - Analytical Receptive Field RF(K)
  - FLOPs(K) per token
  - Marginal Compute Efficiency: delta_PPL / delta_MFLOP
  - Difficulty Stratification: PPL on Top 20% highest-entropy tokens vs Bottom 80%
"""

import math
from typing import Dict, List, Optional, Union, Iterable
import numpy as np
import torch
import torch.nn.functional as F

from eval.perplexity import loss_to_perplexity


def calculate_step_flops(d_model: int, d_hidden: int, kernel_size: int = 3) -> int:
    """
    Computes theoretical FLOPs per token for one micro-step of CausalNCAStep:
    1. Conv1d: 2 * kernel_size * d_model * d_hidden
    2. Update gate (1x1): 2 * (d_hidden + d_model) * d_model
    3. Reset gate (1x1): 2 * (d_hidden + d_model) * d_model
    4. Candidate nhood (1x1): 2 * d_hidden * d_model
    5. Candidate state (1x1): 2 * d_model * d_model
    6. Elementwise ops (silu, sigmoid, tanh, interpolation): ~10 * d_model
    """
    f_conv = 2 * kernel_size * d_model * d_hidden
    f_update = 2 * (d_hidden + d_model) * d_model
    f_reset = 2 * (d_hidden + d_model) * d_model
    f_cand_nhood = 2 * d_hidden * d_model
    f_cand_state = 2 * d_model * d_model
    f_elem = 10 * d_model
    return f_conv + f_update + f_reset + f_cand_nhood + f_cand_state + f_elem


def calculate_model_flops(
    vocab_size: int,
    d_embed: int,
    d_model: int,
    d_hidden: int,
    K: int,
    kernel_size: int = 3,
) -> int:
    """Computes total forward FLOPs per token for NCA-LM at depth K."""
    f_readout = 2 * d_model * vocab_size
    f_step = calculate_step_flops(d_model, d_hidden, kernel_size)
    return f_readout + K * f_step


@torch.no_grad()
def evaluate_depth_scaling(
    model: torch.nn.Module,
    dataloader: Iterable,
    k_values: Optional[List[int]] = None,
    device: Union[str, torch.device] = "cpu",
    is_shared: bool = True,
    max_k_allowed: Optional[int] = None,
    calculate_entropy_splits: bool = True,
    num_batches: Optional[int] = None,
) -> Dict[str, Union[Dict, List]]:
    """
    Evaluates test-time compute depth scaling across k_values.

    Args:
        model: Trained NCA_LM or baseline model supporting override_K.
        dataloader: Test dataloader yielding (inputs, targets).
        k_values: List of micro-step counts, e.g. [1, 2, 3, 4, 5, 6, 7, 8, 10, 12].
        device: Device to run evaluation on.
        is_shared: Whether model has shared weights (True for NCA, False for fixed CNN).
        max_k_allowed: Maximum K permitted for unshared models (cannot exceed physical depth).
        calculate_entropy_splits: Whether to compute high-entropy subset PPL.
        num_batches: Optional cap on batches for fast evaluation / smoke testing.

    Returns:
        Structured dictionary containing per-K metrics, scaling curve, and efficiency.
    """
    if k_values is None:
        k_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]

    model.eval()
    model.to(device)

    # Extract model architecture parameters
    d_model = getattr(model, "d_model", 288)
    d_embed = getattr(model, "d_embed", 288)
    d_hidden = d_model * 2
    vocab_size = getattr(model, "vocab_size", 8192)
    radius = getattr(model.step, "radius", 2) if hasattr(model, "step") else 2
    physical_k = getattr(model, "K", 6)

    if max_k_allowed is None and not is_shared:
        max_k_allowed = physical_k

    results_by_k = {}
    curve = []

    for k in k_values:
        if not is_shared and max_k_allowed is not None and k > max_k_allowed:
            # Unshared model cannot physically execute deeper than its parameter depth
            continue

        total_loss = 0.0
        total_tokens = 0

        # For difficulty stratification
        token_losses = []
        token_entropies = []

        batch_count = 0
        for inputs, targets in dataloader:
            if num_batches is not None and batch_count >= num_batches:
                break
            batch_count += 1

            inputs = inputs.to(device)
            targets = targets.to(device)

            if hasattr(model, "forward"):
                # Pass override_K to dynamic NCA
                try:
                    logits = model(inputs, override_K=k)
                except TypeError:
                    logits = model(inputs)
            else:
                logits = model(inputs)

            B, T, V = logits.shape
            logits_flat = logits.view(-1, V)
            targets_flat = targets.view(-1)

            loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            total_loss += loss_per_token.sum().item()
            total_tokens += targets_flat.numel()

            if calculate_entropy_splits:
                # Calculate predictive distribution entropy: H = -sum(p * log(p))
                probs = F.softmax(logits_flat, dim=-1)
                log_probs = F.log_softmax(logits_flat, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)

                token_losses.extend(loss_per_token.detach().cpu().numpy().tolist())
                token_entropies.extend(entropy.detach().cpu().numpy().tolist())

        if total_tokens == 0:
            continue

        avg_loss = total_loss / total_tokens
        ppl = loss_to_perplexity(avg_loss)
        rf = 1 + radius * (2**k - 1)
        flops = calculate_model_flops(vocab_size, d_embed, d_model, d_hidden, k, kernel_size=radius + 1)

        point = {
            "K": k,
            "loss_nll": round(avg_loss, 4),
            "perplexity": round(ppl, 2),
            "receptive_field": rf,
            "flops_per_token": flops,
            "mflops_per_token": round(flops / 1e6, 2),
        }

        # Stratify by entropy quartiles if collected
        if calculate_entropy_splits and len(token_losses) > 0:
            losses_arr = np.array(token_losses)
            entropies_arr = np.array(token_entropies)

            # Top 20% highest entropy tokens (hard / ambiguous tokens)
            p80 = np.percentile(entropies_arr, 80)
            hard_mask = entropies_arr >= p80
            easy_mask = ~hard_mask

            hard_loss = float(np.mean(losses_arr[hard_mask])) if np.any(hard_mask) else avg_loss
            easy_loss = float(np.mean(losses_arr[easy_mask])) if np.any(easy_mask) else avg_loss

            point["hard_tokens_top20_ppl"] = round(loss_to_perplexity(hard_loss), 2)
            point["easy_tokens_bottom80_ppl"] = round(loss_to_perplexity(easy_loss), 2)

        results_by_k[k] = point
        curve.append(point)

    # Calculate compute-normalized marginal efficiencies: delta_PPL / delta_MFLOP
    for i in range(1, len(curve)):
        prev_pt = curve[i - 1]
        curr_pt = curve[i]
        d_ppl = curr_pt["perplexity"] - prev_pt["perplexity"]
        d_mflop = curr_pt["mflops_per_token"] - prev_pt["mflops_per_token"]
        efficiency = (d_ppl / d_mflop) if d_mflop > 0 else 0.0
        curr_pt["delta_ppl"] = round(d_ppl, 2)
        curr_pt["delta_mflops"] = round(d_mflop, 2)
        curr_pt["ppl_per_mflop"] = round(efficiency, 4)

    if curve:
        curve[0]["delta_ppl"] = 0.0
        curve[0]["delta_mflops"] = 0.0
        curve[0]["ppl_per_mflop"] = 0.0

    return {
        "is_shared": is_shared,
        "d_model": d_model,
        "curve": curve,
        "results_by_k": results_by_k,
    }
