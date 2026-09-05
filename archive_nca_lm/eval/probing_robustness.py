"""
Probe 3C: Surface Input Noise & Typo Tolerance Module.

Evaluates model degradation under controlled in-vocabulary token substitutions.
Computes relative degradation to eliminate clean baseline perplexity confounds:
  - Relative Degradation Ratio: R(p) = PPL(p) / PPL(0)
  - Log Perplexity Shift: Delta log PPL(p) = log PPL(p) - log PPL(0)
  - Linear Degradation Slope: beta = d R(p) / dp
"""

import math
from typing import Dict, List, Optional, Union, Iterable
import numpy as np
import torch
import torch.nn.functional as F

from eval.perplexity import loss_to_perplexity


def corrupt_token_tensor(
    tokens: torch.Tensor,
    p: float,
    vocab_size: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Randomly corrupts input tokens with probability p using valid in-vocabulary token IDs.
    Does not introduce out-of-distribution special tokens.
    """
    if p <= 0.0:
        return tokens

    corrupted = tokens.clone()
    mask = torch.rand(tokens.shape, device=tokens.device, generator=generator) < p

    # Sample uniformly from valid vocabulary
    random_tokens = torch.randint(
        0, vocab_size, tokens.shape, device=tokens.device, dtype=tokens.dtype, generator=generator
    )
    corrupted[mask] = random_tokens[mask]
    return corrupted


@torch.no_grad()
def evaluate_noise_robustness(
    model: torch.nn.Module,
    dataloader: Iterable,
    corruption_rates: Optional[List[float]] = None,
    device: Union[str, torch.device] = "cpu",
    vocab_size: Optional[int] = None,
    num_batches: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Union[Dict, List, float]]:
    """
    Runs noise robustness evaluation across corruption rates p.

    Args:
        model: Trained language model.
        dataloader: Test dataloader yielding (inputs, targets).
        corruption_rates: List of probabilities in [0.0, 1.0]. Default: [0.0, 0.02, 0.05, 0.10, 0.15, 0.20].
        device: Evaluation device.
        vocab_size: Vocabulary size (inferred from model if None).
        num_batches: Max batches to evaluate for smoke testing.
        seed: Random seed for corruption generator.

    Returns:
        Dictionary with per-p metrics, relative degradation curve, and fitted slope beta.
    """
    if corruption_rates is None:
        corruption_rates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

    model.eval()
    model.to(device)

    if vocab_size is None:
        vocab_size = getattr(model, "vocab_size", 8192)

    gen = torch.Generator(device=device if torch.device(device).type == "cuda" else "cpu")
    gen.manual_seed(seed)

    curve = []
    clean_ppl = None

    for p in corruption_rates:
        total_loss = 0.0
        total_tokens = 0
        batch_idx = 0

        for inputs, targets in dataloader:
            if num_batches is not None and batch_idx >= num_batches:
                break
            batch_idx += 1

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Apply in-vocabulary corruption to inputs only (targets remain ground truth)
            corrupted_inputs = corrupt_token_tensor(inputs, p=p, vocab_size=vocab_size, generator=gen)

            logits = model(corrupted_inputs)
            V = logits.shape[-1]
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction="sum")

            total_loss += loss.item()
            total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
        ppl = loss_to_perplexity(avg_loss)

        if p == 0.0 or clean_ppl is None:
            clean_ppl = ppl
            clean_loss = avg_loss

        rel_ratio = (ppl / clean_ppl) if clean_ppl and not math.isnan(clean_ppl) else 1.0
        delta_log_ppl = (math.log(ppl) - math.log(clean_ppl)) if clean_ppl and ppl > 0 else 0.0

        point = {
            "corruption_rate_p": float(p),
            "loss_nll": round(avg_loss, 4),
            "perplexity": round(ppl, 2),
            "relative_degradation_ratio": round(rel_ratio, 4),
            "delta_log_ppl": round(delta_log_ppl, 4),
        }
        curve.append(point)

    # Fit linear slope: R(p) = 1.0 + beta * p
    p_vals = np.array([pt["corruption_rate_p"] for pt in curve])
    r_vals = np.array([pt["relative_degradation_ratio"] for pt in curve])

    if len(p_vals) > 1 and np.var(p_vals) > 0:
        # Constrained linear regression through (0, 1.0)
        # R(p) - 1.0 = beta * p  => beta = sum(p * (R - 1)) / sum(p^2)
        beta_slope = float(np.sum(p_vals * (r_vals - 1.0)) / np.sum(p_vals**2))
    else:
        beta_slope = 0.0

    return {
        "clean_perplexity": round(clean_ppl, 2) if clean_ppl else None,
        "degradation_slope_beta": round(beta_slope, 4),
        "curve": curve,
    }
