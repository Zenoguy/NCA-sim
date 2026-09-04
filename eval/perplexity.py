"""
Unified Perplexity & Evaluation Module.
Shared across all baselines, n-grams, and future neural models to ensure consistent metrics.
"""

import math
from typing import Dict, Union, Iterable
import numpy as np
import torch
import torch.nn.functional as F


def loss_to_perplexity(loss: float, max_loss: float = 50.0) -> float:
    """
    Safely convert cross-entropy loss to perplexity with overflow guards.
    e^50 is ~5e21, safely preventing float overflow while indicating divergence.
    """
    if math.isnan(loss) or math.isinf(loss):
        return float('inf')
    clamped_loss = min(max(loss, 0.0), max_loss)
    return math.exp(clamped_loss)


def compute_discrete_nll(log_probs: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Compute average negative log-likelihood (base e) and perplexity from an array of log probabilities.
    Args:
        log_probs: array of log P(x_i | x_{<i}) in natural log base.
    Returns:
        {"loss": float, "perplexity": float, "total_tokens": int}
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.cpu().numpy()

    total_tokens = len(log_probs)
    if total_tokens == 0:
        return {"loss": float('nan'), "perplexity": float('nan'), "total_tokens": 0}

    avg_nll = -float(np.mean(log_probs))
    ppl = loss_to_perplexity(avg_nll)
    return {
        "loss": round(avg_nll, 4),
        "perplexity": round(ppl, 2),
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def evaluate_neural_perplexity(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: Union[str, torch.device] = "cpu",
    pad_id: int = None,
) -> Dict[str, float]:
    """
    Standardized evaluation of PyTorch neural language models over a DataLoader.
    Computes exact token-weighted cross entropy and perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass: model outputs logits [B, T, V]
        logits = model(inputs)
        
        # Flatten for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        if pad_id is not None:
            mask = targets_flat != pad_id
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            batch_loss = (loss * mask).sum().item()
            batch_tokens = mask.sum().item()
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum')
            batch_loss = loss.item()
            batch_tokens = targets_flat.numel()

        total_loss += batch_loss
        total_tokens += batch_tokens

    if total_tokens == 0:
        return {"loss": float('nan'), "perplexity": float('nan'), "total_tokens": 0}

    avg_loss = total_loss / total_tokens
    ppl = loss_to_perplexity(avg_loss)

    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "total_tokens": total_tokens,
    }
