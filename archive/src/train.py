"""
Recurrent Multi-Step Macro Rollout Trainer for KdV NCA and Baselines.

Enforces the hard invariant:
    K NCA updates == Delta T physical time
Only macro-states u(t), u(t + Delta T), u(t + 2*Delta T), ... are supervised.
Intermediate NCA states are internal unobserved micro-steps.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.metrics import relative_l2_error


def normalized_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalized MSE loss: ||pred - target||_2^2 / (||target||_2^2 + eps).
    """
    diff_sq = torch.sum((pred - target) ** 2, dim=(-2, -1))
    target_sq = torch.sum(target ** 2, dim=(-2, -1)) + eps
    return torch.mean(diff_sq / target_sq)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    K: int = 2,
    rollout_steps: int = 8,
    grad_clip: float = 1.0,
) -> float:
    """
    Train model for one epoch using recurrent multi-step rollout.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch, _ in dataloader:
        batch = batch.to(device)  # (B, num_macro_steps + 1, 1, N)
        B, max_steps, _, N = batch.shape

        steps_to_roll = min(rollout_steps, max_steps - 1)
        if steps_to_roll < 1:
            continue

        optimizer.zero_grad()

        # Multi-step autonomous rollout over M macro steps
        u_curr = batch[:, 0]  # initial condition (B, 1, N)

        # Initialize model hidden/memory states
        if hasattr(model, "total_channels"):
            h = torch.zeros(B, model.hidden_dim, N, device=device, dtype=batch.dtype)
            s = torch.cat([u_curr, h], dim=1)
        else:
            s = u_curr


        if hasattr(model, "init_memory"):
            m = model.init_memory(B, N, device, batch.dtype)
        else:
            m = None

        loss = 0.0

        for m_step in range(1, steps_to_roll + 1):
            target_macro = batch[:, m_step]

            # In MemoryNCA no_persistence mode, reset memory at macro boundary
            if hasattr(model, "mode") and model.mode == "no_persistence" and m is not None:
                m = torch.zeros_like(m)

            # Advance K micro-steps corresponding to 1 physical Delta T
            for _ in range(K):
                if hasattr(model, "memory_dim"):
                    s, m = model.step(s, m)
                elif hasattr(model, "step"):
                    s = model.step(s)
                else:
                    s, _ = model.forward(s, K=1)


            u_pred = s[:, :1, :]
            step_loss = normalized_mse_loss(u_pred, target_macro)
            loss = loss + step_loss

        loss = loss / steps_to_roll
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    K: int = 2,
    rollout_steps: int = 8,
) -> float:
    """
    Evaluate autonomous rollout loss on validation data.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            B, max_steps, _, N = batch.shape
            steps_to_roll = min(rollout_steps, max_steps - 1)
            if steps_to_roll < 1:
                continue

            u0 = batch[:, 0]
            if hasattr(model, "rollout"):
                pred_traj, _ = (
                    model.rollout(u0, num_macro_steps=steps_to_roll, K=K)
                    if hasattr(model, "memory_dim")
                    else (model.rollout(u0, num_macro_steps=steps_to_roll, K=K), None)
                )
            else:
                pred_traj = model.rollout(u0, num_macro_steps=steps_to_roll, K=K)

            # Target trajectory over the same steps
            target_traj = batch[:, : steps_to_roll + 1]
            loss = normalized_mse_loss(pred_traj[:, 1:], target_traj[:, 1:])
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.002,
    weight_decay: float = 1e-4,
    K: int = 2,
    rollout_steps: int = 8,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    save_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Full training pipeline for a given model.
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    iterator = range(1, epochs + 1)
    if verbose:
        iterator = tqdm(iterator, desc="Training")

    for epoch in iterator:
        train_loss = train_epoch(
            model, train_loader, optimizer, device, K=K, rollout_steps=rollout_steps, grad_clip=grad_clip
        )
        val_loss = validate(model, val_loader, device, K=K, rollout_steps=rollout_steps)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss and save_path is not None:
            best_val_loss = val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )

        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"TrLoss": f"{train_loss:.4e}", "ValLoss": f"{val_loss:.4e}"})

    return history
