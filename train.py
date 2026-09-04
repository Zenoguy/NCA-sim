"""
Unified Training Script for Language Model Baselines.
Config-driven, reproducible, supporting AMP, Cosine schedule with warmup, and validation logging.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.dataset import get_dataloader
from eval.perplexity import evaluate_neural_perplexity, loss_to_perplexity
from models.transformer_baseline import TransformerLM
from models.mamba_baseline import MambaLM
from models.rnn_baseline import GRULM


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def build_model(cfg: dict) -> nn.Module:
    m_cfg = cfg["model"]
    m_type = m_cfg["type"].lower()
    if m_type == "transformer":
        return TransformerLM(
            vocab_size=m_cfg["vocab_size"],
            d_model=m_cfg["d_model"],
            num_layers=m_cfg["num_layers"],
            num_heads=m_cfg["num_heads"],
            mlp_ratio=m_cfg.get("mlp_ratio", 4.0),
            attention_mode=m_cfg.get("attention_mode", "causal"),
            window_size=m_cfg.get("window_size", 128),
            dropout=m_cfg.get("dropout", 0.1),
            tie_weights=m_cfg.get("tie_weights", True),
        )
    elif m_type == "mamba":
        return MambaLM(
            vocab_size=m_cfg["vocab_size"],
            d_model=m_cfg["d_model"],
            num_layers=m_cfg["num_layers"],
            d_state=m_cfg.get("d_state", 16),
            expand=m_cfg.get("expand", 2),
            d_conv=m_cfg.get("d_conv", 4),
            dropout=m_cfg.get("dropout", 0.1),
            tie_weights=m_cfg.get("tie_weights", True),
        )
    elif m_type == "gru":
        return GRULM(
            vocab_size=m_cfg["vocab_size"],
            d_model=m_cfg["d_model"],
            num_layers=m_cfg["num_layers"],
            dropout=m_cfg.get("dropout", 0.1),
            tie_weights=m_cfg.get("tie_weights", True),
        )
    else:
        raise ValueError(f"Unsupported model type: {m_type}")


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = min_lr / max_lr
        return min_factor + (1.0 - min_factor) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config_path: str, override_device: str = None, max_steps: int = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set seeds for strict reproducibility
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Select Device
    if override_device:
        device = torch.device(override_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    # Build Model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Instantiated {cfg['model']['type'].upper()} model with {total_params:,} trainable parameters.")

    # Load Data
    d_cfg = cfg["data"]
    seq_len = d_cfg.get("seq_len", 128)
    batch_size = d_cfg.get("batch_size", 32)

    train_tokens = np.load(d_cfg["train_npy"])
    valid_tokens = np.load(d_cfg["valid_npy"])
    test_tokens = np.load(d_cfg["test_npy"])

    train_loader = get_dataloader(train_tokens, seq_len=seq_len, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(valid_tokens, seq_len=seq_len, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_tokens, seq_len=seq_len, batch_size=batch_size, shuffle=False)

    print(f"Data Batches: Train={len(train_loader):,}, Valid={len(val_loader):,}, Test={len(test_loader):,}")

    # Setup Optimizer & Scheduler
    t_cfg = cfg["training"]
    lr = float(t_cfg.get("lr", 5e-4))
    min_lr = float(t_cfg.get("min_lr", 5e-5))
    weight_decay = float(t_cfg.get("weight_decay", 0.1))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    epochs = int(t_cfg.get("epochs", 5))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

    total_steps = epochs * len(train_loader) if max_steps is None else max_steps
    warmup_steps = int(t_cfg.get("warmup_steps", 200))
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr, lr)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(t_cfg.get("save_dir", "outputs/level1/run"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initial Validation
    print("Running initial validation check...")
    val_metrics = evaluate_neural_perplexity(model, val_loader, device=device)
    print(f"Initial Val Loss: {val_metrics['loss']:.4f} | Perplexity: {val_metrics['perplexity']:.2f}")

    best_val_loss = float("inf")
    history = []
    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        tokens_processed = 0
        t0 = time.time()

        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item() * targets.numel()
            tokens_processed += targets.numel()

            if max_steps and global_step >= max_steps:
                break

        epoch_time = time.time() - t0
        train_loss = epoch_loss / max(1, tokens_processed)
        train_ppl = loss_to_perplexity(train_loss)
        tok_per_sec = tokens_processed / max(0.1, epoch_time)

        # Validate
        val_metrics = evaluate_neural_perplexity(model, val_loader, device=device)
        val_loss = val_metrics["loss"]
        val_ppl = val_metrics["perplexity"]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:6.2f}) | "
            f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:6.2f}) | "
            f"Throughput: {tok_per_sec:,.0f} tok/s {'[*BEST*]' if is_best else ''}"
        )

        history.append({
            "epoch": epoch,
            "step": global_step,
            "train_loss": round(train_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "throughput_tok_per_sec": round(tok_per_sec, 1),
        })

        if max_steps and global_step >= max_steps:
            print(f"Reached max_steps ({max_steps}), ending training.")
            break

    # Save final model & evaluate on Test set
    torch.save(model.state_dict(), save_dir / "last_model.pt")

    # Load best model for official test evaluation
    if (save_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate_neural_perplexity(model, test_loader, device=device)
    print(f"\nFinal Test Loss: {test_metrics['loss']:.4f} | Test Perplexity: {test_metrics['perplexity']:.2f}")

    # Log complete provenance
    summary = {
        "git_commit": get_git_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": cfg["model"]["type"],
        "config_file": str(config_path),
        "seed": seed,
        "total_parameters": total_params,
        "device": str(device),
        "total_training_time_sec": round(time.time() - start_time, 2),
        "best_val_perplexity": round(loss_to_perplexity(best_val_loss), 2),
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
        "history": history,
    }

    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu/mps)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps for smoke tests")
    args = parser.parse_args()

    train(args.config, override_device=args.device, max_steps=args.max_steps)
