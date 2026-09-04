"""
Decoder-Only Transformer Baseline with RoPE and Configurable Attention Modes.
Supports both Full Causal Attention and Sliding-Window Attention (W=128).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, x: torch.Tensor, seq_len: int):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            # shape: [seq_len, dim // 2] -> [seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
            self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k shape: [B, num_heads, T, head_dim]
        T = q.size(2)
        self._update_cache(q, T)
        cos = self._cos_cached[:, :, :T, :].to(q.dtype)
        sin = self._sin_cached[:, :, :T, :].to(k.dtype)

        def rotate_half(tensor):
            d_half = tensor.shape[-1] // 2
            x1 = tensor[..., :d_half]
            x2 = tensor[..., d_half:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention supporting both Full Causal and Sliding-Window Causal masking.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_mode: str = "causal",
        window_size: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_mode = attention_mode
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def _build_attention_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Builds boolean attention mask where True indicates allowed positions.
        """
        # Causal mask: query position i can attend to key position j if j <= i
        row_indices = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
        col_indices = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        mask = col_indices <= row_indices  # [T, T] causal

        if self.attention_mode == "sliding":
            # Sliding window: key position j must be >= i - window_size + 1
            # Exactly window_size tokens: [i - window_size + 1, ..., i]
            sliding_mask = col_indices >= (row_indices - self.window_size + 1)
            mask = mask & sliding_mask

        return mask  # [T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, T, T]

        mask = self._build_attention_mask(T, x.device)
        # Apply mask: fill disallowed positions with large negative value
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        # Numerical guard: if an entire row was masked (e.g. invalid bounds), softmax produces NaNs
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """SwiGLU / Gated MLP block."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer decoder block."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_mode: str = "causal",
        window_size: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            attention_mode=attention_mode,
            window_size=window_size,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = FeedForward(d_model, hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Complete Decoder-Only Transformer Language Model.
    Supports weight tying and both full causal and sliding-window attention.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        attention_mode: str = "causal",
        window_size: int = 128,
        mlp_ratio: float = 3.5,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.attention_mode = attention_mode
        self.window_size = window_size

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                attention_mode=attention_mode,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.tok_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] token IDs
        B, T = x.shape
        h = self.dropout(self.tok_embed(x))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)  # [B, T, vocab_size]
        return logits
