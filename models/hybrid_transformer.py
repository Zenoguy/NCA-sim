"""
Hybrid NCA-Transformer Language Model (Phase 4).

Combines a lightweight, weight-shared Causal NCA cellular adaptor (<5% parameter budget)
as a pre-attention local smoothing front-end with a decoder-only Transformer backbone.

Architectures:
- NCAAdaptorBlock: Weight-shared 2-step cellular adaptor with causal dilation and GRU gating.
- CNNAdaptorBlock: Matched-parameter unshared 2-layer causal convolutional control.
- HybridTransformerLM: Complete LM integrating token embedding, adaptor stem, Transformer blocks,
  and tied output head. Supports --adaptor_type [nca|cnn|none].
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_baseline import TransformerBlock, RotaryEmbedding
from models.nca_lm import SinusoidalStepEmbedding


class NCAAdaptorBlock(nn.Module):
    """
    Causal Weight-Shared Cellular Adaptor (Pre-Attention Smoothing Stem).
    
    Operates in a bottleneck latent space (default d_adaptor=160 for d_model=384):
    1. LayerNorm + Linear down-projection: d_model -> d_adaptor
    2. K iterative cellular update steps (default K=2):
       - Causal left padding with exponential dilation d_k = 2^k (receptive field RF=7 for K=2).
       - Weight-shared causal convolution (kernel_size=3, radius=2).
       - Parameter-free sinusoidal step embedding.
       - SiLU activation.
       - Full channel GRU recurrence (update gate, reset gate, candidate state).
    3. Linear up-projection: d_adaptor -> d_model
    4. Residual skip connection: y = x + Adaptor(x).
    
    Total parameters for d_model=384, d_adaptor=160: ~355k (<3.5% overhead on 10.2M Transformer).
    """

    def __init__(
        self,
        d_model: int = 384,
        d_adaptor: int = 160,
        radius: int = 2,
        K: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_adaptor = d_adaptor
        self.radius = radius
        self.kernel_size = radius + 1  # 3 for radius=2
        self.K = K

        self.ln_in = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_adaptor, bias=True)

        # Exponential dilation schedule: d_0 = 1, d_1 = 2
        self.dilations = [2**k for k in range(K)]

        # Sinusoidal step embedding conditioning
        self.step_embed = SinusoidalStepEmbedding(d_adaptor)
        self.step_scale = nn.Parameter(torch.tensor(0.05))

        # Weight-shared causal NCA convolution
        self.conv_weight = nn.Parameter(
            torch.randn(d_adaptor, d_adaptor, self.kernel_size)
            * (2.0 / (d_adaptor * self.kernel_size)) ** 0.5
        )
        self.conv_bias = nn.Parameter(torch.zeros(d_adaptor))

        # Shared GRU channel recurrence gates
        self.update_gate = nn.Conv1d(2 * d_adaptor, d_adaptor, kernel_size=1)
        self.reset_gate = nn.Conv1d(2 * d_adaptor, d_adaptor, kernel_size=1)
        self.candidate_nhood = nn.Conv1d(d_adaptor, d_adaptor, kernel_size=1)
        self.candidate_state = nn.Conv1d(d_adaptor, d_adaptor, kernel_size=1)

        # Up-projection and residual output
        self.proj_out = nn.Linear(d_adaptor, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Small initialization for output projection so initial forward pass is close to identity
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def _cellular_step(self, s: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        Executes one microstep of the causal cellular rule.
        s: [B, d_adaptor, T]
        """
        d = self.dilations[step_idx] if step_idx < len(self.dilations) else 2**step_idx
        pad_len = self.radius * d

        # Strict left-padding: (pad_left, pad_right) = (pad_len, 0)
        s_padded = F.pad(s, (pad_len, 0))

        # Weight-shared causal convolution
        nhood = F.conv1d(s_padded, self.conv_weight, self.conv_bias, dilation=d)

        # Step conditioning
        emb = self.step_embed(step_idx, device=s.device, dtype=s.dtype) * self.step_scale
        nhood = F.silu(nhood + emb)

        # GRU gating across channels
        joint = torch.cat([nhood, s], dim=1)
        z = torch.sigmoid(self.update_gate(joint))
        r = torch.sigmoid(self.reset_gate(joint))

        cand = torch.tanh(self.candidate_nhood(nhood) + self.candidate_state(r * s))
        s_next = (1.0 - z) * s + z * cand
        return s_next

    def forward(self, x: torch.Tensor, override_K: Optional[int] = None) -> torch.Tensor:
        """
        x: [B, T, d_model]
        Returns: [B, T, d_model]
        """
        B, T, C = x.shape
        num_steps = override_K if override_K is not None else self.K

        # Project to adaptor bottleneck: [B, T, d_adaptor] -> [B, d_adaptor, T]
        h = self.proj_in(self.ln_in(x))
        s = h.transpose(1, 2).contiguous()

        # Iterate cellular automaton
        for step_idx in range(num_steps):
            s = self._cellular_step(s, step_idx)

        # Project back to d_model: [B, d_adaptor, T] -> [B, T, d_model]
        s = s.transpose(1, 2).contiguous()
        out = self.dropout(self.proj_out(s))

        # Residual connection
        return x + out


class CNNAdaptorBlock(nn.Module):
    """
    Matched-Parameter Unshared Causal Conv Control (Phase 4 Control).
    
    A 2-layer causal convolutional feed-forward block matched in parameter budget
    and receptive field (RF=7) to NCAAdaptorBlock, but without cellular weight sharing
    or GRU recurrence.
    
    Total parameters for d_model=384, d_adaptor=160, d_mid=240: ~355k params.
    """

    def __init__(
        self,
        d_model: int = 384,
        d_adaptor: int = 160,
        d_mid: int = 240,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_adaptor = d_adaptor
        self.kernel_size = kernel_size
        self.radius = kernel_size - 1  # 2

        self.ln_in = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_adaptor, bias=True)

        # Layer 1: dilation 1, receptive field 3
        self.conv1 = nn.Conv1d(d_adaptor, d_mid, kernel_size=kernel_size, dilation=1)
        # Layer 2: dilation 2, receptive field 1 + 2 + 4 = 7
        self.conv2 = nn.Conv1d(d_mid, d_adaptor, kernel_size=kernel_size, dilation=2)

        self.proj_out = nn.Linear(d_adaptor, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, override_K: Optional[int] = None) -> torch.Tensor:
        """
        x: [B, T, d_model]
        Returns: [B, T, d_model]
        """
        B, T, C = x.shape
        h = self.proj_in(self.ln_in(x))
        s = h.transpose(1, 2).contiguous()  # [B, d_adaptor, T]

        # Conv 1: left pad = 2 * 1 = 2
        s = F.pad(s, (self.radius * 1, 0))
        s = F.silu(self.conv1(s))

        # Conv 2: left pad = 2 * 2 = 4
        s = F.pad(s, (self.radius * 2, 0))
        s = F.silu(self.conv2(s))

        s = s.transpose(1, 2).contiguous()
        out = self.dropout(self.proj_out(s))

        return x + out


class HybridTransformerLM(nn.Module):
    """
    Hybrid NCA-Transformer Language Model.
    
    Token Embedding -> [NCA / CNN Adaptor Stem] -> Transformer Blocks -> LayerNorm -> Readout Head.
    Supports weight tying and seamless ablation via adaptor_type="none" or bypass_adaptor=True.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_layers: int = 3,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        attention_mode: str = "causal",
        window_size: int = 128,
        dropout: float = 0.1,
        tie_weights: bool = True,
        adaptor_type: str = "nca",  # "nca", "cnn", or "none"
        adaptor_dim: int = 160,
        adaptor_K: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.attention_mode = attention_mode
        self.adaptor_type = adaptor_type.lower()
        self.adaptor_dim = adaptor_dim
        self.adaptor_K = adaptor_K

        # Token Embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Pre-Attention Stem Adaptor
        if self.adaptor_type == "nca":
            self.adaptor = NCAAdaptorBlock(
                d_model=d_model,
                d_adaptor=adaptor_dim,
                radius=2,
                K=adaptor_K,
                dropout=dropout,
            )
        elif self.adaptor_type == "cnn":
            self.adaptor = CNNAdaptorBlock(
                d_model=d_model,
                d_adaptor=adaptor_dim,
                d_mid=240,
                kernel_size=3,
                dropout=dropout,
            )
        elif self.adaptor_type == "none":
            self.adaptor = None
        else:
            raise ValueError(f"Unknown adaptor_type: {adaptor_type}. Expected 'nca', 'cnn', or 'none'.")

        # Transformer Backbone
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

    def count_parameters(self) -> Tuple[int, int, float]:
        """
        Returns (total_params, adaptor_params, overhead_percentage).
        """
        total = sum(p.numel() for p in self.parameters())
        adaptor = sum(p.numel() for p in self.adaptor.parameters()) if self.adaptor is not None else 0
        backbone = total - adaptor
        overhead_pct = (adaptor / backbone) * 100.0 if backbone > 0 else 0.0
        return total, adaptor, overhead_pct

    def forward(
        self,
        x: torch.Tensor,
        bypass_adaptor: bool = False,
        override_K: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        x: [B, T] Token IDs
        bypass_adaptor: If True, skips adaptor stem (for instant ablation testing).
        override_K: Override cellular steps in NCA adaptor.
        """
        h = self.dropout(self.tok_embed(x))

        if self.adaptor is not None and not bypass_adaptor:
            if isinstance(self.adaptor, NCAAdaptorBlock):
                h = self.adaptor(h, override_K=override_K)
            else:
                h = self.adaptor(h)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits
