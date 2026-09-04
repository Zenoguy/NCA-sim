"""
Causal Neural Cellular Automata Language Model (NCA-LM).

Implements:
1. SinusoidalStepEmbedding: Parameter-free deterministic step conditioning for arbitrary K.
2. CausalNCAStep: Causal, dilated local cell update operator supporting:
   - Weight-shared causal NCA rule with multiscale dilation (shared_weights=True, single F_theta)
   - Step-specific CNN stack baseline (shared_weights=False, separate F_{theta_k})
   - Strict left-padding: zero future information leakage.
   - Full GRU gating across channel dimension.
3. NCA_LM: Autoregressive language model with dynamic microstep scaling (override_K).
"""

import math
from typing import List, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalStepEmbedding(nn.Module):
    """
    Deterministic sinusoidal step embedding.
    Maps an integer step index k to a fixed vector of dimension d_hidden.
    Valid for any arbitrary k >= 0 with zero learned parameters.
    """

    def __init__(self, d_hidden: int):
        super().__init__()
        self.d_hidden = d_hidden
        half_dim = d_hidden // 2
        # Compute frequencies: exp(-log(10000) * i / (half_dim - 1))
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("freqs", freqs)

    def forward(self, step_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns tensor of shape [1, d_hidden, 1] suitable for adding to 1D conv features [B, d_hidden, T].
        """
        freqs = self.freqs.to(device=device, dtype=torch.float32)
        args = float(step_idx) * freqs
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        if emb.shape[-1] < self.d_hidden:
            # Pad with zero if odd dimension
            emb = F.pad(emb, (0, self.d_hidden - emb.shape[-1]))
        return emb.view(1, self.d_hidden, 1).to(dtype=dtype)


class CausalNCAStep(nn.Module):
    """
    Causal, dilated local update operator for 1D sequences.
    
    Receptive Field Convention:
      Kernel size k_s = radius + 1 = 3 (with radius=2).
      At step k with dilation d_k = 2^k, the convolution accesses:
        {t - 2*d_k, t - d_k, t}
      Left padding pad_k = radius * d_k = 2 * 2^k guarantees causality.
      Receptive field after K steps:
        RF(K) = 1 + radius * sum_{k=0}^{K-1} 2^k = 1 + 2 * (2^K - 1)
        RF(6) = 1 + 2 * 63 = 127 tokens (126 past tokens + 1 current token).
    """

    def __init__(
        self,
        d_model: int,
        radius: int = 2,
        d_hidden: Optional[int] = None,
        max_K: int = 12,
        shared_weights: bool = True,
        step_embed_type: str = "sinusoidal",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden or (d_model * 2)
        self.radius = radius
        self.kernel_size = radius + 1  # 3 for radius=2
        self.max_K = max_K
        self.shared_weights = shared_weights
        self.step_embed_type = step_embed_type

        # Exponential dilation schedule: [1, 2, 4, 8, 16, 32, 64, ...]
        self.dilations = [2**i for i in range(max_K)]

        # Step conditioning
        if step_embed_type == "sinusoidal":
            self.step_embed = SinusoidalStepEmbedding(self.d_hidden)
            # Learnable scale initialized small (0.05) so it conditions without drowning local features
            self.step_scale = nn.Parameter(torch.tensor(0.05))
        elif step_embed_type == "learned":
            self.step_embed = nn.Parameter(torch.randn(max_K, self.d_hidden, 1) * 0.02)
            self.step_scale = None
        else:
            self.step_embed = None
            self.step_scale = None

        if self.shared_weights:
            # Single weight-shared causal NCA rule F_theta
            self.conv_weight = nn.Parameter(
                torch.randn(self.d_hidden, d_model, self.kernel_size)
                * (2.0 / (d_model * self.kernel_size)) ** 0.5
            )
            self.conv_bias = nn.Parameter(torch.zeros(self.d_hidden))

            # Shared GRU gates across iterations
            self.update_gate = nn.Conv1d(self.d_hidden + d_model, d_model, kernel_size=1)
            self.reset_gate = nn.Conv1d(self.d_hidden + d_model, d_model, kernel_size=1)
            self.candidate_nhood = nn.Conv1d(self.d_hidden, d_model, kernel_size=1)
            self.candidate_state = nn.Conv1d(d_model, d_model, kernel_size=1)
        else:
            # Untied baseline: independent convolution and GRU gates per step
            self.convs = nn.ModuleList([
                nn.Conv1d(d_model, self.d_hidden, kernel_size=self.kernel_size, padding=0)
                for _ in range(max_K)
            ])
            self.update_gates = nn.ModuleList([
                nn.Conv1d(self.d_hidden + d_model, d_model, kernel_size=1) for _ in range(max_K)
            ])
            self.reset_gates = nn.ModuleList([
                nn.Conv1d(self.d_hidden + d_model, d_model, kernel_size=1) for _ in range(max_K)
            ])
            self.cand_nhoods = nn.ModuleList([
                nn.Conv1d(self.d_hidden, d_model, kernel_size=1) for _ in range(max_K)
            ])
            self.cand_states = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=1) for _ in range(max_K)
            ])

    def get_dilation(self, step_idx: int) -> int:
        if step_idx < len(self.dilations):
            return self.dilations[step_idx]
        return 2**step_idx

    def forward(self, s: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        Forward pass for micro-step step_idx.
        s: State tensor of shape [B, d_model, T]
        Returns: Updated state tensor s_new of shape [B, d_model, T]
        """
        d = self.get_dilation(step_idx)
        pad_len = self.radius * d

        # STRICT LEFT PADDING: (pad_left, pad_right) = (pad_len, 0)
        # Guarantee: position t in s only depends on positions <= t
        s_padded = F.pad(s, (pad_len, 0))

        if self.shared_weights:
            neighborhood = F.conv1d(s_padded, self.conv_weight, self.conv_bias, dilation=d)
            update_gate = self.update_gate
            reset_gate = self.reset_gate
            cand_nhood = self.candidate_nhood
            cand_state = self.candidate_state
        else:
            conv_layer = self.convs[step_idx]
            neighborhood = F.conv1d(s_padded, conv_layer.weight, conv_layer.bias, dilation=d)
            update_gate = self.update_gates[step_idx]
            reset_gate = self.reset_gates[step_idx]
            cand_nhood = self.cand_nhoods[step_idx]
            cand_state = self.cand_states[step_idx]

        # Step conditioning
        if self.step_embed_type == "sinusoidal":
            emb = self.step_embed(step_idx, device=s.device, dtype=s.dtype) * self.step_scale
            neighborhood = neighborhood + emb
        elif self.step_embed_type == "learned":
            neighborhood = neighborhood + self.step_embed[step_idx]

        neighborhood = F.silu(neighborhood)

        # Full channel GRU recurrence
        joint = torch.cat([neighborhood, s], dim=1)
        z = torch.sigmoid(update_gate(joint))
        r = torch.sigmoid(reset_gate(joint))

        cand = torch.tanh(cand_nhood(neighborhood) + cand_state(r * s))
        s_new = (1.0 - z) * s + z * cand
        return s_new


class NCA_LM(nn.Module):
    """
    Autoregressive Language Model based on Causal Neural Cellular Automata.
    
    Supports:
    - Weight-shared causal NCA rule with multiscale dilation (Variant A, Variant D)
    - Step-specific CNN stack (Variant B, Variant C)
    - Tied token embedding and readout weights when d_hidden_channels == 0
    - Truncated and extrapolated test-time compute depth scaling via override_K
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 288,
        d_hidden_channels: int = 0,
        radius: int = 2,
        K: int = 6,
        max_K: int = 12,
        shared_weights: bool = True,
        step_embed_type: str = "sinusoidal",
        tie_weights: bool = True,
        use_norm: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_hidden_channels = d_hidden_channels
        self.d_model = d_embed + d_hidden_channels
        self.K = K
        self.max_K = max_K
        self.shared_weights = shared_weights
        self.use_norm = use_norm

        self.embed = nn.Embedding(vocab_size, d_embed)

        self.step = CausalNCAStep(
            d_model=self.d_model,
            radius=radius,
            d_hidden=self.d_model * 2,
            max_K=max_K,
            shared_weights=shared_weights,
            step_embed_type=step_embed_type,
        )

        self.norm = nn.LayerNorm(self.d_model) if use_norm else nn.Identity()
        self.readout = nn.Linear(self.d_model, vocab_size, bias=False)

        if tie_weights and (self.d_model == self.d_embed):
            self.readout.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if self.readout.weight is not self.embed.weight:
            nn.init.normal_(self.readout.weight, mean=0.0, std=0.02)

    @staticmethod
    def compute_receptive_field(K: int, radius: int = 2) -> int:
        """
        Analytical receptive field: RF = 1 + radius * sum_{k=0}^{K-1} 2^k = 1 + radius * (2^K - 1).
        Represents: (RF - 1) previous tokens + 1 current token.
        """
        return 1 + radius * (2**K - 1)

    def forward(self, x: torch.Tensor, override_K: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.
        x: Input token indices of shape [B, T]
        override_K: Optional integer overriding number of micro-step iterations
        Returns: Logits of shape [B, T, vocab_size]
        """
        B, T = x.shape
        e = self.embed(x).transpose(1, 2)  # [B, d_embed, T]

        if self.d_hidden_channels > 0:
            h0 = torch.zeros(B, self.d_hidden_channels, T, device=x.device, dtype=e.dtype)
            s = torch.cat([e, h0], dim=1)  # [B, d_model, T]
        else:
            s = e

        steps = override_K if override_K is not None else self.K
        for k in range(steps):
            s = self.step(s, step_idx=k)

        # Permute back to [B, T, d_model] for layer norm and readout
        s_out = s.transpose(1, 2)
        s_norm = self.norm(s_out)
        logits = self.readout(s_norm)
        return logits

    def forward_intermediates(self, x: torch.Tensor, override_K: Optional[int] = None) -> List[torch.Tensor]:
        """
        Returns list of internal sequence states [s_0, s_1, ..., s_K] of shape [B, d_model, T].
        Used by Probe 3E to measure latent error contraction over microsteps.
        """
        B, T = x.shape
        e = self.embed(x).transpose(1, 2)

        if self.d_hidden_channels > 0:
            h0 = torch.zeros(B, self.d_hidden_channels, T, device=x.device, dtype=e.dtype)
            s = torch.cat([e, h0], dim=1)
        else:
            s = e

        intermediates = [s.clone()]
        steps = override_K if override_K is not None else self.K
        for k in range(steps):
            s = self.step(s, step_idx=k)
            intermediates.append(s.clone())

        return intermediates

