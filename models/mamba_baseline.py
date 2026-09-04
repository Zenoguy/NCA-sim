"""
Pure PyTorch Selective State-Space Model (Mamba / S6 Baseline).
Portable, verified, and runs natively on CPU and GPU without custom C++ CUDA compilation.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_loop(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Exact mathematical reference implementation of the selective scan via unrolled time loop.
    Args:
        u: [B, T, D_in] input
        delta: [B, T, D_in] discretization step
        A: [D_in, N] continuous state matrix (negative real)
        B: [B, T, N] input projection
        C: [B, T, N] output projection
        D: [D_in] optional skip parameter
    Returns:
        y: [B, T, D_in] output sequence
    """
    B_sz, T, D_in = u.shape
    N = A.shape[1]

    # Pre-discretize A: dA = exp(delta * A) -> [B, T, D_in, N]
    # delta: [B, T, D_in, 1], A: [1, 1, D_in, N]
    delta_expanded = delta.unsqueeze(-1)
    A_expanded = A.view(1, 1, D_in, N)
    dA = torch.exp(delta_expanded * A_expanded)  # [B, T, D_in, N]

    # Pre-discretize B: dB_u = (delta * u) * B
    # delta * u: [B, T, D_in, 1], B: [B, T, 1, N] -> [B, T, D_in, N]
    delta_u = (delta * u).unsqueeze(-1)
    B_expanded = B.unsqueeze(2)
    dB_u = delta_u * B_expanded  # [B, T, D_in, N]

    h = torch.zeros(B_sz, D_in, N, device=u.device, dtype=u.dtype)
    y_list = []

    C_expanded = C.unsqueeze(2)  # [B, T, 1, N]

    for t in range(T):
        h = dA[:, t] * h + dB_u[:, t]  # [B, D_in, N]
        # Contract with C: sum over N
        y_t = (h * C_expanded[:, t]).sum(dim=-1)  # [B, D_in]
        y_list.append(y_t)

    y = torch.stack(y_list, dim=1)  # [B, T, D_in]

    if D is not None:
        y = y + u * D.view(1, 1, D_in)

    return y


class MambaBlock(nn.Module):
    """
    Single Mamba (S6) Selective State Space Layer.
    Combines 1D causal convolution, input-dependent discretization, and selective scan.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # In-projection: projects input to (u, z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal 1D Convolution over sequence
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=0,
        )

        # Projection to selective parameters (delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4 parameter A: parameterized as -exp(A_log) for guaranteed numerical stability
        A = torch.repeat_interleave(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0), self.d_inner, dim=0)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Initialize dt_proj bias to span log(dt_min) to log(dt_max)
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            # Initialize dt_proj.weight with small uniform to let bias dominate initial discretization
            nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
            nn.init.uniform_(self.x_proj.weight, -0.1, 0.1)

        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)


    def forward(self, x: torch.Tensor, use_reference_loop: bool = False) -> torch.Tensor:
        # x: [B, T, d_model]
        B, T, D = x.shape
        xz = self.in_proj(x)  # [B, T, 2 * d_inner]
        u, z = xz.chunk(2, dim=-1)  # each [B, T, d_inner]

        # 1D Causal convolution with strict left-padding
        u_t = u.transpose(1, 2)  # [B, d_inner, T]
        u_padded = F.pad(u_t, (self.d_conv - 1, 0))
        u_conv = self.conv1d(u_padded)[:, :, :T].transpose(1, 2)  # [B, T, d_inner]
        u_conv = F.silu(u_conv)

        # Project to selective parameters
        x_dbl = self.x_proj(u_conv)  # [B, T, dt_rank + 2 * d_state]
        dt, B_mat, C_mat = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # [B, T, d_inner]

        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Execute selective scan
        y = selective_scan_loop(u_conv, dt, A, B_mat, C_mat, self.D)

        # Gate with z
        y = y * F.silu(z)

        return self.out_proj(y)


class MambaLM(nn.Module):
    """
    Complete Mamba Language Model.
    Stacks Mamba SSM blocks with LayerNorm and tied embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    d_conv=d_conv,
                ),
            })
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
        B, T = x.shape
        h = self.dropout(self.tok_embed(x))
        for layer in self.layers:
            h = h + layer["mamba"](layer["norm"](h))
        h = self.ln_f(h)
        logits = self.head(h)
        return logits
