"""
Vanilla Neural Cellular Automata (NCA) for 1D KdV Dynamics.

Preserves strictly local cellular computation:
- Receptive field restricted to radius r=1 (kernel_size=3) via circular 1D conv.
- Cell state: s_i = [u_i, h_i] in R^{1 + C_h}.
- Updates: Delta s_i = F_theta( P(s_i) ), s_{t+1} = s_t + Delta s_t.
- No global spatial pooling or all-to-all layers.

Includes an automated parameter-matching solver and FLOPs/MACs calculator.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class LocalPerception(nn.Module):
    """
    Pure local 1D perception using circular convolution (radius r=1).
    Maps state (B, C, N) -> (B, C_perc, N).
    """

    def __init__(self, channels: int, kernel_size: int = 3, physics_informed: bool = False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.radius = kernel_size // 2
        self.physics_informed = physics_informed

        if not physics_informed:
            # Pure local perception: learned depthwise circular conv
            self.conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels * 2,
                kernel_size=kernel_size,
                padding=self.radius,
                padding_mode="circular",
                groups=1,  # Allows cross-channel local interaction
            )
            self.out_channels = channels * 2
        else:
            # Physics-informed perception: Identity + Central Difference + Discrete Laplacian
            # Fixed stencil kernels + learned conv
            self.conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels * 2,
                kernel_size=kernel_size,
                padding=self.radius,
                padding_mode="circular",
            )
            # 3 extra channels per state: identity, 1st diff, 2nd diff
            self.out_channels = channels * 2 + channels * 3

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        p_learned = self.conv(s)
        if not self.physics_informed:
            return p_learned

        # Finite difference stencils with circular padding
        s_pad = torch.cat([s[:, :, -1:], s, s[:, :, :1]], dim=2)
        diff1 = 0.5 * (s_pad[:, :, 2:] - s_pad[:, :, :-2])
        diff2 = s_pad[:, :, 2:] - 2.0 * s + s_pad[:, :, :-2]
        return torch.cat([p_learned, s, diff1, diff2], dim=1)


class VanillaNCA(nn.Module):
    """
    Vanilla Local Neural Cellular Automaton.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        kernel_size: int = 3,
        mlp_hidden: Optional[int] = None,
        physics_informed: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.total_channels = 1 + hidden_dim  # 1 physical channel u + hidden_dim
        self.kernel_size = kernel_size
        self.physics_informed = physics_informed

        # Perception layer
        self.perception = LocalPerception(
            channels=self.total_channels,
            kernel_size=kernel_size,
            physics_informed=physics_informed,
        )

        # Cell-wise shared MLP (1x1 Conv)
        in_dim = self.perception.out_channels
        mid_dim = mlp_hidden or max(64, in_dim * 2)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(mid_dim, self.total_channels, kernel_size=1),
        )

        # Initialize last layer with small weights for stable residual steps
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def step(self, s: torch.Tensor) -> torch.Tensor:
        """
        Single NCA micro-update:
            s^{t+1} = s^t + F_theta( Perception(s^t) )
        """
        p = self.perception(s)
        ds = self.mlp(p)
        return s + ds

    def forward(self, u0: torch.Tensor, K: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance state by K NCA micro-steps corresponding to 1 physical delta_T:
            K micro-updates = delta_T
        Args:
            u0: (B, 1, N) initial physical field.
            K: Number of micro-steps.
        Returns:
            u_next: (B, 1, N)
            s_next: (B, total_channels, N)
        """
        B, _, N = u0.shape
        # Initialize full state s with zero hidden state
        h0 = torch.zeros(B, self.hidden_dim, N, device=u0.device, dtype=u0.dtype)
        s = torch.cat([u0, h0], dim=1)

        for _ in range(K):
            s = self.step(s)

        u_next = s[:, :1, :]
        return u_next, s

    def rollout(self, u0: torch.Tensor, num_macro_steps: int, K: int = 2) -> torch.Tensor:
        """
        Autonomous multi-step rollout:
        Produces states at [0, delta_T, 2*delta_T, ..., num_macro_steps*delta_T].
        Returns:
            trajectory: (B, num_macro_steps + 1, 1, N)
        """
        B, _, N = u0.shape
        h = torch.zeros(B, self.hidden_dim, N, device=u0.device, dtype=u0.dtype)
        s = torch.cat([u0, h], dim=1)

        trajectory = [s[:, :1, :]]

        for _ in range(num_macro_steps):
            for _ in range(K):
                s = self.step(s)
            trajectory.append(s[:, :1, :])

        return torch.stack(trajectory, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_nca_macs(
    model: nn.Module, N: int, K: int, is_memory_nca: bool = False
) -> int:
    """
    Compute total multiply-accumulate operations (MACs) per physical macro interval Delta T.
    MACs = K * (Perception MACs + MLP MACs [+ Memory MACs]) across all N cells.
    """
    total_macs_per_step = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            # MACs = in_channels * (out_channels / groups) * kernel_size * N
            k = m.kernel_size[0]
            macs = m.in_channels * (m.out_channels // m.groups) * k * N
            total_macs_per_step += macs

    return int(total_macs_per_step * K)


def find_matched_vanilla_channels(
    target_params: int,
    kernel_size: int = 3,
    mlp_ratio: int = 2,
    tolerance: float = 0.01,
) -> Tuple[int, int, int]:
    """
    Automatically search candidate (hidden_dim, mlp_hidden) pairs for VanillaNCA
    to match target_params within < 1% relative difference.

    Returns:
        (best_hidden_dim, best_mlp_hidden, actual_params)
    """
    best_c = None
    best_mlp = None
    best_diff = float("inf")
    best_params = 0

    # Search candidate hidden channels
    for c in range(4, 64):
        tot_chan = 1 + c
        in_dim = tot_chan * 2
        nominal_mlp = in_dim * mlp_ratio

        # Fine sweep mlp_hidden around nominal_mlp
        for mlp_hidden in range(max(16, nominal_mlp - 32), nominal_mlp + 48):
            model = VanillaNCA(hidden_dim=c, kernel_size=kernel_size, mlp_hidden=mlp_hidden)
            p = count_parameters(model)
            diff = abs(p - target_params)
            if diff < best_diff:
                best_diff = diff
                best_c = c
                best_mlp = mlp_hidden
                best_params = p

            if best_diff / target_params < 0.005:
                break
        if best_diff / target_params < 0.005:
            break

    return best_c, best_mlp, best_params

