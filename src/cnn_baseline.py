"""
Recurrent 1D CNN Baseline Surrogate for KdV Dynamics.

4-layer convolutional network with periodic circular padding and residual connection:
    u_{t+Delta T} = u_t + CNN(u_t)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    Recurrent 1D Convolutional Neural Network baseline.
    """

    def __init__(self, hidden_dim: int = 32, kernel_size: int = 5, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        radius = kernel_size // 2

        layers = []
        # Input layer
        layers.append(
            nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=radius, padding_mode="circular")
        )
        layers.append(nn.GELU())

        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=radius,
                    padding_mode="circular",
                )
            )
            layers.append(nn.GELU())

        # Output layer
        layers.append(
            nn.Conv1d(
                hidden_dim, 1, kernel_size=kernel_size, padding=radius, padding_mode="circular"
            )
        )
        self.net = nn.Sequential(*layers)

        # Initialize output layer to zero for identity residual start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u: torch.Tensor, K: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance by 1 macro step (K is ignored or repeated)."""
        u_curr = u
        for _ in range(K):
            u_curr = u_curr + self.net(u_curr)
        return u_curr, u_curr

    def rollout(self, u0: torch.Tensor, num_macro_steps: int, K: int = 1) -> torch.Tensor:
        """Autonomous rollout over num_macro_steps."""
        trajectory = [u0]
        u = u0
        for _ in range(num_macro_steps):
            u, _ = self.forward(u, K=K)
            trajectory.append(u)
        return torch.stack(trajectory, dim=1)
