"""
Memory-Augmented Neural Cellular Automaton (Memory-NCA).

Implements strictly local cellular computation with persistent recurrent memory.
Update sequence within each micro-step:
    P_t  -->  m_{t+1}  -->  s_{t+1}

Supported modes:
1. 'persistent': (Main benchmark) m_i^0 = 0, memory persists across macro steps.
2. 'no_persistence': (Control) Identical architecture, but m_i is reset to 0 every Delta T.
3. 'random_persistence': (Control) Persistent random static memory m_i ~ N(0, 1), not updated.
4. 'contextual': Memory can be initialized from a warm-up sequence for regime testing.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from src.nca import LocalPerception


class GatedMemoryCell(nn.Module):
    """
    Per-cell gated recurrent memory unit:
        tilde_m = tanh( W_m [P_t, m_t] + b_m )
        g       = sigmoid( W_g [P_t, m_t] + b_g )
        m_{t+1} = g * m_t + (1 - g) * tilde_m
    Operates with 1x1 convolutions (shared across all spatial cells).
    """

    def __init__(self, perc_dim: int, memory_dim: int):
        super().__init__()
        self.perc_dim = perc_dim
        self.memory_dim = memory_dim

        in_dim = perc_dim + memory_dim
        # Candidate memory generator
        self.cand_conv = nn.Conv1d(in_dim, memory_dim, kernel_size=1)
        # Gate generator
        self.gate_conv = nn.Conv1d(in_dim, memory_dim, kernel_size=1)

        # Initialize gate bias to 1.0 (retention bias, standard in GRU/LSTM)
        nn.init.constant_(self.gate_conv.bias, 1.0)

    def forward(self, p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        pm = torch.cat([p, m], dim=1)
        cand = torch.tanh(self.cand_conv(pm))
        g = torch.sigmoid(self.gate_conv(pm))
        m_next = g * m + (1.0 - g) * cand
        return m_next


class MemoryNCA(nn.Module):
    """
    Memory-Augmented Neural Cellular Automaton.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        memory_dim: int = 16,
        kernel_size: int = 3,
        mlp_hidden: Optional[int] = None,
        mode: str = "persistent",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.total_channels = 1 + hidden_dim  # Physical u + hidden
        self.kernel_size = kernel_size
        self.mode = mode

        # Local perception layer
        self.perception = LocalPerception(
            channels=self.total_channels,
            kernel_size=kernel_size,
            physics_informed=False,
        )
        perc_dim = self.perception.out_channels

        # Memory module (if memory_dim > 0)
        if memory_dim > 0:
            self.memory_cell = GatedMemoryCell(perc_dim=perc_dim, memory_dim=memory_dim)
        else:
            self.memory_cell = None

        # Cell-wise shared state update MLP
        # Input to MLP is: [P_t, m_{t+1}] if memory_dim > 0 else P_t
        mlp_in_dim = perc_dim + (memory_dim if memory_dim > 0 else 0)
        mid_dim = mlp_hidden or max(64, mlp_in_dim * 2)

        self.mlp = nn.Sequential(
            nn.Conv1d(mlp_in_dim, mid_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(mid_dim, self.total_channels, kernel_size=1),
        )

        # Initialize last layer with zeros for identity residual start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def init_memory(self, B: int, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize memory state based on active mode."""
        if self.memory_dim == 0:
            return torch.zeros(B, 0, N, device=device, dtype=dtype)

        if self.mode == "random_persistence":
            # Persistent fixed Gaussian noise
            return torch.randn(B, self.memory_dim, N, device=device, dtype=dtype)
        else:
            # Standard endogenous zero initialization
            return torch.zeros(B, self.memory_dim, N, device=device, dtype=dtype)

    def step(
        self, s: torch.Tensor, m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single micro-update following sequence:
            P_t --> m_{t+1} --> s_{t+1}
        """
        # 1. Perception
        p = self.perception(s)

        # 2. Memory update
        if self.memory_dim > 0:
            if self.mode == "random_persistence":
                # Static memory: retained without updating
                m_next = m
            else:
                m_next = self.memory_cell(p, m)
            # Combine perception and updated memory
            mlp_in = torch.cat([p, m_next], dim=1)
        else:
            m_next = m
            mlp_in = p

        # 3. State update
        ds = self.mlp(mlp_in)
        s_next = s + ds

        return s_next, m_next

    def forward(
        self,
        u0: torch.Tensor,
        K: int = 1,
        m0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advance state by K NCA micro-steps corresponding to 1 physical interval delta_T.
        """
        B, _, N = u0.shape
        h0 = torch.zeros(B, self.hidden_dim, N, device=u0.device, dtype=u0.dtype)
        s = torch.cat([u0, h0], dim=1)

        if m0 is None:
            m = self.init_memory(B, N, u0.device, u0.dtype)
        else:
            m = m0

        for _ in range(K):
            s, m = self.step(s, m)

        u_next = s[:, :1, :]
        return u_next, s, m

    def rollout(
        self,
        u0: torch.Tensor,
        num_macro_steps: int,
        K: int = 2,
        m0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autonomous multi-step rollout:
        Advances through num_macro_steps, executing K micro-steps per macro interval.

        Returns:
            trajectory: (B, num_macro_steps + 1, 1, N)
            final_memory: (B, memory_dim, N)
        """
        B, _, N = u0.shape
        h = torch.zeros(B, self.hidden_dim, N, device=u0.device, dtype=u0.dtype)
        s = torch.cat([u0, h], dim=1)

        if m0 is None:
            m = self.init_memory(B, N, u0.device, u0.dtype)
        else:
            m = m0

        trajectory = [s[:, :1, :]]

        for _ in range(num_macro_steps):
            if self.mode == "no_persistence":
                # Control: reset memory at every macro-step boundary
                m = torch.zeros_like(m)

            for _ in range(K):
                s, m = self.step(s, m)

            trajectory.append(s[:, :1, :])

        return torch.stack(trajectory, dim=1), m
