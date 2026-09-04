"""
Transport-Augmented Neural Cellular Automaton (Adv-NCA).

Implements differentiable semi-Lagrangian transport of persistent memory
along physical characteristics or learned coherent-structure velocity fields:
    m_t + v(x, t) m_x = F(m, P)

Supports:
1. Dual-Memory Decomposition:
   m = [m_transport, m_local]
   m_transport is advected by v(x, t), while m_local remains Eulerian.
2. Five Transport Modes:
   - 'stationary': v = 0 (Eulerian baseline control)
   - 'characteristic': v = 6u (Nonlinear PDE convective characteristic)
   - 'learned': v = V_theta(s, m) (Learned coherent-structure velocity field)
   - 'oracle_estimated': v = 2 * max_x(u) (Soliton velocity estimated from current state)
   - 'oracle_true': v = 2 * A_true (Soliton velocity from privileged ground-truth amplitude)
3. Causal Velocity Interventions:
   - Normal v
   - Zero v (v -> 0)
   - Reversed v (v -> -v)
   - Random Gaussian v
   - Magnitude-matched phase-randomized v
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from src.nca import LocalPerception
from src.memory_nca import GatedMemoryCell


class DifferentiableSemiLagrangian1D(nn.Module):
    """
    Differentiable 1D Semi-Lagrangian Transport Operator with Periodic Boundary.
    Avoids discrete sign-branching and hard clamps, providing smooth departure
    gradients with respect to continuous velocity displacements almost everywhere.
    """

    def __init__(self, Lx: float = 50.0):
        super().__init__()
        self.Lx = Lx

    def forward(
        self,
        m: torch.Tensor,
        v: torch.Tensor,
        delta_t: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Advect memory m by velocity field v over time interval delta_t:
            m^*(x) = m(x - v(x)*delta_t)
        Args:
            m: (B, C_trans, N) Transported memory channels
            v: (B, 1, N) Velocity field
            delta_t: Time step duration (Delta T / K)
        Returns:
            m_star: (B, C_trans, N) Advected memory
            diagnostics: Dictionary of CFL and transport diagnostic metrics
        """
        B, C_trans, N = m.shape
        dx = self.Lx / N

        # Departure displacement in grid cells
        disp = v * (delta_t / dx)  # (B, 1, N)

        # Coordinate grid indices [0, 1, ..., N-1]
        i_coords = torch.arange(N, device=m.device, dtype=m.dtype).view(1, 1, N)

        # Continuous departure coordinates with periodic wrapping
        x_star = (i_coords - disp) % float(N)

        # Base integer departure cell and fractional offset
        j = torch.floor(x_star).long() % N
        j_next = (j + 1) % N
        lam = x_star - torch.floor(x_star)  # in [0, 1)

        # Vectorized gather across channels
        j_expanded = j.expand(-1, C_trans, -1)
        j_next_expanded = j_next.expand(-1, C_trans, -1)
        lam_expanded = lam.expand(-1, C_trans, -1)

        m_j = torch.gather(m, 2, j_expanded)
        m_j_next = torch.gather(m, 2, j_next_expanded)

        m_star = (1.0 - lam_expanded) * m_j + lam_expanded * m_j_next

        # CFL and transport diagnostics (detached from computation graph)
        with torch.no_grad():
            abs_disp = torch.abs(disp)
            mean_abs_v = float(torch.mean(torch.abs(v)).item())
            max_abs_v = float(torch.max(torch.abs(v)).item())
            mean_disp = float(torch.mean(abs_disp).item())
            max_disp = float(torch.max(abs_disp).item())
            frac_disp_gt_1 = float((abs_disp > 1.0).float().mean().item())
            frac_disp_gt_half_N = float((abs_disp > (N / 2.0)).float().mean().item())

            # Mass conservation divergence error: |sum(m*) - sum(m)| / (sum(|m|) + eps)
            mass_initial = torch.sum(m, dim=-1)
            mass_advected = torch.sum(m_star, dim=-1)
            mass_error = float(
                torch.mean(
                    torch.abs(mass_advected - mass_initial)
                    / (torch.sum(torch.abs(m), dim=-1) + 1e-7)
                ).item()
            )

        diagnostics = {
            "mean_abs_v": mean_abs_v,
            "max_abs_v": max_abs_v,
            "mean_disp": mean_disp,
            "max_disp": max_disp,
            "frac_disp_gt_1": frac_disp_gt_1,
            "frac_disp_gt_half_N": frac_disp_gt_half_N,
            "mass_conservation_error": mass_error,
        }

        return m_star, diagnostics


class TransportVelocityNet(nn.Module):
    """
    Compact 1x1 Convolutional Network predicting local transport velocity:
        v_hat(x) = V_theta(s_i, m_i)
    Initialized with zero weights and zero bias so v_hat = 0 at epoch 0,
    enabling a smooth continuous departure from the stationary memory baseline.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        # Initialize last layer with zeros
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdvectiveMemoryNCA(nn.Module):
    """
    Transport-Augmented Neural Cellular Automaton (Adv-NCA).
    Integrates dual-memory partitioning, differentiable semi-Lagrangian transport,
    and multiple physical/learned transport modes.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        memory_dim: int = 16,
        transport_dim: int = 8,
        kernel_size: int = 3,
        mlp_hidden: Optional[int] = None,
        mode: str = "learned",
        Lx: float = 50.0,
        delta_T: float = 0.1,
        K: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.transport_dim = min(transport_dim, memory_dim)
        self.local_dim = memory_dim - self.transport_dim
        self.total_channels = 1 + hidden_dim  # Physical u + hidden h
        self.kernel_size = kernel_size
        self.mode = mode
        self.Lx = Lx
        self.delta_T = delta_T
        self.K = K
        self.delta_t = delta_T / float(K)

        # Local perception layer (radius r=1)
        self.perception = LocalPerception(
            channels=self.total_channels,
            kernel_size=kernel_size,
            physics_informed=False,
        )
        perc_dim = self.perception.out_channels

        # Differentiable Semi-Lagrangian Transport module
        self.transport_module = DifferentiableSemiLagrangian1D(Lx=Lx)

        # Velocity prediction network (instantiated only when mode == 'learned')
        # Takes current physical state s = [u, h] and memory m
        if self.mode == "learned":
            vel_in_dim = self.total_channels + memory_dim
            self.velocity_net = TransportVelocityNet(in_channels=vel_in_dim, hidden_dim=8)
        else:
            self.velocity_net = None

        # Gated memory module (operates on advected memory m* and perception P)
        if memory_dim > 0:
            self.memory_cell = GatedMemoryCell(perc_dim=perc_dim, memory_dim=memory_dim)
        else:
            self.memory_cell = None

        # Cell-wise shared state update MLP
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

        # Optional velocity override hook for causal interventions
        self.velocity_override: Optional[
            Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]
        ] = None

    def init_memory(
        self, B: int, N: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Initialize memory state with zeros."""
        return torch.zeros(B, self.memory_dim, N, device=device, dtype=dtype)

    def compute_velocity(
        self,
        s: torch.Tensor,
        m: torch.Tensor,
        oracle_A: Optional[torch.Tensor] = None,
        true_A: Optional[torch.Tensor] = None,
        apply_override: bool = True,
    ) -> torch.Tensor:
        """
        Compute or lookup the local transport velocity field v(x, t) for the current micro-step.
        """
        B, _, N = s.shape

        # 1. Check for active causal intervention hook
        if apply_override and self.velocity_override is not None:
            if callable(self.velocity_override):
                return self.velocity_override(s)
            elif isinstance(self.velocity_override, torch.Tensor):
                return self.velocity_override.to(device=s.device, dtype=s.dtype)
            elif isinstance(self.velocity_override, (int, float)):
                return torch.full((B, 1, N), float(self.velocity_override), device=s.device, dtype=s.dtype)

        # 2. Compute velocity based on active mode
        if self.mode == "stationary":
            return torch.zeros(B, 1, N, device=s.device, dtype=s.dtype)

        elif self.mode == "characteristic":
            # Nonlinear convective transport: v(x) = 6 * u(x)
            u = s[:, :1, :]
            return 6.0 * u

        elif self.mode == "learned":
            # Learned transport velocity from local state and memory
            sm = torch.cat([s, m], dim=1)
            return self.velocity_net(sm)

        elif self.mode == "oracle_estimated":
            # Oracle coherent-structure speed estimated from current peak amplitude: v = 2 * max_x(u)
            u = s[:, :1, :]
            u_max = torch.amax(u, dim=-1, keepdim=True).view(B, 1, 1)
            return 2.0 * u_max.expand(B, 1, N)

        elif self.mode == "oracle_true":
            # Oracle coherent-structure speed from privileged true amplitude: v = 2 * A_true
            if true_A is not None:
                true_A_t = true_A.to(device=s.device, dtype=s.dtype).view(B, 1, 1)
                return 2.0 * true_A_t.expand(B, 1, N)
            elif oracle_A is not None:
                oracle_A_t = oracle_A.to(device=s.device, dtype=s.dtype).view(B, 1, 1)
                return 2.0 * oracle_A_t.expand(B, 1, N)
            else:
                u = s[:, :1, :]
                u_max = torch.amax(u, dim=-1, keepdim=True).view(B, 1, 1)
                return 2.0 * u_max.expand(B, 1, N)

        else:
            raise ValueError(f"Unknown transport mode: {self.mode}")

    def step(
        self,
        s: torch.Tensor,
        m: torch.Tensor,
        oracle_A: Optional[torch.Tensor] = None,
        true_A: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Single micro-step executing:
            Compute v -> Transport m_trans -> Perceive -> Gate m -> Update s
        """
        B, _, N = s.shape

        # 1. Compute velocity field
        v = self.compute_velocity(s, m, oracle_A=oracle_A, true_A=true_A)

        # 2. Dual-Memory Decomposition & Transport
        if self.transport_dim > 0 and self.mode != "stationary":
            m_trans = m[:, : self.transport_dim, :]
            m_local = m[:, self.transport_dim :, :]

            # Advect transported channels
            m_trans_star, diag = self.transport_module(m_trans, v, self.delta_t)
            # Local channels remain strictly Eulerian
            m_star = torch.cat([m_trans_star, m_local], dim=1)
        else:
            m_star = m
            diag = {
                "mean_abs_v": float(torch.mean(torch.abs(v)).item()),
                "max_abs_v": float(torch.max(torch.abs(v)).item()),
                "mean_disp": 0.0,
                "max_disp": 0.0,
                "frac_disp_gt_1": 0.0,
                "frac_disp_gt_half_N": 0.0,
                "mass_conservation_error": 0.0,
            }

        diag["velocity_field"] = v.detach()

        # 3. Local Perception
        p = self.perception(s)

        # 4. Gated Memory Update
        if self.memory_cell is not None:
            m_next = self.memory_cell(p, m_star)
            mlp_in = torch.cat([p, m_next], dim=1)
        else:
            m_next = m_star
            mlp_in = p

        # 5. State Update
        ds = self.mlp(mlp_in)
        s_next = s + ds

        return s_next, m_next, diag

    def forward(
        self,
        u0: torch.Tensor,
        K: Optional[int] = None,
        m0: Optional[torch.Tensor] = None,
        oracle_A: Optional[torch.Tensor] = None,
        true_A: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
        """Advance state by K NCA micro-steps corresponding to 1 physical interval Delta T."""
        K_steps = K or self.K
        B, _, N = u0.shape
        h0 = torch.zeros(B, self.hidden_dim, N, device=u0.device, dtype=u0.dtype)
        s = torch.cat([u0, h0], dim=1)

        m = self.init_memory(B, N, u0.device, u0.dtype) if m0 is None else m0
        step_diags = []

        for _ in range(K_steps):
            s, m, diag = self.step(s, m, oracle_A=oracle_A, true_A=true_A)
            step_diags.append(diag)

        u_next = s[:, :1, :]
        return u_next, s, m, step_diags

    def rollout(
        self,
        u0: torch.Tensor,
        num_macro_steps: int,
        K: Optional[int] = None,
        m0: Optional[torch.Tensor] = None,
        oracle_A: Optional[torch.Tensor] = None,
        true_A: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
        """
        Autonomous multi-step rollout over num_macro_steps.
        Returns:
            trajectory: (B, num_macro_steps + 1, 1, N)
            final_memory: (B, memory_dim, N)
            rollout_diags: List of transport diagnostics per step
        """
        K_steps = K or self.K
        B, _, N = u0.shape
        h = torch.zeros(B, self.hidden_dim, N, device=u0.device, dtype=u0.dtype)
        s = torch.cat([u0, h], dim=1)

        m = self.init_memory(B, N, u0.device, u0.dtype) if m0 is None else m0
        trajectory = [s[:, :1, :]]
        rollout_diags = []

        for _ in range(num_macro_steps):
            for _ in range(K_steps):
                s, m, diag = self.step(s, m, oracle_A=oracle_A, true_A=true_A)
                rollout_diags.append(diag)
            trajectory.append(s[:, :1, :])

        return torch.stack(trajectory, dim=1), m, rollout_diags


def compute_advective_macs(
    model: AdvectiveMemoryNCA, N: int, K: int
) -> int:
    """Compute total MACs per physical macro interval Delta T."""
    total_macs_per_step = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            k = m.kernel_size[0]
            macs = m.in_channels * (m.out_channels // m.groups) * k * N
            total_macs_per_step += macs

    # Add semi-Lagrangian interpolation MACs (approx 2 FLOPs/MACs per cell per transported channel)
    transport_macs = model.transport_dim * 2 * N
    total_macs_per_step += transport_macs

    return int(total_macs_per_step * K)


def find_matched_advective_mlp(
    target_params: int = 7765,
    hidden_dim: int = 16,
    memory_dim: int = 16,
    transport_dim: int = 8,
    mode: str = "learned",
) -> Tuple[int, int]:
    """
    Search mlp_hidden for AdvectiveMemoryNCA to match target_params within < 0.2%.
    Returns:
        (best_mlp_hidden, actual_params)
    """
    best_mlp = 64
    best_diff = float("inf")
    best_params = 0

    for mlp_h in range(32, 128):
        model = AdvectiveMemoryNCA(
            hidden_dim=hidden_dim,
            memory_dim=memory_dim,
            transport_dim=transport_dim,
            mlp_hidden=mlp_h,
            mode=mode,
        )
        p = sum(param.numel() for param in model.parameters() if param.requires_grad)
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_mlp = mlp_h
            best_params = p

    return best_mlp, best_params
