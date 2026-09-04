"""
High-Precision Fourier Pseudospectral KdV Solver using ETDRK4.

Solves the 1D Korteweg-de Vries (KdV) equation on a periodic domain:
    u_t + alpha * u * u_x + beta * u_{xxx} = 0

Canonical normalized form: alpha = 6.0, beta = 1.0.

Time integration uses Exponential Time Differencing 4th-Order Runge-Kutta
(ETDRK4, Kassam & Trefethen 2005) with 2/3 dealiasing to eliminate high-frequency
aliasing instability while avoiding the stiff linear dispersion time-step constraint.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class KdVSolver:
    """
    Fourier Pseudospectral ETDRK4 solver for 1D KdV.
    """

    def __init__(
        self,
        N: int = 256,
        Lx: float = 50.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        dt: float = 0.005,
    ):
        """
        Initialize the KdV solver.

        Args:
            N: Number of spatial grid points (must be even, preferably power of 2).
            Lx: Spatial domain length [-Lx/2, Lx/2).
            alpha: Nonlinear advection parameter (canonical = 6.0).
            beta: Linear dispersion parameter (canonical = 1.0).
            dt: Default internal simulation timestep.
        """
        self.N = N
        self.Lx = Lx
        self.alpha = alpha
        self.beta = beta
        self.dt = dt

        # Spatial grid
        self.dx = Lx / N
        self.x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)

        # Fourier wavenumbers
        self.k = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)

        # Dealiasing mask (2/3 rule)
        k_max = np.max(np.abs(self.k))
        self.dealias = np.abs(self.k) < (2.0 / 3.0) * k_max

        # Linear dispersion operator: u_t = L(u) + N(u)
        # u_t + beta * u_xxx = 0 => u_t = -beta * u_xxx
        # In Fourier: F{u_xxx} = (ik)^3 u_hat = -i k^3 u_hat
        # Therefore: L(k) = -beta * (-i k^3) = i * beta * k^3
        self.L_op = 1j * self.beta * (self.k**3)

        # Precompute ETDRK4 coefficients for the default dt
        self._precompute_etdrk4(self.dt)

    def _precompute_etdrk4(self, dt: float) -> None:
        """
        Precompute ETDRK4 operators using complex contour integration
        (Kassam & Trefethen 2005) for numerical stability around z=0.
        """
        self.dt = dt
        z = self.L_op * dt
        self.E = np.exp(z)
        self.E2 = np.exp(z / 2.0)

        # Contour integration on unit circle around each z
        M = 32
        r = 1.0
        theta = 2.0 * np.pi * (np.arange(1, M + 1) - 0.5) / M
        w = z[:, None] + r * np.exp(1j * theta)[None, :]

        # ETDRK4 functions:
        # Q = dt * (exp(z/2) - 1) / z
        # f1 = (-4 - z + exp(z)*(4 - 3z + z^2)) / z^3
        # f2 = (2 + z + exp(z)*(-2 + z)) / z^3
        # f3 = (-4 - 3z - z^2 + exp(z)*(4 - z)) / z^3
        self.Q = dt * np.mean((np.exp(w / 2.0) - 1.0) / w, axis=1)
        self.f1 = np.mean(
            (-4.0 - w + np.exp(w) * (4.0 - 3.0 * w + w**2)) / (w**3), axis=1
        )
        self.f2 = np.mean((2.0 + w + np.exp(w) * (-2.0 + w)) / (w**3), axis=1)
        self.f3 = np.mean(
            (-4.0 - 3.0 * w - w**2 + np.exp(w) * (4.0 - w)) / (w**3), axis=1
        )

    def _nonlinear_op(self, u_hat: np.ndarray) -> np.ndarray:
        """
        Evaluate dealiased nonlinear convection:
            N(u) = -alpha * u * u_x = -alpha/2 * d/dx(u^2)
        """
        u = np.real(np.fft.ifft(u_hat))
        u2_hat = np.fft.fft(u**2)
        return -0.5 * self.alpha * 1j * self.k * u2_hat * self.dealias

    def step_fourier(self, u_hat: np.ndarray) -> np.ndarray:
        """
        Advance one internal time step in Fourier space using ETDRK4.
        """
        Na = self._nonlinear_op(u_hat)
        a = self.E2 * u_hat + self.Q * Na

        Nb = self._nonlinear_op(a)
        b = self.E2 * u_hat + self.Q * Nb

        Nc = self._nonlinear_op(b)
        c = self.E2 * a + self.Q * (2.0 * Nc - Na)

        Nd = self._nonlinear_op(c)
        u_hat_next = self.E * u_hat + self.dt * (
            self.f1 * Na + 2.0 * self.f2 * (Nb + Nc) + self.f3 * Nd
        )
        return u_hat_next

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Advance one internal time step in physical space.
        """
        u_hat = np.fft.fft(u)
        u_hat_next = self.step_fourier(u_hat)
        return np.real(np.fft.ifft(u_hat_next))

    def solve(
        self,
        u0: np.ndarray,
        t_eval: np.ndarray,
        dt_internal: Optional[float] = None,
    ) -> np.ndarray:
        """
        Solve KdV equation from t_eval[0] to t_eval[-1], outputting states
        strictly at t_eval observation points.

        Args:
            u0: Initial physical state u(x, t_eval[0]).
            t_eval: 1D array of output times [t0, t1, ..., tM]. Must be monotonically increasing.
            dt_internal: Internal step size for ETDRK4. If different from self.dt,
                         ETDRK4 operators are recomputed.

        Returns:
            Array of shape (len(t_eval), N) containing physical states u(x, t_m).
        """
        if dt_internal is not None and abs(dt_internal - self.dt) > 1e-12:
            self._precompute_etdrk4(dt_internal)

        num_times = len(t_eval)
        trajectory = np.zeros((num_times, self.N), dtype=np.float64)
        trajectory[0] = u0.copy()

        u_hat = np.fft.fft(u0)
        current_t = t_eval[0]

        for i in range(1, num_times):
            target_t = t_eval[i]
            delta_t = target_t - current_t
            if delta_t < 0:
                raise ValueError("t_eval must be monotonically increasing.")

            # Number of internal steps to target_t
            n_substeps = max(1, int(round(delta_t / self.dt)))
            sub_dt = delta_t / n_substeps

            if abs(sub_dt - self.dt) > 1e-12:
                # Temporarily adapt operators for exact matching
                self._precompute_etdrk4(sub_dt)

            for _ in range(n_substeps):
                u_hat = self.step_fourier(u_hat)

            current_t = target_t
            trajectory[i] = np.real(np.fft.ifft(u_hat))

        return trajectory

    def exact_soliton(
        self,
        x: Optional[np.ndarray] = None,
        t: float = 0.0,
        A: float = 1.0,
        x0: float = 0.0,
    ) -> np.ndarray:
        """
        Evaluate the exact single traveling soliton solution on the periodic domain:
            u(x, t) = A * sech^2( (x - v*t - x0) / L )
        with:
            L = sqrt( 12 * beta / (alpha * A) )
            v = alpha * A / 3
        """
        if x is None:
            x = self.x

        L = np.sqrt(12.0 * self.beta / (self.alpha * A))
        v = (self.alpha * A) / 3.0

        # Periodic shift on domain [-Lx/2, Lx/2)
        xi = (x - v * t - x0 + self.Lx / 2.0) % self.Lx - self.Lx / 2.0
        return A / (np.cosh(xi / L) ** 2)

    def exact_soliton_L(self, A: float) -> float:
        """Return analytical width L for amplitude A on exact soliton manifold."""
        return float(np.sqrt(12.0 * self.beta / (self.alpha * A)))

    def exact_soliton_velocity(self, A: float) -> float:
        """Return analytical speed v for amplitude A."""
        return float((self.alpha * A) / 3.0)

    def compute_invariants(self, u: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute conserved quantities for KdV with periodic boundary:
            I1: Zeroth-order integral  = integral(u dx)
            I2: Quadratic invariant    = integral(u^2 dx)
            I3: Hamiltonian invariant  = integral( (alpha/3)*u^3 - beta*(u_x)^2 dx )

        Args:
            u: 1D array of physical values u(x).

        Returns:
            (I1, I2, I3) as floats.
        """
        I1 = float(np.sum(u) * self.dx)
        I2 = float(np.sum(u**2) * self.dx)

        # Spectral derivative for u_x
        u_hat = np.fft.fft(u)
        ux = np.real(np.fft.ifft(1j * self.k * u_hat))
        I3 = float(
            np.sum((self.alpha / 3.0) * (u**3) - self.beta * (ux**2)) * self.dx
        )

        return I1, I2, I3

    def subgrid_peak(self, u: np.ndarray) -> Tuple[float, float]:
        """
        Estimate the continuous sub-grid peak position x_peak and peak amplitude A_peak
        using Fourier sinc-interpolation around the discrete maximum.

        Returns:
            (x_peak, A_peak)
        """
        idx = int(np.argmax(u))
        # High-resolution Fourier zero-padding around the peak
        # Upsample by factor of 16 locally
        upsample = 16
        u_hat = np.fft.fft(u)
        padded_hat = np.zeros(self.N * upsample, dtype=np.complex128)
        half = self.N // 2
        padded_hat[:half] = u_hat[:half] * upsample
        padded_hat[-half:] = u_hat[-half:] * upsample
        fine_u = np.real(np.fft.ifft(padded_hat))
        fine_dx = self.Lx / (self.N * upsample)
        fine_x = np.linspace(
            -self.Lx / 2.0, self.Lx / 2.0, self.N * upsample, endpoint=False
        )

        fine_idx = int(np.argmax(fine_u))
        return float(fine_x[fine_idx]), float(fine_u[fine_idx])
