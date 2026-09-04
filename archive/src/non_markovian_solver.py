"""
Coupled Non-Markovian KdV Solver with Hidden Latent Relaxation Field (Mori-Zwanzig System).

Solves the coupled system:
    u_t + 6 * u * u_x + u_{xxx} = w(x, t)
    w_t = -lambda_rel * w(x, t) + kappa * u(x, t)

By Mori-Zwanzig projection, integrating out the unobserved field w(x, t)
yields an exact Volterra integro-differential equation for u(x, t) with
an exponential memory kernel:
    u_t + 6 * u * u_x + u_{xxx} = kappa * integral_0^t exp(-lambda_rel*(t-s)) * u(x, s) ds + w0*exp(-lambda_rel*t)

The neural simulator ONLY observes u(x, t); w(x, t) is strictly latent.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class CoupledNonMarkovianKdVSolver:
    """
    Fourier pseudospectral solver for coupled (u, w) KdV system using ETDRK4 on u
    coupled with RK4 on the latent relaxation field w.
    """

    def __init__(
        self,
        N: int = 128,
        Lx: float = 40.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        lambda_rel: float = 1.0,
        kappa: float = 1.5,
        dt: float = 0.005,
    ):
        self.N = N
        self.Lx = Lx
        self.alpha = alpha
        self.beta = beta
        self.lambda_rel = lambda_rel
        self.kappa = kappa
        self.dt = dt

        self.dx = Lx / N
        self.x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)
        self.k = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)

        k_max = np.max(np.abs(self.k))
        self.dealias = np.abs(self.k) < (2.0 / 3.0) * k_max
        self.L_op = 1j * self.beta * (self.k**3)

        self._precompute_etdrk4(self.dt)

    def _precompute_etdrk4(self, dt: float) -> None:
        self.dt = dt
        z = self.L_op * dt
        self.E = np.exp(z)
        self.E2 = np.exp(z / 2.0)

        M = 32
        r = 1.0
        theta = 2.0 * np.pi * (np.arange(1, M + 1) - 0.5) / M
        w = z[:, None] + r * np.exp(1j * theta)[None, :]

        self.Q = dt * np.mean((np.exp(w / 2.0) - 1.0) / w, axis=1)
        self.f1 = np.mean(
            (-4.0 - w + np.exp(w) * (4.0 - 3.0 * w + w**2)) / (w**3), axis=1
        )
        self.f2 = np.mean((2.0 + w + np.exp(w) * (-2.0 + w)) / (w**3), axis=1)
        self.f3 = np.mean(
            (-4.0 - 3.0 * w - w**2 + np.exp(w) * (4.0 - w)) / (w**3), axis=1
        )

    def _nonlinear_op(self, u_hat: np.ndarray, w_field: np.ndarray) -> np.ndarray:
        """Evaluate nonlinear convection + latent forcing: -alpha/2 * d/dx(u^2) + w."""
        u = np.real(np.fft.ifft(u_hat))
        u2_hat = np.fft.fft(u**2)
        conv_term = -0.5 * self.alpha * 1j * self.k * u2_hat * self.dealias
        w_hat = np.fft.fft(w_field)
        return conv_term + w_hat

    def step(self, u_hat: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one coupled internal step using ETDRK4 on u and RK4 on w:
            u_t = L(u) + N(u, w)
            w_t = -lambda_rel * w + kappa * u
        """
        dt = self.dt
        u_phys = np.real(np.fft.ifft(u_hat))

        # Substep 1
        Na = self._nonlinear_op(u_hat, w)
        kw1 = -self.lambda_rel * w + self.kappa * u_phys

        # Substep 2
        a = self.E2 * u_hat + self.Q * Na
        w_b = w + 0.5 * dt * kw1
        Nb = self._nonlinear_op(a, w_b)
        u_b = np.real(np.fft.ifft(a))
        kw2 = -self.lambda_rel * w_b + self.kappa * u_b

        # Substep 3
        b = self.E2 * u_hat + self.Q * Nb
        w_c = w + 0.5 * dt * kw2
        Nc = self._nonlinear_op(b, w_c)
        u_c = np.real(np.fft.ifft(b))
        kw3 = -self.lambda_rel * w_c + self.kappa * u_c

        # Substep 4
        c = self.E2 * a + self.Q * (2.0 * Nc - Na)
        w_d = w + dt * kw3
        Nd = self._nonlinear_op(c, w_d)
        u_d = np.real(np.fft.ifft(c))
        kw4 = -self.lambda_rel * w_d + self.kappa * u_d

        # Final state updates
        u_hat_next = self.E * u_hat + dt * (
            self.f1 * Na + 2.0 * self.f2 * (Nb + Nc) + self.f3 * Nd
        )
        w_next = w + (dt / 6.0) * (kw1 + 2.0 * kw2 + 2.0 * kw3 + kw4)

        return u_hat_next, w_next

    def solve(
        self,
        u0: np.ndarray,
        w0: Optional[np.ndarray],
        t_eval: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve coupled system, outputting u(x, t) and latent w(x, t) at t_eval times.
        """
        num_times = len(t_eval)
        u_traj = np.zeros((num_times, self.N), dtype=np.float64)
        w_traj = np.zeros((num_times, self.N), dtype=np.float64)

        u_traj[0] = u0.copy()
        if w0 is None:
            w0 = np.zeros(self.N, dtype=np.float64)
        w_traj[0] = w0.copy()

        u_hat = np.fft.fft(u0)
        w = w0.copy()
        current_t = t_eval[0]

        for i in range(1, num_times):
            target_t = t_eval[i]
            delta_t = target_t - current_t
            n_substeps = max(1, int(round(delta_t / self.dt)))
            sub_dt = delta_t / n_substeps

            if abs(sub_dt - self.dt) > 1e-12:
                self._precompute_etdrk4(sub_dt)

            for _ in range(n_substeps):
                u_hat, w = self.step(u_hat, w)

            current_t = target_t
            u_traj[i] = np.real(np.fft.ifft(u_hat))
            w_traj[i] = w.copy()

        return u_traj, w_traj


def build_non_markovian_datasets(
    solver: CoupledNonMarkovianKdVSolver,
    delta_T: float = 0.1,
    train_horizon: int = 16,
    n_train: int = 32,
    n_val: int = 8,
    seed: int = 42,
) -> Dict[str, Dict[str, Union[np.ndarray, List[Dict]]]]:
    """
    Generate partition-isolated datasets for the Coupled Non-Markovian KdV system (Environment C).
    Latent field w is hidden; only u(x, t) is exposed in the dataset.
    """
    t_eval = np.linspace(0.0, train_horizon * delta_T, train_horizon + 1)
    datasets = {}

    # 1. Training Set
    rng = np.random.default_rng(seed)
    train_A = rng.uniform(0.6, 1.2, n_train)
    train_x0 = rng.uniform(-15.0, 15.0, n_train)
    train_trajs, train_metas = [], []

    for a, x0 in zip(train_A, train_x0):
        L = np.sqrt(12.0 * solver.beta / (solver.alpha * a))
        xi = (solver.x - x0 + solver.Lx / 2.0) % solver.Lx - solver.Lx / 2.0
        u0 = a / (np.cosh(xi / L) ** 2)
        u_traj, _ = solver.solve(u0, w0=None, t_eval=t_eval)
        train_trajs.append(u_traj)
        train_metas.append({"A": float(a), "x0": float(x0), "type": "non_markovian_coupled"})

    datasets["train"] = {"data": np.array(train_trajs), "metadata": train_metas}

    # 2. Validation Set
    val_rng = np.random.default_rng(seed + 1000)
    val_A = val_rng.uniform(0.65, 1.15, n_val)
    val_x0 = val_rng.uniform(-15.0, 15.0, n_val)
    val_trajs, val_metas = [], []

    for a, x0 in zip(val_A, val_x0):
        L = np.sqrt(12.0 * solver.beta / (solver.alpha * a))
        xi = (solver.x - x0 + solver.Lx / 2.0) % solver.Lx - solver.Lx / 2.0
        u0 = a / (np.cosh(xi / L) ** 2)
        u_traj, _ = solver.solve(u0, w0=None, t_eval=t_eval)
        val_trajs.append(u_traj)
        val_metas.append({"A": float(a), "x0": float(x0), "type": "non_markovian_coupled"})

    datasets["val"] = {"data": np.array(val_trajs), "metadata": val_metas}
    return datasets

