"""
Dataset generation and management for KdV Soliton Dynamics.

Enforces strict partition boundaries between train, validation, and test sets
by trajectory and physical parameter configurations (never shuffling frames).

Supports:
1. Exact on-manifold solitons: L = sqrt(12 * beta / (alpha * A))
2. Off-manifold perturbed pulses: A and L varied independently
3. Two-pulse collision initial conditions
4. Long-horizon trajectories
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from src.kdv_solver import KdVSolver


class KdVTrajectoryDataset(Dataset):
    """
    PyTorch Dataset wrapping multi-step KdV macro-trajectories.
    Each item is a tensor of shape (num_macro_steps, 1, N).
    """

    def __init__(self, data: Union[np.ndarray, torch.Tensor], metadata: Optional[List[Dict]] = None):
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()

        # Ensure shape: (num_trajectories, num_macro_steps, 1, N)
        if self.data.ndim == 3:
            self.data = self.data.unsqueeze(2)

        self.metadata = metadata or [{} for _ in range(len(self.data))]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        return self.data[idx], self.metadata[idx]


def generate_single_trajectory(
    solver: KdVSolver,
    u0: np.ndarray,
    delta_T: float,
    num_macro_steps: int,
    dt_internal: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a single trajectory sampled strictly at macro observation times:
        t = 0, delta_T, 2*delta_T, ..., num_macro_steps*delta_T
    """
    t_eval = np.linspace(0.0, num_macro_steps * delta_T, num_macro_steps + 1)
    trajectory = solver.solve(u0, t_eval, dt_internal=dt_internal)
    return trajectory


def generate_on_manifold_soliton(
    solver: KdVSolver,
    A: float,
    x0: float,
    delta_T: float,
    num_macro_steps: int,
    dt_internal: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate exact analytical soliton on-manifold trajectory."""
    L = solver.exact_soliton_L(A)
    v = solver.exact_soliton_velocity(A)
    u0 = solver.exact_soliton(t=0.0, A=A, x0=x0)
    traj = generate_single_trajectory(solver, u0, delta_T, num_macro_steps, dt_internal)
    meta = {
        "type": "on_manifold_soliton",
        "A": float(A),
        "L": float(L),
        "x0": float(x0),
        "v": float(v),
        "delta_T": float(delta_T),
        "num_macro_steps": int(num_macro_steps),
    }
    return traj, meta


def generate_off_manifold_pulse(
    solver: KdVSolver,
    A: float,
    L: float,
    x0: float,
    delta_T: float,
    num_macro_steps: int,
    dt_internal: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate off-manifold perturbed pulse trajectory:
        u(x, 0) = A * sech^2( (x - x0) / L )
    with L != sqrt(12*beta/(alpha*A)), producing soliton formation + radiation.
    """
    xi = (solver.x - x0 + solver.Lx / 2.0) % solver.Lx - solver.Lx / 2.0
    u0 = A / (np.cosh(xi / L) ** 2)
    traj = generate_single_trajectory(solver, u0, delta_T, num_macro_steps, dt_internal)
    meta = {
        "type": "off_manifold_pulse",
        "A": float(A),
        "L": float(L),
        "L_equilibrium": float(solver.exact_soliton_L(A)),
        "x0": float(x0),
        "delta_T": float(delta_T),
        "num_macro_steps": int(num_macro_steps),
    }
    return traj, meta


def generate_two_pulse_collision(
    solver: KdVSolver,
    A1: float,
    x1: float,
    A2: float,
    x2: float,
    delta_T: float,
    num_macro_steps: int,
    dt_internal: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate two-pulse collision trajectory:
        u(x, 0) = A1 * sech^2((x-x1)/L1) + A2 * sech^2((x-x2)/L2)
    where A1 > A2 and x1 < x2, so pulse 1 catches up and passes pulse 2.
    """
    L1 = solver.exact_soliton_L(A1)
    L2 = solver.exact_soliton_L(A2)
    xi1 = (solver.x - x1 + solver.Lx / 2.0) % solver.Lx - solver.Lx / 2.0
    xi2 = (solver.x - x2 + solver.Lx / 2.0) % solver.Lx - solver.Lx / 2.0
    u0 = (A1 / (np.cosh(xi1 / L1) ** 2)) + (A2 / (np.cosh(xi2 / L2) ** 2))
    traj = generate_single_trajectory(solver, u0, delta_T, num_macro_steps, dt_internal)
    meta = {
        "type": "two_pulse_collision",
        "A1": float(A1),
        "x1": float(x1),
        "L1": float(L1),
        "A2": float(A2),
        "x2": float(x2),
        "L2": float(L2),
        "delta_T": float(delta_T),
        "num_macro_steps": int(num_macro_steps),
    }
    return traj, meta


def build_experiment_datasets(
    solver: KdVSolver,
    delta_T: float = 0.1,
    train_horizon: int = 16,
    long_horizon: int = 100,
    seed: int = 42,
) -> Dict[str, Dict[str, Union[np.ndarray, List[Dict]]]]:
    """
    Generate complete set of partition-isolated datasets for the benchmark.

    Returns dict of datasets:
    - 'train': On-manifold solitons A in [0.6, 1.2], x0 randomized
    - 'val': Independent seed on-manifold solitons in [0.6, 1.2]
    - 'test_interp': Held-out interior parameter combinations
    - 'test_extrap': Amplitudes A in [1.3, 1.8] (outside training distribution)
    - 'test_unseen_params' (Test A): Unseen on-manifold parameters
    - 'test_perturbed_pulses' (Test B): Off-manifold pulses (L != L(A))
    - 'test_two_pulses' (Test C): Nonlinear two-pulse collision ICs
    - 'test_long_horizon': Extended rollout over long_horizon steps
    """
    rng = np.random.default_rng(seed)

    datasets = {}

    # 1. Training Set (On-Manifold Solitons)
    # Range: A in [0.6, 1.2], x0 in [-15, 15]
    n_train = 32
    train_A = rng.uniform(0.6, 1.2, n_train)
    train_x0 = rng.uniform(-15.0, 15.0, n_train)
    train_trajs, train_metas = [], []
    for a, x0 in zip(train_A, train_x0):
        traj, meta = generate_on_manifold_soliton(solver, a, x0, delta_T, train_horizon)
        train_trajs.append(traj)
        train_metas.append(meta)
    datasets["train"] = {"data": np.array(train_trajs), "metadata": train_metas}

    # 2. Validation Set (Independent configurations in training envelope)
    n_val = 8
    val_rng = np.random.default_rng(seed + 1000)
    val_A = val_rng.uniform(0.65, 1.15, n_val)
    val_x0 = val_rng.uniform(-15.0, 15.0, n_val)
    val_trajs, val_metas = [], []
    for a, x0 in zip(val_A, val_x0):
        traj, meta = generate_on_manifold_soliton(solver, a, x0, delta_T, train_horizon)
        val_trajs.append(traj)
        val_metas.append(meta)
    datasets["val"] = {"data": np.array(val_trajs), "metadata": val_metas}

    # 3. Interpolation Test Set (Fixed held-out interior amplitudes)
    interp_A = [0.70, 0.85, 1.00, 1.10]
    interp_x0 = [-12.0, -4.0, 4.0, 12.0]
    interp_trajs, interp_metas = [], []
    for a, x0 in zip(interp_A, interp_x0):
        traj, meta = generate_on_manifold_soliton(solver, a, x0, delta_T, train_horizon)
        interp_trajs.append(traj)
        interp_metas.append(meta)
    datasets["test_interp"] = {"data": np.array(interp_trajs), "metadata": interp_metas}

    # 4. Extrapolation Test Set (Amplitudes strictly outside training envelope)
    extrap_A = [1.35, 1.50, 1.65, 1.80]
    extrap_x0 = [-10.0, -3.0, 5.0, 11.0]
    extrap_trajs, extrap_metas = [], []
    for a, x0 in zip(extrap_A, extrap_x0):
        traj, meta = generate_on_manifold_soliton(solver, a, x0, delta_T, train_horizon)
        extrap_trajs.append(traj)
        extrap_metas.append(meta)
    datasets["test_extrap"] = {"data": np.array(extrap_trajs), "metadata": extrap_metas}

    # 5. Test A: Unseen Single-Soliton Parameters
    unseen_A = [0.62, 0.78, 0.92, 1.08, 1.18]
    unseen_x0 = [-14.0, -7.0, 0.0, 7.0, 14.0]
    unseen_trajs, unseen_metas = [], []
    for a, x0 in zip(unseen_A, unseen_x0):
        traj, meta = generate_on_manifold_soliton(solver, a, x0, delta_T, train_horizon)
        unseen_trajs.append(traj)
        unseen_metas.append(meta)
    datasets["test_unseen_params"] = {"data": np.array(unseen_trajs), "metadata": unseen_metas}

    # 6. Test B: Off-Manifold Perturbed Pulses (L != L(A))
    # Compressing or broadening pulse induces dispersive wave train
    pert_configs = [
        (0.8, 0.7 * solver.exact_soliton_L(0.8), -8.0),
        (0.8, 1.3 * solver.exact_soliton_L(0.8), -4.0),
        (1.0, 0.75 * solver.exact_soliton_L(1.0), 0.0),
        (1.0, 1.25 * solver.exact_soliton_L(1.0), 6.0),
    ]
    pert_trajs, pert_metas = [], []
    for a, l, x0 in pert_configs:
        traj, meta = generate_off_manifold_pulse(solver, a, l, x0, delta_T, train_horizon)
        pert_trajs.append(traj)
        pert_metas.append(meta)
    datasets["test_perturbed_pulses"] = {"data": np.array(pert_trajs), "metadata": pert_metas}

    # 7. Test C: Two-Pulse Collision Initial Condition
    # Fast pulse A1=1.4 at x1=-15, Slower pulse A2=0.7 at x2=0
    two_pulse_configs = [
        (1.4, -16.0, 0.7, 0.0),
        (1.2, -14.0, 0.6, 2.0),
        (1.5, -18.0, 0.8, -2.0),
    ]
    tp_trajs, tp_metas = [], []
    for a1, x1, a2, x2 in two_pulse_configs:
        traj, meta = generate_two_pulse_collision(solver, a1, x1, a2, x2, delta_T, train_horizon)
        tp_trajs.append(traj)
        tp_metas.append(meta)
    datasets["test_two_pulses"] = {"data": np.array(tp_trajs), "metadata": tp_metas}

    # 8. Long-Horizon Test Set
    long_A = [0.8, 1.0, 1.2]
    long_x0 = [-15.0, -10.0, -5.0]
    long_trajs, long_metas = [], []
    for a, x0 in zip(long_A, long_x0):
        traj, meta = generate_on_manifold_soliton(solver, a, x0, delta_T, long_horizon)
        long_trajs.append(traj)
        long_metas.append(meta)
    datasets["test_long_horizon"] = {"data": np.array(long_trajs), "metadata": long_metas}

    return datasets


def save_datasets(datasets: Dict[str, Dict], save_dir: Union[str, Path]) -> None:
    """Save datasets dictionary to disk as .npz and .pt files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, ds in datasets.items():
        np.savez_compressed(
            save_dir / f"{name}.npz",
            data=ds["data"],
            metadata=ds["metadata"],
        )


def load_datasets(data_dir: Union[str, Path]) -> Dict[str, KdVTrajectoryDataset]:
    """Load saved datasets into PyTorch KdVTrajectoryDataset objects."""
    data_dir = Path(data_dir)
    datasets = {}
    for file_path in data_dir.glob("*.npz"):
        name = file_path.stem
        with np.load(file_path, allow_pickle=True) as loaded:
            data = loaded["data"]
            metadata = loaded["metadata"].tolist()
            datasets[name] = KdVTrajectoryDataset(data, metadata)
    return datasets
