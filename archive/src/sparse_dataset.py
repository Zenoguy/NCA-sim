"""
Sparse Probe Dataset Wrapper for Partially-Observed KdV (Environment B).

Applies a spatial observation mask M with P sparse probe locations:
    y(x, t) = M(x) * u(x, t)
The model observes only y(x, t) at sparse locations and must reconstruct/roll out
the continuous wave field u(x, t + Delta T).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class SparseProbeDataset(Dataset):
    """
    Dataset where input is masked with sparse probes:
        input:  y(t) = M * u(t)
        target: u(t) (full field ground truth)
    """

    def __init__(
        self,
        full_trajectories: Union[np.ndarray, torch.Tensor],
        num_probes: int = 16,
        probe_indices: Optional[List[int]] = None,
    ):
        if isinstance(full_trajectories, np.ndarray):
            self.full_data = torch.from_numpy(full_trajectories).float()
        else:
            self.full_data = full_trajectories.float()

        if self.full_data.ndim == 3:
            self.full_data = self.full_data.unsqueeze(2)

        B, T, C, N = self.full_data.shape
        self.N = N
        self.num_probes = num_probes

        # Generate uniform probe indices if not provided
        if probe_indices is None:
            step = max(1, N // num_probes)
            self.probe_indices = list(range(0, N, step))[:num_probes]
        else:
            self.probe_indices = probe_indices

        # Binary spatial mask
        self.mask = torch.zeros(1, 1, N)
        self.mask[0, 0, self.probe_indices] = 1.0

        # Sparse observed data
        self.sparse_data = self.full_data * self.mask.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.full_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sparse_traj: (T, 1, N) with values only at probe locations
            full_traj:   (T, 1, N) full continuous ground truth
        """
        return self.sparse_data[idx], self.full_data[idx]
