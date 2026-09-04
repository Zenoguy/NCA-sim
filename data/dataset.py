"""
Autoregressive Dataset Module for NCA-LM.
Provides chunking, causal shift alignment, and PyTorch DataLoaders.
"""

from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AutoregressiveDataset(Dataset):
    """
    Chunks a continuous stream of token IDs into autoregressive input/target pairs.
    For sequence length T, extracts chunks of length T + 1:
        inputs:  chunk[:-1] (length T)
        targets: chunk[1:]  (length T)
    """
    def __init__(self, token_array: Union[np.ndarray, torch.Tensor], seq_len: int = 128, stride: int = None):
        super().__init__()
        if isinstance(token_array, np.ndarray):
            self.tokens = torch.from_numpy(token_array.astype(np.int64))
        else:
            self.tokens = token_array.long()

        self.seq_len = seq_len
        self.stride = stride or seq_len  # Non-overlapping by default

        # Number of complete samples we can extract
        total_tokens = len(self.tokens)
        required_len = seq_len + 1
        if total_tokens < required_len:
            raise ValueError(f"Total tokens ({total_tokens}) is less than required ({required_len})")

        self.num_samples = (total_tokens - required_len) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        
        inputs = chunk[:-1]
        targets = chunk[1:]
        return inputs, targets

    def verify_causal_alignment(self) -> bool:
        """
        Strict mathematical check verifying that:
        targets[i] == inputs[i+1] for all i in [0, seq_len - 2].
        """
        if len(self) == 0:
            return True
        sample_in, sample_tgt = self[0]
        match = torch.equal(sample_in[1:], sample_tgt[:-1])
        if not match:
            raise AssertionError("Causal shift mismatch detected: targets are not strictly inputs shifted by 1!")
        return True


def get_dataloader(
    token_array: Union[np.ndarray, torch.Tensor],
    seq_len: int = 128,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    stride: int = None,
) -> DataLoader:
    """Create a standard DataLoader for autoregressive training/eval."""
    dataset = AutoregressiveDataset(token_array, seq_len=seq_len, stride=stride)
    dataset.verify_causal_alignment()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
