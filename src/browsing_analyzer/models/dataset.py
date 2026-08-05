"""Category sequence dataset for LSTM training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CategorySequenceDataset(Dataset):
    """Dataset of ``(input_window, next_category)`` samples.

    Each sample predicts the category at step ``t`` from the ``seq_len``
    categories that preceded it.

    Args:
        X: Integer tensor of shape ``(n_samples, seq_len)``.
        y: Integer tensor of shape ``(n_samples,)``.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
