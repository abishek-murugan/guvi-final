"""Category sequence dataset for LSTM training.

Sessions are flattened into ordered category sequences. Each sample predicts
the category at step ``t`` from the ``sequence_length`` categories that
preceded it. Padding is applied with index 0.

A precomputed sample index maps dataset indices to ``(sequence, position)``
pairs so ``__getitem__`` is O(1).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CategorySequenceDataset(Dataset):
    """PyTorch dataset built from category-index sequences.

    Samples are produced in temporal order: session 0's events first, then
    session 1's, etc., so a leading slice corresponds to earlier browsing.

    Args:
        sequences: List of 1-D integer category index sequences.
        sequence_length: Window length for the LSTM input.
    """

    def __init__(self, sequences: list[list[int]], sequence_length: int) -> None:
        self.seq_len = sequence_length
        self._sequences = [s for s in sequences if len(s) >= 2]
        # (seq_index, position_in_seq) for every (x, y) sample.
        self._index: list[tuple[int, int]] = []
        for i, seq in enumerate(self._sequences):
            for pos in range(len(seq) - 1):
                self._index.append((i, pos))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, pos = self._index[index]
        seq = self._sequences[seq_idx]
        start = max(0, pos + 1 - self.seq_len)
        x = seq[start : pos + 1]
        x = [0] * (self.seq_len - len(x)) + x
        y = seq[pos + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
