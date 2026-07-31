"""PyTorch LSTM next-category predictor.

Architecture: ``Embedding -> LSTM (2 layers) -> Dropout -> Linear -> Softmax``.
The model consumes the last ``sequence_length`` category indices of a session
and outputs a probability distribution over the next category.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CategoryLSTM(nn.Module):
    """Embedding-LSTM classifier for next-category prediction.

    Args:
        vocab_size: Number of distinct categories (+1 for the padding token).
        embedding_dim: Dense embedding size.
        hidden_dim: LSTM hidden size.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between layers / before the head.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits for the last timestep.

        Args:
            x: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Logits of shape ``(batch, vocab_size)``.
        """
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        last = self.dropout(out[:, -1, :])
        logits: torch.Tensor = self.fc(last)
        return logits


class NextCategoryPredictor:
    """High-level wrapper: tokenizer, training surface, inference helpers.

    Args:
        settings: Application settings (model hyper-parameters).
        categories: Ordered list of category names.
    """

    def __init__(self, settings: Settings, categories: list[str]) -> None:
        self.settings = settings
        self.categories = sorted(set(categories))
        self.cat_to_idx = {cat: i + 1 for i, cat in enumerate(self.categories)}
        self.idx_to_cat = {i + 1: cat for i, cat in enumerate(self.categories)}
        self.vocab_size = len(self.categories) + 1  # +1 for padding token 0
        torch.manual_seed(settings.model.seed)
        self.model = CategoryLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=settings.model.embedding_dim,
            hidden_dim=settings.model.hidden_dim,
            num_layers=settings.model.num_layers,
            dropout=settings.model.dropout,
        )

    def to_sequences(self, sessions: list[list[str]]) -> list[list[int]]:
        """Convert per-session category lists into integer index sequences."""
        return [
            [self.cat_to_idx[c] for c in session if c in self.cat_to_idx] for session in sessions
        ]

    def predict_proba(self, sequence: list[str]) -> dict[str, float]:
        """Predict the next-category probability distribution for a sequence.

        Args:
            sequence: Ordered category names (the recent history).

        Returns:
            Mapping of category name -> probability.
        """
        self.model.eval()
        indices = self.to_sequences([sequence])[0]
        if not indices:
            return {cat: 0.0 for cat in self.categories}
        pad = self.settings.model.sequence_length - len(indices)
        tensor = torch.tensor(
            [[0] * max(pad, 0) + indices[-self.settings.model.sequence_length :]], dtype=torch.long
        )
        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return {self.idx_to_cat[i]: float(probs[i].item()) for i in range(1, self.vocab_size)}

    def save(self, path: Path) -> None:
        """Persist the trained model, vocab, and full model configuration.

        Args:
            path: Destination ``.pt`` file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "categories": self.categories,
                "config": {
                    "sequence_length": self.settings.model.sequence_length,
                    "embedding_dim": self.settings.model.embedding_dim,
                    "hidden_dim": self.settings.model.hidden_dim,
                    "num_layers": self.settings.model.num_layers,
                    "dropout": self.settings.model.dropout,
                    "batch_size": self.settings.model.batch_size,
                    "learning_rate": self.settings.model.learning_rate,
                    "seed": self.settings.model.seed,
                },
            },
            path,
        )
        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path, settings: Settings | None = None) -> NextCategoryPredictor:
        """Reconstruct a predictor from a saved checkpoint.

        The model architecture is rebuilt from the checkpoint's full config,
        so no external settings are required for a faithful reload.

        Args:
            path: Path to a checkpoint written by :meth:`save`.
            settings: Optional settings; when omitted, ``Settings()`` defaults
                are used (the checkpoint config takes precedence).

        Returns:
            A :class:`NextCategoryPredictor` with the saved weights loaded.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        settings = settings or Settings()
        settings.model.sequence_length = checkpoint["config"]["sequence_length"]
        settings.model.embedding_dim = checkpoint["config"]["embedding_dim"]
        settings.model.hidden_dim = checkpoint["config"]["hidden_dim"]
        settings.model.num_layers = checkpoint["config"]["num_layers"]
        settings.model.dropout = checkpoint["config"]["dropout"]

        predictor = cls(settings, checkpoint["categories"])
        predictor.model.load_state_dict(checkpoint["model_state_dict"])
        predictor.model.eval()
        logger.info("model_loaded", path=str(path))
        return predictor
