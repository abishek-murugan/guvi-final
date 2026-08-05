"""PyTorch LSTM next-category predictor.

Architecture (from the notebook): ``Embedding -> LSTM (2 layers) -> Dropout ->
Linear``. The model consumes the last ``sequence_length`` category indices of
a session and outputs logits over the next category.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CategoryPredictor(nn.Module):
    """Embedding-LSTM classifier for next-category prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits over the next category."""
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        last_hidden_state = hidden[-1, :, :]
        return torch.as_tensor(self.fc(self.dropout(last_hidden_state)))


class NextCategoryPredictor:
    """Wraps the LSTM with the category vocabulary and training surface.

    Args:
        settings: Model hyper-parameters.
        categories: Ordered list of category names (order defines the ids).
    """

    def __init__(self, settings: Settings, categories: list[str]) -> None:
        self.categories = list(categories)
        self.category_to_id = {category: i for i, category in enumerate(self.categories)}
        self.id_to_category = {i: category for i, category in enumerate(self.categories)}
        self.sequence_length = settings.model.sequence_length
        self.config = {
            "sequence_length": settings.model.sequence_length,
            "embedding_dim": settings.model.embedding_dim,
            "hidden_dim": settings.model.hidden_dim,
            "num_layers": settings.model.num_layers,
            "dropout": settings.model.dropout,
        }
        torch.manual_seed(settings.model.random_state)
        self.model = CategoryPredictor(
            vocab_size=len(self.categories),
            embed_dim=settings.model.embedding_dim,
            hidden_dim=settings.model.hidden_dim,
            num_layers=settings.model.num_layers,
            output_dim=len(self.categories),
            dropout_prob=settings.model.dropout,
        )

    def build_samples(self, session_sequences: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        """Convert per-session category lists into fixed-length windows.

        Returns ``(X, y)`` where each window of ``sequence_length`` categories
        predicts the category that immediately follows it.
        """
        X: list[list[int]] = []
        y: list[int] = []
        for seq in session_sequences:
            ids = [self.category_to_id[c] for c in seq if c in self.category_to_id]
            if len(ids) > self.sequence_length:
                for i in range(len(ids) - self.sequence_length):
                    X.append(ids[i : i + self.sequence_length])
                    y.append(ids[i + self.sequence_length])
        return np.array(X), np.array(y)

    def predict_proba(self, sequence: list[str]) -> dict[str, float]:
        """Predict the next-category probability distribution for a sequence.

        Args:
            sequence: Ordered category names (the recent history).

        Returns:
            Mapping of category name -> probability.
        """
        ids = [self.category_to_id[c] for c in sequence if c in self.category_to_id]
        if not ids:
            return {category: 0.0 for category in self.categories}

        window = ids[-self.sequence_length :]
        if len(window) < self.sequence_length:
            window = [window[0]] * (self.sequence_length - len(window)) + window

        self.model.eval()
        tensor = torch.tensor([window], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return {self.id_to_category[i]: float(probs[i].item()) for i in range(len(self.categories))}

    def save(self, path: Path) -> None:
        """Persist the trained model, vocabulary and configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "categories": self.categories,
                "config": self.config,
            },
            path,
        )
        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path, settings: Settings | None = None) -> NextCategoryPredictor:
        """Reconstruct a predictor from a saved checkpoint."""
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
