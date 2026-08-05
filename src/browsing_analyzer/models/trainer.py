"""Training loop and evaluation for the next-category LSTM.

Replicates the notebook's training cell: ``CrossEntropyLoss`` + Adam over 10
epochs with per-epoch test accuracy, then a full classification report and
confusion matrix on the held-out set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from ..config import Settings
from ..utils.logging import get_logger
from .dataset import CategorySequenceDataset
from .lstm_predictor import NextCategoryPredictor

logger = get_logger(__name__)


@dataclass
class TrainerResult:
    """Bundle of training outputs."""

    history: dict[str, list[float]] = field(default_factory=dict)
    test_accuracy: float = 0.0
    classification_report: dict = field(default_factory=dict)
    confusion: np.ndarray | None = None


def _train_test_split(
    X: torch.Tensor, y: torch.Tensor, test_size: float, random_state: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split with stratification, falling back to a random split."""
    from sklearn.model_selection import train_test_split

    try:
        split = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        logger.warning("stratification_failed_using_random_split")
        split = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return split[0], split[1], split[2], split[3]


def train_model(
    predictor: NextCategoryPredictor,
    session_sequences: list[list[str]],
    settings: Settings,
) -> TrainerResult:
    """Train the LSTM and evaluate on a held-out test split.

    Args:
        predictor: The predictor wrapping the model and vocabulary.
        session_sequences: Category sequences, one per session.
        settings: Model hyper-parameters.

    Returns:
        A :class:`TrainerResult` with loss/accuracy history, the test
        classification report and confusion matrix.
    """
    torch.manual_seed(settings.model.random_state)
    X, y = predictor.build_samples(session_sequences)
    X_tensor = torch.LongTensor(X)
    y_tensor = torch.LongTensor(y)

    X_train, X_test, y_train, y_test = _train_test_split(
        X_tensor,
        y_tensor,
        settings.model.test_size,
        settings.model.random_state,
    )

    train_loader = DataLoader(
        CategorySequenceDataset(X_train, y_train),
        batch_size=settings.model.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        CategorySequenceDataset(X_test, y_test),
        batch_size=settings.model.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = predictor.model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.model.learning_rate)

    history: dict[str, list[float]] = {"train_loss": [], "test_accuracy": []}

    for epoch in range(settings.model.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        history["train_loss"].append(running_loss / len(train_loader))

        test_accuracy, _, _ = _evaluate(model, test_loader, device)
        history["test_accuracy"].append(test_accuracy)
        logger.info(
            "epoch",
            epoch=epoch + 1,
            train_loss=round(history["train_loss"][-1], 4),
            test_accuracy=round(test_accuracy, 4),
        )

    test_accuracy, all_labels, all_predictions = _evaluate(model, test_loader, device)
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=predictor.categories,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        all_labels, all_predictions, labels=list(range(len(predictor.categories)))
    )
    model.to("cpu")

    logger.info("training_complete", test_accuracy=round(test_accuracy, 4))
    return TrainerResult(
        history=history,
        test_accuracy=test_accuracy,
        classification_report=report,
        confusion=confusion,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Return ``(accuracy, labels, predictions)`` over a dataloader."""
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predicted.cpu().numpy().tolist())
    accuracy = accuracy_score(all_labels, all_predictions)
    return float(accuracy), all_labels, all_predictions
