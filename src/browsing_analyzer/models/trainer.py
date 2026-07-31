"""Training loop and evaluation for the next-category LSTM.

Includes a simple most-common-category baseline so the deep model's gains are
quantified, plus accuracy / macro F1 metrics and a confusion matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset

from ..config import Settings
from ..utils.logging import get_logger
from .dataset import CategorySequenceDataset
from .lstm_predictor import NextCategoryPredictor

logger = get_logger(__name__)


class TrainerResult:
    """Bundle of training outputs for reporting."""

    def __init__(
        self,
        history: dict[str, list[float]],
        test_accuracy: float,
        macro_f1: float,
        confusion: np.ndarray,
        baseline_accuracy: float,
        baseline_f1: float,
    ) -> None:
        self.history = history
        self.test_accuracy = test_accuracy
        self.macro_f1 = macro_f1
        self.confusion = confusion
        self.baseline_accuracy = baseline_accuracy
        self.baseline_f1 = baseline_f1


def train_model(
    predictor: NextCategoryPredictor,
    sequences: list[list[int]],
    settings: Settings,
    seed: int | None = None,
) -> TrainerResult:
    """Train the LSTM and evaluate on a temporal holdout.

    Args:
        predictor: The predictor wrapping the model + vocab.
        sequences: Integer category sequences (one per session).
        settings: Model hyper-parameters.
        seed: Random seed (falls back to ``settings.model.seed``).

    Returns:
        A :class:`TrainerResult` with loss curves, metrics, confusion matrix
        and the most-common-category baseline comparison.
    """
    torch.manual_seed(seed or settings.model.seed)
    rng = torch.Generator().manual_seed(seed or settings.model.seed)

    dataset = CategorySequenceDataset(sequences, settings.model.sequence_length)
    if len(dataset) < 10:
        raise ValueError("Not enough samples to train the LSTM")

    # Temporal split: samples are in session order, so a leading slice is the
    # *earlier* browsing. Reserve the last 20% for test, last 20% of the rest
    # for validation (validation_split=0.2 -> 20% val, 60% train, 20% test).
    n_test = max(1, int(len(dataset) * (1 - settings.model.validation_split * 2)))
    n_train = len(dataset) - n_test
    n_val = int(n_train * settings.model.validation_split)
    n_train -= n_val

    train_ds = Subset(dataset, range(0, n_train))
    val_ds = Subset(dataset, range(n_train, n_train + n_val))
    test_ds = Subset(dataset, range(n_train + n_val, len(dataset)))

    train_loader = DataLoader(
        train_ds, batch_size=settings.model.batch_size, shuffle=True, generator=rng
    )
    val_loader = DataLoader(val_ds, batch_size=settings.model.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=settings.model.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = predictor.model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.model.learning_rate)
    criterion = CrossEntropyLoss(ignore_index=0)

    best_val_loss = float("inf")
    patience = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(settings.model.epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_losses.append(criterion(logits, y).item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= settings.model.patience:
                logger.info("early_stopping", epoch=epoch, val_loss=round(val_loss, 4))
                break

        if (epoch + 1) % 10 == 0:
            logger.info(
                "epoch",
                epoch=epoch + 1,
                train_loss=round(history["train_loss"][-1], 4),
                val_loss=round(val_loss, 4),
            )

    # Evaluate on the test split.
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().tolist()
            y_true.extend(y.tolist())
            y_pred.extend(preds)

    labels = list(range(1, predictor.vocab_size))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    conf = confusion_matrix(y_true, y_pred, labels=labels)

    # Most-common-category baseline: always predict the training majority class.
    all_labels = [dataset[i][1].item() for i in range(0, n_train + n_val)]
    if all_labels:
        majority = max(set(all_labels), key=all_labels.count)
        baseline_pred = [majority] * len(y_true)
        baseline_acc = accuracy_score(y_true, baseline_pred)
        baseline_f1 = f1_score(
            y_true, baseline_pred, average="macro", labels=labels, zero_division=0
        )
    else:
        baseline_acc = baseline_f1 = 0.0

    result = TrainerResult(
        history=history,
        test_accuracy=acc,
        macro_f1=f1,
        confusion=conf,
        baseline_accuracy=baseline_acc,
        baseline_f1=baseline_f1,
    )
    logger.info(
        "training_complete",
        test_accuracy=round(acc, 4),
        macro_f1=round(f1, 4),
        baseline_accuracy=round(baseline_acc, 4),
    )
    return result


def confusion_df(confusion: np.ndarray, categories: list[str]) -> pd.DataFrame:
    """Wrap a confusion matrix in a labelled DataFrame."""
    cats = sorted(categories)
    return pd.DataFrame(confusion, index=cats, columns=cats)
