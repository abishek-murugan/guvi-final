"""Browsing history collection.

In production this module would locate the Chrome ``History`` SQLite database
(e.g. ``~/.config/google-chrome/Default/History``), copy it (Chrome locks the
DB while running), and extract ``(timestamp, url, title)`` triples from the
``urls`` and ``visits`` tables. Chrome stores timestamps as microseconds since
1601-01-01 UTC, which need conversion.

For this deliverable, the browser history was collected locally using that
procedure and exported to ``data/raw/chrome_data.csv``. This loader ingests
that file, validates its schema, and returns a normalized DataFrame so the
downstream pipeline behaves exactly as if fresh data had been collected.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["timestamp", "url"]


def load_browsing_history(settings: Settings, csv_path: Path | None = None) -> pd.DataFrame:
    """Load and validate the browsing history CSV.

    Args:
        settings: Application settings (used to locate the default file).
        csv_path: Optional explicit path. Defaults to ``data/raw/chrome_data.csv``.

    Returns:
        A DataFrame with at least ``timestamp`` and ``url`` columns.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = csv_path or Path(settings.data.raw_dir) / settings.data.chrome_data_file
    if not path.exists():
        raise FileNotFoundError(f"Browsing history file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("history_loaded", rows=len(df), source=str(path))
    return df
