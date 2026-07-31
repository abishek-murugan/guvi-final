"""RAM usage collection.

In production a background thread samples system and browser memory every
``ram_logger.interval_seconds`` (5-10s) using :mod:`psutil`:

* ``ram_used_mb`` / ``ram_available_mb`` from ``psutil.virtual_memory()``.
* ``browser_ram_mb`` as the sum of resident memory of all Chrome processes.
* ``cpu_percent`` from ``psutil.cpu_percent(interval=None)``.

Each sample is appended to ``data/raw/ram_log.csv``. For this deliverable the
RAM log was collected locally with that procedure; this loader ingests the
exported CSV and returns a normalized DataFrame for the downstream pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["timestamp", "ram_used_mb", "ram_available_mb", "browser_ram_mb"]


def load_ram_log(settings: Settings, csv_path: Path | None = None) -> pd.DataFrame:
    """Load and validate the RAM log CSV.

    Args:
        settings: Application settings (used to locate the default file).
        csv_path: Optional explicit path. Defaults to ``data/raw/ram_data.csv``.

    Returns:
        A DataFrame with ``timestamp`` plus RAM columns, sorted by time.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = csv_path or Path(settings.data.raw_dir) / settings.data.ram_data_file
    if not path.exists():
        raise FileNotFoundError(f"RAM log file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("ram_log_loaded", rows=len(df), source=str(path))
    return df
