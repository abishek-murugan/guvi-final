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

# Alternative schema: RAM figures exported in gigabytes instead of megabytes.
# `used_ram_gb` -> `ram_used_mb`, `available_ram_gb` -> `ram_available_mb`,
# `chrome_ram_gb` -> `browser_ram_mb`. Values are converted to MB (x1024).
GB_SCHEMA_COLUMNS = {
    "used_ram_gb": "ram_used_mb",
    "available_ram_gb": "ram_available_mb",
    "chrome_ram_gb": "browser_ram_mb",
}


def load_ram_log(settings: Settings, csv_path: Path | None = None) -> pd.DataFrame:
    """Load and validate the RAM log CSV.

    Accepts either the native MB schema (``ram_used_mb``, ``ram_available_mb``,
    ``browser_ram_mb``) or an exported GB schema (``used_ram_gb``,
    ``available_ram_gb``, ``chrome_ram_gb``) which is normalized to MB.

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
    is_gb_schema = all(c in df.columns for c in GB_SCHEMA_COLUMNS)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing and not is_gb_schema:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if is_gb_schema:
        df = df.rename(columns=GB_SCHEMA_COLUMNS)
        for col in REQUIRED_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce") * 1024.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "ram_log_loaded",
        rows=len(df),
        source=str(path),
        schema="gb" if is_gb_schema else "mb",
    )
    return df
