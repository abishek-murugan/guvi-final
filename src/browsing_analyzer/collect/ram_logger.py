"""System RAM usage collection.

``collect_ram_log`` samples ``psutil.virtual_memory`` every ``interval``
seconds and appends readings to a CSV. It mirrors the notebook's ``log_ram``
cell and is intentionally never run by the pipeline (running it would append
to the raw data). The pipeline reads the exported CSV via
:func:`load_ram_log`.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["timestamp", "total_mb", "used_mb", "available_mb", "usage_percent"]

_MB = 1024**2


def collect_ram_log(
    interval: int = 5,
    duration_hours: float | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Sample system RAM every ``interval`` seconds and append to a CSV.

    Args:
        interval: Seconds between samples.
        duration_hours: How long to keep sampling. Defaults to running until
            interrupted (matching the notebook).
        output_path: Destination CSV path. Defaults to ``data/raw``.

    Returns:
        A DataFrame of collected samples.
    """
    output_path = output_path or Path("data") / "raw" / "ram_log_last5days.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    deadline = time.time() + (duration_hours * 3600.0 if duration_hours else float("inf"))
    rows: list[dict[str, float | str]] = []

    with output_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(["timestamp", "total_mb", "used_mb", "available_mb", "usage_percent"])

        while time.time() < deadline:
            mem = psutil.virtual_memory()
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_mb": round(mem.total / _MB, 2),
                "used_mb": round(mem.used / _MB, 2),
                "available_mb": round(mem.available / _MB, 2),
                "usage_percent": mem.percent,
            }
            writer.writerow(row.values())
            fh.flush()
            rows.append(row)
            time.sleep(interval)

    df = pd.DataFrame(rows)
    logger.info("ram_log_exported", rows=len(df), path=str(output_path))
    return df


def load_ram_log(settings: Settings, csv_path: Path | None = None) -> pd.DataFrame:
    """Load and normalize the RAM log CSV.

    Args:
        settings: Application settings (used to locate the default file).
        csv_path: Optional explicit path.

    Returns:
        A DataFrame with ``timestamp`` plus RAM columns, sorted by time.
    """
    path = csv_path or Path(settings.data.raw_dir) / settings.data.ram_log_file
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
