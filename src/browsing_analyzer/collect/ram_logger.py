"""RAM usage collection.

Two code paths live here:

1. :func:`collect_ram_log` — samples system and browser memory via
   :mod:`psutil` every ``interval_seconds`` and appends each reading to
   ``data/raw/ram_data.csv``.
2. :func:`load_ram_log` — ingests that exported CSV and returns a normalized
   DataFrame for the downstream pipeline.

The collection step runs once on the user's machine (locally, privately);
the pipeline itself only reads the CSV via :func:`load_ram_log`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import psutil

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

_BYTES_TO_MB = 1024 * 1024


def _chrome_processes() -> list[psutil.Process]:
    """Return running Chrome processes, or an empty list if none are found."""
    chrome = []
    for proc in psutil.process_iter(["name"]):
        try:
            if "chrome" in proc.info["name"].lower():
                chrome.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return chrome


def _chrome_ram_mb() -> float:
    """Total resident memory (MB) used by all Chrome processes."""
    total = 0.0
    for proc in _chrome_processes():
        try:
            total += proc.memory_info().rss / _BYTES_TO_MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def collect_ram_log(
    settings: Settings,
    duration_hours: float | None = None,
    interval_seconds: int | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Sample system + browser RAM via psutil and export to CSV.

    Samples are taken every ``interval_seconds`` for ``duration_hours`` hours.
    Each row captures system memory (used/available), the total resident
    memory of all Chrome processes, and overall CPU utilisation.

    Args:
        settings: Application settings (RAM logger knobs).
        duration_hours: How long to keep sampling. Defaults to
            ``settings.ram_logger.duration_hours``.
        interval_seconds: Seconds between samples. Defaults to
            ``settings.ram_logger.interval_seconds``.
        output_path: Destination CSV path. Defaults to ``data/raw/ram_data.csv``.

    Returns:
        A DataFrame with ``timestamp``, ``ram_used_mb``, ``ram_available_mb``,
        ``browser_ram_mb`` and ``cpu_percent`` columns.
    """
    interval = interval_seconds or settings.ram_logger.interval_seconds
    duration = duration_hours or settings.ram_logger.duration_hours
    deadline = time.time() + duration * 3600.0

    rows: list[dict] = []
    while time.time() < deadline:
        mem = psutil.virtual_memory()
        rows.append(
            {
                "timestamp": pd.Timestamp.now().isoformat(),
                "ram_used_mb": round(mem.used / _BYTES_TO_MB, 1),
                "ram_available_mb": round(mem.available / _BYTES_TO_MB, 1),
                "browser_ram_mb": round(_chrome_ram_mb(), 1),
                "cpu_percent": round(psutil.cpu_percent(interval=0.5), 1),
            }
        )
        time.sleep(interval)

    df = pd.DataFrame(rows)
    out = output_path or (Path("data") / "raw" / "ram_data.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("ram_log_exported", rows=len(df), path=str(out))
    return df


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
