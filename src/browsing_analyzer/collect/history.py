"""Chrome browsing history collection.

``extract_chrome_history`` reads Chrome's SQLite ``History`` database and
exports ``timestamp, url, title`` rows. It is the offline collection step and
is intentionally never run by the pipeline (running it could overwrite the raw
data). The pipeline reads the exported CSV via :func:`load_browsing_history`.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["timestamp", "url"]


def extract_chrome_history(
    profile: str = "Default",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Extract browsing history from Chrome's SQLite database and export it.

    Chrome locks the ``History`` database while running, so the database is
    copied to a temporary file before querying.

    Args:
        profile: Chrome profile directory name.
        output_path: Destination CSV path. Defaults to ``data/raw``.

    Returns:
        A DataFrame with ``timestamp``, ``url`` and ``title`` columns.
    """
    history_path = os.path.expanduser(f"~/.config/google-chrome/{profile}/History")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Chrome history database not found: {history_path}")

    temp_history = Path("/tmp") / "temp_chrome_history"
    shutil.copy2(history_path, temp_history)

    try:
        conn = sqlite3.connect(temp_history)
        query = """
            SELECT
                datetime(last_visit_time/1000000 - 11644473600, 'unixepoch', 'localtime') as timestamp,
                url,
                title
            FROM urls
            ORDER BY last_visit_time DESC
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
        temp_history.unlink(missing_ok=True)

    out = output_path or Path("data") / "raw" / "browsing_history_last5days.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("chrome_history_exported", rows=len(df), path=str(out))
    return df


def load_browsing_history(settings: Settings, csv_path: Path | None = None) -> pd.DataFrame:
    """Load and normalize the browsing history CSV.

    Args:
        settings: Application settings (used to locate the default file).
        csv_path: Optional explicit path.

    Returns:
        A DataFrame with at least ``timestamp`` and ``url`` columns.
    """
    path = csv_path or Path(settings.data.raw_dir) / settings.data.browser_history_file
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
