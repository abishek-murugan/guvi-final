"""Browsing history collection.

Two code paths live here:

1. :func:`extract_chrome_history` — extracts raw history from Chrome's SQLite
   database (``~/.config/google-chrome/Default/History``), converts the
   WebKit microsecond timestamps, and exports the result to
   ``data/raw/chrome_data.csv``.
2. :func:`load_browsing_history` — ingests that exported CSV and returns a
   normalized DataFrame for the downstream pipeline.

The collection step runs once on the user's machine (locally, privately);
the pipeline itself only reads the CSV via :func:`load_browsing_history`.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["timestamp", "url"]

# Chrome stores timestamps as microseconds since 1601-01-01 00:00:00 UTC
# (the WebKit/Chromium epoch), whereas Unix time starts at 1970-01-01.
_WEBKIT_TO_UNIX_OFFSET_US = 11_644_473_600_000_000


def _chrome_history_db_path(profile: str = "Default") -> Path:
    """Return the path to the Chrome ``History`` SQLite database."""
    return Path.home() / ".config" / "google-chrome" / profile / "History"


def extract_chrome_history(
    profile: str = "Default",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Extract browsing history from Chrome's SQLite database and export it.

    Chrome locks the ``History`` database while running, so the DB is first
    copied to a temporary file before querying. Rows are pulled by joining the
    ``urls`` and ``visits`` tables and ordered by visit time.

    Args:
        profile: Chrome profile directory name (default ``Default``).
        output_path: Destination CSV path. Defaults to ``data/raw/chrome_data.csv``.

    Returns:
        A DataFrame with ``timestamp``, ``url`` and ``title`` columns.

    Raises:
        FileNotFoundError: If the Chrome History database does not exist.
    """
    db_path = _chrome_history_db_path(profile)
    if not db_path.exists():
        raise FileNotFoundError(f"Chrome history database not found: {db_path}")

    # Copy the DB so Chrome's exclusive lock does not block the query.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        shutil.copy2(db_path, tmp.name)
        tmp_db = tmp.name

    try:
        conn = sqlite3.connect(tmp_db)
        query = """
            SELECT urls.url AS url,
                   urls.title AS title,
                   visits.visit_time AS visit_time
            FROM visits
            JOIN urls ON visits.url = urls.id
            ORDER BY visits.visit_time ASC
        """
        rows = conn.execute(query).fetchall()
        conn.close()
    finally:
        Path(tmp_db).unlink(missing_ok=True)

    df = pd.DataFrame(rows, columns=["url", "title", "visit_time"])
    if df.empty:
        logger.warning("chrome_history_empty")
        return df

    # Convert WebKit microseconds (since 1601-01-01) to normal datetimes.
    df["timestamp"] = pd.to_datetime(
        (df["visit_time"] - _WEBKIT_TO_UNIX_OFFSET_US) // 1_000_000,
        unit="s",
    )
    df = df.drop(columns=["visit_time"]).dropna(subset=["url"])

    out = output_path or (Path("data") / "raw" / "chrome_data.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("chrome_history_exported", rows=len(df), path=str(out))
    return df


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
