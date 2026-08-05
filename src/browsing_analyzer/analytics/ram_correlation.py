"""Correlate browsing events with system RAM usage.

Each event is aligned with the nearest *previous* RAM sample via a backward
``pd.merge_asof`` (the memory level that was in effect when the visit
happened), then RAM statistics are aggregated per session and per category.
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

_RAM_AGG = {
    "mean_used_mb": ("used_mb", "mean"),
    "peak_used_mb": ("used_mb", "max"),
    "mean_usage_percent": ("usage_percent", "mean"),
    "peak_usage_percent": ("usage_percent", "max"),
}


def align_ram_with_events(events: pd.DataFrame, ram_log: pd.DataFrame) -> pd.DataFrame:
    """Merge each event with the most recent RAM reading at or before it."""
    events_sorted = events.sort_values("timestamp").reset_index(drop=True)
    ram_sorted = ram_log.sort_values("timestamp").reset_index(drop=True)
    return pd.merge_asof(
        events_sorted,
        ram_sorted,
        on="timestamp",
        direction="backward",
        suffixes=("_chrome", "_ram"),
    )


def session_ram_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RAM usage statistics per session."""
    return events.groupby("session_id").agg(**_RAM_AGG).reset_index()


def category_ram_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RAM usage statistics per category."""
    return events.groupby("category").agg(**_RAM_AGG).reset_index()
