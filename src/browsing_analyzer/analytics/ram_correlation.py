"""Correlate browsing events with RAM usage.

RAM samples are captured every few seconds while the browser is open. We
align each browsing event with the nearest RAM sample in time (backward
search via ``pd.merge_asof``) and then aggregate RAM statistics per session
and per category.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

RAM_COLUMNS = ["ram_used_mb", "ram_available_mb", "browser_ram_mb"]


def align_ram_with_events(events: pd.DataFrame, ram_log: pd.DataFrame) -> pd.DataFrame:
    """Merge each browsing event with the nearest *previous* RAM sample.

    Uses a backward ``merge_asof`` so an event at time ``t`` is paired with
    the most recent RAM reading ``<= t`` (realistic: the browser was using
    that memory level when the visit happened).

    Args:
        events: Browsing events with ``timestamp``.
        ram_log: RAM samples with ``timestamp``.

    Returns:
        ``events`` with RAM columns joined on (events keep their row order).
    """
    if ram_log.empty:
        logger.warning("empty_ram_log_returning_unmerged_events")
        for col in RAM_COLUMNS:
            events[col] = np.nan
        return events

    events_sorted = events.sort_values("timestamp").reset_index(drop=True)
    ram_sorted = ram_log.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        events_sorted,
        ram_sorted[["timestamp", *RAM_COLUMNS]],
        on="timestamp",
        direction="backward",
    )
    # Restore original ordering and index.
    merged = merged.set_index(events_sorted.index)
    merged = merged.loc[events.index]
    return merged


def session_ram_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RAM statistics per session.

    Args:
        events: Events already aligned with RAM columns and carrying
            ``session_id``.

    Returns:
        Per-session DataFrame with ``avg_ram_mb``, ``peak_ram_mb``,
        ``ram_std_mb`` (system + browser) and peak ``cpu_percent`` if present.
    """
    if events.empty:
        return pd.DataFrame()

    groups = events.groupby("session_id")

    result = pd.DataFrame(
        {
            "avg_ram_mb": groups["ram_used_mb"].mean(),
            "peak_ram_mb": groups["ram_used_mb"].max(),
            "avg_browser_ram_mb": groups["browser_ram_mb"].mean(),
            "peak_browser_ram_mb": groups["browser_ram_mb"].max(),
            "ram_std_mb": groups["ram_used_mb"].std().fillna(0.0),
        }
    )
    if "cpu_percent" in events.columns:
        result["peak_cpu_percent"] = groups["cpu_percent"].max()
    return result.reset_index()
