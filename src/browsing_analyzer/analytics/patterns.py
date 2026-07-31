"""Temporal browsing pattern discovery.

Extracts hourly/day-wise usage patterns, identifies peak activity blocks and
builds a category transition matrix used to reason about browsing flow.
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

_HOURS = list(range(24))
_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _hour_activity(events: pd.DataFrame) -> pd.DataFrame:
    """Count events per hour of day (0-23)."""
    counts = events.groupby(["date", "hour"]).size().reset_index(name="visits")
    pivot = counts.pivot_table(index="hour", columns="date", values="visits", fill_value=0)
    pivot = pivot.reindex(_HOURS, fill_value=0)
    pivot["mean_visits"] = pivot.mean(axis=1)
    return pivot


def _day_activity(events: pd.DataFrame) -> pd.DataFrame:
    """Count events per day of week."""
    counts = events.groupby(["date", "day_name"]).size().reset_index(name="visits")
    pivot = counts.pivot_table(index="date", columns="day_name", values="visits", fill_value=0)
    pivot = pivot.reindex(columns=_DAY_ORDER, fill_value=0)
    return pivot


def discover_time_patterns(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute temporal pattern tables used in reports and the dashboard.

    Args:
        events: Cleaned, categorized browsing events with ``hour``, ``date``,
            ``day_name``, ``category`` columns.

    Returns:
        Dictionary of DataFrames: ``hourly``, ``daily``, ``category_by_hour``,
        ``transitions`` and ``peak_hours``.
    """
    hourly = _hour_activity(events)
    daily = _day_activity(events)

    category_by_hour = (events.groupby(["hour", "category"]).size().unstack(fill_value=0)).reindex(
        _HOURS, fill_value=0
    )

    # Category transition matrix: P(cat_t+1 | cat_t) across consecutive visits.
    categories = events["category"]
    transitions = (
        pd.crosstab(categories, categories.shift(-1), normalize="index")
        if len(events) > 1
        else pd.DataFrame()
    )

    # Peak hours: top hours by event count with a label.
    hour_counts = hourly["mean_visits"].sort_values(ascending=False)
    peak_hours = pd.DataFrame(
        {
            "hour": hour_counts.index.astype(int),
            "mean_visits": hour_counts.values,
        }
    ).head(8)

    logger.info("time_patterns_computed", hours=len(hourly), days=len(daily))
    return {
        "hourly": hourly,
        "daily": daily,
        "category_by_hour": category_by_hour,
        "transitions": transitions,
        "peak_hours": peak_hours,
    }
