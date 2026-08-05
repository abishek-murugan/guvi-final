"""Temporal browsing pattern discovery.

Extracts hourly and daily usage patterns, a category-by-hour distribution and
a category transition matrix used to reason about browsing flow.
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def discover_time_patterns(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute temporal pattern tables used in reports and the dashboard.

    Args:
        events: Cleaned, categorized events with ``hour``, ``date``,
            ``day_name`` and ``category`` columns.

    Returns:
        Dictionary of DataFrames: ``hourly``, ``daily``, ``category_by_hour``,
        ``transitions`` and ``peak_hours``.
    """
    if events.empty:
        return {
            "hourly": pd.DataFrame(),
            "daily": pd.DataFrame(),
            "category_by_hour": pd.DataFrame(),
            "transitions": pd.DataFrame(),
            "peak_hours": pd.DataFrame(),
        }

    hourly_counts = events.groupby(["date", "hour"]).size().reset_index(name="visits")
    hourly = hourly_counts.pivot_table(index="hour", columns="date", values="visits", fill_value=0)
    hourly = hourly.reindex(range(24), fill_value=0)
    hourly["mean_visits"] = hourly.mean(axis=1)

    daily_counts = events.groupby(["date", "day_name"]).size().reset_index(name="visits")
    daily = daily_counts.pivot_table(
        index="date", columns="day_name", values="visits", fill_value=0
    )
    daily = daily.reindex(columns=_DAY_ORDER, fill_value=0)

    category_by_hour = (events.groupby(["hour", "category"]).size().unstack(fill_value=0)).reindex(
        range(24), fill_value=0
    )

    categories = events["category"]
    transitions = (
        pd.crosstab(categories, categories.shift(-1), normalize="index")
        if len(events) > 1
        else pd.DataFrame()
    )

    peak_hours = hourly["mean_visits"].sort_values(ascending=False).head(8).reset_index()
    peak_hours.columns = ["hour", "mean_visits"]

    logger.info("time_patterns_computed", hours=len(hourly), days=len(daily))
    return {
        "hourly": hourly,
        "daily": daily,
        "category_by_hour": category_by_hour,
        "transitions": transitions,
        "peak_hours": peak_hours,
    }
