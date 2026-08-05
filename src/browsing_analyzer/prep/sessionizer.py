"""Sessionization of browsing events.

A *session* is a consecutive run of events where the time gap between
neighbouring events does not exceed ``inactivity_threshold_minutes``. Each
session is summarized with counts, switching behaviour and pacing metrics
used downstream by clustering and the LSTM.
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def _primary_category(series: pd.Series) -> str:
    """Return the most frequent category in a series."""
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else "Unknown"


def _switches(series: pd.Series) -> int:
    """Count how many times consecutive values change within a series."""
    return int((series != series.shift(1)).sum() - 1)


class Sessionizer:
    """Builds sessions from timestamped events and session-level features.

    Args:
        inactivity_threshold_minutes: Maximum gap between events in a session.
    """

    def __init__(self, inactivity_threshold_minutes: int = 15) -> None:
        self.threshold_seconds = inactivity_threshold_minutes * 60

    def sessionize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Assign ``session_id`` to events and compute session summaries.

        Args:
            df: Cleaned, categorized history sorted by time.

        Returns:
            ``(events, session_summary)`` where ``events`` carries
            ``time_diff``, ``new_session`` and ``session_id`` columns, and
            ``session_summary`` is the per-session feature table.
        """
        events = df.sort_values("timestamp").reset_index(drop=True).copy()
        events["time_diff"] = events["timestamp"].diff().dt.total_seconds()
        events["new_session"] = (events["time_diff"] > self.threshold_seconds) | events[
            "time_diff"
        ].isna()
        events["session_id"] = events["new_session"].cumsum()

        session_summary = self._session_summary(events)
        logger.info("sessions_built", sessions=len(session_summary), events=len(events))
        return events, session_summary

    def _session_summary(self, events: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-session statistics into a summary table."""
        grouped = events.groupby("session_id", sort=True)

        summary = pd.DataFrame(
            {
                "session_start": grouped["timestamp"].min(),
                "session_end": grouped["timestamp"].max(),
                "page_count": grouped["url"].count(),
                "unique_domains": grouped["domain"].nunique(),
                "unique_categories": grouped["category"].nunique(),
                "primary_category": grouped["category"].agg(_primary_category),
            }
        )
        summary["session_duration_minutes"] = (
            summary["session_end"] - summary["session_start"]
        ).dt.total_seconds() / 60.0

        switches = pd.DataFrame(
            {
                "domain_switches": grouped["domain"].agg(_switches),
                "category_switches": grouped["category"].agg(_switches),
            }
        )
        summary = summary.join(switches)
        summary["domain_switches"] = summary["domain_switches"].clip(lower=0)
        summary["category_switches"] = summary["category_switches"].clip(lower=0)
        summary["avg_time_per_page_seconds"] = (
            summary["session_duration_minutes"] * 60.0
        ) / summary["page_count"]
        return summary.reset_index()
