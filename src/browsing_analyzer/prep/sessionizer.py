"""Sessionization of browsing events.

A *session* is a consecutive run of browsing events where the time gap between
neighbouring events does not exceed ``inactivity_threshold_minutes``. For each
session we compute summary statistics (counts, ratios, switching behaviour,
RAM aggregates) used later by clustering and the LSTM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Sessionizer:
    """Builds sessions from timestamped events and computes session features.

    Args:
        settings: Application settings (sessionization thresholds).
    """

    def __init__(self, settings: Settings) -> None:
        self.threshold_min = settings.sessionization.inactivity_threshold_minutes
        self.min_events = settings.sessionization.min_session_events
        self.max_session_hours = settings.sessionization.max_session_hours

    def _compute_session_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-session statistics into a feature table."""
        session_groups = sessions.groupby("session_id", sort=True)

        features = pd.DataFrame(
            {
                "session_start": session_groups["timestamp"].min(),
                "session_end": session_groups["timestamp"].max(),
                "event_count": session_groups["timestamp"].count(),
                "unique_domains": session_groups["domain"].nunique(),
                "unique_categories": session_groups["category"].nunique(),
                "switching_rate": session_groups["category"].nunique()
                / session_groups["timestamp"].count(),
            }
        )

        features["duration_minutes"] = (
            features["session_end"] - features["session_start"]
        ).dt.total_seconds() / 60.0
        features["session_span_hours"] = features["duration_minutes"] / 60.0
        features["median_hour"] = session_groups["hour"].median()
        features["is_weekend"] = (
            session_groups["day_name"].first().isin({"Saturday", "Sunday"}).astype(int)
        )

        # Category counts (vectorized crosstab) drive both entropy and ratios.
        counts = pd.crosstab(sessions["session_id"], sessions["category"])

        # Category entropy (how "scattered" the browsing was within the session).
        probs = counts.div(counts.sum(axis=1), axis=0)
        features["category_entropy"] = -(probs * np.log(probs + 1e-12)).sum(axis=1)

        # Category ratios relative to event count.
        n_events = features["event_count"]
        for cat in sorted(set(sessions["category"]) | {"other"}):
            col = f"{cat}_ratio"
            if cat in counts.columns:
                features[col] = counts[cat].reindex(features.index).fillna(0) / n_events
            else:
                features[col] = 0.0

        features = features.reset_index()
        return features

    def sessionize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Assign session ids to events and compute session-level features.

        Args:
            df: Cleaned, categorized history with ``timestamp``, ``domain``,
                ``category``, ``hour``, ``day_name`` columns, sorted by time.

        Returns:
            ``(events, sessions)`` where ``events`` carries a ``session_id``
            column and ``sessions`` is the per-session feature table.
        """
        if df.empty:
            return df.copy(), pd.DataFrame()

        events = df.sort_values("timestamp").reset_index(drop=True).copy()
        gaps = events["timestamp"].diff().dt.total_seconds() / 60.0
        new_session = gaps > self.threshold_min

        # Duration cap: even when no inactivity gap occurs (e.g. data sampled
        # continuously at 1Hz), force a break whenever a session would span more
        # than ``max_session_hours`` so sessions stay meaningful downstream.
        if self.max_session_hours > 0:
            session_start = events["timestamp"].where(new_session | (events.index == 0)).ffill()
            elapsed_hours = (events["timestamp"] - session_start).dt.total_seconds() / 3600.0
            bucket = elapsed_hours // self.max_session_hours
            duration_break = (bucket != bucket.shift(1)).fillna(False)
            new_session = new_session | duration_break

        events["session_id"] = new_session.cumsum().astype(int)

        sessions = self._compute_session_features(events)

        # Drop sessions below the minimum event count.
        keep = sessions["event_count"] >= self.min_events
        kept_ids = set(sessions.loc[keep, "session_id"])
        sessions = sessions[keep].reset_index(drop=True)
        events = events[events["session_id"].isin(kept_ids)].reset_index(drop=True)

        logger.info(
            "sessions_built",
            sessions=len(sessions),
            events=len(events),
            threshold_minutes=self.threshold_min,
        )
        return events, sessions
