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
            session_groups["day_name"]
            .apply(lambda s: s.iloc[0] in {"Saturday", "Sunday"})
            .astype(int)
        )

        # Category entropy (how "scattered" the browsing was within the session).
        def _entropy(cats: pd.Series) -> float:
            probs = cats.value_counts(normalize=True).values
            return float(-(probs * np.log(probs + 1e-12)).sum())

        features["category_entropy"] = session_groups["category"].apply(_entropy)

        # Category ratios relative to event count.
        counts = session_groups["category"].value_counts().unstack(fill_value=0)
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
