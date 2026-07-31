"""Unit tests for sessionization."""

from __future__ import annotations

import pandas as pd

from browsing_analyzer.config import Settings
from browsing_analyzer.prep.categorizer import Categorizer
from browsing_analyzer.prep.cleaner import clean_history
from browsing_analyzer.prep.sessionizer import Sessionizer


def _prepare(sample_history_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_history(sample_history_df)
    cleaned["category"] = Categorizer().categorize_series(cleaned["domain"])
    return cleaned


def test_session_count_with_15_min_threshold(settings: Settings, sample_history_df: pd.DataFrame):
    df = _prepare(sample_history_df)
    sessionizer = Sessionizer(settings)
    events, sessions = sessionizer.sessionize(df)
    # Gaps: 10:12 -> 11:30 (78min) = new session; 11:35 -> 14:00 (145min) = new session.
    assert len(sessions) == 3


def test_session_events_get_session_id(settings: Settings, sample_history_df: pd.DataFrame):
    df = _prepare(sample_history_df)
    sessionizer = Sessionizer(settings)
    events, _ = sessionizer.sessionize(df)
    assert "session_id" in events.columns
    assert events["session_id"].nunique() == 3


def test_session_features_computed(settings: Settings, sample_history_df: pd.DataFrame):
    df = _prepare(sample_history_df)
    sessionizer = Sessionizer(settings)
    events, sessions = sessionizer.sessionize(df)
    for col in [
        "event_count",
        "unique_domains",
        "duration_minutes",
        "switching_rate",
        "median_hour",
    ]:
        assert col in sessions.columns
    assert sessions["event_count"].sum() == len(events)


def test_min_session_events_filter(settings: Settings):
    df = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-07-28 10:00:00"), "url": "https://www.facebook.com/"},
            {"timestamp": pd.Timestamp("2026-07-29 10:00:00"), "url": "https://www.facebook.com/"},
        ]
    )
    prepared = _prepare(df)
    sessionizer = Sessionizer(settings)
    events, sessions = sessionizer.sessionize(prepared)
    # Single-event sessions dropped (min_session_events = 2).
    assert sessions.empty
    assert events.empty


def test_custom_threshold(settings: Settings, sample_history_df: pd.DataFrame):
    settings.sessionization.inactivity_threshold_minutes = 120
    df = _prepare(sample_history_df)
    sessionizer = Sessionizer(settings)
    events, sessions = sessionizer.sessionize(df)
    # With 120 min threshold, only the 145-min gap splits sessions -> 2 sessions.
    assert len(sessions) == 2
