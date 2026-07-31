"""Unit tests for RAM-browsing correlation."""

from __future__ import annotations

from browsing_analyzer.analytics.ram_correlation import align_ram_with_events, session_ram_stats
from browsing_analyzer.config import Settings
from browsing_analyzer.prep.categorizer import Categorizer
from browsing_analyzer.prep.cleaner import clean_history
from browsing_analyzer.prep.sessionizer import Sessionizer


def _prepared(settings: Settings, sample_history_df, sample_ram_df):
    cleaned = clean_history(sample_history_df)
    cleaned["category"] = Categorizer().categorize_series(cleaned["domain"])
    sessionizer = Sessionizer(settings)
    events, _ = sessionizer.sessionize(cleaned)
    aligned = align_ram_with_events(events, sample_ram_df)
    return aligned


def test_align_ram_backward_merge(settings: Settings, sample_history_df, sample_ram_df):
    aligned = _prepared(settings, sample_history_df, sample_ram_df)
    assert "ram_used_mb" in aligned.columns
    # Every event must have a non-null RAM value (RAM covers the whole span).
    assert aligned["ram_used_mb"].notna().all()


def test_align_ram_values_sane(settings: Settings, sample_history_df, sample_ram_df):
    aligned = _prepared(settings, sample_history_df, sample_ram_df)
    assert (aligned["ram_used_mb"] > 0).all()
    assert (aligned["browser_ram_mb"] > 0).all()


def test_session_ram_stats_aggregation(settings: Settings, sample_history_df, sample_ram_df):
    aligned = _prepared(settings, sample_history_df, sample_ram_df)
    stats = session_ram_stats(aligned)
    assert {"avg_ram_mb", "peak_ram_mb", "avg_browser_ram_mb"}.issubset(stats.columns)
    assert stats["peak_ram_mb"].ge(stats["avg_ram_mb"]).all()


def test_align_ram_empty_log_returns_nan(settings: Settings, sample_history_df):
    cleaned = clean_history(sample_history_df)
    cleaned["category"] = Categorizer().categorize_series(cleaned["domain"])
    events, _ = Sessionizer(settings).sessionize(cleaned)
    import pandas as pd

    aligned = align_ram_with_events(events, pd.DataFrame(columns=["timestamp", "ram_used_mb"]))
    assert aligned["ram_used_mb"].isna().all()
