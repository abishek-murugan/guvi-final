"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from browsing_analyzer.config import Settings


@pytest.fixture
def settings() -> Settings:
    """A default Settings instance for tests."""
    return Settings()


@pytest.fixture
def sample_history_df() -> pd.DataFrame:
    """A small, deterministic browsing history frame (3 sessions)."""
    rows = [
        {
            "timestamp": pd.Timestamp("2026-07-28 10:00:00"),
            "url": "https://www.facebook.com/feed?ref=1",
            "title": "fb",
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 10:05:00"),
            "url": "https://www.youtube.com/watch?v=abc",
            "title": "yt",
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 10:12:00"),
            "url": "https://stackoverflow.com/questions/1",
            "title": "so",
        },
        # 1 hour gap -> new session
        {
            "timestamp": pd.Timestamp("2026-07-28 11:30:00"),
            "url": "https://github.com/user/repo",
            "title": "gh",
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 11:35:00"),
            "url": "https://www.amazon.in/shop",
            "title": "amz",
        },
        # 2 hour gap -> new session
        {
            "timestamp": pd.Timestamp("2026-07-28 14:00:00"),
            "url": "https://www.netflix.com/browse",
            "title": "nf",
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 14:10:00"),
            "url": "https://www.twitter.com/home",
            "title": "tw",
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_ram_df() -> pd.DataFrame:
    """A small RAM log aligned with the sample history."""
    rows = [
        {
            "timestamp": pd.Timestamp("2026-07-28 10:00:00"),
            "ram_used_mb": 4200.0,
            "ram_available_mb": 3992.0,
            "browser_ram_mb": 1200.0,
            "cpu_percent": 30.0,
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 10:05:00"),
            "ram_used_mb": 4350.0,
            "ram_available_mb": 3842.0,
            "browser_ram_mb": 1350.0,
            "cpu_percent": 45.0,
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 10:12:00"),
            "ram_used_mb": 4400.0,
            "ram_available_mb": 3792.0,
            "browser_ram_mb": 1400.0,
            "cpu_percent": 55.0,
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 11:30:00"),
            "ram_used_mb": 4100.0,
            "ram_available_mb": 4092.0,
            "browser_ram_mb": 1100.0,
            "cpu_percent": 25.0,
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 11:35:00"),
            "ram_used_mb": 4250.0,
            "ram_available_mb": 3942.0,
            "browser_ram_mb": 1300.0,
            "cpu_percent": 40.0,
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 14:00:00"),
            "ram_used_mb": 4800.0,
            "ram_available_mb": 3392.0,
            "browser_ram_mb": 1900.0,
            "cpu_percent": 70.0,
        },
        {
            "timestamp": pd.Timestamp("2026-07-28 14:10:00"),
            "ram_used_mb": 4700.0,
            "ram_available_mb": 3492.0,
            "browser_ram_mb": 1800.0,
            "cpu_percent": 65.0,
        },
    ]
    return pd.DataFrame(rows)
