"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from browsing_analyzer.config import Settings

DOMAINS: list[tuple[str, str]] = [
    ("facebook.com", "social"),
    ("youtube.com", "media"),
    ("stackoverflow.com", "learning"),
    ("amazon.in", "shopping"),
    ("gmail.com", "productivity"),
    ("bbc.com", "news"),
]


@pytest.fixture
def settings() -> Settings:
    """A default Settings instance for tests."""
    return Settings()


@pytest.fixture
def synthetic_history_df() -> pd.DataFrame:
    """Deterministic browsing history spanning 3 days (~15 sessions)."""
    rng = random.Random(7)
    rows: list[dict] = []
    start = pd.Timestamp("2026-07-28 08:00:00")
    for day in range(3):
        for s in range(5):
            session_start = start + pd.Timedelta(days=day, hours=7 + s * 3 + rng.randint(0, 1))
            n_events = rng.randint(5, 10)
            prev_domain = None
            for i in range(n_events):
                if prev_domain is None or rng.random() < 0.4:
                    prev_domain = rng.choice(DOMAINS)
                domain = prev_domain
                t = session_start + pd.Timedelta(minutes=i * rng.randint(1, 4))
                rows.append(
                    {
                        "timestamp": t,
                        "url": f"https://www.{domain[0]}/page/{i}",
                        "title": f"{domain[0]} page",
                    }
                )
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


@pytest.fixture
def synthetic_ram_df() -> pd.DataFrame:
    """Deterministic RAM log in the GB schema spanning the synthetic history."""
    start = pd.Timestamp("2026-07-28 08:00:00")
    index = pd.date_range(start, start + pd.Timedelta(days=3), freq="10min")
    rows = []
    for i, t in enumerate(index):
        spike = 0.5 if i % 137 == 0 else 0.0
        evening = 0.3 if t.hour in (20, 21, 22, 23) else 0.0
        used = round(4.2 + evening + spike, 2)
        rows.append(
            {
                "timestamp": t,
                "total_ram_gb": 16.0,
                "used_ram_gb": used,
                "available_ram_gb": round(16.0 - used, 2),
                "ram_percent": round(used / 16.0 * 100.0, 1),
                "chrome_tabs": 8,
                "chrome_ram_gb": round(1.2 + 0.05 * (t.hour % 24), 2),
                "cpu_percent": round(30.0 + 10.0 * (t.hour % 5), 1),
                "network_kbps": 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_data_files(
    tmp_path: Path, synthetic_history_df: pd.DataFrame, synthetic_ram_df: pd.DataFrame
) -> tuple[Path, Path]:
    """Write the synthetic history + RAM log (GB schema) as raw CSVs."""
    out = tmp_path / "raw"
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "chrome_data.csv"
    ram_path = out / "ram_data.csv"
    synthetic_history_df.to_csv(history_path, index=False)
    synthetic_ram_df.to_csv(ram_path, index=False)
    return history_path, ram_path


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
