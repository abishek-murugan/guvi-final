"""Generate synthetic (but realistic) browsing + RAM data for demo/testing.

The pipeline expects two CSVs:
    data/raw/chrome_data.csv
    data/raw/ram_data.csv

This script synthesizes 5 days of Chrome history and matching RAM samples so
the full pipeline can be exercised end-to-end without real browsing data. The
synthetic history includes:

* realistic hourly/weekend patterns,
* Markov-style category transitions (e.g. social -> media -> shopping),
* category-dependent browser RAM (media/social are memory-heavy),
* occasional RAM spikes to make clustering, RAM correlation, and
  recommendations meaningful.

Usage:
    uv run python scripts/generate_synthetic_data.py [--days 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Domains keyed by category (subset of config/domain_categories.yaml).
SITES: dict[str, list[str]] = {
    "social": [
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "reddit.com",
        "linkedin.com",
        "discord.com",
    ],
    "media": ["youtube.com", "netflix.com", "spotify.com", "twitch.tv"],
    "learning": [
        "github.com",
        "stackoverflow.com",
        "wikipedia.org",
        "medium.com",
        "w3schools.com",
        "coursera.org",
    ],
    "shopping": ["amazon.in", "flipkart.com", "myntra.com", "meesho.com"],
    "productivity": [
        "gmail.com",
        "docs.google.com",
        "calendar.google.com",
        "notion.so",
    ],
    "news": ["bbc.com", "thehindu.com", "ndtv.com", "reuters.com"],
    "entertainment": ["imdb.com", "9gag.com", "quora.com"],
    "finance": ["groww.in", "moneycontrol.com", "zerodha.com"],
    "search": ["google.com", "bing.com", "duckduckgo.com"],
}

# Category weights per hour of day (evening leans social/media, day leans learning).
def _category_weight(hour: int) -> dict[str, float]:
    base: dict[str, float] = {"learning": 0.25, "social": 0.2, "media": 0.15}
    if hour < 8:
        base = {"social": 0.15, "media": 0.35, "news": 0.3, "learning": 0.1}
    elif hour < 12:
        base = {"learning": 0.4, "productivity": 0.2, "news": 0.15, "social": 0.1}
    elif hour < 17:
        base = {
            "learning": 0.3,
            "productivity": 0.25,
            "social": 0.15,
            "shopping": 0.15,
        }
    elif hour < 21:
        base = {"social": 0.25, "media": 0.3, "shopping": 0.15, "learning": 0.1}
    else:
        base = {"social": 0.4, "media": 0.35, "entertainment": 0.1, "learning": 0.05}
    weights = {c: w for c, w in base.items() if c in SITES}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


# Markov-style next-category preferences (natural browsing flow). Used with
# probability ``follow_prev``; otherwise we fall back to hour-based weights.
TRANSITIONS: dict[str, dict[str, float]] = {
    "social": {"media": 0.35, "social": 0.3, "shopping": 0.15, "learning": 0.1, "news": 0.1},
    "media": {"media": 0.3, "social": 0.3, "learning": 0.15, "entertainment": 0.15, "shopping": 0.1},
    "learning": {"learning": 0.35, "productivity": 0.2, "social": 0.15, "news": 0.15, "search": 0.15},
    "productivity": {"learning": 0.3, "productivity": 0.2, "news": 0.2, "social": 0.15, "finance": 0.15},
    "shopping": {"social": 0.3, "media": 0.2, "shopping": 0.2, "search": 0.15, "finance": 0.15},
    "news": {"social": 0.3, "learning": 0.25, "news": 0.2, "media": 0.15, "finance": 0.1},
    "entertainment": {"media": 0.4, "social": 0.2, "learning": 0.2, "news": 0.1, "shopping": 0.1},
    "finance": {"news": 0.3, "social": 0.25, "shopping": 0.2, "productivity": 0.15, "learning": 0.1},
    "search": {"learning": 0.3, "social": 0.2, "shopping": 0.15, "news": 0.15, "media": 0.2},
}

# Browser RAM footprint per category (MB) — media/social are heavy.
CATEGORY_BROWSER_RAM: dict[str, float] = {
    "media": 2200.0,
    "social": 1600.0,
    "entertainment": 1300.0,
    "shopping": 1200.0,
    "news": 1000.0,
    "learning": 950.0,
    "search": 900.0,
    "productivity": 800.0,
    "finance": 850.0,
}


def _pick_category(prev_category: str | None, hour: int, rng: random.Random) -> str:
    """Choose the next category using transitions (if available) or hour weights."""
    if prev_category and prev_category in TRANSITIONS and rng.random() < 0.7:
        options = TRANSITIONS[prev_category]
        return rng.choices(list(options.keys()), weights=list(options.values()))[0]
    weights = _category_weight(hour)
    return rng.choices(list(weights.keys()), weights=list(weights.values()))[0]


def generate_history(days: int, seed: int) -> pd.DataFrame:
    """Generate a browsing history DataFrame covering ``days`` days."""
    rng = random.Random(seed)
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    rows: list[dict] = []

    day = start
    while day < end:
        weekday_factor = 0.7 if day.weekday() >= 5 else 1.0
        for hour in range(24):
            if hour < 6:
                continue
            # Visit density per hour: 5-14 on active hours, scaled by weekday.
            density = rng.uniform(5, 14) * weekday_factor
            if hour in (9, 10, 15, 16, 20, 21, 22):
                density *= 1.5
            n_visits = int(round(density)) + rng.randint(0, 3)
            prev_category: str | None = None
            for _ in range(n_visits):
                t = day + timedelta(
                    hours=hour, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59)
                )
                category = _pick_category(prev_category, hour, rng)
                domain = rng.choice(SITES[category])
                path = rng.choice(["/", "/feed", "/watch?v=abc", "/post", "/page", "/search?q=x"])
                url = f"https://www.{domain}{path}?ref={rng.randint(100, 999)}"
                rows.append(
                    {
                        "timestamp": t.isoformat(),
                        "url": url,
                        "title": f"{domain} {category} page",
                    }
                )
                prev_category = category
        day += timedelta(days=1)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def _domain_to_category(history: pd.DataFrame) -> np.ndarray:
    """Map each history row to a category via its URL host (fast local mapping)."""
    domain_to_cat: dict[str, str] = {}
    for cat, domains in SITES.items():
        for domain in domains:
            domain_to_cat[domain] = cat
            domain_to_cat[f"www.{domain}"] = cat

    categories: list[str] = []
    for url in history["url"]:
        host = (url or "").split("?", 1)[0].split("/", 3)[2]
        categories.append(domain_to_cat.get(host, "other"))
    return np.array(categories)


def generate_ram_log(
    history: pd.DataFrame, interval_s: int, seed: int
) -> pd.DataFrame:
    """Generate RAM samples every ``interval_s`` across the history span.

    Browser RAM is derived from the *active category* during the sample window
    (media/social are heavier), so RAM correlates with what the user was doing.
    """
    rng = random.Random(seed + 1)
    timestamps = pd.to_datetime(history["timestamp"]).to_numpy()
    categories = _domain_to_category(history)
    per_row_browser_ram = np.array(
        [CATEGORY_BROWSER_RAM.get(cat, 900.0) for cat in categories]
    ) + np.random.default_rng(seed + 2).uniform(-60, 60, size=len(categories))

    start = timestamps.min()
    end = timestamps.max()
    t_index = pd.date_range(start, end, freq=f"{interval_s}s")

    base = 3800.0
    total_mb = 8192.0
    rows = []
    # Index of the last history event at or before each sample.
    sample_ts = t_index.values.astype("datetime64[ns]").astype(np.int64)
    event_ts = timestamps.astype("datetime64[ns]").astype(np.int64)
    positions = np.searchsorted(event_ts, sample_ts, side="right") - 1

    for i, t in enumerate(t_index):
        hour_effect = 250.0 if t.hour in (20, 21, 22, 23) else 100.0
        spike = 500.0 if rng.random() < 0.02 else 0.0
        pos = positions[i]
        browser_ram = (
            (per_row_browser_ram[pos] if pos >= 0 else 700.0)
            + hour_effect * 0.5
            + spike * 0.4
            + rng.uniform(-40, 60)
        )
        ram_used = base + hour_effect + spike + rng.uniform(-40, 60)
        rows.append(
            {
                "timestamp": t.isoformat(),
                "ram_used_mb": round(max(ram_used, 1500), 1),
                "ram_available_mb": round(max(total_mb - ram_used, 500), 1),
                "browser_ram_mb": round(max(browser_ram, 200), 1),
                "cpu_percent": round(max(min(rng.gauss(32, 18), 99), 1), 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=5, help="Days of history to synthesize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--interval-s", type=int, default=10, help="RAM sample interval (s).")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="Output directory.")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    history = generate_history(args.days, args.seed)
    ram = generate_ram_log(history, args.interval_s, args.seed)

    history_path = out_dir / "chrome_data.csv"
    ram_path = out_dir / "ram_data.csv"
    history.to_csv(history_path, index=False)
    ram.to_csv(ram_path, index=False)
    print(f"Wrote {len(history)} history rows -> {history_path}")
    print(f"Wrote {len(ram)} RAM samples    -> {ram_path}")


if __name__ == "__main__":
    main()
