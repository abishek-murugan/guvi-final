# Browsing Analyzer

AI system that analyzes browsing history for a selectable time window (last 3/4/5 days),
identifies browsing patterns, behavior clusters, correlates browsing with RAM usage, trains a
PyTorch LSTM to predict the next browsing category, and produces actionable recommendations
with a Streamlit dashboard.

## Features

- **Browsing history ingestion** — loads Chrome history exported as `chrome_data.csv`
  (timestamp, url, title) and normalizes it.
- **RAM usage ingestion** — loads system + browser RAM samples from `ram_data.csv`.
- **Privacy-first preprocessing** — every URL is reduced to its registrable domain; query
  strings and paths are stripped; raw URLs are dropped from all downstream artifacts.
- **Domain categorization** — curated `config/domain_categories.yaml` table + keyword
  heuristics (social / media / learning / shopping / productivity / news / …).
- **Sessionization** — 15-minute inactivity gap (configurable) with rich per-session features
  (duration, event count, switching rate, category entropy, RAM aggregates).
- **RAM correlation** — nearest-timestamp (backward) merge of RAM samples onto browsing
  events; per-session and per-category RAM statistics.
- **Behavior clustering** — KMeans / GMM with silhouette scoring and auto-generated
  human-readable cluster labels.
- **Temporal patterns** — hourly / daily usage heatmaps, peak-hour detection, category
  transition matrix.
- **Deep learning (PyTorch)** — Embedding → LSTM → Linear next-category predictor with
  accuracy / macro F1 / confusion matrix and a most-common-category baseline comparison.
- **Recommendation engine** — every recommendation is traceable to the metric that triggered
  it (e.g. `social_ratio > 0.45 after 22:00`).
- **Reporting** — Markdown report and an interactive Streamlit dashboard.

## Architecture

```
Chrome history CSV ─┐                       ┌─ RAM log CSV
 (chrome_data.csv)  ├─> clean -> categorize ├─ (ram_data.csv)
                    │     (domain only)     │
                    ▼                       ▼
             sessionize (15-min gap)  nearest-time RAM merge
                    │                       │
                    ▼                       ▼
              session features       per-session/category RAM stats
                    │                       │
        ┌───────────┼───────────┐           │
        ▼           ▼           ▼           ▼
   clustering   time        LSTM        recommendations
   (KMeans/GMM) patterns    (PyTorch)   (traceable rules)
        │           │           │           │
        └───────────┴───────────┴───────────┘
                    ▼
        Markdown report + Streamlit dashboard
```

## Project Layout

```
├── config/
│   ├── config.yaml             # All runtime configuration
│   └── domain_categories.yaml  # Domain -> category table
├── data/                       # Raw + processed data (gitignored)
├── logs/                       # Structured log output
├── reports/                    # Generated markdown reports
├── scripts/
│   └── run_pipeline.py             # One-shot pipeline runner
├── src/
│   └── browsing_analyzer/
│       ├── cli.py              # Typer CLI
│       ├── config.py           # Pydantic settings
│       ├── pipeline.py         # End-to-end orchestrator
│       ├── collect/            # history + RAM ingestion
│       ├── prep/               # cleaning, categorization, sessionization
│       ├── analytics/          # RAM correlation, clustering, patterns
│       ├── models/             # PyTorch LSTM + trainer
│       ├── recommendations/    # traceable recommendation engine
│       ├── reporting/          # markdown report + Streamlit dashboard
│       └── utils/              # logging, time helpers
└── src/tests/                  # unit + integration tests
```

## Setup

Requirements: **Python 3.11+**, **uv** (https://astral.sh/uv), Linux (Chrome history paths).

```bash
# 1. Install dependencies
uv sync

# 2. Place your collected data in place
#    data/raw/chrome_data.csv
#    data/raw/ram_data.csv
```

### Data format

`data/raw/chrome_data.csv` (minimum `timestamp`, `url` columns):

| timestamp               | url                                | title     |
|-------------------------|------------------------------------|-----------|
| 2026-07-28T10:00:00     | https://www.facebook.com/feed?ref=1 | My feed   |
| 2026-07-28T10:05:00     | https://www.youtube.com/watch?v=abc  | A video   |

`data/raw/ram_data.csv` — either a GB schema (converted to MB automatically) or an MB schema:

| timestamp           | total_ram_gb | used_ram_gb | available_ram_gb | chrome_ram_gb | cpu_percent |
|---------------------|--------------|-------------|------------------|---------------|-------------|
| 2026-07-28T10:00:00 | 16.0         | 9.24        | 6.76             | 1.46          | 30.0        |

| timestamp           | ram_used_mb | ram_available_mb | browser_ram_mb | cpu_percent |
|---------------------|-------------|------------------|----------------|-------------|
| 2026-07-28T10:00:00 | 4200.0      | 3992.0           | 1200.0         | 30.0        |

## Usage

```bash
# Full pipeline (collect -> preprocess -> analyze -> train -> recommend -> report)
uv run browsing-analyzer pipeline --window 4

# Individual steps
uv run browsing-analyzer preprocess --window 4
uv run browsing-analyzer train --window 4
uv run browsing-analyzer report --window 4

# Streamlit dashboard
uv run browsing-analyzer dashboard
# or: uv run streamlit run -m browsing_analyzer.reporting.dashboard

# Script-based runner (same as `pipeline`)
uv run python scripts/run_pipeline.py --window 4
```

## Testing

```bash
uv run pytest -q
uv run pytest -q --cov=browsing_analyzer --cov-report=term-missing
```

## Code Quality

```bash
uv run ruff check src
uv run ruff format --check src
uv run mypy src
```

## Privacy

- Query strings and paths are stripped during URL cleaning; only domains are stored.
- Raw URLs are dropped before any processed artifact is written.
- All data stays local — nothing is uploaded.
- `data/raw/`, `data/processed/`, `data/models/`, and `logs/` are gitignored.

## License

MIT
