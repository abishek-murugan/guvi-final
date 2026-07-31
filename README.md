# Browsing Analyzer

AI system that analyzes browsing history for a selectable time window (last 3/4/5 days),
identifies browsing patterns, behavior clusters, correlates browsing with RAM usage, trains a
PyTorch LSTM to predict the next browsing category, and produces actionable recommendations
with a Markdown report.

## Features

- **Browsing history ingestion** — `extract_chrome_history()` reads Chrome's SQLite
  `History` database, converts WebKit timestamps, and exports `data/raw/chrome_data.csv`;
  `load_browsing_history()` ingests that file and normalizes it.
- **RAM usage ingestion** — `collect_ram_log()` samples system + browser RAM via `psutil`
  and exports `data/raw/ram_data.csv`; `load_ram_log()` ingests that file.
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
  accuracy / macro F1 / confusion matrix and a most-common-category baseline comparison;
  the trained model, category vocab, and full architecture config are saved to
  `data/models/lstm_model.pt`.
- **Recommendation engine** — every recommendation is traceable to the metric that triggered
  it (e.g. `social_ratio > 0.45 after 22:00`).
- **Reporting** — Markdown report.

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
               Markdown report
```

## Project Layout

```
├── config/
│   ├── config.yaml             # All runtime configuration
│   └── domain_categories.yaml  # Domain -> category table
├── data/
│   ├── raw/                    # chrome_data.csv + ram_data.csv (collection outputs)
│   ├── processed/              # sanitized events/sessions/clusters (domain level only)
│   └── models/                 # trained LSTM model (lstm_model.pt)
├── logs/                       # Structured log output
├── reports/                    # Generated markdown report
└── src/
    └── browsing_analyzer/
        ├── cli.py              # Typer CLI
        ├── config.py           # Pydantic settings
        ├── pipeline.py         # End-to-end orchestrator
        ├── collect/            # Chrome SQLite extraction, psutil RAM logger, loaders
        ├── prep/               # cleaning, categorization, sessionization
        ├── analytics/          # RAM correlation, clustering, patterns
        ├── models/             # PyTorch LSTM + trainer + checkpointing
        ├── recommendations/    # traceable recommendation engine
        ├── reporting/          # markdown report generator
        └── utils/              # logging, time helpers
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
```

## Collecting data (optional)

Data collection runs once on your machine, privately, and is not part of the
pipeline (the pipeline reads the exported CSVs):

```bash
# Extract browsing history from Chrome's SQLite DB -> data/raw/chrome_data.csv
uv run python -c "from browsing_analyzer.collect.history import extract_chrome_history; extract_chrome_history()"

# Log system + Chrome RAM every 5s for 24h -> data/raw/ram_data.csv
uv run python -c "from browsing_analyzer.collect.ram_logger import collect_ram_log; from browsing_analyzer.config import load_settings; collect_ram_log(load_settings())"
```

## Code Quality

```bash
uvx ruff check src
uvx ruff format --check src
uv run --with mypy mypy src
```

## Privacy

- Query strings and paths are stripped during URL cleaning; only domains are stored.
- Raw URLs are dropped before any processed artifact is written.
- All data stays local — nothing is uploaded.
- `data/raw/`, `data/processed/`, `data/models/`, and `logs/` are gitignored.

## License

MIT
