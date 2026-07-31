# AGENTS.md

Guidance for AI coding assistants working in this repository.

## Project Overview

`browsing-analyzer` is a privacy-first AI pipeline that analyzes local browsing
history (Chrome) correlated with system/browser RAM usage. It produces
clusters, time patterns, a PyTorch LSTM next-category model, recommendations,
a Markdown report, and a Streamlit dashboard.

## Commands

- Install deps: `uv sync`
- Run tests: `uv run pytest -q`
- Lint: `uv run ruff check src`
- Format check: `uv run ruff format --check src`
- Type check: `uv run mypy src`
- Full pipeline: `uv run browsing-analyzer pipeline --window 4`
- Runner script: `uv run python scripts/run_pipeline.py --window 4`

## Code Layout

- `src/browsing_analyzer/` — the importable package (src-layout, entry point
  `browsing-analyzer` in `pyproject.toml`).
- `src/browsing_analyzer/config.py` — Pydantic settings loaded from
  `config/config.yaml`.
- `src/browsing_analyzer/pipeline.py` — end-to-end orchestrator returning a
  `PipelineResult`.
- `src/browsing_analyzer/{collect,prep,analytics,models,recommendations,reporting,utils}/`
  — modular responsibilities matching the plan.
- `src/tests/` — pytest suite (unit + integration); synthetic fixtures live in
  `src/tests/conftest.py`.
- `scripts/` — standalone runners.
- `config/` — configuration YAML files (never treat as code).

## Conventions

- Follow existing style; run `ruff check` before finishing.
- Add type hints to public functions; use `from __future__ import annotations`.
- Log key steps via `get_logger(__name__)` (structlog) — do not use `print` in
  library code.
- Keep configuration in `config/config.yaml` (or Pydantic defaults), never
  hardcode paths in modules.
- **Privacy invariant:** raw URLs must never appear in processed data,
  reports, or dashboards — only domains/categories. `pipeline.py` drops the
  `url` column after domain extraction.
- Tests must be deterministic: use fixed seeds and synthetic data fixtures in
  `src/tests/conftest.py`.

## Config

Key runtime knobs (all validated by Pydantic):

- `sessionization.inactivity_threshold_minutes` (default 15)
- `clustering.algorithm` / `n_clusters` (kmeans | gmm | dbscan)
- `model.*` — LSTM hyper-parameters
- `recommendations.*` — thresholds that must stay in sync with
  `recommendations/engine.py`

## Verification

Before considering a change done:

1. `uv run ruff check src && uv run ruff format --check src`
2. `uv run mypy src`
3. `uv run pytest -q`
4. If you touched pipeline data flows, run `uv run python scripts/run_pipeline.py --window 3`
   and confirm the report generates without raw URLs.
