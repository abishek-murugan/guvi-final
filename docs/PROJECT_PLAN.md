# Browsing Analyzer — Project Plan

## Overview

Build a production-quality AI system that analyzes Chrome browsing history for
selectable time windows (3/4/5 days), identifies browsing patterns and behavior
clusters, correlates browsing with system/browser RAM usage, trains a PyTorch
LSTM for next-category prediction, and generates actionable, evidence-backed
recommendations via a Streamlit dashboard.

## Architecture & Tech Stack

| Layer | Technology |
|-------|------------|
| Language / env | Python 3.11+ managed with astral `uv` |
| Data | pandas, numpy, sqlite3 |
| Config | Pydantic settings backed by `config/config.yaml` |
| Logging | structlog (JSON to file, pretty to console) |
| Preprocessing | `tldextract`, urllib |
| ML | scikit-learn (KMeans/GMM/DBSCAN + silhouette), PyTorch (LSTM) |
| Reporting | Markdown report, Plotly + Streamlit dashboard |
| Quality | ruff, mypy, pytest, GitHub Actions |

## Data Flow

```
Chrome history CSV (chrome_data.csv) ──┐
                                      ├──► clean (domain only, privacy)
RAM log CSV (ram_data.csv) ───────────┤
                                      ▼
                               categorize (domain→category)
                                      ▼
                               sessionize (15-min gap)
                                      ▼
                     session features + RAM alignment (merge_asof)
                          │          │            │
                   clustering    patterns      LSTM (PyTorch)
                   (KMeans/GMM)  (hour/day)   next-category
                          │          │            │
                          └──────────┴────────────┘
                                      ▼
                       recommendation engine (traceable rules)
                                      ▼
                  Markdown report + Streamlit dashboard
```

## Implementation Phases

1. **Foundation** — uv project, pydantic config, structlog, collectors,
   domain→category table, CLI scaffold.
2. **Preprocessing** — URL cleaning/domain extraction, categorization,
   sessionization with feature engineering, synthetic data generator.
3. **Analytics** — RAM↔browsing correlation (nearest-timestamp merge),
   KMeans/GMM clustering with silhouette + auto labels, temporal patterns.
4. **Deep learning** — PyTorch LSTM (Embedding → LSTM → Linear → Softmax),
   trainer with accuracy / macro F1 / confusion matrix and most-common-category
   baseline, deterministic seeds.
5. **Recommendations + Reporting** — traceable rule engine (≥5 rules), Markdown
   report generator, Streamlit dashboard.
6. **Testing & Docs** — unit + integration tests, README/AGENTS, GitHub Actions,
   privacy audit.

## Git Workflow

- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`.
- Feature branches off `develop`, merged with `--no-ff`.
- `main` holds production-ready state.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Browser / OS | Chrome on Linux | scope per requirements |
| DL approach | PyTorch LSTM next-category | predictive value, baseline comparison |
| Session gap | 15 minutes | standard inactivity threshold |
| RAM merge | backward `merge_asof` | most recent sample ≤ event time |
| Privacy | drop raw URLs after domain extraction | no sensitive data downstream |

## Evaluation Coverage

- Data pipeline quality: parse completeness, domain extraction accuracy,
  sessionization correctness.
- Clustering: silhouette score, interpretable labels, top drivers.
- Deep learning: accuracy / macro F1, confusion matrix, baseline comparison.
- Recommendation quality: ≥5 actionable, traceable recommendations.
- RAM analysis: category-wise RAM mean/peak, top-3 memory-heavy categories.
