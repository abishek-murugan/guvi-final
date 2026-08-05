# Time-Based Browsing Pattern Analyzer

Analyzes browser history and correlates it with system RAM usage to uncover
behavioral patterns, cluster session types, and predict the next browsing
category with an LSTM. Developed as the DS105 final project.

## Project Structure

```
config/                         YAML configuration (notebook-exact hyper-params)
notebooks/guvi-final.ipynb      Authoritative pipeline (single source of truth)
src/browsing_analyzer/
  cli.py                        Typer CLI (pipeline / preprocess / train / report)
  dashboard.py                  Streamlit + Plotly interactive dashboard
  pipeline.py                   End-to-end orchestrator + artifact persistence
  collect/                      Chrome history + RAM log loaders (collectors, never run)
  prep/                         Cleaning, domain categorization, sessionization
  analytics/                    RAM correlation, KMeans clustering, temporal patterns
  models/                       PyTorch LSTM next-category predictor + trainer
  recommendations/              Traceable, evidence-backed recommendations
  reporting/                    Markdown report generator + static plots
data/raw/                       Input CSVs (protected, never modified)
data/processed/                 Regenerated artifacts (sessions, features, metrics)
data/models/                    Trained LSTM, cluster model, training results
reports/                        Final report + images
```

## Setup

```bash
uv sync
```

## Usage

```bash
# Run the full pipeline (preprocess -> cluster -> LSTM -> report)
uv run browsing-analyzer pipeline

# Or run the stages individually
uv run browsing-analyzer preprocess
uv run browsing-analyzer train
uv run browsing-analyzer report

# Interactive dashboard
uv run streamlit run src/browsing_analyzer/dashboard.py
```

The dashboard loads persisted artifacts from `data/processed` and
`data/models` by default; a sidebar toggle re-runs the full pipeline
(including ~2 min of LSTM training) before rendering.

## Configuration

All hyper-parameters live in `config/config.yaml` and mirror the notebook:
15-minute sessionization, KMeans with `k=2` over 13 scaled features, and the
LSTM (sequence length 5, embedding 128, hidden 256, 2 layers, dropout 0.5,
batch 64, lr 0.001, 10 epochs). The domain→category mapping (654 domains,
16 categories) is extracted at runtime from the notebook and cached to
`config/domain_category_map.yaml`.

## Key Findings (current run)

- **Sequential behavior:** The LSTM predicts the next browsing category with
  **~85% test accuracy**.
- **Performance impact:** Search/Reference and Social Media drive the highest
  peak RAM usage (~6.6 GB); entertainment and social sites dominate memory
  consumption.
- **Behavioral clusters:** KMeans identifies **2 session types** (low vs high
  RAM sessions, silhouette ≈ 0.62).

## Deliverables

- `reports/Final_Project_Report.md`: full analysis report
- `reports/images/*.png`: 13 visualizations (clusters, RAM correlation, LSTM curves, …)
- `data/processed/`: session features, RAM stats, LSTM sequences, metrics, recommendations
- `data/models/`: `lstm_model.pt`, `lstm_metadata.json`, `cluster_model.pkl`, `lstm_result.pkl`

## Quality

```bash
uvx ruff check src
uvx ruff format --check src
uv run --with mypy mypy src
```
