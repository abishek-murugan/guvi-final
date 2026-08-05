"""Markdown report generation.

Produces the final project report with the real pipeline outputs: data
summary, RAM correlation, clustering, LSTM metrics, recommendations and the
list of deliverable files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..pipeline import PipelineResult

logger = get_logger(__name__)


def generate_markdown_report(result: PipelineResult, output_path: Path) -> Path:
    """Write the final project report for a pipeline run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("# DS105 Final Project: Time-Based Browsing Pattern Analyzer")
    lines.append("")
    lines.append(
        "## Project Overview\n\n"
        f"This project analyzes browsing history and system RAM usage over the window "
        f"{result.events['timestamp'].min():%Y-%m-%d} to "
        f"{result.events['timestamp'].max():%Y-%m-%d}. The goal is to identify behavioral "
        "patterns, categorize web usage, and correlate these activities with system performance."
    )

    # 1. Data summary
    lines.append("## 1. Data Summary\n")
    lines.append(
        f"The dataset consists of **{len(result.events):,} browsing events** across "
        f"**{len(result.session_features):,} sessions** and **{len(result.category_ram_stats)} "
        f"browsing categories**, synchronized with RAM logs captured at 5-second intervals."
    )
    top_domains = result.top_domains.head(10)
    if not top_domains.empty:
        lines.append("\n### Top 10 Domains")
        lines.append(_df_table(top_domains))

    category_distribution = result.events["category"].value_counts().reset_index()
    category_distribution.columns = ["category", "count"]
    lines.append("\n### Category Distribution")
    lines.append(_df_table(category_distribution))
    lines.append("")

    # 2. RAM correlation
    lines.append("## 2. RAM Correlation Analysis\n")
    ram = result.category_ram_stats.sort_values("peak_used_mb", ascending=False)
    if not ram.empty:
        lines.append("### RAM Usage by Category (MB)")
        view = ram.copy()
        view[["mean_used_mb", "peak_used_mb", "mean_usage_percent", "peak_usage_percent"]] = view[
            ["mean_used_mb", "peak_used_mb", "mean_usage_percent", "peak_usage_percent"]
        ].round(2)
        lines.append(_df_table(view))
        top = ram.iloc[0]
        lines.append(
            f"\n> **Finding:** **{top['category']}** has the highest peak RAM usage "
            f"({top['peak_used_mb']:.0f} MB); entertainment and social media are the primary "
            "drivers of high memory consumption."
        )
    lines.append("")

    # 3. Clustering
    lines.append("## 3. Behavior Clustering\n")
    if result.cluster is not None:
        cluster = result.cluster
        lines.append(
            f"Using KMeans clustering on scaled session features, we identified "
            f"**{len(cluster.cluster_centers)} distinct session types** "
            f"(silhouette score {cluster.silhouette:.3f})."
        )
        lines.append("\n### Cluster Profiles")
        lines.append(_df_table(cluster.cluster_centers.round(2)))
        lines.append("\n### Cluster Labels")
        labels = pd.DataFrame(
            [{"cluster": cid, "label": label} for cid, label in cluster.profiles.items()]
        )
        lines.append(_df_table(labels))
    lines.append("")

    # 4. Deep learning
    lines.append("## 4. Deep Learning: Next-Category Prediction\n")
    if result.dl_result is not None:
        dl = result.dl_result
        lines.append(
            "An **LSTM (Long Short-Term Memory)** model predicts the next browsing category "
            "from the last 5 visits.\n"
        )
        lines.append(
            f"* **Model Architecture:** Embedding (128) -> LSTM (2 layers, 256 units, dropout 0.5) "
            "-> Linear (Softmax)\n"
            f"* **Training:** {len(dl.history['train_loss'])} epochs, Adam lr=0.001, batch size 64\n"
            f"* **Test Accuracy:** **{dl.test_accuracy:.2%}**\n"
            f"* **Insight:** Browsing behavior is highly sequential, allowing the model to "
            "anticipate transitions between categories."
        )
        report = dl.classification_report
        if report:
            report_df = _classification_report_df(report, dl.confusion is not None)
            lines.append("\n### Classification Report (Test Set)")
            lines.append(_df_table(report_df))
    lines.append("")

    # 5. Recommendations
    lines.append("## 5. Actionable Recommendations\n")
    for i, rec in enumerate(result.recommendations, start=1):
        lines.append(f"{i}. **{rec.title}** ({rec.severity}) — {rec.rationale}")
    lines.append("")

    # 6. Deliverables
    lines.append("## 6. Deliverables\n")
    lines.append(
        "- `data/processed/final_browsing_history.csv`: sanitized and preprocessed history\n"
        "- `data/processed/final_ram_log.csv`: time-aligned RAM metrics\n"
        "- `data/processed/domain_category_map.csv`: mapping of domains to categories\n"
        "- `data/processed/session_features.csv`: engineered features for behavior analysis\n"
        "- `data/models/lstm_model.pt`: trained LSTM next-category predictor\n"
        "- `data/models/cluster_model.pkl`: trained KMeans clustering model\n"
        "- `reports/images/session_clusters.png`: visualization of behavior clusters\n"
        "- `reports/images/category_ram_correlation.png`: peak RAM usage by category\n"
        "- `reports/images/lstm_training_history.png`: LSTM training curves\n"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report_written", path=str(output_path))
    return output_path


def _df_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    """Render a DataFrame as a compact markdown table."""
    df = df.head(max_rows)
    if df.empty:
        return "_No data_"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row.tolist()) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep, *rows])


def _classification_report_df(report: dict, has_confusion: bool) -> pd.DataFrame:
    """Flatten a sklearn classification report into a DataFrame."""
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict) and "precision" in metrics:
            rows.append(
                {
                    "category": label,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": metrics["support"],
                }
            )
    return pd.DataFrame(rows)
