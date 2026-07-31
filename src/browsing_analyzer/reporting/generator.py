"""Markdown report generation.

Produces a self-contained report covering: top domains/categories, time-based
insights, cluster summaries, RAM correlation results, LSTM metrics, and the
recommendation summary. Only domain/category level data is included (privacy).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..pipeline import PipelineResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


def generate_markdown_report(result: PipelineResult, output_path: Path) -> Path:
    """Write a Markdown report for a finished pipeline run.

    Args:
        result: The pipeline result to summarize.
        output_path: Destination file path.

    Returns:
        The path the report was written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("# Browsing History Analysis Report")
    lines.append("")
    lines.append("> Generated locally. All data is domain/category level; no raw URLs.")
    lines.append("")
    window = result.window_days
    lines.append(f"**Analysis window:** last {window} days")
    lines.append(f"**Events analyzed:** {len(result.events)}")
    lines.append(f"**Sessions identified:** {len(result.sessions)}")
    lines.append("")

    # 1. Top domains/categories
    lines.append("## 1. Top Domains & Categories")
    lines.append("")
    if not result.top_domains.empty:
        lines.append("### Top domains")
        lines.append(_df_table(result.top_domains.head(10)))
    if not result.category_stats.empty:
        lines.append("### Category distribution")
        cat_table = result.category_stats.copy()
        cat_table["share"] = cat_table["event_count"] / cat_table["event_count"].sum()
        cat_table["share"] = cat_table["share"].map(lambda x: f"{x:.1%}")
        for col in cat_table.columns:
            if pd.api.types.is_float_dtype(cat_table[col]):
                cat_table[col] = cat_table[col].round(0)
        lines.append(_df_table(cat_table))
    lines.append("")

    # 2. Time patterns
    lines.append("## 2. Time-Based Usage Patterns")
    lines.append("")
    hourly = result.patterns.get("hourly")
    if hourly is not None and not hourly.empty:
        top_hours = hourly["mean_visits"].sort_values(ascending=False).head(5)
        lines.append(
            "**Peak hours:** "
            + ", ".join(f"{int(h)}:00 ({v:.1f} visits)" for h, v in top_hours.items())
        )
        lines.append("")
        lines.append("### Hourly activity (mean visits)")
        hourly_view = pd.DataFrame(
            {"hour": hourly.index, "mean_visits": hourly["mean_visits"].round(2)}
        )
        lines.append(_df_table(hourly_view))
    daily = result.patterns.get("daily")
    if daily is not None and not daily.empty:
        lines.append("### Daily activity")
        daily_view = daily.sum().to_frame("visits").reset_index()
        daily_view = daily_view.rename(columns={"index": "day_name"})
        lines.append(_df_table(daily_view))
    lines.append("")

    # 3. Clusters
    lines.append("## 3. Session Clusters")
    lines.append("")
    if result.cluster is not None:
        lines.append(f"- **Algorithm:** {result.settings.clustering.algorithm}")
        lines.append(f"- **Silhouette score:** {result.cluster.silhouette:.3f}")
        lines.append("")
        lines.append("### Cluster profiles")
        rows = []
        for cid, label in result.cluster.profiles.items():
            rows.append({"cluster": cid, "label": label})
        lines.append(_df_table(pd.DataFrame(rows)))
        lines.append("")
        lines.append("### Cluster centers (feature means)")
        lines.append(_df_table(result.cluster.cluster_centers.round(2)))
    else:
        lines.append("_No clusters computed (insufficient sessions)._")
    lines.append("")

    # 4. RAM correlation
    lines.append("## 4. RAM Correlation")
    lines.append("")
    if not result.category_stats.empty and not result.category_stats["avg_ram_mb"].isna().all():
        if "avg_browser_ram_mb" in result.category_stats.columns:
            ram_cols = [
                "category",
                "event_count",
                "avg_ram_mb",
                "avg_browser_ram_mb",
                "peak_browser_ram_mb",
            ]
        else:
            ram_cols = ["category", "event_count", "avg_ram_mb", "peak_ram_mb"]
        ram_view = result.category_stats[ram_cols].copy().round(0)
        lines.append("### Category-wise RAM")
        lines.append(_df_table(ram_view))
        if "avg_browser_ram_mb" in ram_view.columns:
            heavy = ram_view.sort_values("avg_browser_ram_mb", ascending=False).head(3)
            label = "Top 3 memory-heavy categories (browser RAM)"
        else:
            heavy = ram_view.sort_values("avg_ram_mb", ascending=False).head(3)
            label = "Top 3 memory-heavy categories (system RAM)"
        lines.append("")
        lines.append(f"**{label}:** " + ", ".join(heavy["category"]))
    else:
        lines.append("_RAM alignment produced no values for the current window._")
    lines.append("")

    # 5. Deep learning
    lines.append("## 5. Deep Learning (LSTM Next-Category Prediction)")
    lines.append("")
    if result.dl_result is not None:
        dl = result.dl_result
        lines.append(f"- **Test accuracy:** {dl.test_accuracy:.3f}")
        lines.append(f"- **Macro F1:** {dl.macro_f1:.3f}")
        lines.append(f"- **Baseline accuracy (most-common):** {dl.baseline_accuracy:.3f}")
        lines.append(f"- **Baseline macro F1:** {dl.baseline_f1:.3f}")
        lines.append("")
        lines.append("### Confusion matrix")
        labels = sorted(set(result.events["category"]))
        lines.append(_df_table(pd.DataFrame(dl.confusion, index=labels, columns=labels)))
    else:
        lines.append("_Model not trained (no data or flag disabled)._")
    lines.append("")

    # 6. Recommendations
    lines.append("## 6. Recommendations")
    lines.append("")
    for i, rec in enumerate(result.recommendations, start=1):
        lines.append(f"### {i}. {rec.title}")
        lines.append("")
        lines.append(f"- **Rationale:** {rec.rationale}")
        lines.append(f"- **Evidence:** {rec.evidence}")
        lines.append(f"- **Severity:** {rec.severity}")
        lines.append(f"- **Metric:** `{rec.metric}`")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report_written", path=str(output_path))
    return output_path


def _df_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame as a compact markdown table (no external deps)."""
    df = df.head(max_rows)
    if df.empty:
        return "_No data_"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join([header, sep, *rows])
