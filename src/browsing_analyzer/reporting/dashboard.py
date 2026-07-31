"""Streamlit dashboard.

Run with: ``streamlit run -m browsing_analyzer.reporting.dashboard`` or via the
CLI ``browsing-analyzer dashboard``. The dashboard re-runs the pipeline for a
selectable window (3/4/5 days) and renders interactive charts for patterns,
clusters, RAM correlation, LSTM output, and recommendations.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..config import load_settings
from ..pipeline import run_pipeline
from ..utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Browsing Analyzer", layout="wide")


def _section_header(title: str) -> None:
    st.subheader(title)


def render_time_patterns(result) -> None:
    """Render hourly/daily usage heatmaps."""
    _section_header("Time-Based Patterns")

    hourly = result.patterns.get("hourly")
    if hourly is None or hourly.empty:
        st.info("No temporal data in the selected window.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            hourly.reset_index().melt(id_vars="hour", value_vars="mean_visits"),
            x="hour",
            y="value",
            title="Mean visits by hour of day",
        )
        st.plotly_chart(fig, width="stretch")
    with col2:
        daily = result.patterns.get("daily")
        if daily is not None and not daily.empty:
            daily_view = daily.sum().to_frame("visits").reset_index()
            daily_view = daily_view.rename(columns={"index": "day_name"})
            fig = px.bar(
                daily_view,
                x="day_name",
                y="visits",
                title="Visits by day of week",
            )
            st.plotly_chart(fig, width="stretch")


def render_clusters(result) -> None:
    """Render cluster summaries."""
    _section_header("Behavior Clusters")
    if result.cluster is None:
        st.info("Insufficient sessions to cluster.")
        return

    st.write(f"**Silhouette score:** `{result.cluster.silhouette:.3f}`")
    profiles = pd.DataFrame(
        [{"cluster": cid, "label": label} for cid, label in result.cluster.profiles.items()]
    )
    st.dataframe(profiles, width="stretch")
    centers = result.cluster.cluster_centers
    if "duration_minutes" in centers.columns and "event_count" in centers.columns:
        fig = px.scatter(
            centers.reset_index(),
            x="duration_minutes",
            y="event_count",
            color="index",
            size="avg_ram_mb" if "avg_ram_mb" in centers.columns else None,
            title="Cluster centers (duration vs events, size = avg RAM)",
        )
        st.plotly_chart(fig, width="stretch")


def render_ram_correlation(result) -> None:
    """Render category-wise RAM stats."""
    _section_header("RAM Correlation")
    stats = result.category_stats
    if stats.empty or stats["avg_ram_mb"].isna().all():
        st.info("No RAM-aligned data in the window.")
        return

    if "avg_browser_ram_mb" in stats.columns:
        fig = px.bar(
            stats.sort_values("avg_browser_ram_mb"),
            x="category",
            y="avg_browser_ram_mb",
            title="Average browser RAM by category",
            color="category",
        )
        st.plotly_chart(fig, width="stretch")
        heavy = stats.sort_values("avg_browser_ram_mb", ascending=False).head(3)
        st.markdown(
            "**Top 3 memory-heavy categories (browser RAM):** "
            + ", ".join(
                f"{r['category']} ({r['avg_browser_ram_mb']:.0f} MB)" for _, r in heavy.iterrows()
            )
        )
    else:
        fig = px.bar(
            stats.sort_values("avg_ram_mb"),
            x="category",
            y="avg_ram_mb",
            title="Average system RAM by category",
            color="category",
        )
        st.plotly_chart(fig, width="stretch")
        heavy = stats.sort_values("avg_ram_mb", ascending=False).head(3)
        st.markdown(
            "**Top 3 memory-heavy categories (system RAM):** "
            + ", ".join(f"{r['category']} ({r['avg_ram_mb']:.0f} MB)" for _, r in heavy.iterrows())
        )


def render_lstm(result) -> None:
    """Render LSTM metrics and next-category prediction."""
    _section_header("Deep Learning - LSTM Next-Category Prediction")
    if result.dl_result is None:
        st.info("LSTM not trained (no data or disabled).")
        return

    dl = result.dl_result
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{dl.test_accuracy:.3f}")
    c2.metric("Macro F1", f"{dl.macro_f1:.3f}")
    c3.metric("Baseline Accuracy", f"{dl.baseline_accuracy:.3f}")
    c4.metric("Baseline F1", f"{dl.baseline_f1:.3f}")

    labels = sorted(set(result.events["category"]))
    conf_df = pd.DataFrame(dl.confusion, index=labels, columns=labels)
    fig = px.imshow(
        conf_df, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix"
    )
    st.plotly_chart(fig, width="stretch")

    # Next-category prediction
    if result.predictor is not None and not result.events.empty:
        last_seq = result.events.groupby("session_id")["category"].apply(list).iloc[-1]
        probs = result.predictor.predict_proba(list(last_seq))
        prob_df = pd.DataFrame(
            {"category": list(probs.keys()), "probability": list(probs.values())}
        )
        prob_df = prob_df.sort_values("probability", ascending=False)
        st.write("**Next-category probability distribution** (based on the most recent session)")
        st.bar_chart(prob_df.set_index("category"))


def render_recommendations(result) -> None:
    """Render the recommendation summary."""
    _section_header("Recommendations")
    for i, rec in enumerate(result.recommendations, start=1):
        severity_color = {"high": ":red[**]", "medium": ":orange[**]", "low": ":green[**]"}.get(
            rec.severity, "**"
        )
        st.markdown(f"{severity_color}{i}. {rec.title}{severity_color}")
        st.markdown(f"- *Rationale:* {rec.rationale}")
        st.markdown(f"- *Evidence:* {rec.evidence}")
        st.markdown(f"- *Metric:* `{rec.metric}`")


def run_dashboard() -> None:
    """Entry point launched by ``streamlit run``."""
    configure_logging()
    settings = load_settings()

    st.title("Browsing History Analyzer")
    st.caption("Analyze your Chrome browsing patterns, RAM usage, and next-category predictions.")

    window = st.sidebar.selectbox(
        "Analysis window (days)",
        options=settings.time_windows,
        index=settings.time_windows.index(settings.default_window),
    )

    if st.sidebar.button("Run analysis", type="primary"):
        with st.spinner("Running the full pipeline..."):
            result = run_pipeline(settings=settings, window_days=window)
        st.session_state["result"] = result
        st.session_state["window"] = window

    if "result" not in st.session_state:
        st.info("Select a time window and press **Run analysis** to begin.")
        return

    result = st.session_state["result"]
    st.success(
        f"Analyzed **{len(result.events)}** events across **{len(result.sessions)}** sessions "
        f"in the last {st.session_state['window']} days."
    )

    render_time_patterns(result)
    render_clusters(result)
    render_ram_correlation(result)
    render_lstm(result)
    render_recommendations(result)


if __name__ == "__main__":
    run_dashboard()
