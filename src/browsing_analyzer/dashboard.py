"""Interactive Streamlit dashboard for the browsing analyzer.

Run with:

.. code-block:: bash

    uv run streamlit run src/browsing_analyzer/dashboard.py

The dashboard loads persisted artifacts from ``data/processed`` and
``data/models`` by default; use the sidebar toggle to re-run the full
pipeline (including LSTM training) before rendering.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from browsing_analyzer.config import load_settings
from browsing_analyzer.pipeline import PipelineResult, load_pipeline_result, run_pipeline

st.set_page_config(page_title="Browsing Analyzer", layout="wide")


@st.cache_resource(show_spinner=False)
def _cached_load(source: str) -> PipelineResult:
    settings = load_settings()
    if source == "run":
        return run_pipeline(settings)
    return load_pipeline_result(settings)


def _kpi(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


def _render_kpis(result: PipelineResult) -> None:
    assert result.predictor is not None
    assert result.dl_result is not None
    assert result.cluster is not None
    cols = st.columns(5)
    cols[0].metric("Events", f"{len(result.events):,}")
    cols[1].metric("Sessions", f"{len(result.session_features):,}")
    cols[2].metric("Categories", len(result.predictor.categories))
    cols[3].metric("LSTM Accuracy", f"{result.dl_result.test_accuracy:.1%}")
    cols[4].metric("Clusters", result.cluster.model.n_clusters)


def _render_overview(result: PipelineResult) -> None:
    st.subheader("Category distribution")
    dist = result.events["category"].value_counts().reset_index()
    dist.columns = ["category", "events"]
    fig = px.bar(
        dist,
        x="category",
        y="events",
        color="category",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top domains")
    top = result.top_domains.head(15)
    fig = px.bar(top, x="event_count", y="domain", orientation="h", color="category")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Hourly activity by weekday")
    pivot = result.events.groupby(["hour", "day_name"]).size().unstack(fill_value=0)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(columns=[d for d in days if d in pivot.columns], fill_value=0)
    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale="YlOrRd", labels={"x": "Day", "y": "Hour"}
    )
    st.plotly_chart(fig, width="stretch")


def _render_ram(result: PipelineResult) -> None:
    st.subheader("Peak RAM by category")
    df = result.category_ram_stats.sort_values("peak_used_mb", ascending=False)
    fig = px.bar(df, x="category", y="peak_used_mb", color="category")
    st.plotly_chart(fig, width="stretch")

    st.subheader("RAM usage over time")
    ram = result.ram_log
    fig = go.Figure(go.Scatter(x=ram["timestamp"], y=ram["used_mb"], mode="lines", name="used_mb"))
    fig.update_layout(xaxis_title="Time", yaxis_title="Used RAM (MB)")
    st.plotly_chart(fig, width="stretch")


def _render_clusters(result: PipelineResult) -> None:
    assert result.cluster is not None
    st.subheader("Session clusters (PCA-reduced features)")
    df = result.session_features.copy()
    pca = result.cluster.X_pca
    df["PCA1"] = pca[:, 0]
    df["PCA2"] = pca[:, 1]
    fig = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color=df["cluster"].astype(str),
        hover_data=["session_duration_minutes", "page_count", "peak_used_mb"],
    )
    st.plotly_chart(fig, width="stretch")


def _render_lstm(result: PipelineResult) -> None:
    assert result.predictor is not None
    st.subheader("Next-category prediction")
    recent = result.events.groupby("session_id")["category"].apply(list).tolist()
    last_seq = recent[-1] if recent else []
    history = last_seq[-result.predictor.sequence_length :]

    st.caption("Adjust the recent category sequence and see the predicted next category.")
    seq = []
    cols = st.columns(len(history))
    for i, (col, value) in enumerate(zip(cols, history, strict=True)):
        seq.append(
            col.selectbox(
                f"t-{len(history) - i}",
                result.predictor.categories,
                index=result.predictor.categories.index(value),
            )
        )

    probs = result.predictor.predict_proba(seq)
    prob_df = pd.DataFrame(
        {"category": list(probs), "probability": list(probs.values())}
    ).sort_values("probability", ascending=False)
    fig = px.bar(prob_df, x="category", y="probability", color="category")
    st.plotly_chart(fig, width="stretch")
    st.info(
        f"Most likely next category: **{prob_df.iloc[0]['category']}** ({prob_df.iloc[0]['probability']:.0%})"
    )


def _render_recommendations(result: PipelineResult) -> None:
    st.subheader("Recommendations")
    if not result.recommendations:
        st.write("No recommendations.")
        return
    for rec in result.recommendations:
        color = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(rec.severity, "⚪")
        with st.container(border=True):
            st.markdown(f"### {color} {rec.title}")
            st.write(rec.rationale)
            st.caption(f"Evidence: {rec.evidence} | Metric: `{rec.metric}`")


def main() -> None:
    st.title("Browsing & RAM Analyzer")
    st.sidebar.header("Controls")
    source = (
        "run"
        if st.sidebar.radio("Data source", ["Load saved artifacts", "Re-run pipeline"])
        == "Re-run pipeline"
        else "load"
    )
    result = _cached_load(source)

    tab_overview, tab_ram, tab_cluster, tab_lstm, tab_recs = st.tabs(
        ["Overview", "RAM correlation", "Clustering", "LSTM prediction", "Recommendations"]
    )
    with tab_overview:
        _render_kpis(result)
        _render_overview(result)
    with tab_ram:
        _render_ram(result)
    with tab_cluster:
        _render_clusters(result)
    with tab_lstm:
        _render_lstm(result)
    with tab_recs:
        _render_recommendations(result)


if __name__ == "__main__":
    main()
