"""Static report visualizations.

Renders the notebook's plots plus additional evaluation charts (top domains,
hourly/weekly activity, session duration, RAM boxplots, confusion matrix and
cluster profiles) into the configured images directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import Settings
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..pipeline import PipelineResult

logger = get_logger(__name__)

_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_PLOTS: dict[str, str] = {}


def generate_plots(result: PipelineResult, settings: Settings) -> list[Path]:
    """Render all report plots and return the written file paths."""
    images_dir = Path(settings.output.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3})

    calls: list[tuple[str, Callable[[PipelineResult, Path], None]]] = [
        ("category_distribution.png", _plot_category_distribution),
        ("category_ram_correlation.png", _plot_ram_by_category),
        ("elbow_silhouette.png", _plot_elbow_silhouette),
        ("session_clusters.png", _plot_clusters),
        ("lstm_training_history.png", _plot_training_history),
        ("top_domains.png", _plot_top_domains),
        ("hourly_activity_heatmap.png", _plot_hourly_heatmap),
        ("weekday_activity.png", _plot_weekday_activity),
        ("session_duration_histogram.png", _plot_session_duration),
        ("category_ram_boxplot.png", _plot_category_ram_boxplot),
        ("ram_usage_over_time.png", _plot_ram_over_time),
        ("lstm_confusion_matrix.png", _plot_confusion_matrix),
        ("cluster_profiles.png", _plot_cluster_profiles),
    ]

    written: list[Path] = []
    for filename, plotter in calls:
        try:
            plotter(result, images_dir / filename)
            written.append(images_dir / filename)
        except Exception as exc:  # pragma: no cover - best-effort rendering
            logger.warning("plot_failed", plot=filename, error=str(exc))

    logger.info("plots_generated", count=len(written), dir=str(images_dir))
    return written


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_category_distribution(result: PipelineResult, path: Path) -> None:
    counts = result.events["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x="count", y="category", data=counts, palette="viridis", hue="category", ax=ax)
    ax.set_title("Distribution of Browsing Categories")
    ax.set_xlabel("Number of Visits")
    ax.set_ylabel("Category")
    _save(fig, path)


def _plot_ram_by_category(result: PipelineResult, path: Path) -> None:
    stats = result.category_ram_stats.sort_values("peak_used_mb", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x="peak_used_mb", y="category", data=stats, palette="plasma", hue="category", ax=ax)
    ax.set_title("Peak RAM Usage per Browsing Category")
    ax.set_xlabel("Peak RAM Used (MB)")
    ax.set_ylabel("Category")
    _save(fig, path)


def _plot_elbow_silhouette(result: PipelineResult, path: Path) -> None:
    assert result.cluster is not None
    elbow = result.cluster.elbow
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS (Inertia)", color="tab:blue")
    ax1.plot(elbow["k"], elbow["wcss"], marker="o", linestyle="--", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Elbow Method vs Silhouette Score")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:red")
    valid = elbow.dropna(subset=["silhouette"])
    ax2.plot(
        valid["k"],
        valid["silhouette"],
        marker="s",
        linestyle="-",
        color="tab:red",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    _save(fig, path)


def _plot_clusters(result: PipelineResult, path: Path) -> None:
    assert result.cluster is not None
    cluster = result.cluster
    plot_df = pd.DataFrame(cluster.X_pca, columns=["PCA1", "PCA2"])
    plot_df["Cluster"] = cluster.labels
    fig, ax = plt.subplots(figsize=(10, 7))
    for label in sorted(plot_df["Cluster"].unique()):
        subset = plot_df[plot_df["Cluster"] == label]
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            label=f"Cluster {label}",
            alpha=0.6,
            edgecolors="w",
            s=50,
        )
    ax.set_title("Session Clusters Visualization (PCA Reduced 2D Space)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Clusters")
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, path)


def _plot_training_history(result: PipelineResult, path: Path) -> None:
    assert result.dl_result is not None
    history = result.dl_result.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["train_loss"], marker="o")
    ax1.set_title("Training Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(history["test_accuracy"], marker="o")
    ax2.set_title("Test Accuracy Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    _save(fig, path)


def _plot_top_domains(result: PipelineResult, path: Path) -> None:
    top = result.top_domains.head(15).sort_values("event_count")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x="event_count", y="domain", data=top, palette="mako", hue="domain", ax=ax)
    ax.set_title("Top 15 Domains by Visit Count")
    ax.set_xlabel("Number of Visits")
    ax.set_ylabel("Domain")
    _save(fig, path)


def _plot_hourly_heatmap(result: PipelineResult, path: Path) -> None:
    pivot = (
        result.events.groupby(["hour", "day_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=_DAY_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Visits"})
    ax.set_title("Hourly Activity by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    _save(fig, path)


def _plot_weekday_activity(result: PipelineResult, path: Path) -> None:
    counts = result.events.groupby("day_name").size().reindex(_DAY_ORDER)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette="crest", hue=counts.index, ax=ax)
    ax.set_title("Visits by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Visits")
    _save(fig, path)


def _plot_session_duration(result: PipelineResult, path: Path) -> None:
    durations = result.session_features["session_duration_minutes"].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(durations, bins=40, kde=True, color="steelblue", ax=ax)
    ax.set_title("Session Duration Distribution")
    ax.set_xlabel("Session Duration (minutes)")
    ax.set_ylabel("Number of Sessions")
    ax.set_xlim(0, durations.quantile(0.99))
    _save(fig, path)


def _plot_category_ram_boxplot(result: PipelineResult, path: Path) -> None:
    data = result.merged_data[["category", "used_mb"]].dropna()
    order = data.groupby("category")["used_mb"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x="category", y="used_mb", data=data, order=order, palette="viridis", hue="category", ax=ax
    )
    ax.set_title("System RAM Used by Browsing Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Used RAM (MB)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def _plot_ram_over_time(result: PipelineResult, path: Path) -> None:
    ram = result.ram_log
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ram["timestamp"], ram["used_mb"], color="crimson", linewidth=0.8)
    ax.set_title("System RAM Usage Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Used RAM (MB)")
    _save(fig, path)


def _plot_confusion_matrix(result: PipelineResult, path: Path) -> None:
    assert result.dl_result is not None
    assert result.predictor is not None
    confusion = result.dl_result.confusion
    categories = result.predictor.categories
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        ax=ax,
    )
    ax.set_title("LSTM Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("True Category")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def _plot_cluster_profiles(result: PipelineResult, path: Path) -> None:
    assert result.cluster is not None
    centers = result.cluster.cluster_centers
    features = [
        f
        for f in ["session_duration_minutes", "page_count", "peak_used_mb", "unique_categories"]
        if f in centers.columns
    ]
    normalized = (centers[features] - centers[features].min()) / (
        centers[features].max() - centers[features].min()
    ).replace(0, 1)
    normalized = normalized.reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = normalized.melt(
        id_vars=normalized.columns[0], var_name="feature", value_name="scaled"
    ).rename(columns={normalized.columns[0]: "cluster"})
    sns.barplot(
        x="feature",
        y="scaled",
        hue="cluster",
        data=plot_df,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Cluster Profiles (Min-Max Scaled Feature Means)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Scaled Mean Value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Cluster")
    _save(fig, path)
