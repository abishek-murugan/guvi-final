"""End-to-end pipeline orchestrator.

Glues together: collection -> cleaning -> categorization -> sessionization ->
RAM correlation -> clustering -> patterns -> LSTM training -> recommendations.
The pipeline is exposed as a single ``PipelineResult`` for the CLI, report
generator, and dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .analytics import BehaviorClusterer, discover_time_patterns, session_ram_stats
from .analytics.clustering import ClusterResult
from .collect import load_browsing_history, load_ram_log
from .config import Settings, load_settings
from .models import NextCategoryPredictor, train_model
from .models.trainer import TrainerResult
from .prep import Categorizer, Sessionizer, clean_history
from .recommendations import generate_recommendations
from .recommendations.engine import Recommendation
from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Container holding every artifact produced by the pipeline."""

    events: pd.DataFrame
    sessions: pd.DataFrame
    raw_history: pd.DataFrame
    ram_log: pd.DataFrame
    patterns: dict[str, pd.DataFrame]
    cluster: ClusterResult | None
    category_stats: pd.DataFrame
    top_domains: pd.DataFrame
    dl_result: TrainerResult | None
    recommendations: list[Recommendation]
    settings: Settings = field(default_factory=load_settings)
    predictor: NextCategoryPredictor | None = None
    window_days: int = 4


def run_pipeline(
    settings: Settings | None = None,
    window_days: int | None = None,
    history_path: Path | None = None,
    ram_path: Path | None = None,
    train_model_flag: bool = True,
    output_dir: Path | None = None,
) -> PipelineResult:
    """Execute the full analysis pipeline.

    Args:
        settings: Optional settings override (defaults to repo config).
        window_days: Time window in days (3, 4 or 5).
        history_path: Optional path to the browsing history CSV.
        ram_path: Optional path to the RAM log CSV.
        train_model_flag: Whether to train the LSTM model.
        output_dir: Where processed data / models are written.

    Returns:
        A :class:`PipelineResult` with all downstream artifacts.
    """
    settings = settings or load_settings()
    window = window_days or settings.default_window
    if window not in settings.time_windows:
        raise ValueError(f"window_days must be one of {settings.time_windows}")

    logger.info("pipeline_started", window_days=window)

    # 1. Collection
    raw_history = load_browsing_history(settings, history_path)
    ram_log = load_ram_log(settings, ram_path)

    # 2. Window filter
    cutoff = raw_history["timestamp"].max() - pd.Timedelta(days=window)
    history = raw_history[raw_history["timestamp"] >= cutoff]

    # 3. Preprocessing
    cleaned = clean_history(history)
    categorizer = Categorizer()
    cleaned["category"] = categorizer.categorize_series(cleaned["domain"])

    # Privacy: drop raw URLs after domain extraction — downstream artifacts
    # contain only domain/category-level data.
    cleaned = cleaned.drop(columns=["url"])

    # 4. Sessionization
    sessionizer = Sessionizer(settings)
    events, sessions = sessionizer.sessionize(cleaned)

    # 5. RAM correlation
    events = align_ram_events(events, ram_log)
    ram_stats = session_ram_stats(events)
    if not ram_stats.empty:
        sessions = sessions.merge(ram_stats, on="session_id", how="left")
    events = events.merge(ram_stats, on="session_id", how="left", suffixes=("", "_s"))

    # 6. Clustering + patterns
    clusterer = BehaviorClusterer(settings)
    cluster = clusterer.fit(sessions) if not sessions.empty else None
    if cluster is not None:
        sessions["cluster_label"] = [
            cluster.profiles.get(int(label), f"cluster {label}") for label in cluster.labels
        ]
        sessions["cluster_id"] = cluster.labels

    patterns = discover_time_patterns(events) if not events.empty else {}

    # 7. Category / domain stats
    category_stats = build_category_stats(events)
    top_domains = build_top_domains(events)

    # 8. Deep learning (LSTM next-category)
    dl_result = None
    predictor = None
    if train_model_flag and not events.empty:
        predictor, dl_result = run_lstm(settings, events)

    # 9. Recommendations
    dl_signals = {}
    if predictor is not None and not sessions.empty:
        last_seq = events.groupby("session_id")["category"].apply(list).iloc[-1]
        probs = predictor.predict_proba(list(last_seq))
        if probs:
            best = max(probs, key=probs.get)
            dl_signals = {"next_category": best, "next_prob": probs[best]}

    recommendations = generate_recommendations(
        settings=settings,
        sessions=sessions,
        category_stats=category_stats,
        top_domains=top_domains,
        hour_stats=patterns.get("hourly", pd.DataFrame()),
        dl_signals=dl_signals,
    )

    # Persist processed artifacts
    out = output_dir or Path(settings.data.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "browsing_history_processed.csv", index=False)
    sessions.to_csv(out / "sessions.csv", index=False)
    if cluster is not None:
        pd.DataFrame({"session_id": sessions["session_id"], "cluster_id": cluster.labels}).to_csv(
            out / "cluster_assignments.csv", index=False
        )

    logger.info("pipeline_finished", sessions=len(sessions), events=len(events))
    return PipelineResult(
        events=events,
        sessions=sessions,
        raw_history=raw_history,
        ram_log=ram_log,
        patterns=patterns,
        cluster=cluster,
        category_stats=category_stats,
        top_domains=top_domains,
        dl_result=dl_result,
        recommendations=recommendations,
        settings=settings,
        predictor=predictor,
        window_days=window,
    )


def align_ram_events(events: pd.DataFrame, ram_log: pd.DataFrame) -> pd.DataFrame:
    """Nearest-timestamp (backward) merge of RAM samples onto events."""
    from .analytics import align_ram_with_events

    return align_ram_with_events(events, ram_log)


def build_category_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-category event counts and RAM figures."""
    if events.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "event_count",
                "avg_ram_mb",
                "peak_ram_mb",
                "avg_browser_ram_mb",
                "peak_browser_ram_mb",
            ]
        )
    g = events.groupby("category")
    stats = pd.DataFrame(
        {
            "category": g.size().index,
            "event_count": g.size().values,
            "avg_ram_mb": g["ram_used_mb"].mean().values,
            "peak_ram_mb": g["ram_used_mb"].max().values,
            "avg_browser_ram_mb": g["browser_ram_mb"].mean().values,
            "peak_browser_ram_mb": g["browser_ram_mb"].max().values,
        }
    )
    stats = stats.sort_values("event_count", ascending=False).reset_index(drop=True)
    return stats


def build_top_domains(events: pd.DataFrame) -> pd.DataFrame:
    """Top domains by visit count."""
    if events.empty:
        return pd.DataFrame(columns=["domain", "category", "event_count"])
    top = (
        events.groupby(["domain", "category"])
        .size()
        .reset_index(name="event_count")
        .sort_values("event_count", ascending=False)
        .reset_index(drop=True)
    )
    return top


def run_lstm(settings: Settings, events: pd.DataFrame):
    """Train the LSTM next-category model on per-session sequences."""

    categories = sorted(set(events["category"]))
    predictor = NextCategoryPredictor(settings, categories)
    sessions_sequences = events.groupby("session_id")["category"].apply(list).tolist()
    sequences = predictor.to_sequences(sessions_sequences)
    result = train_model(predictor, sequences, settings)
    return predictor, result
