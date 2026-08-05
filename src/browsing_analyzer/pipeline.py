"""End-to-end pipeline orchestrator.

Reproduces the notebook flow (preprocessing -> sessionization -> RAM
correlation -> clustering -> LSTM training) and persists every artifact to
``data/processed``, ``data/models`` and ``reports/images``. ``run_pipeline``
executes the full flow; ``load_pipeline_result`` reconstructs the same result
from persisted artifacts without retraining.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import BehaviorClusterer, align_ram_with_events, category_ram_stats
from .analytics.clustering import ClusterResult
from .analytics.ram_correlation import session_ram_stats
from .collect import load_browsing_history, load_ram_log
from .config import Settings, load_settings
from .models import NextCategoryPredictor, train_model
from .models.trainer import TrainerResult
from .prep import Categorizer, Sessionizer, clean_history
from .recommendations import generate_recommendations
from .recommendations.engine import Recommendation
from .reporting.visualize import generate_plots
from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Container holding every artifact produced by the pipeline."""

    settings: Settings
    events: pd.DataFrame
    ram_log: pd.DataFrame
    merged_data: pd.DataFrame
    session_summary: pd.DataFrame
    session_features: pd.DataFrame
    category_ram_stats: pd.DataFrame
    top_domains: pd.DataFrame
    cluster: ClusterResult | None = None
    predictor: NextCategoryPredictor | None = None
    dl_result: TrainerResult | None = None
    recommendations: list[Recommendation] = field(default_factory=list)
    plots: list[Path] = field(default_factory=list)


def run_pipeline(
    settings: Settings | None = None,
    train_model_flag: bool = True,
) -> PipelineResult:
    """Execute the full analysis pipeline and persist all artifacts.

    Args:
        settings: Optional settings override (defaults to repo config).
        train_model_flag: Whether to train the LSTM next-category model.

    Returns:
        A :class:`PipelineResult` with all downstream artifacts.
    """
    settings = settings or load_settings()

    raw_history = load_browsing_history(settings)
    ram_log = load_ram_log(settings)

    cleaned = clean_history(raw_history)
    categorizer = Categorizer(
        notebook_path=Path(settings.notebook_path),
        cache_path=Path(settings.domain_categories_yaml),
    )
    cleaned["category"] = categorizer.categorize_series(cleaned["domain"])

    sessionizer = Sessionizer(settings.sessionization.inactivity_threshold_minutes)
    events, session_summary = sessionizer.sessionize(cleaned)

    merged_data = align_ram_with_events(events, ram_log)
    session_ram = session_ram_stats(merged_data)
    category_ram = category_ram_stats(merged_data)

    session_features = pd.merge(session_summary, session_ram, on="session_id", how="left")
    session_features["session_start_hour"] = session_features["session_start"].dt.hour
    session_features["session_start_dayofweek"] = session_features["session_start"].dt.dayofweek

    clusterer = BehaviorClusterer(settings)
    cluster = clusterer.fit(session_features)
    session_features["cluster"] = cluster.labels

    top_domains = _build_top_domains(events)

    predictor = None
    dl_result = None
    if train_model_flag:
        categories = list(events["category"].unique())
        predictor = NextCategoryPredictor(settings, categories)
        session_sequences = events.groupby("session_id")["category"].apply(list).tolist()
        dl_result = train_model(predictor, session_sequences, settings)
        events["category_id"] = events["category"].map(predictor.category_to_id)
        dl_signals = _build_dl_signals(predictor, session_sequences)
    else:
        dl_signals = {}

    recommendations = generate_recommendations(
        settings=settings,
        sessions=session_features,
        category_stats=category_ram,
        top_domains=top_domains,
        hour_stats=pd.DataFrame(),
        dl_signals=dl_signals,
    )

    result = PipelineResult(
        settings=settings,
        events=events,
        ram_log=ram_log,
        merged_data=merged_data,
        session_summary=session_summary,
        session_features=session_features,
        category_ram_stats=category_ram,
        top_domains=top_domains,
        cluster=cluster,
        predictor=predictor,
        dl_result=dl_result,
        recommendations=recommendations,
    )

    if train_model_flag:
        result.plots = generate_plots(result, settings)
    _persist_artifacts(result, settings)
    logger.info("pipeline_finished", sessions=len(session_features), events=len(events))
    return result


def load_pipeline_result(settings: Settings | None = None) -> PipelineResult:
    """Reconstruct a :class:`PipelineResult` from persisted artifacts."""
    settings = settings or load_settings()
    data = Path(settings.data.processed_dir)
    models = Path(settings.data.models_dir)

    events = pd.read_csv(data / settings.data.processed_history_file)
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    ram_log = pd.read_csv(data / settings.data.processed_ram_log_file)
    ram_log["timestamp"] = pd.to_datetime(ram_log["timestamp"])
    merged_data = pd.read_csv(data / settings.data.merged_data_file)
    merged_data["timestamp"] = pd.to_datetime(merged_data["timestamp"])
    session_summary = pd.read_csv(data / settings.data.session_summary_file)
    session_summary["session_start"] = pd.to_datetime(session_summary["session_start"])
    session_summary["session_end"] = pd.to_datetime(session_summary["session_end"])
    session_features = pd.read_csv(data / settings.data.session_features_file)
    category_ram = pd.read_csv(data / settings.data.category_ram_stats_file)

    with (models / settings.data.cluster_model_file).open("rb") as fh:
        cluster = pickle.load(fh)
    with (models / settings.data.lstm_result_file).open("rb") as fh:
        dl_result = pickle.load(fh)

    predictor = NextCategoryPredictor.load(models / settings.data.model_output_file, settings)
    recommendations = _load_recommendations(data / settings.data.recommendations_file)
    top_domains = _build_top_domains(events)

    return PipelineResult(
        settings=settings,
        events=events,
        ram_log=ram_log,
        merged_data=merged_data,
        session_summary=session_summary,
        session_features=session_features,
        category_ram_stats=category_ram,
        top_domains=top_domains,
        cluster=cluster,
        predictor=predictor,
        dl_result=dl_result,
        recommendations=recommendations,
    )


def _build_top_domains(events: pd.DataFrame) -> pd.DataFrame:
    """Top domains by visit count."""
    return (
        events.groupby(["domain", "category"])
        .size()
        .reset_index(name="event_count")
        .sort_values("event_count", ascending=False)
        .reset_index(drop=True)
    )


def _build_dl_signals(predictor: NextCategoryPredictor, session_sequences: list[list[str]]) -> dict:
    """Predicted next category for the most recent session."""
    if not session_sequences:
        return {}
    probs = predictor.predict_proba(session_sequences[-1])
    best = max(probs, key=lambda category: probs[category])
    return {"next_category": best, "next_prob": probs[best]}


def _load_recommendations(path: Path) -> list[Recommendation]:
    """Load persisted recommendations from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Recommendation(**item) for item in data]


def _persist_artifacts(result: PipelineResult, settings: Settings) -> None:
    """Write processed data, models and metrics to disk."""
    data = Path(settings.data.processed_dir)
    models = Path(settings.data.models_dir)
    data.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    result.events.to_csv(data / settings.data.processed_history_file, index=False)
    result.ram_log.to_csv(data / settings.data.processed_ram_log_file, index=False)
    result.merged_data.to_csv(data / settings.data.merged_data_file, index=False)
    result.session_summary.to_csv(data / settings.data.session_summary_file, index=False)
    result.session_features.to_csv(data / settings.data.session_features_file, index=False)
    result.category_ram_stats.to_csv(data / settings.data.category_ram_stats_file, index=False)

    Categorizer(
        notebook_path=Path(settings.notebook_path), cache_path=Path(settings.domain_categories_yaml)
    ).to_dataframe().to_csv(data / settings.data.domain_category_map_file, index=False)

    if result.cluster is not None and result.dl_result is not None and result.predictor is not None:
        _persist_cluster_viz(result, data, settings)
        _persist_sequences(result, data, settings)
        _persist_confusion(result, data, settings)
        _persist_metrics(result, data, settings)
        _persist_models(result, models, settings)
        _persist_recommendations(result, data, settings)

    logger.info("artifacts_persisted", processed_dir=str(data), models_dir=str(models))


def _persist_cluster_viz(result: PipelineResult, data: Path, settings: Settings) -> None:
    assert result.cluster is not None
    viz = pd.DataFrame(
        {
            "session_id": result.session_features["session_id"],
            "PCA1": result.cluster.X_pca[:, 0],
            "PCA2": result.cluster.X_pca[:, 1],
            "cluster": result.cluster.labels,
        }
    )
    viz.to_csv(data / settings.data.cluster_viz_file, index=False)


def _persist_sequences(result: PipelineResult, data: Path, settings: Settings) -> None:
    assert result.predictor is not None
    predictor = result.predictor
    session_sequences = result.events.groupby("session_id")["category"].apply(list).tolist()
    X, y = predictor.build_samples(session_sequences)
    np.savez(
        data / settings.data.sequences_file,
        X=X,
        y=y,
        categories=np.array(predictor.categories),
    )


def _persist_confusion(result: PipelineResult, data: Path, settings: Settings) -> None:
    assert result.dl_result is not None
    assert result.predictor is not None
    if result.dl_result.confusion is None:
        return
    confusion = pd.DataFrame(
        result.dl_result.confusion,
        index=result.predictor.categories,
        columns=result.predictor.categories,
    )
    confusion.to_csv(data / settings.data.lstm_confusion_file)


def _persist_metrics(result: PipelineResult, data: Path, settings: Settings) -> None:
    assert result.predictor is not None
    assert result.dl_result is not None
    assert result.cluster is not None
    metrics = {
        "events": int(len(result.events)),
        "sessions": int(len(result.session_features)),
        "categories": len(result.predictor.categories),
        "window_start": str(result.events["timestamp"].min()),
        "window_end": str(result.events["timestamp"].max()),
        "test_accuracy": float(result.dl_result.test_accuracy),
        "silhouette": float(result.cluster.silhouette),
        "clusters": int(result.cluster.model.n_clusters),
        "peak_ram_mb": float(result.category_ram_stats["peak_used_mb"].max()),
    }
    (data / settings.data.metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _persist_models(result: PipelineResult, models: Path, settings: Settings) -> None:
    assert result.predictor is not None
    assert result.dl_result is not None
    assert result.cluster is not None
    result.predictor.save(models / settings.data.model_output_file)
    with (models / settings.data.cluster_model_file).open("wb") as fh:
        pickle.dump(result.cluster, fh)
    with (models / settings.data.lstm_result_file).open("wb") as fh:
        pickle.dump(result.dl_result, fh)

    metadata = {
        "config": result.predictor.config,
        "history": result.dl_result.history,
        "test_accuracy": result.dl_result.test_accuracy,
        "classification_report": result.dl_result.classification_report,
    }
    (models / settings.data.model_metadata_file).write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )


def _persist_recommendations(result: PipelineResult, data: Path, settings: Settings) -> None:
    (data / settings.data.recommendations_file).write_text(
        json.dumps([rec.to_dict() for rec in result.recommendations], indent=2),
        encoding="utf-8",
    )
