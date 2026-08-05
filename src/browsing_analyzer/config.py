"""Typed configuration loaded from ``config/config.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class DataConfig(BaseModel):
    """Paths for raw, processed and model artifacts."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "data/models"
    browser_history_file: str = "browsing_history_last5days.csv"
    ram_log_file: str = "ram_log_last5days.csv"
    browser_history_output: str = "browsing_history_last5days.csv"
    ram_log_output: str = "ram_log_last5days.csv"
    processed_history_file: str = "final_browsing_history.csv"
    processed_ram_log_file: str = "final_ram_log.csv"
    domain_category_map_file: str = "domain_category_map.csv"
    merged_data_file: str = "merged_data.csv"
    session_summary_file: str = "session_summary.csv"
    session_features_file: str = "session_features.csv"
    category_ram_stats_file: str = "category_ram_stats.csv"
    cluster_viz_file: str = "cluster_viz.csv"
    lstm_confusion_file: str = "lstm_confusion.csv"
    sequences_file: str = "lstm_sequences.npz"
    metrics_file: str = "pipeline_metrics.json"
    recommendations_file: str = "recommendations.json"
    model_output_file: str = "lstm_model.pt"
    model_metadata_file: str = "lstm_metadata.json"
    lstm_result_file: str = "lstm_result.pkl"
    cluster_model_file: str = "cluster_model.pkl"


class SessionizationConfig(BaseModel):
    """Session boundary settings."""

    inactivity_threshold_minutes: int = 15


class ClusteringConfig(BaseModel):
    """KMeans clustering settings."""

    n_clusters: int = 2
    random_state: int = 42
    features: list[str] = Field(
        default_factory=lambda: [
            "session_duration_minutes",
            "page_count",
            "unique_domains",
            "unique_categories",
            "domain_switches",
            "category_switches",
            "avg_time_per_page_seconds",
            "mean_used_mb",
            "peak_used_mb",
            "mean_usage_percent",
            "peak_usage_percent",
            "session_start_hour",
            "session_start_dayofweek",
        ]
    )


class ModelConfig(BaseModel):
    """PyTorch LSTM hyper-parameters (from the notebook)."""

    type: str = "lstm"
    sequence_length: int = 5
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.5
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    test_size: float = 0.2
    random_state: int = 42


class RecommendationsConfig(BaseModel):
    """Thresholds driving the recommendation rules."""

    late_night_hour: int = 22
    social_media_threshold: float = 0.45
    ram_spike_threshold_mb: float = 500.0
    category_ram_threshold_mb: float = 300.0
    high_switch_threshold: int = 6


class OutputConfig(BaseModel):
    """Report and visualization output settings."""

    images_dir: str = "reports/images"
    report_path: str = "reports"
    report_filename: str = "Final_Project_Report.md"


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    log_dir: str = "logs"


class Settings(BaseModel):
    """Top-level application settings container."""

    notebook_path: str = "notebooks/guvi-final.ipynb"
    domain_categories_yaml: str = "config/domain_category_map.yaml"
    data: DataConfig = Field(default_factory=DataConfig)
    sessionization: SessionizationConfig = Field(default_factory=SessionizationConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    recommendations: RecommendationsConfig = Field(default_factory=RecommendationsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


@lru_cache(maxsize=1)
def load_settings(config_path: Path | None = None) -> Settings:
    """Load and cache settings from the configuration YAML."""
    path = config_path or CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings.model_validate(raw)
