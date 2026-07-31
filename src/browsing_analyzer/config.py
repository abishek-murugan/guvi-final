"""Pydantic-based configuration management.

Loads ``config/config.yaml`` (or an overridden path) and exposes typed,
validated settings to the rest of the application.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class BrowserConfig(BaseModel):
    """Browser selection settings."""

    name: str = "chrome"
    profile: str = "Default"


class DataConfig(BaseModel):
    """Paths for raw and processed data."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "data/models"
    chrome_data_file: str = "chrome_data.csv"
    ram_data_file: str = "ram_data.csv"
    browser_history_output: str = "browsing_history.csv"
    ram_log_output: str = "ram_log.csv"


class SessionizationConfig(BaseModel):
    """Session boundary settings."""

    inactivity_threshold_minutes: int = 15
    min_session_events: int = 2


class RamLoggerConfig(BaseModel):
    """RAM collection methodology description (interval, duration)."""

    interval_seconds: int = 5
    duration_hours: int = 24


class ClusteringConfig(BaseModel):
    """Clustering hyper-parameters and feature list."""

    algorithm: Literal["kmeans", "gmm", "dbscan"] = "kmeans"
    n_clusters: int = 5
    random_state: int = 42
    features: list[str] = Field(default_factory=lambda: ["event_count"])


class ModelConfig(BaseModel):
    """PyTorch LSTM hyper-parameters."""

    type: str = "lstm"
    sequence_length: int = 20
    embedding_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    patience: int = 10
    seed: int = 42


class RecommendationsConfig(BaseModel):
    """Thresholds driving the recommendation rules."""

    social_media_threshold: float = 0.45
    late_night_hour: int = 22
    ram_spike_threshold_mb: float = 500.0
    category_ram_threshold_mb: float = 300.0


class OutputConfig(BaseModel):
    """Report and dashboard output settings."""

    report_path: str = "reports"
    report_filename: str = "browsing_report.md"
    dashboard_port: int = 8501


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    log_dir: str = "logs"
    console_output: bool = True


class Settings(BaseModel):
    """Top-level application settings container."""

    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    time_windows: list[int] = Field(default_factory=lambda: [3, 4, 5])
    default_window: int = 4
    sessionization: SessionizationConfig = Field(default_factory=SessionizationConfig)
    ram_logger: RamLoggerConfig = Field(default_factory=RamLoggerConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    recommendations: RecommendationsConfig = Field(default_factory=RecommendationsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("default_window")
    @classmethod
    def _validate_default_window(cls, value: int) -> int:
        if value not in (3, 4, 5):
            raise ValueError("default_window must be one of 3, 4, 5")
        return value


@lru_cache(maxsize=1)
def load_settings(config_path: Path | None = None) -> Settings:
    """Load and cache settings from a YAML file.

    Args:
        config_path: Optional path to the config YAML. Defaults to the
            repository-level ``config/config.yaml``.

    Returns:
        A validated :class:`Settings` object.
    """
    path = config_path or CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings.model_validate(raw)
