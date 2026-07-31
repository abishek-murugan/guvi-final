"""Unit tests for session clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from browsing_analyzer.analytics.clustering import BehaviorClusterer
from browsing_analyzer.config import Settings


def _session_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for i in range(40):
        kind = i % 3
        if kind == 0:  # long, heavy, late-night sessions
            rows.append(
                {
                    "session_id": i,
                    "duration_minutes": rng.uniform(60, 180),
                    "event_count": int(rng.integers(20, 60)),
                    "unique_domains": int(rng.integers(5, 10)),
                    "category_entropy": rng.uniform(1.5, 2.0),
                    "switching_rate": rng.uniform(0.4, 0.7),
                    "avg_ram_mb": rng.uniform(4500, 5200),
                    "peak_ram_mb": rng.uniform(5200, 6500),
                    "median_hour": rng.uniform(21, 23),
                    "is_weekend": 1,
                    "session_span_hours": rng.uniform(1, 3),
                }
            )
        elif kind == 1:  # short, low RAM daytime sessions
            rows.append(
                {
                    "session_id": i,
                    "duration_minutes": rng.uniform(5, 25),
                    "event_count": int(rng.integers(2, 8)),
                    "unique_domains": int(rng.integers(1, 3)),
                    "category_entropy": rng.uniform(0.1, 0.8),
                    "switching_rate": rng.uniform(0.1, 0.4),
                    "avg_ram_mb": rng.uniform(3500, 4000),
                    "peak_ram_mb": rng.uniform(3800, 4200),
                    "median_hour": rng.uniform(9, 13),
                    "is_weekend": 0,
                    "session_span_hours": rng.uniform(0.1, 0.5),
                }
            )
        else:  # medium midday
            rows.append(
                {
                    "session_id": i,
                    "duration_minutes": rng.uniform(20, 60),
                    "event_count": int(rng.integers(8, 20)),
                    "unique_domains": int(rng.integers(3, 6)),
                    "category_entropy": rng.uniform(0.8, 1.5),
                    "switching_rate": rng.uniform(0.3, 0.6),
                    "avg_ram_mb": rng.uniform(4000, 4500),
                    "peak_ram_mb": rng.uniform(4300, 5000),
                    "median_hour": rng.uniform(14, 17),
                    "is_weekend": 0,
                    "session_span_hours": rng.uniform(0.3, 1.0),
                }
            )
    return pd.DataFrame(rows)


def test_clustering_produces_labels_and_profiles(settings: Settings):
    clusterer = BehaviorClusterer(settings)
    result = clusterer.fit(_session_table())
    assert len(result.labels) == 40
    assert result.cluster_centers.shape[0] == settings.clustering.n_clusters
    assert len(result.profiles) == settings.clustering.n_clusters


def test_clustering_silhouette_is_bounded(settings: Settings):
    clusterer = BehaviorClusterer(settings)
    result = clusterer.fit(_session_table())
    assert -1.0 <= result.silhouette <= 1.0


def test_clustering_profiles_human_readable(settings: Settings):
    clusterer = BehaviorClusterer(settings)
    result = clusterer.fit(_session_table())
    for label in result.profiles.values():
        assert isinstance(label, str) and len(label) > 0


def test_clustering_missing_features_warns(settings: Settings):
    settings.clustering.features = ["duration_minutes", "event_count", "not_a_feature"]
    clusterer = BehaviorClusterer(settings)
    result = clusterer.fit(_session_table())
    assert "not_a_feature" not in result.feature_names
