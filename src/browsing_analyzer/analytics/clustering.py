"""Unsupervised clustering of browsing sessions.

Sessions are embedded in a fixed feature space (duration, event count,
switching rate, RAM aggregates, hour-of-day, weekend flag). KMeans (or GMM)
groups them into interpretable behavior clusters which are then labelled from
their top feature drivers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterResult:
    """Container for clustering output."""

    labels: np.ndarray
    silhouette: float
    feature_names: list[str]
    cluster_centers: pd.DataFrame
    profiles: dict[int, str] = field(default_factory=dict)
    model: object | None = None


class BehaviorClusterer:
    """Clusters session feature tables into labelled behavior groups.

    Args:
        settings: Application settings (algorithm, n_clusters, features).
    """

    def __init__(self, settings: Settings) -> None:
        self.algorithm = settings.clustering.algorithm
        self.n_clusters = settings.clustering.n_clusters
        self.random_state = settings.clustering.random_state
        self.features = list(settings.clustering.features)

    def fit(self, sessions: pd.DataFrame) -> ClusterResult:
        """Fit the clustering model and produce labelled profiles.

        Args:
            sessions: Session feature table (from the sessionizer).

        Returns:
            A :class:`ClusterResult` with labels, silhouette score, centers and
            human-readable cluster profiles.
        """
        available = [f for f in self.features if f in sessions.columns]
        missing = set(self.features) - set(available)
        if missing:
            logger.warning("clustering_features_missing", missing=sorted(missing))

        X = sessions[available].fillna(0.0).to_numpy(dtype=np.float64)
        if len(X) < 2:
            raise ValueError("Need at least 2 sessions to cluster")

        n_clusters = min(self.n_clusters, len(X))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.algorithm == "gmm":
            model = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        elif self.algorithm == "dbscan":
            model = None
            labels = _dbscan_fallback(X_scaled)
        else:
            model = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=self.random_state,
            )
            labels = model.fit_predict(X_scaled)

        if model is not None:
            labels = model.fit_predict(X_scaled)

        if len(np.unique(labels)) < 2:
            silhouette = float("nan")
        else:
            silhouette = float(silhouette_score(X_scaled, labels))

        centers_df = _cluster_centers(X, labels, available)
        profiles = _label_clusters(centers_df, available)

        logger.info(
            "clustering_done",
            algorithm=self.algorithm,
            n_clusters=n_clusters,
            silhouette=round(silhouette, 4),
        )
        return ClusterResult(
            labels=labels,
            silhouette=silhouette,
            feature_names=available,
            cluster_centers=centers_df,
            profiles=profiles,
            model=model,
        )


def _dbscan_fallback(X: np.ndarray) -> np.ndarray:
    """DBSCAN-style density clustering with adaptive epsilon (simple impl)."""
    from sklearn.cluster import DBSCAN

    # Heuristic epsilon: 0.3 * mean pairwise distance of scaled features.
    n = min(len(X), 2000)
    sample = X[np.random.default_rng(0).choice(len(X), n, replace=False)]
    dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    eps = float(0.3 * dists[np.triu_indices(n, 1)].mean())
    return np.asarray(DBSCAN(eps=eps, min_samples=2).fit_predict(X))


def _cluster_centers(X: np.ndarray, labels: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean feature values per cluster (unscaled, for interpretation)."""
    rows = {}
    for cluster in np.unique(labels):
        rows[int(cluster)] = X[labels == cluster].mean(axis=0)
    return pd.DataFrame.from_dict(rows, orient="index", columns=features).sort_index()


def _label_clusters(centers: pd.DataFrame, features: list[str]) -> dict[int, str]:
    """Generate a human-readable label per cluster from its top drivers.

    Labels combine the two most distinguishing feature deviations, e.g.
    ``"High duration + high RAM"``.
    """
    if centers.empty:
        return {}
    normalized = centers.sub(centers.mean(), axis=1).div(centers.std().replace(0, 1), axis=1)
    labels: dict[int, str] = {}
    for cluster in centers.index:
        row = normalized.loc[cluster].dropna()
        top = row.abs().sort_values(ascending=False).head(2)
        parts = []
        for feature, z in top.items():
            direction = "high" if z >= 0 else "low"
            parts.append(f"{direction} {_humanize_feature(feature)}")
        labels[int(cluster)] = " + ".join(parts) if parts else "balanced session"
    return labels


def _humanize_feature(name: str) -> str:
    """Convert a snake_case feature name into a readable label."""
    mapping = {
        "duration_minutes": "duration",
        "event_count": "activity",
        "unique_domains": "domain variety",
        "category_entropy": "topic mix",
        "switching_rate": "switching",
        "avg_ram_mb": "avg RAM",
        "peak_ram_mb": "peak RAM",
        "median_hour": "late hour",
        "is_weekend": "weekend use",
        "session_span_hours": "session span",
    }
    return mapping.get(name, name.replace("_", " "))
