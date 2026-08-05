"""Unsupervised clustering of browsing sessions.

Sessions are embedded in a fixed feature space, scaled with
``StandardScaler`` and grouped with KMeans (``k=2``, matching the notebook's
elbow/silhouette analysis). A 2-D PCA projection and a silhouette sweep are
kept for evaluation and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterResult:
    """Container for clustering outputs."""

    labels: np.ndarray
    silhouette: float
    feature_names: list[str]
    cluster_centers: pd.DataFrame
    profiles: dict[int, str]
    elbow: pd.DataFrame
    model: KMeans
    scaler: StandardScaler
    pca: PCA
    X_scaled: np.ndarray
    X_pca: np.ndarray


_PROFILE_FEATURES = [
    "session_duration_minutes",
    "page_count",
    "peak_used_mb",
    "unique_categories",
]


class BehaviorClusterer:
    """Clusters session feature tables into labelled behavior groups.

    Args:
        settings: Application settings (features, n_clusters, random state).
    """

    def __init__(self, settings: Settings) -> None:
        self.n_clusters = settings.clustering.n_clusters
        self.random_state = settings.clustering.random_state
        self.features = list(settings.clustering.features)

    def fit(self, sessions: pd.DataFrame) -> ClusterResult:
        """Fit the clustering model and produce labelled cluster profiles."""
        available = [f for f in self.features if f in sessions.columns]
        X = sessions[available].fillna(0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        elbow = self._elbow_analysis(X_scaled)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(X_scaled)
        silhouette = (
            silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else float("nan")
        )

        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)

        centers = pd.DataFrame(X, columns=available).groupby(labels).mean()
        profiles = self._label_clusters(centers)

        logger.info(
            "clustering_done",
            n_clusters=self.n_clusters,
            silhouette=round(silhouette, 4),
        )
        return ClusterResult(
            labels=labels,
            silhouette=silhouette,
            feature_names=available,
            cluster_centers=centers,
            profiles=profiles,
            elbow=elbow,
            model=kmeans,
            scaler=scaler,
            pca=pca,
            X_scaled=X_scaled,
            X_pca=X_pca,
        )

    def _elbow_analysis(self, X_scaled: np.ndarray, max_k: int = 9) -> pd.DataFrame:
        """Silhouette + inertia sweep over candidate cluster counts."""
        rows: list[dict[str, float | int | None]] = []
        kmeans_1 = KMeans(n_clusters=1, init="k-means++", random_state=self.random_state, n_init=10)
        kmeans_1.fit(X_scaled)
        rows.append({"k": 1, "wcss": float(kmeans_1.inertia_), "silhouette": None})

        for k in range(2, min(max_k, len(X_scaled))):
            kmeans = KMeans(
                n_clusters=k, init="k-means++", random_state=self.random_state, n_init=10
            )
            labels = kmeans.fit_predict(X_scaled)
            rows.append(
                {
                    "k": k,
                    "wcss": float(kmeans.inertia_),
                    "silhouette": float(silhouette_score(X_scaled, labels)),
                }
            )
        return pd.DataFrame(rows)

    def _label_clusters(self, centers: pd.DataFrame) -> dict[int, str]:
        """Generate a human-readable label per cluster from its drivers."""
        means = centers.mean()
        labels: dict[int, str] = {}
        for cluster in centers.index:
            parts = []
            for feature in _PROFILE_FEATURES:
                if feature not in centers.columns:
                    continue
                ratio = centers.loc[cluster, feature] / means[feature] if means[feature] else 1.0
                if ratio > 1.2:
                    parts.append(f"High {_humanize(feature)}")
                elif ratio < 0.8:
                    parts.append(f"Low {_humanize(feature)}")
            labels[int(cluster)] = " + ".join(parts) if parts else "Balanced session"
        return labels


def _humanize(name: str) -> str:
    """Convert a snake_case feature name into a readable label."""
    mapping = {
        "session_duration_minutes": "duration",
        "page_count": "activity",
        "peak_used_mb": "RAM",
        "unique_categories": "topic variety",
    }
    return mapping.get(name, name.replace("_", " "))
