from __future__ import annotations

import importlib
from collections.abc import Sequence

import numpy as np


def evaluate_kmeans(embeddings: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    try:
        cluster_module = importlib.import_module("sklearn.cluster")
        metrics_module = importlib.import_module("sklearn.metrics")
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for clustering evaluation") from exc

    kmeans_class = cluster_module.KMeans
    adjusted_rand_score = metrics_module.adjusted_rand_score
    normalized_mutual_info_score = metrics_module.normalized_mutual_info_score
    num_clusters = len(set(labels))
    predictions = kmeans_class(n_clusters=num_clusters, n_init="auto", random_state=42).fit_predict(
        embeddings
    )
    return {
        "nmi": float(normalized_mutual_info_score(labels, predictions)),
        "ari": float(adjusted_rand_score(labels, predictions)),
    }
