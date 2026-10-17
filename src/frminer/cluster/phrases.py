"""Method A: cluster the distinct extracted feature phrases.

Phrase embeddings (sentence-transformers) → UMAP (cosine, few dimensions) → HDBSCAN.
HDBSCAN leaves low-density phrases as noise; noise phrases close enough to a cluster
centroid are then attached to it, the rest stay unassigned (-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score


def embed(texts: list[str], model_name: str, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True
    )


def reduce(emb: np.ndarray, n_components: int = 5, n_neighbors: int = 15, seed: int = 0):
    import umap

    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    ).fit_transform(emb)


def hdbscan_labels(x: np.ndarray, min_cluster_size: int, min_samples: int | None) -> np.ndarray:
    return HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True
    ).fit_predict(x)


def centroids(emb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cluster ids (sorted, no -1) and their L2-normalised mean embeddings."""
    ids = np.array(sorted(set(labels) - {-1}))
    cents = np.stack([emb[labels == c].mean(axis=0) for c in ids]) if len(ids) else emb[:0]
    cents = cents / np.linalg.norm(cents, axis=1, keepdims=True).clip(1e-12)
    return ids, cents


def assign_outliers(emb: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Attach noise points to the nearest centroid when cosine similarity ≥ threshold."""
    labels = labels.copy()
    ids, cents = centroids(emb, labels)
    noise = np.where(labels == -1)[0]
    if len(ids) and len(noise):
        sim = emb[noise] @ cents.T
        best = sim.argmax(axis=1)
        ok = sim[np.arange(len(noise)), best] >= threshold
        labels[noise[ok]] = ids[best[ok]]
    return labels


@dataclass
class Quality:
    n_clusters: int
    outlier_rate: float  # share of phrases left unassigned
    outlier_mentions: float  # share of mentions (phrase weight) left unassigned
    silhouette: float  # cosine, on the original embeddings, assigned points only
    largest_share: float  # share of assigned mentions in the biggest cluster


def quality(
    emb: np.ndarray, labels: np.ndarray, weights: np.ndarray, max_points: int = 5000, seed: int = 0
) -> Quality:
    assigned = labels != -1
    n = len(set(labels[assigned]))
    sil = float("nan")
    if n >= 2:
        idx = np.where(assigned)[0]
        if len(idx) > max_points:
            idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
        sil = float(silhouette_score(emb[idx], labels[idx], metric="cosine"))
    w_assigned = weights[assigned]
    sizes = np.bincount(labels[assigned], weights=w_assigned) if assigned.any() else np.array([0.0])
    return Quality(
        n_clusters=n,
        outlier_rate=float(1 - assigned.mean()),
        outlier_mentions=float(1 - w_assigned.sum() / weights.sum()),
        silhouette=sil,
        largest_share=float(sizes.max() / max(w_assigned.sum(), 1e-12)),
    )
