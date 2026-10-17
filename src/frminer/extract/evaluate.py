"""Extraction metrics: predicted feature phrases against the gold phrases written during
annotation. Phrases match when their embeddings are similar enough (paraphrases count);
a one-to-one matching keeps one prediction from matching two gold phrases."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from frminer.extract.normalize import normalize

Embed = Callable[[list[str]], np.ndarray]  # phrases → L2-normalised row vectors


def split_gold(features: str | None) -> list[str]:
    if not isinstance(features, str):
        return []
    return [f.strip() for f in features.split(";") if f.strip()]


def matches(sim: np.ndarray, threshold: float) -> int:
    """Largest number of one-to-one (pred, gold) pairs with similarity ≥ threshold
    (optimal assignment; ties broken towards higher similarity)."""
    from scipy.optimize import linear_sum_assignment

    ok = sim >= threshold
    if not ok.any():
        return 0
    rows, cols = linear_sum_assignment(np.where(ok, 1.0 + sim, 0.0), maximize=True)
    return int(ok[rows, cols].sum())


def item_counts(
    preds: Sequence[list[str]], golds: Sequence[list[str]], embed: Embed, threshold: float
) -> pd.DataFrame:
    """Per item: number of predicted, gold and matched phrases, and the best similarity
    reached by each gold phrase (averaged), for micro scores and bootstrapping."""
    phrases = sorted({p for ps in (*preds, *golds) for p in ps})
    vecs = dict(zip(phrases, embed(phrases), strict=True)) if phrases else {}
    rows = []
    for pred, gold in zip(preds, golds, strict=True):
        if pred and gold:
            sim = np.array([[vecs[p] @ vecs[g] for g in gold] for p in pred])
            matched, best = matches(sim, threshold), sim.max(axis=0).mean()
        else:
            matched, best = 0, 0.0
        rows.append({"n_pred": len(pred), "n_gold": len(gold), "matched": matched, "best": best})
    return pd.DataFrame(rows)


def scores(counts: pd.DataFrame) -> dict[str, float]:
    pred, gold, m = counts["n_pred"].sum(), counts["n_gold"].sum(), counts["matched"].sum()
    p = m / pred if pred else 0.0
    r = m / gold if gold else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "item_hit": float((counts["matched"] > 0).mean()),
        "empty": float((counts["n_pred"] == 0).mean()),
        "phrases_per_item": float(counts["n_pred"].mean()),
        "soft_recall": float(counts["best"].mean()),
    }


def bootstrap_f1(counts: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    f1s = [scores(counts.iloc[i])["f1"] for i in idx]
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token overlap after normalisation (lemmas, no filler)."""
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum(min(p.count(w), g.count(w)) for w in set(p) & set(g))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def best_token_f1(preds: Sequence[list[str]], golds: Sequence[list[str]]) -> float:
    """Mean over gold phrases of the best token-F1 any prediction for that item reaches."""
    vals = [
        max((token_f1(p, g) for p in pred), default=0.0)
        for pred, gold in zip(preds, golds, strict=True)
        for g in gold
    ]
    return float(np.mean(vals)) if vals else 0.0


def sentence_embedder(model_name: str) -> Embed:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(phrases: list[str]) -> np.ndarray:
        return model.encode(phrases, normalize_embeddings=True, batch_size=256)

    return embed
