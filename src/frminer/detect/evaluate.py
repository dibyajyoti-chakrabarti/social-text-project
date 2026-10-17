"""Classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from frminer.annotate import LABELS

TARGET = "feature_request"


def point_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true == TARGET, y_pred == TARGET, average="binary", zero_division=0
    )
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "accuracy": float((y_true == y_pred).mean()),
        "fr_precision": p,
        "fr_recall": r,
        "fr_f1": f,
        "kappa": cohen_kappa_score(y_true, y_pred, labels=LABELS),
    }


def bootstrap_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Point estimate plus 95% percentile bootstrap CI for every metric (rows)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    point = point_metrics(y_true, y_pred)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        samples.append(point_metrics(y_true[idx], y_pred[idx]))
    boot = pd.DataFrame(samples)
    return pd.DataFrame(
        {
            "value": pd.Series(point),
            "ci_low": boot.quantile(0.025),
            "ci_high": boot.quantile(0.975),
        }
    )


def per_class(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": n}, index=LABELS)


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Rows = gold label, columns = predicted label."""
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=LABELS), index=LABELS, columns=LABELS
    )
