"""Theme time series and emerging / declining detection.

A theme's monthly volume is shown as its share of all corpus items that month, which
removes swings in how much feedback was collected. Emerging / declining compares the
recent window with the one before it (two-proportion z-test on the theme's share of
corpus items, Benjamini–Hochberg across themes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def monthly(items: pd.DataFrame, corpus_created: pd.Series) -> pd.DataFrame:
    """theme_id × month: items and share of that month's corpus items."""
    month = items["created_at"].dt.tz_localize(None).dt.to_period("M")
    counts = items.assign(month=month).groupby(["theme_id", "month"])["id"].nunique()
    totals = corpus_created.dt.tz_localize(None).dt.to_period("M").value_counts()
    out = counts.rename("items").reset_index()
    out["share"] = out["items"] / out["month"].map(totals)
    out["month"] = out["month"].dt.to_timestamp()
    return out


def two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided p-value for equal proportions x1/n1 vs x2/n2."""
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x2 / n2 - x1 / n1) / se
    return float(2 * norm.sf(abs(z)))


def bh(p: pd.Series) -> pd.Series:
    """Benjamini–Hochberg adjusted p-values."""
    order = p.sort_values()
    m = len(order)
    adj = (order * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return adj.reindex(p.index)


def classify_trends(
    stats: pd.DataFrame,
    tot_prev: int,
    tot_recent: int,
    *,
    min_ratio: float,
    min_recent: int,
    alpha: float,
) -> pd.DataFrame:
    """Adds p, q (BH) and trend ∈ {emerging, declining, stable} to per-theme stats."""
    out = stats.copy()
    out["p"] = [
        two_proportion_p(int(a), tot_prev, int(b), tot_recent)
        for a, b in zip(out["prev_items"], out["recent_items"], strict=True)
    ]
    out["q"] = bh(out["p"])
    ratio = (out["recent_items"] / tot_recent + 1e-9) / (out["prev_items"] / tot_prev + 1e-9)
    out["ratio"] = ratio
    sig = out["q"] < alpha
    out["trend"] = "stable"
    out.loc[sig & (ratio >= min_ratio) & (out["recent_items"] >= min_recent), "trend"] = "emerging"
    out.loc[sig & (ratio <= 1 / min_ratio) & (out["prev_items"] >= min_recent), "trend"] = (
        "declining"
    )
    return out
