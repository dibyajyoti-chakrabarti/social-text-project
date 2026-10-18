"""Per-theme statistics and the priority score.

priority = w_volume·z(log items) + w_engagement·z(engagement)
         + w_dissatisfaction·z(share of ≤2★ reviews) + w_growth·z(growth)

Engagement is each item's percentile rank of thumbs-up / upvotes within its own source
(so Play and Reddit are comparable), averaged per theme. Growth is the log2 ratio of the
theme's share of all corpus items in the recent window vs the window before it (5★ Play
reviews left out, see `trend_items`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def item_table(themed: pd.DataFrame) -> pd.DataFrame:
    """One row per (theme, item) with the item's metadata and engagement percentile."""
    items = themed[themed["theme_id"] >= 0].drop_duplicates(["theme_id", "id"]).copy()
    eng = themed.drop_duplicates("id")[["id", "source", "engagement"]].copy()
    eng["engagement_pct"] = eng.groupby("source")["engagement"].rank(pct=True)
    return items.merge(eng[["id", "engagement_pct"]], on="id", how="left")


def trend_items(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that count towards growth, trends and monthly shares: all but 5★ Play reviews.

    Google Play only returns 5★ reviews for the last weeks of the window (pagination limit),
    so counting them would inflate recent volume and make every theme look like it shrinks.
    """
    five_star_play = (df["source"] == "play") & df["rating"].eq(5).fillna(False).astype(bool)
    return df[~five_star_play]


def windows(end: pd.Timestamp, weeks: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(previous start, recent start, end) for two back-to-back windows of `weeks`."""
    recent = end - pd.Timedelta(weeks=weeks)
    return recent - pd.Timedelta(weeks=weeks), recent, end


def window_counts(created: pd.Series, bounds) -> tuple[int, int]:
    prev_start, recent_start, end = bounds
    prev = int(((created >= prev_start) & (created < recent_start)).sum())
    recent = int(((created >= recent_start) & (created < end)).sum())
    return prev, recent


def theme_stats(
    themed: pd.DataFrame, corpus_created: pd.Series, *, weeks: int, min_support: int
) -> pd.DataFrame:
    items = item_table(themed)
    end = corpus_created.max() + pd.Timedelta(seconds=1)
    bounds = windows(end, weeks)
    tot_prev, tot_recent = window_counts(corpus_created, bounds)
    rows = []
    for theme, g in items.groupby("theme_id"):
        play = g[(g["source"] == "play") & g["rating"].notna()]
        prev, recent = window_counts(trend_items(g)["created_at"], bounds)
        growth = (
            np.log2((recent / tot_recent + EPS) / (prev / tot_prev + EPS))
            if prev + recent >= min_support
            else np.nan
        )
        rows.append(
            {
                "theme_id": theme,
                "items": g["id"].nunique(),
                "play_items": int((g["source"] == "play").sum()),
                "reddit_items": int((g["source"] != "play").sum()),
                "engagement": g["engagement_pct"].mean(),
                "low_rating_share": float((play["rating"] <= 2).mean()) if len(play) else np.nan,
                "mean_rating": float(play["rating"].mean()) if len(play) else np.nan,
                "prev_items": prev,
                "recent_items": recent,
                "growth": growth,
            }
        )
    return pd.DataFrame(rows)


def zscore(s: pd.Series) -> pd.Series:
    """Standardised values; missing values (too little data) count as average (0)."""
    sd = s.std()
    return ((s - s.mean()) / sd if sd and sd > 0 else s * 0).fillna(0.0)


def priority(stats: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    parts = {
        "volume": zscore(np.log(stats["items"])),
        "engagement": zscore(stats["engagement"]),
        "dissatisfaction": zscore(stats["low_rating_share"]),
        "growth": zscore(stats["growth"]),
    }
    return sum(weights[k] * v for k, v in parts.items())
