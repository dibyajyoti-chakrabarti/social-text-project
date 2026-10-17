"""Platform and rating comparisons of request themes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def platform(items: pd.DataFrame, min_items: int = 10) -> pd.DataFrame:
    """Theme share among Play requests vs among Reddit requests (posts + comments).

    `log2_ratio` > 0: relatively more common on Reddit. Shares are of the items with any
    theme on that platform; themes with fewer than `min_items` in total are left out.
    """
    plat = np.where(items["source"] == "play", "play", "reddit")
    counts = items.assign(platform=plat).groupby(["theme_id", "platform"])["id"].nunique()
    wide = counts.unstack(fill_value=0).reindex(columns=["play", "reddit"], fill_value=0)
    totals = items.assign(platform=plat).groupby("platform")["id"].nunique()
    out = wide.rename(columns={"play": "play_items", "reddit": "reddit_items"})
    out["play_share"] = out["play_items"] / totals.get("play", 1)
    out["reddit_share"] = out["reddit_items"] / totals.get("reddit", 1)
    out["log2_ratio"] = np.log2((out["reddit_share"] + 1e-4) / (out["play_share"] + 1e-4))
    out = out[out["play_items"] + out["reddit_items"] >= min_items]
    return out.reset_index()


def ratings(items: pd.DataFrame, min_items: int = 10) -> pd.DataFrame:
    """Star-rating profile of each theme's Play reviews."""
    play = items[(items["source"] == "play") & items["rating"].notna()]
    dist = pd.crosstab(play["theme_id"], play["rating"].astype(int), normalize="index")
    dist.columns = [f"r{c}" for c in dist.columns]
    out = dist.assign(
        n=play.groupby("theme_id")["id"].nunique(),
        mean_rating=play.groupby("theme_id")["rating"].mean(),
        low_share=play.groupby("theme_id")["rating"].apply(lambda r: float((r <= 2).mean())),
        high_share=play.groupby("theme_id")["rating"].apply(lambda r: float((r >= 4).mean())),
    )
    return out[out["n"] >= min_items].reset_index()


def competitors(items: pd.DataFrame) -> pd.DataFrame:
    """Competitor × theme counts of items that cite the competitor."""
    cited = items[items["competitor"].notna()]
    return (
        cited.groupby(["competitor", "theme_id"])["id"]
        .nunique()
        .rename("items")
        .reset_index()
        .sort_values("items", ascending=False, ignore_index=True)
    )
