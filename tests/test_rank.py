import numpy as np
import pandas as pd
import pytest

from frminer.insights import compare, existing
from frminer.rank import score, trends


def themed_frame():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    # theme 0: 10 old Play items (rating 1); theme 1: 10 recent Reddit items
    for i in range(10):
        rows.append((f"p{i}", "play", t0, 1, i, 0, None))
        rows.append((f"r{i}", "reddit_post", t0 + pd.Timedelta(weeks=20), pd.NA, i, 1, "Tidal"))
    rows.append(("p0", "play", t0, 1, 0, 1, None))  # p0 mentions both themes
    return pd.DataFrame(
        rows,
        columns=["id", "source", "created_at", "rating", "engagement", "theme_id", "competitor"],
    ).astype({"rating": "Int64"})


def corpus_created():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.Series([t0 + pd.Timedelta(weeks=w) for w in range(21) for _ in range(10)])


def test_theme_stats_volume_ratings_growth():
    stats = score.theme_stats(themed_frame(), corpus_created(), weeks=10, min_support=5)
    s0 = stats.set_index("theme_id").loc[0]
    s1 = stats.set_index("theme_id").loc[1]
    assert s0["items"] == 10 and s1["items"] == 11
    assert s0["low_rating_share"] == 1.0
    assert s1["growth"] > 0  # all of theme 1 is in the recent window


def test_priority_and_zscore():
    stats = pd.DataFrame(
        {"items": [100, 10], "engagement": [0.5, 0.5], "low_rating_share": [np.nan, 0.2]}
    ).assign(growth=[0.0, 1.0])
    w = {"volume": 1.0, "engagement": 1.0, "dissatisfaction": 0.0, "growth": 0.0}
    p = score.priority(stats, w)
    assert p[0] > p[1]
    assert score.zscore(pd.Series([1.0, np.nan])).tolist() == [0.0, 0.0]


def test_bh_and_two_proportion():
    q = trends.bh(pd.Series([0.01, 0.04, 0.03, 0.5]))
    assert q.iloc[0] == pytest.approx(0.04) and q.iloc[3] == pytest.approx(0.5)
    assert trends.two_proportion_p(10, 1000, 10, 1000) == pytest.approx(1.0)
    assert trends.two_proportion_p(10, 1000, 60, 1000) < 1e-6


def test_classify_trends():
    stats = pd.DataFrame({"prev_items": [10, 100, 50], "recent_items": [80, 20, 52]})
    out = trends.classify_trends(stats, 10000, 10000, min_ratio=1.5, min_recent=15, alpha=0.05)
    assert out["trend"].tolist() == ["emerging", "declining", "stable"]


def test_monthly_share():
    items = score.item_table(themed_frame())
    m = trends.monthly(items, corpus_created())
    jan = m[(m["theme_id"] == 0)].iloc[0]
    assert jan["items"] == 10 and 0 < jan["share"] <= 1


def test_platform_ratings_competitors():
    items = score.item_table(themed_frame())
    plat = compare.platform(items, min_items=1).set_index("theme_id")
    assert plat.loc[1, "log2_ratio"] > 0 > plat.loc[0, "log2_ratio"]
    rat = compare.ratings(items, min_items=1).set_index("theme_id")
    assert rat.loc[0, "mean_rating"] == 1 and rat.loc[0, "low_share"] == 1
    comp = compare.competitors(items)
    assert comp.iloc[0].to_dict() == {"competitor": "Tidal", "theme_id": 1, "items": 10}


def test_existing_candidates_and_validator():
    t = np.eye(3)[:2]
    c = np.eye(3)
    idx, sim = existing.candidates(t, c, k=2)
    assert idx[:, 0].tolist() == [0, 1] and sim[:, 0].tolist() == [1.0, 1.0]
    v = existing._validator(1)
    v({"items": {"1": {"match": 2, "status": "premium_only"}}})
    with pytest.raises(ValueError):
        v({"items": {"1": {"match": "2", "status": "exists"}}})
    with pytest.raises(ValueError):
        v({"items": {"1": {"match": 1, "status": "maybe"}}})


def test_catalogue_loads():
    cat = existing.load_catalogue()
    assert len(cat) >= 30
    assert all({"name", "description", "availability"} <= set(c) for c in cat)
