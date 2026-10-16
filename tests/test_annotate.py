import numpy as np
import pandas as pd
import pytest

from frminer.annotate import LabelStore, sample_gold


def fake_corpus(n=600, seed=0):
    rng = np.random.default_rng(seed)
    sources = rng.choice(["play", "reddit_post", "reddit_comment"], size=n, p=[0.6, 0.3, 0.1])
    return pd.DataFrame(
        {
            "id": [f"x{i}" for i in range(n)],
            "source": sources,
            "text": ["some feedback text here"] * n,
            "rating": pd.array(
                [rng.integers(1, 6) if s == "play" else pd.NA for s in sources], dtype="Int64"
            ),
            "flair": [rng.choice(["Rant", "Question"]) if s != "play" else None for s in sources],
            "n_words": rng.integers(4, 400, size=n),
            "created_at": pd.Timestamp("2026-01-01", tz="UTC"),
        }
    )


QUOTAS = {"play": 40, "reddit_post": 20, "reddit_comment": 10}


def test_sample_respects_quotas_split_and_length():
    gold = sample_gold(fake_corpus(), quotas=QUOTAS, n_dev=14, max_words=250)
    assert gold["source"].value_counts().to_dict() == QUOTAS
    assert gold["id"].is_unique
    assert (gold["split"] == "dev").sum() == 14
    assert (
        gold.loc[gold["split"] == "dev", "order"].max()
        < gold.loc[gold["split"] == "test", "order"].min()
    )
    assert set(gold["stratum"]) == {"random", "enriched"}


def test_sample_is_reproducible():
    a = sample_gold(fake_corpus(), quotas=QUOTAS, n_dev=14)
    b = sample_gold(fake_corpus(), quotas=QUOTAS, n_dev=14)
    assert a["id"].tolist() == b["id"].tolist()


def test_label_store_last_write_wins(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    store.save("a", "complaint")
    store.save("b", "feature_request", features=" sleep timer ", unsure=True)
    store.save("a", "bug_report")
    latest = store.latest()
    assert latest.loc["a", "label"] == "bug_report"
    assert latest.loc["b", "features"] == "sleep timer"
    assert latest.loc["b", "unsure"] == 1


def test_label_store_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError):
        LabelStore(tmp_path / "l.csv").save("a", "feature")
