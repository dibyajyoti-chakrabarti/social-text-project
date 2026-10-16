"""Gold-set sampling and label storage for the human annotation task.

Sampling design (see PLAN.md §7 and annotations/guidelines.md):

* Per-source quotas, so all three sources are represented.
* Half of each quota is a uniform random sample ("random" stratum): unbiased estimates.
* The other half ("enriched" stratum) over-samples items likely to be informative
  using METADATA ONLY (star rating, Reddit flair, length), never the text content, so
  that no text classifier (rules, ML or LLM) is favoured by how the gold set was built.
* Items longer than `max_words` are excluded to keep annotation fast.
* Dev/test split stratified by source × stratum. Dev items come first in labelling order.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["feature_request", "bug_report", "complaint", "praise", "question", "other"]
QUOTAS = {"play": 250, "reddit_post": 110, "reddit_comment": 40}
ENRICH_FLAIRS = {"Feature Request", "Rant", "Discussion", "Suggestion"}
LABEL_FIELDS = ["id", "label", "features", "unsure", "labelled_at"]


def enrichment_weight(df: pd.DataFrame) -> pd.Series:
    """Metadata-only sampling weight for the enriched stratum."""
    w = np.log1p(df["n_words"].clip(upper=150)).astype(float)
    mid_rating = df["rating"].isin([2, 3, 4]).fillna(False).astype(bool)
    flair = df["flair"].isin(ENRICH_FLAIRS).fillna(False).astype(bool)
    return w * np.where(mid_rating, 2.0, 1.0) * np.where(flair, 3.0, 1.0)


def sample_gold(
    corpus: pd.DataFrame,
    quotas: dict[str, int] = QUOTAS,
    n_dev: int = 100,
    max_words: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = corpus[corpus["n_words"] <= max_words]
    parts = []
    for source, quota in quotas.items():
        src = pool[pool["source"] == source]
        n_random = quota // 2
        random = src.sample(n=n_random, random_state=rng.integers(1 << 31))
        rest = src.drop(random.index)
        enriched = rest.sample(
            n=quota - n_random,
            weights=enrichment_weight(rest),
            random_state=rng.integers(1 << 31),
        )
        parts += [random.assign(stratum="random"), enriched.assign(stratum="enriched")]
    gold = pd.concat(parts)

    # Stratified dev/test split: dev share is the same within every source × stratum cell.
    dev_share = n_dev / len(gold)
    gold["split"] = "test"
    for _, cell in gold.groupby(["source", "stratum"]):
        n = round(len(cell) * dev_share)
        dev_idx = cell.sample(n=n, random_state=rng.integers(1 << 31)).index
        gold.loc[dev_idx, "split"] = "dev"

    # Shuffle within split; dev first.
    gold = gold.sample(frac=1, random_state=rng.integers(1 << 31))
    gold = gold.sort_values("split", key=lambda s: s.map({"dev": 0, "test": 1}), kind="stable")
    gold["order"] = range(len(gold))
    cols = ["order", "id", "split", "stratum", "source", "rating", "flair", "created_at", "text"]
    return gold[cols].reset_index(drop=True)


class LabelStore:
    """Append-only CSV of labels; the last row for an id wins (so edits are just appends)."""

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(LABEL_FIELDS)

    def save(self, item_id: str, label: str, features: str = "", unsure: bool = False) -> None:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        row = [item_id, label, features.strip(), int(unsure), datetime.now(UTC).isoformat()]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def latest(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, dtype={"features": str}, keep_default_na=False)
        return df.drop_duplicates("id", keep="last").set_index("id")


def load_gold(annotations_dir: Path, split: str | None = None) -> pd.DataFrame:
    """Gold sample joined with its latest labels; optionally restricted to one split."""
    sample = pd.read_csv(annotations_dir / "gold_sample.csv")
    labels = LabelStore(annotations_dir / "gold_labels.csv").latest()
    gold = sample.merge(labels[["label", "features", "unsure"]], left_on="id", right_index=True)
    if len(gold) != len(sample):
        raise ValueError(f"{len(sample) - len(gold)} gold items are unlabelled")
    if split is not None:
        gold = gold[gold["split"] == split]
    return gold.sort_values("order").reset_index(drop=True)
