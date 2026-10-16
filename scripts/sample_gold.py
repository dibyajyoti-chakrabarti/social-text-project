"""Draw the gold annotation sample → annotations/gold_sample.csv.

uv run python scripts/sample_gold.py [--force]

Refuses to overwrite an existing sample unless --force is given, because labels are
keyed to the sampled ids.
"""

import sys

import pandas as pd

from frminer.annotate import sample_gold
from frminer.config import path


def main() -> None:
    out = path("annotations") / "gold_sample.csv"
    if out.exists() and "--force" not in sys.argv:
        sys.exit(f"{out} exists; pass --force to resample (this invalidates existing labels)")
    corpus = pd.read_parquet(path("interim") / "corpus.parquet")
    gold = sample_gold(corpus)
    gold.to_csv(out, index=False)
    print(gold.groupby(["split", "source", "stratum"]).size().unstack(fill_value=0))
    print(f"wrote {len(gold)} items to {out}")


if __name__ == "__main__":
    main()
