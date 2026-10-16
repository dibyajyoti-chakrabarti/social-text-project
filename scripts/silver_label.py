"""LLM silver labels for a sample of the corpus → data/processed/silver.parquet.

uv run python scripts/silver_label.py

Samples items per source (excluding every gold item, so nothing trained on silver has
seen the test set), and labels them with the best dev configuration of the LLM
classifier. Runs in chunks and stops cleanly when the provider's daily quota is reached;
re-run the next day to continue (finished chunks come from the LLM cache).
"""

import pandas as pd

from frminer.config import load_config, path
from frminer.detect import llm_classify
from frminer.llm import LLM, QuotaExceeded

CHUNK = 500


def main() -> None:
    cfg = load_config()["detect"]["silver"]
    corpus = pd.read_parquet(path("interim") / "corpus.parquet")
    gold_ids = set(pd.read_csv(path("annotations") / "gold_sample.csv")["id"])
    pool = corpus[~corpus["id"].isin(gold_ids)]
    sample = pd.concat(
        pool[pool["source"] == src].sample(n=n, random_state=cfg["seed"])
        for src, n in cfg["per_source"].items()
    ).sample(frac=1, random_state=cfg["seed"])[["id", "source", "created_at", "rating", "text"]]

    llm = LLM(cfg["profile"])
    labels: list[str | None] = []
    for start in range(0, len(sample), CHUNK):
        chunk = sample["text"].iloc[start : start + CHUNK]
        try:
            res = llm_classify.classify(
                llm,
                chunk,
                batch_size=cfg["batch_size"],
                prompt=cfg["prompt"],
                desc=f"silver {start}-{start + len(chunk)}",
            )
        except QuotaExceeded as e:
            print(f"stopped at item {start}: {e}. Re-run tomorrow to continue.")
            break
        labels += res.labels

    out = sample.iloc[: len(labels)].assign(label=labels).dropna(subset=["label"])
    out.to_parquet(path("processed") / "silver.parquet", index=False)
    print(f"{len(out)} / {len(sample)} items labelled")
    print(out.groupby("source")["label"].value_counts(normalize=True).unstack().round(3))


if __name__ == "__main__":
    main()
