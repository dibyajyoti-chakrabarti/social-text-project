"""Clean and merge raw scraped data into data/interim/corpus.parquet.

uv run python scripts/build_corpus.py
"""

import json
from datetime import UTC, datetime

import pandas as pd

from frminer.config import load_config, path
from frminer.ingest import clean


def main() -> None:
    cfg = load_config()["data"]
    raw = path("raw")
    # One file per star-rating stream (plus any unfiltered file); overlaps are deduped by id.
    play = pd.concat(
        [
            pd.read_json(f, lines=True, dtype={"reviewId": str})
            for f in sorted(raw.glob("playstore*.jsonl"))
        ],
        ignore_index=True,
    )
    posts = pd.read_json(raw / "reddit_posts.jsonl", lines=True, dtype={"id": str})
    comments_path = raw / "reddit_comments.jsonl"
    parts = [clean.from_playstore(play), clean.from_reddit_posts(posts)]
    if comments_path.exists() and comments_path.stat().st_size:
        comments = pd.read_json(comments_path, lines=True, dtype={"id": str, "link_id": str})
        parts.append(clean.from_reddit_comments(comments, posts))

    since = datetime.fromisoformat(cfg["since"]).replace(tzinfo=UTC)
    until = datetime.fromisoformat(cfg["until"]).replace(tzinfo=UTC)
    corpus, stats = clean.build_corpus(parts, since, until, cfg["min_words"])
    stats["by_source"] = corpus["source"].value_counts().to_dict()

    out = path("interim") / "corpus.parquet"
    corpus.to_parquet(out, index=False)
    (path("reports") / "tables").mkdir(exist_ok=True)
    with open(path("reports") / "tables" / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(corpus)} rows to {out}")


if __name__ == "__main__":
    main()
