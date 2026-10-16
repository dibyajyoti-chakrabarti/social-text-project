"""Scrape r/truespotify posts, then top-level comments on feature-relevant posts (resumable).

uv run python scripts/scrape_reddit.py
"""

import json
from datetime import UTC, datetime

from frminer.config import load_config, path
from frminer.ingest import reddit


def main() -> None:
    cfg = load_config()["data"]
    rcfg = cfg["reddit"]
    since = datetime.fromisoformat(cfg["since"]).replace(tzinfo=UTC)
    until = datetime.fromisoformat(cfg["until"]).replace(tzinfo=UTC)
    posts_path = path("raw") / "reddit_posts.jsonl"
    n = reddit.scrape_posts(rcfg["subreddit"], since, until, posts_path)
    print(f"wrote {n} new posts to {posts_path}")

    with open(posts_path) as f:
        posts = [json.loads(line) for line in f]
    eligible = [
        p
        for p in posts
        if p.get("link_flair_text") in rcfg["comment_flairs"]
        and p["num_comments"] >= rcfg["comment_min_post_comments"]
        and p.get("selftext") != "[removed]"
    ]
    eligible.sort(key=lambda p: p["score"], reverse=True)
    ids = [p["id"] for p in eligible[: rcfg["comment_max_posts"]]]
    print(f"fetching comments for {len(ids)} of {len(eligible)} eligible posts")
    comments_path = path("raw") / "reddit_comments.jsonl"
    n = reddit.scrape_top_level_comments(ids, comments_path, rcfg["comment_min_score"])
    print(f"wrote {n} new comments to {comments_path}")


if __name__ == "__main__":
    main()
