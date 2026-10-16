"""Reddit collection via the Arctic Shift archive API (no API key required).

Posts are paged oldest-to-newest by `created_utc`, so a re-run resumes from the last
post already on disk. Author names are never written to disk.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import httpx
from tqdm import tqdm

API = "https://arctic-shift.photon-reddit.com/api"
POST_FIELDS = "id,title,selftext,score,num_comments,link_flair_text,created_utc"
COMMENT_FIELDS = "id,body,score,parent_id,link_id,created_utc,author"
BOT_AUTHORS = {"AutoModerator", "[deleted]"}


class _Retryable(Exception):
    pass


class ArcticShift:
    def __init__(self, pause_s: float = 0.3):
        self._client = httpx.Client(timeout=90, headers={"User-Agent": "frminer-uni-project"})
        self.pause_s = pause_s

    def get(self, endpoint: str, **params) -> list[dict]:
        for attempt in range(6):
            try:
                r = self._client.get(f"{API}/{endpoint}", params=params)
                # Arctic Shift answers 422 for transient server-side query timeouts.
                if r.status_code in (422, 429) or r.status_code >= 500:
                    raise _Retryable(f"HTTP {r.status_code}")
                r.raise_for_status()  # other 4xx = our bug: fail fast
                time.sleep(self.pause_s)
                return r.json()["data"]
            except (httpx.TransportError, _Retryable) as e:
                wait = 2**attempt * 3
                tqdm.write(f"{endpoint} failed ({e!r}); retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"Arctic Shift {endpoint} failed repeatedly; re-run to resume")


def _last_created(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last = None
    with open(path) as f:
        for line in f:
            last = line
    return json.loads(last)["created_utc"] if last else None


def scrape_posts(subreddit: str, since: datetime, until: datetime, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    api = ArcticShift()
    after = _last_created(out_path) or int(since.timestamp()) - 1
    before = int(until.timestamp())
    written = 0
    bar = tqdm(desc=f"r/{subreddit} posts", unit="post")
    with open(out_path, "a") as out:
        while True:
            page = api.get(
                "posts/search",
                subreddit=subreddit,
                after=after,
                before=before,
                sort="asc",
                limit=100,
                fields=POST_FIELDS,
            )
            if not page:
                break
            for p in page:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
            out.flush()
            written += len(page)
            bar.update(len(page))
            after = page[-1]["created_utc"]
            bar.set_postfix(at=datetime.fromtimestamp(after).date().isoformat())
    bar.close()
    return written


def scrape_top_level_comments(post_ids: list[str], out_path: Path, min_score: int) -> int:
    """Fetch top-level comments for the given posts. Resumable per post."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = out_path.with_suffix(".done.txt")
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [pid for pid in post_ids if pid not in done]
    api = ArcticShift()
    written = 0
    with open(out_path, "a") as out, open(done_path, "a") as done_f:
        for pid in tqdm(todo, desc="comments", unit="post"):
            for c in api.get("comments/search", link_id=pid, limit=100, fields=COMMENT_FIELDS):
                top_level = c["parent_id"].startswith("t3_")
                if top_level and c["score"] >= min_score and c["author"] not in BOT_AUTHORS:
                    c.pop("author")
                    out.write(json.dumps(c, ensure_ascii=False) + "\n")
                    written += 1
            done_f.write(pid + "\n")
            out.flush()
            done_f.flush()
    return written
