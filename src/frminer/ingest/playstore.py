"""Resumable Google Play review scraper (one stream per star rating).

Reviews are fetched newest-first, one page at a time, and appended to a JSONL file.
The pagination token is pickled after every page, so an interrupted run resumes
where it stopped. Usernames and avatars are never written to disk.
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path

from google_play_scraper import Sort, reviews
from tqdm import tqdm

KEEP = ("reviewId", "content", "score", "thumbsUpCount", "at", "appVersion", "repliedAt")


def _record(r: dict) -> dict:
    out = {k: r.get(k) for k in KEEP}
    for k in ("at", "repliedAt"):
        if out[k] is not None:
            out[k] = out[k].replace(tzinfo=UTC).isoformat()
    return out


def scrape(
    app_id: str,
    since: datetime,
    out_path: Path,
    lang: str = "en",
    country: str = "us",
    page_size: int = 200,
    pause_s: float = 0.5,
    score: int | None = None,
) -> int:
    """Scrape reviews newer than `since` into `out_path`. Returns reviews written this run.

    Google caps newest-first pagination at roughly 34k reviews per stream. Filtering by
    star rating (`score`) gives five independent streams, each with its own cap, so
    scraping per rating reaches much further back than one unfiltered stream.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(".state.pkl")
    token = None
    if state_path.exists():
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        if state.get("done"):
            print(f"{out_path.name}: already complete")
            return 0
        token = state["token"]

    written = 0
    bar = tqdm(desc=f"play:{app_id}:{score or 'all'}★", unit="rev")
    with open(out_path, "a") as out:
        while True:
            for attempt in range(6):
                try:
                    page, token = reviews(
                        app_id,
                        lang=lang,
                        country=country,
                        sort=Sort.NEWEST,
                        count=page_size,
                        filter_score_with=score,
                        continuation_token=token,
                    )
                    break
                except Exception as e:  # network hiccups / throttling
                    wait = 2**attempt * 5
                    tqdm.write(f"page failed ({e!r}); retrying in {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError("Play Store scraping failed repeatedly; re-run to resume")

            fresh = [r for r in page if r["at"].replace(tzinfo=UTC) >= since]
            for r in fresh:
                out.write(json.dumps(_record(r), ensure_ascii=False) + "\n")
            out.flush()
            written += len(fresh)
            bar.update(len(fresh))
            if page:
                bar.set_postfix(oldest=page[-1]["at"].date().isoformat())

            done = not page or len(fresh) < len(page) or token is None or token.token is None
            with open(state_path, "wb") as f:
                pickle.dump({"token": token, "done": done}, f)
            if done:
                break
            time.sleep(pause_s)
    bar.close()
    return written
