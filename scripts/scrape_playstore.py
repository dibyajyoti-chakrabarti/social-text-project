"""Scrape Spotify Google Play reviews for the configured window (resumable).

uv run python scripts/scrape_playstore.py
"""

from datetime import UTC, datetime

from frminer.config import load_config, path
from frminer.ingest import playstore


def main() -> None:
    cfg = load_config()["data"]
    pcfg = cfg["playstore"]
    since = datetime.fromisoformat(cfg["since"]).replace(tzinfo=UTC)
    for score in (1, 2, 3, 4, 5):
        out = path("raw") / f"playstore_{score}star.jsonl"
        n = playstore.scrape(
            pcfg["app_id"], since, out, lang=pcfg["lang"], country=pcfg["country"], score=score
        )
        print(f"wrote {n} reviews to {out}")


if __name__ == "__main__":
    main()
