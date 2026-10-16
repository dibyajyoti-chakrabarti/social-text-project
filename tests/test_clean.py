from datetime import UTC, datetime

import pandas as pd

from frminer.ingest import clean

SINCE = datetime(2025, 10, 16, tzinfo=UTC)
UNTIL = datetime(2026, 10, 16, tzinfo=UTC)


def play_raw(rows):
    return pd.DataFrame(
        [
            {
                "reviewId": rid,
                "content": text,
                "score": 5,
                "thumbsUpCount": thumbs,
                "at": at,
                "appVersion": "9.0",
            }
            for rid, text, thumbs, at in rows
        ]
    )


def test_clean_text_scrubs_usernames_and_whitespace():
    raw = "thanks u/some_user and /u/Other-1!   great   tip\r\n\n\n\nok"
    assert clean.clean_text(raw) == "thanks u/[user] and u/[user]! great tip\n\nok"


def test_clean_text_handles_missing_values():
    assert clean.clean_text(None) == ""
    assert clean.clean_text(float("nan")) == ""


def test_anon_id_is_stable_and_hides_raw_id():
    a = clean.anon_id("play", "gp:AOqpTOE123")
    assert a == clean.anon_id("play", "gp:AOqpTOE123")
    assert a.startswith("play_") and "AOqp" not in a


def test_reddit_post_combines_title_and_body_and_drops_removed():
    posts = pd.DataFrame(
        [
            {
                "id": "a",
                "title": "Add a sleep timer",
                "selftext": "for podcasts pls",
                "score": 5,
                "num_comments": 1,
                "link_flair_text": "Feature Request",
                "created_utc": 1780000000,
            },
            {
                "id": "b",
                "title": "spam",
                "selftext": "[removed]",
                "score": 1,
                "num_comments": 0,
                "link_flair_text": None,
                "created_utc": 1780000000,
            },
        ]
    )
    out = clean.from_reddit_posts(posts)
    assert len(out) == 1
    assert out.iloc[0]["text"] == "Add a sleep timer\n\nfor podcasts pls"
    assert out.iloc[0]["flair"] == "Feature Request"


def test_build_corpus_filters_window_length_and_duplicates():
    raw = play_raw(
        [
            ("1", "please add a sleep timer feature", 3, "2026-05-01T10:00:00"),
            ("2", "Please add a sleep timer feature!!", 9, "2026-05-02T10:00:00"),  # dup text
            ("3", "very awesome", 0, "2026-05-01T10:00:00"),  # too short
            ("4", "old review about the app from last year", 0, "2024-01-01T10:00:00"),
        ]
    )
    corpus, stats = clean.build_corpus(
        [clean.from_playstore(raw)], SINCE, UNTIL, min_words=4, detect_language=False
    )
    assert len(corpus) == 1
    assert corpus.iloc[0]["engagement"] == 9  # most-engaged duplicate kept
    assert stats["in_window"] == 3 and stats["min_words"] == 2
    assert list(corpus.columns) == clean.COLUMNS
