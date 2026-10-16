"""Merge raw Play Store and Reddit data into one cleaned, anonymised corpus.

Unified schema (one row per feedback item):
    id            stable anonymised id ("play_…", "rp_…", "rc_…")
    source        play | reddit_post | reddit_comment
    text          cleaned text (Reddit posts: title + body)
    created_at    UTC timestamp
    rating        1–5 stars (Play only)
    engagement    thumbs-up (Play) or score/upvotes (Reddit)
    flair         Reddit post flair (comments inherit their post's flair)
    app_version   Play only
    parent_id     Reddit comments: id of the post they reply to
    n_words       whitespace token count
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

REMOVED = {"[removed]", "[deleted]", ""}
USER_MENTION = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+")
WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")
COLUMNS = [
    "id",
    "source",
    "text",
    "created_at",
    "rating",
    "engagement",
    "flair",
    "app_version",
    "parent_id",
    "n_words",
]


def anon_id(prefix: str, raw_id: str) -> str:
    return f"{prefix}_{hashlib.sha1(raw_id.encode()).hexdigest()[:12]}"


def clean_text(text: object) -> str:
    if not isinstance(text, str):  # None / NaN for empty reviews
        return ""
    text = USER_MENTION.sub("u/[user]", text)
    text = text.replace("\r\n", "\n").replace("&amp;", "&").replace("&gt;", ">")
    text = WHITESPACE.sub(" ", text)
    text = BLANK_LINES.sub("\n\n", text)
    return text.strip()


def from_playstore(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": raw["reviewId"].map(lambda r: anon_id("play", r)),
            "source": "play",
            "text": raw["content"].map(clean_text),
            "created_at": pd.to_datetime(raw["at"], utc=True),
            "rating": raw["score"].astype("Int64"),
            "engagement": raw["thumbsUpCount"].astype("Int64"),
            "flair": None,
            "app_version": raw["appVersion"],
            "parent_id": None,
        }
    )


def from_reddit_posts(raw: pd.DataFrame) -> pd.DataFrame:
    # "[removed]" = taken down by moderators (mostly spam / rule breaks): drop the post.
    # "[deleted]" = author deleted the body: keep the title.
    raw = raw[raw["selftext"] != "[removed]"]
    body = raw["selftext"].fillna("").where(~raw["selftext"].isin(REMOVED), "")
    text = (raw["title"].fillna("") + "\n\n" + body).map(clean_text)
    return pd.DataFrame(
        {
            "id": raw["id"].map(lambda r: anon_id("rp", r)),
            "source": "reddit_post",
            "text": text,
            "created_at": pd.to_datetime(raw["created_utc"], unit="s", utc=True),
            "rating": pd.array([pd.NA] * len(raw), dtype="Int64"),
            "engagement": raw["score"].astype("Int64"),
            "flair": raw["link_flair_text"],
            "app_version": None,
            "parent_id": None,
        }
    )


def from_reddit_comments(raw: pd.DataFrame, posts_raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[~raw["body"].isin(REMOVED)]
    post_ids = raw["link_id"].str.removeprefix("t3_")
    flair_by_post = posts_raw.set_index("id")["link_flair_text"]
    return pd.DataFrame(
        {
            "id": raw["id"].map(lambda r: anon_id("rc", r)),
            "source": "reddit_comment",
            "text": raw["body"].map(clean_text),
            "created_at": pd.to_datetime(raw["created_utc"], unit="s", utc=True),
            "rating": pd.array([pd.NA] * len(raw), dtype="Int64"),
            "engagement": raw["score"].astype("Int64"),
            "flair": post_ids.map(flair_by_post).values,
            "app_version": None,
            "parent_id": post_ids.map(lambda r: anon_id("rp", r)).values,
        }
    )


def english_mask(texts: pd.Series) -> pd.Series:
    """True where lingua detects English. Uses a common-language subset for speed."""
    languages = [
        Language.ENGLISH,
        Language.SPANISH,
        Language.PORTUGUESE,
        Language.FRENCH,
        Language.GERMAN,
        Language.ITALIAN,
        Language.DUTCH,
        Language.INDONESIAN,
        Language.TAGALOG,
        Language.HINDI,
        Language.ARABIC,
        Language.RUSSIAN,
        Language.TURKISH,
        Language.POLISH,
    ]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    detected = detector.detect_languages_in_parallel_of(texts.tolist())
    return pd.Series([lang == Language.ENGLISH for lang in detected], index=texts.index)


def build_corpus(
    parts: list[pd.DataFrame],
    since: datetime,
    until: datetime,
    min_words: int,
    detect_language: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Concatenate sources, then filter and deduplicate. Returns corpus + filter counts."""
    df = pd.concat(parts, ignore_index=True)
    stats = {"raw": len(df)}

    df = df.drop_duplicates("id")
    stats["after_id_dedup"] = len(df)

    df = df[(df["created_at"] >= pd.Timestamp(since)) & (df["created_at"] < pd.Timestamp(until))]
    stats["in_window"] = len(df)

    df = df.assign(n_words=df["text"].str.split().str.len().fillna(0).astype(int))
    df = df[df["n_words"] >= min_words]
    stats["min_words"] = len(df)

    # Exact-duplicate texts (copy-paste spam, repeated reviews): keep the most engaged copy.
    key = df["text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = (
        df.assign(_key=key)
        .sort_values("engagement", ascending=False, na_position="last")
        .drop_duplicates(["source", "_key"])
        .drop(columns="_key")
    )
    stats["after_text_dedup"] = len(df)

    if detect_language:
        df = df[english_mask(df["text"])]
    stats["english"] = len(df)

    df = df.sort_values("created_at").reset_index(drop=True)
    return df[COLUMNS], stats
