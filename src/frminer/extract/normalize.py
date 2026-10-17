"""Normalise feature phrases so paraphrases look alike before clustering and matching.

"Please add an option to hide songs!" → "hide song". Lowercase, strip punctuation and
emoji, drop request filler ("please", "add", "an option to", "I wish I could"), drop
articles, and lemmatise ("podcasts" → "podcast").
"""

from __future__ import annotations

import re
from functools import lru_cache

# Leading words that frame the request rather than name the feature. Matched repeatedly,
# longest first, so "please add an option to X" loses all three.
FILLER = [
    "please",
    "pls",
    "plz",
    "kindly",
    "i wish i could",
    "i wish",
    "i want to be able to",
    "i want",
    "i need",
    "we need",
    "we want",
    "would love",
    "would like",
    "to be able to",
    "be able to",
    "able to",
    "allow users to",
    "allow us to",
    "allow me to",
    "let users",
    "let us",
    "let me",
    "give us",
    "give users",
    "add",
    "adding",
    "introduce",
    "implement",
    "include",
    "the option to",
    "an option to",
    "option to",
    "the ability to",
    "ability to",
    "a way to",
    "way to",
    "a feature to",
    "a feature that",
    "feature to",
    "an option for",
    "option for",
    "a feature for",
    "feature for",
    "the",
    "a",
    "an",
    "some",
    "more",
]
_FILLER = sorted(FILLER, key=len, reverse=True)
ARTICLES = {"a", "an", "the"}


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean(phrase: str) -> str:
    """Lowercase, keep letters/digits, strip leading filler and articles (no lemmas)."""
    text = re.sub(r"[^a-z0-9'&+/ -]+", " ", str(phrase).lower())
    text = re.sub(r"\s+", " ", text.replace("/", " / ")).strip(" -'/")
    changed = True
    while changed and text:
        changed = False
        for f in _FILLER:
            if text == f:
                return ""
            if text.startswith(f + " "):
                text, changed = text[len(f) + 1 :], True
                break
    return " ".join(w for w in text.split() if w not in ARTICLES)


def normalize(phrase: str) -> str:
    """`clean` plus lemmatisation: the canonical key used for counting and clustering."""
    text = clean(phrase)
    if not text:
        return ""
    return " ".join(t.lemma_ for t in _nlp()(text) if not t.is_space)


def normalize_many(phrases: list[str]) -> list[str]:
    cleaned = [clean(p) for p in phrases]
    docs = _nlp().pipe(cleaned, batch_size=256)
    return [
        " ".join(t.lemma_ for t in d if not t.is_space) if c else ""
        for c, d in zip(cleaned, docs, strict=True)
    ]
