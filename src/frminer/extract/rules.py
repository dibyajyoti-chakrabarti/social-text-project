"""Rule-based extractor: spaCy dependency patterns around request triggers.

Patterns (per sentence):
  1. object of a request verb: "add [lyrics]", "bring back [the old shuffle]",
     "give us [an option to X]"; verbs that describe a change keep the verb
     ("reduce [prices]" → "reduce prices").
  2. complement of want/need/wish/allow/let: "I wish I could [hide songs]",
     "let us [skip ads]", "want to [download podcasts]".
  3. complement of a capability noun: "option to [X]", "ability to [X]", "mode for [X]".
Competitors are found with a keyword list (configs: extract.competitors).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from frminer.config import load_config
from frminer.extract.normalize import clean

# The object names the feature: keep just the object.
OBJECT_VERBS = {"add", "want", "need", "include", "implement", "introduce", "give", "offer"}
OBJECT_VERBS |= {"enable", "get", "provide", "have", "love", "like", "allow", "let"}
# The verb is part of the request: keep verb + particle + object.
CHANGE_VERBS = {"bring", "restore", "return", "remove", "reduce", "lower", "stop", "make"}
CHANGE_VERBS |= {"change", "put", "increase", "improve", "show"}
COMPLEMENT_HEADS = {"want", "need", "wish", "allow", "let", "love", "like", "able"}
CAPABILITY_NOUNS = {"option", "ability", "way", "feature", "setting", "button", "mode"}
CAPABILITY_NOUNS |= {"toggle", "function", "functionality", "choice", "possibility"}
MODALS = {"should", "could", "would", "can", "'d", "must", "pls"}
PLEASE = {"please", "pls", "plz", "kindly"}
FIRST_PERSON = {"i", "we", "u", "you", "users", "people", "everyone"}
PRONOUNS = {"it", "this", "that", "them", "these", "those", "something", "anything"}
MAX_TOKENS = 8


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner"])


def _span_text(tokens) -> str:
    tokens = sorted(tokens, key=lambda t: t.i)
    out = []
    for t in tokens:
        if t.is_punct and t.text not in {"-", "'", "&"}:
            break  # a phrase ends at the first punctuation mark
        if t.dep_ in {"cc", "mark"} and out:
            break  # "lyrics and ..." / "... because ..." starts another clause
        out.append(t.text)
    return " ".join(out[:MAX_TOKENS])


def _strip_subject(tokens) -> list:
    """Drop the leading subject/modal of a complement clause ("I could hide songs")."""
    tokens = sorted(tokens, key=lambda t: t.i)
    while tokens and (tokens[0].dep_ in {"nsubj", "aux", "neg"} or tokens[0].pos_ == "PRON"):
        tokens = tokens[1:]
    return tokens


def _requested(verb) -> bool:
    """Is this verb used to ask for something? Modal ("should add", "could you add"),
    imperative ("add lyrics", "please bring back"), or wished ("I wish they'd add")."""
    kids = {c.lower_ for c in verb.children}
    if kids & MODALS or kids & PLEASE:
        return True
    if verb.tag_ == "VB" and not any(c.dep_ in {"nsubj", "aux"} for c in verb.children):
        return verb.dep_ in {"ROOT", "conj", "ccomp"}
    head = verb.head
    return verb.dep_ in {"xcomp", "ccomp"} and head.lemma_.lower() in {"wish", "want", "need"}


def _complement_head(tok) -> bool:
    lemma = tok.lemma_.lower()
    kids = {c.lower_ for c in tok.children}
    subj = {c.lower_ for c in tok.children if c.dep_ == "nsubj"}
    if lemma == "wish" or (lemma == "able" and tok.dep_ in {"acomp", "xcomp"}):
        return True
    if lemma in {"love", "like"}:
        return bool(kids & {"would", "'d"})
    if lemma in {"want", "need"}:
        return bool(subj & FIRST_PERSON)
    return lemma in {"allow", "let"} and _requested(tok)


def extract_doc(doc) -> list[str]:
    phrases: list[str] = []
    for tok in doc:
        lemma = tok.lemma_.lower()
        if tok.pos_ == "VERB" and lemma in OBJECT_VERBS | CHANGE_VERBS and _requested(tok):
            for obj in (c for c in tok.children if c.dep_ in {"dobj", "attr"}):
                bare = not any(c.dep_ in {"acl", "relcl", "prep"} for c in obj.children)
                if obj.pos_ == "NUM" or (obj.lower_ in PRONOUNS and bare):
                    continue
                span = list(obj.subtree)
                if lemma in CHANGE_VERBS:
                    prt = [c for c in tok.children if c.dep_ == "prt" or c.lower_ == "back"]
                    span = [tok, *prt, *span]
                phrases.append(_span_text(span))
        if lemma in COMPLEMENT_HEADS and _complement_head(tok):
            for comp in tok.children:
                if comp.dep_ in {"xcomp", "ccomp"} and comp.pos_ in {"VERB", "AUX"}:
                    phrases.append(_span_text(_strip_subject(comp.subtree)))
        if tok.pos_ == "NOUN" and lemma in CAPABILITY_NOUNS:
            for comp in tok.children:
                if comp.dep_ in {"acl", "relcl"} and comp.pos_ == "VERB":
                    phrases.append(_span_text(_strip_subject(comp.subtree)))
                elif comp.dep_ == "prep" and comp.lower_ == "for":
                    phrases.append(_span_text(list(comp.subtree)[1:]))
    return _dedupe(phrases)


def _dedupe(phrases: Iterable[str]) -> list[str]:
    """Clean, drop empties and phrases contained in a longer one, keep first-seen order."""
    cleaned = [c for c in (clean(p) for p in phrases) if c and len(c.split()) >= 1]
    unique = list(dict.fromkeys(cleaned))
    return [p for p in unique if not any(p != q and f" {p} " in f" {q} " for q in unique)]


def extract(text: str) -> list[str]:
    return extract_doc(_nlp()(str(text)))


def extract_many(texts: Iterable[str], batch_size: int = 128) -> list[list[str]]:
    return [extract_doc(d) for d in _nlp().pipe(map(str, texts), batch_size=batch_size)]


@lru_cache(maxsize=1)
def _competitor_patterns() -> list[tuple[str, re.Pattern]]:
    names = load_config()["extract"]["competitors"]
    return [
        (name, re.compile(r"\b(" + "|".join(map(re.escape, aliases)) + r")\b", re.I))
        for name, aliases in names.items()
    ]


def canonical_competitor(name: str | None) -> str | None:
    """Map a competitor name to its canonical form ("YT Music" → "YouTube Music")."""
    if not name:
        return None
    found = competitors(name)
    return found[0] if found else name.strip()


def competitors(text: str) -> list[str]:
    """Competing services mentioned in the text (canonical names, config order)."""
    return [name for name, pat in _competitor_patterns() if pat.search(str(text))]
