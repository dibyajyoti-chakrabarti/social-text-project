"""LLM-as-judge for extractions: grades each (text, extracted phrases) pair as correct,
partial or wrong. The judge is a different model family from the extractor (configs:
extract.judge) and never sees which method produced a list."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from frminer.llm import LLM
from frminer.llm.batch import clip, run_batched

GRADES = ("correct", "partial", "wrong")

SYSTEM = """\
You grade an information-extraction system. Each item has a piece of user feedback about \
the Spotify app and the list of feature requests the system extracted from it (sometimes \
with a reference list written by a human annotator).

Grade each item:
- correct: the list captures every feature the user asks for, each phrase is specific \
enough to file as a product request, and nothing is listed that the user did not ask for. \
An empty list is correct when the text asks for nothing.
- partial: the list captures at least one real request but misses another main one, is \
too vague to act on, or also lists something the user did not ask for.
- wrong: the list misses the user's request entirely, or lists only things the user did \
not ask for (including listing anything when the text asks for nothing).
Judge the meaning, not the wording: paraphrases of the user's request are correct. \
Requests to fix something broken are not feature requests.\
"""

INSTRUCTION = """\
Grade each item below. Reply with JSON only, in the form
{"grades": {"<item id>": "correct" | "partial" | "wrong", ...}}
with one entry for every item id.\
"""


def _format(item: dict) -> str:
    lines = [f"Text: {clip(item['text'])}", f"Extracted: {json.dumps(item['extracted'])}"]
    if item.get("reference") is not None:
        lines.append(f"Reference: {json.dumps(item['reference'])}")
    return "\n".join(lines)


def build_prompt(items: Sequence[str]) -> str:
    """`items` are JSON-encoded dicts (run_batched passes strings)."""
    body = "\n\n".join(f"[{i + 1}]\n{_format(json.loads(s))}" for i, s in enumerate(items))
    return INSTRUCTION + "\n\n---\n\nItems:\n\n" + body


def _validator(n: int):
    def validate(data: Any) -> None:
        grades = data.get("grades") if isinstance(data, dict) else None
        if not isinstance(grades, dict):
            raise ValueError("missing 'grades' object")
        missing = {str(i + 1) for i in range(n)} - set(map(str, grades))
        if missing:
            raise ValueError(f"missing ids {sorted(missing)[:5]}")
        bad = {g for g in grades.values() if g not in GRADES}
        if bad:
            raise ValueError(f"unknown grades {bad}")

    return validate


def grade(
    llm: LLM,
    texts: Sequence[str],
    extracted: Sequence[list[str]],
    references: Sequence[list[str] | None] | None = None,
    *,
    batch_size: int = 10,
    desc: str | None = None,
):
    refs = references if references is not None else [None] * len(texts)
    items = [
        json.dumps({"text": t, "extracted": e, "reference": r}, ensure_ascii=False)
        for t, e, r in zip(texts, extracted, refs, strict=True)
    ]
    return run_batched(
        llm,
        items,
        system=SYSTEM,
        build_prompt=build_prompt,
        validator=_validator,
        parse=lambda data, k: data["grades"][str(k + 1)],
        batch_size=batch_size,
        workers=2,
        desc=desc,
    )
