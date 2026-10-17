"""LLM extractor: the requested feature(s) of each item as short canonical phrases, plus
the competitor the user cites, in batched JSON (see frminer.llm.batch)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from frminer.llm import LLM
from frminer.llm.batch import format_items, run_batched

SYSTEM = """\
You read user feedback about the Spotify app (Google Play reviews and Reddit posts) and \
extract the features the user asks for.

For every item return:
- "features": a list of the distinct things the user asks Spotify to add, change, restore \
or remove. Write each as a short canonical phrase (2-6 words, lowercase English) that names \
the capability itself, the way a product manager would file it: "sleep timer for podcasts", \
"hide songs from playlists", "cheaper premium plan", "fewer ads", "bring back old shuffle", \
"lyrics on free tier". Leave out filler such as "please", "add", "an option to".
  - Make implicit requests explicit: "why can't I hide songs??" → "hide songs"; \
"Apple Music has lyrics, just saying" → "lyrics"; "5 stars if you remove the 3-song \
limit" → "remove 3-song skip limit"; "the rewind button disappeared, why??" → "bring back \
rewind button".
  - One phrase per distinct request; merge restatements of the same request.
  - Requests about ads, prices and plans count.
  - Do NOT list requests to fix something that is broken ("fix the crashes"): those are bug \
reports, not features.
  - Return [] if the item asks for nothing (a pure complaint, bug report, praise or \
question).
- "competitor": the competing service the user cites when asking (e.g. "Apple Music", \
"YouTube Music", "Tidal", "Deezer", "Amazon Music", "SoundCloud"), or null.\
"""

INSTRUCTION = """\
Extract from each item below. Reply with JSON only, in the form
{"items": {"<item id>": {"features": ["..."], "competitor": null}, ...}}
with one entry for every item id.\
"""


def build_prompt(texts: Sequence[str]) -> str:
    return INSTRUCTION + "\n\n---\n\nItems:\n\n" + format_items(texts)


def _validator(n: int):
    def validate(data: Any) -> None:
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, dict):
            raise ValueError("missing 'items' object")
        missing = {str(i + 1) for i in range(n)} - set(map(str, items))
        if missing:
            raise ValueError(f"missing ids {sorted(missing)[:5]}")
        for k in range(n):
            item = items[str(k + 1)]
            feats = item.get("features") if isinstance(item, dict) else None
            if not isinstance(feats, list) or not all(isinstance(f, str) for f in feats):
                raise ValueError(f"item {k + 1}: 'features' must be a list of strings")
            comp = item.get("competitor")
            if comp is not None and not isinstance(comp, str):
                raise ValueError(f"item {k + 1}: 'competitor' must be a string or null")

    return validate


@dataclass
class Extraction:
    features: list[str]
    competitor: str | None


def _parse(data: Any, k: int) -> Extraction:
    item = data["items"][str(k + 1)]
    feats = list(dict.fromkeys(f.strip().lower() for f in item["features"] if f.strip()))
    comp = item.get("competitor")
    comp = comp.strip() if isinstance(comp, str) and comp.strip().lower() != "null" else None
    return Extraction(feats, comp or None)


def extract(
    llm: LLM,
    texts: Sequence[str],
    *,
    batch_size: int = 20,
    workers: int = 4,
    desc: str | None = None,
):
    """One `Extraction` (or None if every attempt failed) per text, in a BatchResult."""
    return run_batched(
        llm,
        texts,
        system=SYSTEM,
        build_prompt=build_prompt,
        validator=_validator,
        parse=_parse,
        batch_size=batch_size,
        workers=workers,
        desc=desc,
    )
