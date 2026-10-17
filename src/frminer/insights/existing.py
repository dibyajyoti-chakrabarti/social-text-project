"""Which request themes ask for features Spotify already has?

Each theme is compared (embedding similarity) with a curated catalogue of existing
features (annotations/existing_features.yaml); the LLM then reads the theme and its
closest catalogue entries and decides the status:
  exists        the feature exists for everyone: a discoverability problem
  premium_only  it exists but only in Premium, and users want it free: a paywall
  partly        a limited version exists; users ask for more
  new           not offered
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import yaml

from frminer.config import path
from frminer.llm import LLM
from frminer.llm.batch import run_batched

STATUSES = ("exists", "premium_only", "partly", "new")

SYSTEM = """\
You compare feature requests from Spotify users with features Spotify already offers. \
Each item is one request theme (name, description and typical phrases) with a short \
list of candidate existing features (name, description, availability).

Decide:
- "match": the number of the candidate the theme asks for, or 0 if none fits.
- "status":
  - "exists": the requested capability already exists for the users asking (typically \
available to everyone), so the request signals that users cannot find it.
  - "premium_only": it exists, but only in Premium, and the theme asks for it on the free \
tier or without paying.
  - "partly": a limited version exists and users ask for more than it offers.
  - "new": no candidate offers what is asked (use match 0).
Rules:
- Be strict: a candidate must offer the capability itself, not merely a related one.
- If the matching candidate's availability is "premium", answer "premium_only" unless the \
phrases clearly come from paying Premium users (users on the free tier cannot use it, so \
for them it does not exist).
- Requests to lower prices (cheaper Premium, regional pricing, discounts) are "new" even \
though discounted plan types exist, unless the phrases ask for a plan type that exists \
(e.g. "family plan", "student plan").\
"""

INSTRUCTION = """\
Judge each item below. Reply with JSON only, in the form
{"items": {"<item id>": {"match": <number>, "status": "..."}, ...}}
with one entry for every item id.\
"""


def load_catalogue() -> list[dict]:
    with open(path("annotations") / "existing_features.yaml") as f:
        return yaml.safe_load(f)


def candidates(theme_emb: np.ndarray, cat_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the k most similar catalogue entries per theme (rows sorted by sim)."""
    sim = theme_emb @ cat_emb.T
    return np.argsort(-sim, axis=1)[:, :k], np.sort(sim, axis=1)[:, ::-1][:, :k]


def _format(item: dict) -> str:
    lines = [
        f"Theme: {item['name']}. {item['description']}",
        f"Typical phrases: {'; '.join(item['phrases'])}",
        "Candidates:",
    ]
    lines += [
        f"  {j + 1}. {c['name']}: {c['description']} (availability: {c['availability']})"
        for j, c in enumerate(item["candidates"])
    ]
    return "\n".join(lines)


def build_prompt(items: Sequence[str]) -> str:
    body = "\n\n".join(f"[{i + 1}]\n{_format(json.loads(s))}" for i, s in enumerate(items))
    return INSTRUCTION + "\n\n---\n\nItems:\n\n" + body


def _validator(n: int):
    def validate(data: Any) -> None:
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, dict):
            raise ValueError("missing 'items' object")
        for k in range(n):
            it = items.get(str(k + 1))
            if not isinstance(it, dict) or it.get("status") not in STATUSES:
                raise ValueError(f"item {k + 1}: bad or missing status")
            if not isinstance(it.get("match"), int):
                raise ValueError(f"item {k + 1}: match must be an integer")

    return validate


def verify(llm: LLM, themes: Sequence[dict], *, batch_size: int = 10):
    """`themes`: dicts with name, description, phrases, candidates. One
    {"match": index into candidates or None, "status"} per theme (or None)."""
    items = [json.dumps(t, ensure_ascii=False) for t in themes]

    res = run_batched(
        llm,
        items,
        system=SYSTEM,
        build_prompt=build_prompt,
        validator=_validator,
        parse=lambda data, k: data["items"][str(k + 1)],
        batch_size=batch_size,
        desc="existing features",
    )
    res.outputs = [
        None
        if out is None
        else {
            "match": out["match"] - 1 if 1 <= out["match"] <= len(t["candidates"]) else None,
            "status": out["status"],
        }
        for out, t in zip(res.outputs, themes, strict=True)
    ]
    return res
