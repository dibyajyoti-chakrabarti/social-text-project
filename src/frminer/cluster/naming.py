"""LLM names each cluster from its most frequent member phrases (batched)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from frminer.llm import LLM
from frminer.llm.batch import format_items, run_batched

SYSTEM = """\
You name clusters of feature requests that users made about the Spotify app. Each \
cluster is given as its most frequent request phrases with mention counts.

For every cluster return:
- "name": a short theme name (2-5 words, Title Case) that a product manager would use \
for the shared request, e.g. "Fewer Ads", "Cheaper Premium", "Sleep Timer for Podcasts". \
Name the request, not the complaint.
- "description": one sentence saying what users ask for.
Base both on the bulk of the phrases (weighted by count); ignore stray phrases.\
"""

INSTRUCTION = """\
Name each cluster below. Reply with JSON only, in the form
{"clusters": {"<cluster id>": {"name": "...", "description": "..."}, ...}}
with one entry for every cluster id.\
"""


def describe(phrases: Sequence[str], counts: Sequence[int], top: int = 25) -> str:
    """The text an LLM sees for one cluster: 'phrase (count); ...' by descending count."""
    pairs = sorted(zip(phrases, counts, strict=True), key=lambda pc: -pc[1])[:top]
    return "; ".join(f"{p} ({c})" for p, c in pairs)


def build_prompt(clusters: Sequence[str]) -> str:
    return INSTRUCTION + "\n\n---\n\nClusters:\n\n" + format_items(clusters)


def _validator(n: int):
    def validate(data: Any) -> None:
        clusters = data.get("clusters") if isinstance(data, dict) else None
        if not isinstance(clusters, dict):
            raise ValueError("missing 'clusters' object")
        for k in range(n):
            c = clusters.get(str(k + 1))
            if not isinstance(c, dict) or not all(
                isinstance(c.get(f), str) and c.get(f).strip() for f in ("name", "description")
            ):
                raise ValueError(f"cluster {k + 1}: needs non-empty 'name' and 'description'")

    return validate


def name_clusters(llm: LLM, descriptions: Sequence[str], *, batch_size: int = 10):
    """One {"name", "description"} dict (or None) per cluster description."""
    return run_batched(
        llm,
        descriptions,
        system=SYSTEM,
        build_prompt=build_prompt,
        validator=_validator,
        parse=lambda data, k: {
            f: data["clusters"][str(k + 1)][f].strip() for f in ("name", "description")
        },
        batch_size=batch_size,
        desc="name clusters",
    )
