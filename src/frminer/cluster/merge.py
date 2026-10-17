"""LLM consolidation of over-split clusters.

Density clustering of short phrases splits one request by wording ("remove skip limit"
vs "unlimited song skips"), and centroid similarity cannot tell those apart from merely
related requests. The LLM sees every cluster's top phrases at once and proposes groups
that are the same request; a second model then checks every proposed merge as a single
yes/no question (member vs the group's largest cluster) and only confirmed merges are kept.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from frminer.llm import LLM
from frminer.llm.batch import run_batched

SYSTEM = """\
You consolidate clusters of feature requests that users made about the Spotify app. \
Each cluster is shown as its id and most frequent request phrases with mention counts.

List the groups of clusters that ask for the SAME thing, so that a product manager \
would file them as one request: e.g. "remove skip limit" and "unlimited song skips"; \
"on-demand playback on free tier" and "pick specific songs on free tier". A version of \
a request restricted to the free tier is the same request. Do NOT merge clusters that \
are only related (e.g. "true shuffle" and "turn off shuffle"; "fewer ads" and "remove \
political ads"; "lyrics" and "lyrics translation"). Every cluster may appear in at most \
one group; leave clusters with no duplicate out.\
"""

INSTRUCTION = """\
Reply with JSON only, in the form {"groups": [[<id>, <id>, ...], ...]}.

Clusters:

"""


def build_prompt(ids: Sequence[int], descriptions: Sequence[str]) -> str:
    lines = [f"[{i}] {d}" for i, d in zip(ids, descriptions, strict=True)]
    return INSTRUCTION + "\n".join(lines)


def _validator(valid: set[int]):
    def validate(data: Any) -> None:
        groups = data.get("groups") if isinstance(data, dict) else None
        if not isinstance(groups, list):
            raise ValueError("missing 'groups' list")
        seen: set[int] = set()
        for g in groups:
            if not isinstance(g, list) or not all(isinstance(i, int) for i in g):
                raise ValueError("each group must be a list of integer ids")
            unknown = set(g) - valid
            if unknown:
                raise ValueError(f"unknown ids {sorted(unknown)[:5]}")
            if seen & set(g):
                raise ValueError(f"ids in several groups: {sorted(seen & set(g))[:5]}")
            seen |= set(g)

    return validate


def find_groups(llm: LLM, ids: Sequence[int], descriptions: Sequence[str]) -> list[list[int]]:
    data, _ = llm.complete_json(
        build_prompt(ids, descriptions), system=SYSTEM, validate=_validator(set(ids))
    )
    return [sorted(set(g)) for g in data["groups"] if len(set(g)) >= 2]


def apply_groups(labels: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Relabel every member of a group with the group's smallest id (-1 stays -1)."""
    target = {i: min(g) for g in groups for i in g}
    return np.array([target.get(int(lab), int(lab)) for lab in labels])


VERIFY_SYSTEM = """\
You check proposed merges of feature-request clusters about the Spotify app. Each item \
shows two clusters (A and B) by their most frequent request phrases. Answer true only if \
a product manager would file A and B as the SAME request (a free-tier-only version of a \
request counts as the same request); answer false if they are merely related or about \
different things.\
"""

VERIFY_INSTRUCTION = """\
Judge each item below. Reply with JSON only, in the form
{"same": {"<item id>": true | false, ...}}
with one entry for every item id.\
"""


def _verify_prompt(items: Sequence[str]) -> str:
    body = "\n\n".join(
        f"[{i + 1}]\nA: {json.loads(s)['a']}\nB: {json.loads(s)['b']}" for i, s in enumerate(items)
    )
    return VERIFY_INSTRUCTION + "\n\n---\n\nItems:\n\n" + body


def _verify_validator(n: int):
    def validate(data: Any) -> None:
        same = data.get("same") if isinstance(data, dict) else None
        if not isinstance(same, dict):
            raise ValueError("missing 'same' object")
        for k in range(n):
            if not isinstance(same.get(str(k + 1)), bool):
                raise ValueError(f"item {k + 1}: needs true/false")

    return validate


def verify_groups(
    llm: LLM,
    groups: Sequence[Sequence[int]],
    descriptions: dict[int, str],
    sizes: dict[int, float],
) -> tuple[list[list[int]], list[tuple[int, int, bool]]]:
    """Keep only the members confirmed to be the same request as their group's largest
    cluster. Returns the confirmed groups and every (anchor, member, verdict) checked."""
    pairs = []
    for g in groups:
        anchor = max(g, key=lambda i: sizes[i])
        pairs += [(anchor, i) for i in g if i != anchor]
    if not pairs:
        return [], []
    items = [
        json.dumps({"a": descriptions[a], "b": descriptions[b]}, ensure_ascii=False)
        for a, b in pairs
    ]
    res = run_batched(
        llm,
        items,
        system=VERIFY_SYSTEM,
        build_prompt=_verify_prompt,
        validator=_verify_validator,
        parse=lambda data, k: data["same"][str(k + 1)],
        batch_size=10,
        workers=2,
        desc="verify merges",
    )
    checked = [(a, b, bool(v)) for (a, b), v in zip(pairs, res.outputs, strict=True)]
    kept: dict[int, list[int]] = {}
    for a, b, ok in checked:
        if ok:
            kept.setdefault(a, [a]).append(b)
    return [sorted(g) for g in kept.values()], checked
