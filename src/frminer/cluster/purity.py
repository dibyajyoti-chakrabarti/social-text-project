"""LLM judge for cluster purity: does each sampled member belong to its cluster?

The judge sees a cluster's label (theme name and description for phrase clusters,
top words for BERTopic topics) and its sampled members, and marks each member as
belonging or not. Purity = share of members that belong.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from frminer.llm import LLM
from frminer.llm.batch import run_batched

SYSTEM = """\
You check the quality of clusters of Spotify user feedback. Each cluster comes with a \
label and a numbered list of members (short feature-request phrases or full posts). For \
every member decide whether it asks for the same thing as the bulk of the cluster \
(true) or is an intruder about something else (false). Judge from the members as a \
whole, using the label as a hint; a member belongs if a product manager would file it \
under the same request.\
"""

INSTRUCTION = """\
Judge each cluster below. Reply with JSON only, in the form
{"clusters": {"<cluster id>": {"<member number>": true | false, ...}, ...}}
with an entry for every member of every cluster.\
"""


def _format(cluster: dict) -> str:
    lines = [f"Label: {cluster['label']}"]
    lines += [f"  {j + 1}. {m}" for j, m in enumerate(cluster["members"])]
    return "\n".join(lines)


def build_prompt(clusters: Sequence[str]) -> str:
    """`clusters` are JSON-encoded {"label", "members"} dicts."""
    body = "\n\n".join(f"[{i + 1}]\n{_format(json.loads(c))}" for i, c in enumerate(clusters))
    return INSTRUCTION + "\n\n---\n\nClusters:\n\n" + body


def _validator(sizes: Sequence[int]):
    def make(n: int):
        def validate(data: Any) -> None:
            clusters = data.get("clusters") if isinstance(data, dict) else None
            if not isinstance(clusters, dict):
                raise ValueError("missing 'clusters' object")
            for k in range(n):
                c = clusters.get(str(k + 1))
                if not isinstance(c, dict):
                    raise ValueError(f"cluster {k + 1} missing")
                if not all(isinstance(v, bool) for v in c.values()):
                    raise ValueError(f"cluster {k + 1}: values must be true/false")

        return validate

    return make


def judge(
    llm: LLM,
    labels: Sequence[str],
    members: Sequence[list[str]],
    *,
    batch_size: int = 3,
    desc: str | None = None,
):
    """One list of booleans (member belongs?) per cluster, or None if judging failed.
    Missing member keys count as not judged (None)."""
    items = [
        json.dumps({"label": lab, "members": mem}, ensure_ascii=False)
        for lab, mem in zip(labels, members, strict=True)
    ]
    sizes = [len(m) for m in members]

    def parse(data: Any, k: int) -> dict:
        return data["clusters"][str(k + 1)]

    res = run_batched(
        llm,
        items,
        system=SYSTEM,
        build_prompt=build_prompt,
        validator=_validator(sizes),
        parse=parse,
        batch_size=batch_size,
        workers=2,
        desc=desc,
    )
    res.outputs = [
        None if out is None else [out.get(str(j + 1)) for j in range(n)]
        for out, n in zip(res.outputs, sizes, strict=True)
    ]
    return res
