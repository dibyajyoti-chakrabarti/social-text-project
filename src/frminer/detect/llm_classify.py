"""Tier 4: LLM intent classifier (zero-shot / few-shot), batched.

Several items go into one prompt and the model returns one label per item id. Batching
is what makes silver-labelling thousands of items possible on free-tier daily quotas;
its effect on accuracy is measured against batch_size=1 on dev.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from frminer.annotate import LABELS
from frminer.llm import LLM, QuotaExceeded

MAX_WORDS = 300  # longer texts are truncated in the prompt

SYSTEM = """\
You label user feedback about the Spotify app (Google Play reviews, Reddit posts and \
Reddit comments) by its main communicative intent. Assign exactly ONE label per item, \
judging from the text alone.

Labels:
- feature_request: asks for new functionality, or for a change to how existing \
functionality, content, pricing or ads work. Explicit or implicit ("please add", "I wish", \
"bring back", "why can't I…", "Apple Music has X", "5 stars if you add X"). Requests about \
ads, prices and plans count.
- bug_report: something is broken or not working as intended: crashes, errors, playback \
failures, sync problems, login, billing or account malfunctions.
- complaint: negative opinion or frustration with no concrete request and no concrete \
malfunction ("too many ads", "getting worse every update", "overpriced").
- praise: positive opinion with no request.
- question: seeks information or help (how to do something, whether something exists, \
why something happens).
- other: none of the above: news or info sharing, off-topic, spam, jokes, unintelligible, \
or replies whose meaning depends on unseen context.

Decision rules, applied in order (the first that applies wins):
1. Asks for something to be added, changed, restored or removed → feature_request \
(even inside a complaint, bug report or praise). "Please fix X" asks for a repair, not a \
feature. A rhetorical "why is there STILL no X?" is a request; a genuine "is there a way \
to X?" is a question.
2. Describes a concrete malfunction → bug_report.
3. Seeks information or help → question.
4. Negative opinion → complaint.
5. Positive opinion → praise.
6. Otherwise → other.
Sarcasm: label the intended meaning.\
"""

# v2 adds two labelling conventions that the annotator applied consistently in dev but
# that v1 (written from the guidelines alone) did not state.
CONVENTIONS_V2 = """

Conventions (these override the general rules above):
- feature_request needs an actionable ask: the user says what should be added, changed, \
restored or removed ("please…", "add…", "bring back…", "I wish…", "let us…", \
"5 stars if…"). Merely describing a disliked behaviour or design ("I don't like how it \
plays random songs", "too many ads", "they removed X") is a complaint, not a request.
- A post that describes a problem and asks other users for help or whether they have it \
too ("anyone else?", "any fix?", "how do I get rid of…?", "why is X happening?") is a \
question, even though it describes a malfunction. A bare problem description without \
such an ask stays bug_report.\
"""

PROMPTS = {"v1": SYSTEM, "v2": SYSTEM + CONVENTIONS_V2}

INSTRUCTION = """\
Label each item below. Reply with JSON only, in the form
{"labels": {"<item id>": "<label>", ...}}
with one entry for every item id. Allowed labels: feature_request, bug_report, complaint, \
praise, question, other.\
"""


def _clip(text: str) -> str:
    words = str(text).split()
    return " ".join(words[:MAX_WORDS]) + (" …" if len(words) > MAX_WORDS else "")


def _format_items(texts: Sequence[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {_clip(t)}" for i, t in enumerate(texts))


def build_prompt(texts: Sequence[str], examples: pd.DataFrame | None = None) -> str:
    parts = []
    if examples is not None and len(examples):
        shots = "\n\n".join(
            f"Text: {_clip(t)}\nLabel: {lab}"
            for t, lab in zip(examples["text"], examples["label"], strict=True)
        )
        parts.append("Labelled examples:\n\n" + shots)
    parts.append(INSTRUCTION)
    parts.append("Items:\n\n" + _format_items(texts))
    return "\n\n---\n\n".join(parts)


def _validator(n: int):
    def validate(data: Any) -> None:
        labels = data.get("labels") if isinstance(data, dict) else None
        if not isinstance(labels, dict):
            raise ValueError("missing 'labels' object")
        missing = {str(i + 1) for i in range(n)} - set(map(str, labels))
        if missing:
            raise ValueError(f"missing ids {sorted(missing)[:5]}")
        bad = {v for v in labels.values() if v not in LABELS}
        if bad:
            raise ValueError(f"unknown labels {bad}")

    return validate


@dataclass
class BatchResult:
    labels: list[str | None]  # None where every attempt failed
    prompt_tokens: int
    completion_tokens: int
    latency_s: float  # summed model latency per call (the cache stores the original latency)
    n_calls: int
    n_failed: int  # items left unlabelled


RETRY_BATCH_SIZE = 5  # items of a failed batch are retried in batches this small


def _run(
    llm: LLM,
    texts: Sequence[str],
    groups: list[list[int]],
    examples: pd.DataFrame | None,
    prompt: str,
    workers: int,
    desc: str | None,
) -> list[Any]:
    """One result (or exception) per group of item indices."""
    results: list[Any] = [None] * len(groups)
    # Each prompt has a different item count, so validators differ; run one map per size.
    by_size: dict[int, list[int]] = {}
    for j, g in enumerate(groups):
        by_size.setdefault(len(g), []).append(j)
    for size, idxs in by_size.items():
        out = llm.map(
            [build_prompt([texts[i] for i in groups[j]], examples) for j in idxs],
            system=PROMPTS[prompt],
            json_mode=True,
            validate=_validator(size),
            workers=workers,
            desc=desc,
        )
        for j, r in zip(idxs, out, strict=True):
            results[j] = r
    return results


def classify(
    llm: LLM,
    texts: Sequence[str],
    *,
    batch_size: int = 20,
    examples: pd.DataFrame | None = None,
    prompt: str = "v2",
    workers: int = 4,
    desc: str | None = None,
) -> BatchResult:
    """Label every text. Items of a failed batch (e.g. one blocked by a provider safety
    filter) are retried in smaller batches, so one bad item cannot sink its neighbours."""
    texts = list(texts)
    labels: list[str | None] = [None] * len(texts)
    pt = ct = calls = 0
    lat = 0.0
    pending = list(range(len(texts)))
    for size in sorted({batch_size, min(batch_size, RETRY_BATCH_SIZE)}, reverse=True):
        if not pending:
            break
        groups = [pending[i : i + size] for i in range(0, len(pending), size)]
        results = _run(llm, texts, groups, examples, prompt, workers, desc)
        calls += len(groups)
        pending = []
        for group, r in zip(groups, results, strict=True):
            if isinstance(r, QuotaExceeded):
                raise r
            if isinstance(r, Exception):
                pending += group
                continue
            data, resp = r
            for k, i in enumerate(group):
                labels[i] = data["labels"][str(k + 1)]
            pt += resp.prompt_tokens or 0
            ct += resp.completion_tokens or 0
            lat += resp.latency_s
    return BatchResult(labels, pt, ct, lat, calls, len(pending))


def pick_examples(dev: pd.DataFrame, per_class: int = 2, seed: int = 0) -> pd.DataFrame:
    """Few-shot examples: `per_class` short dev items per label (fixed seed)."""
    short = dev[dev["text"].str.split().str.len() <= 60]
    picked = [
        g.sample(min(per_class, len(g)), random_state=seed) for _, g in short.groupby("label")
    ]
    return pd.concat(picked).sample(frac=1, random_state=seed)[["id", "text", "label"]]
