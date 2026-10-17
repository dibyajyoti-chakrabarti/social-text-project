"""Run one LLM task over many texts, several texts per prompt.

The model sees items numbered [1]..[n] and answers with JSON keyed by item number.
Batching is what makes thousands of items feasible on free-tier daily quotas. Items of a
failed batch (e.g. one blocked by a provider safety filter) are retried in smaller
batches, so one bad item cannot sink its neighbours.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from frminer.llm.client import LLM, QuotaExceeded

MAX_WORDS = 300  # longer texts are truncated in the prompt
RETRY_BATCH_SIZE = 5  # items of a failed batch are retried in batches this small


def clip(text: str) -> str:
    words = str(text).split()
    return " ".join(words[:MAX_WORDS]) + (" …" if len(words) > MAX_WORDS else "")


def format_items(texts: Sequence[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {clip(t)}" for i, t in enumerate(texts))


@dataclass
class BatchResult:
    outputs: list[Any]  # one parsed output per text; None where every attempt failed
    prompt_tokens: int
    completion_tokens: int
    latency_s: float  # summed model latency per call (the cache stores the original latency)
    n_calls: int
    n_failed: int  # items left without output


def run_batched(
    llm: LLM,
    texts: Sequence[str],
    *,
    system: str,
    build_prompt: Callable[[list[str]], str],
    validator: Callable[[int], Callable[[Any], None]],
    parse: Callable[[Any, int], Any],
    batch_size: int = 20,
    workers: int = 4,
    desc: str | None = None,
) -> BatchResult:
    """`validator(n)` checks a reply for an n-item prompt; `parse(data, k)` returns the
    output for the k-th item (0-based) of a validated reply."""
    texts = list(texts)
    outputs: list[Any] = [None] * len(texts)
    pt = ct = calls = 0
    lat = 0.0
    pending = list(range(len(texts)))
    for size in sorted({batch_size, min(batch_size, RETRY_BATCH_SIZE)}, reverse=True):
        if not pending:
            break
        groups = [pending[i : i + size] for i in range(0, len(pending), size)]
        results = _map(llm, texts, groups, system, build_prompt, validator, workers, desc)
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
                outputs[i] = parse(data, k)
            pt += resp.prompt_tokens or 0
            ct += resp.completion_tokens or 0
            lat += resp.latency_s
    return BatchResult(outputs, pt, ct, lat, calls, len(pending))


def _map(llm, texts, groups, system, build_prompt, validator, workers, desc) -> list[Any]:
    """One result (or exception) per group of item indices."""
    results: list[Any] = [None] * len(groups)
    # Each prompt has a different item count, so validators differ; run one map per size.
    by_size: dict[int, list[int]] = {}
    for j, g in enumerate(groups):
        by_size.setdefault(len(g), []).append(j)
    for size, idxs in by_size.items():
        out = llm.map(
            [build_prompt([texts[i] for i in groups[j]]) for j in idxs],
            system=system,
            json_mode=True,
            validate=validator(size),
            workers=workers,
            desc=desc,
        )
        for j, r in zip(idxs, out, strict=True):
            results[j] = r
    return results
