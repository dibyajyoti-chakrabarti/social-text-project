"""Method development on the gold DEV split only → reports/tables/detect_dev.md.

uv run python scripts/detect_dev.py [config ...]

Runs the rule classifier and several LLM configurations on dev, so that prompt, batch
size and few-shot choices are made without touching the test split. LLM calls are
cached, so re-running is free. Few-shot configs are scored on the dev items that are
not used as examples.
"""

import sys
import time

import pandas as pd

from frminer.annotate import load_gold
from frminer.config import path
from frminer.detect import llm_classify, rules
from frminer.detect.evaluate import point_metrics
from frminer.llm import LLM

# name -> (profile, prompt version, batch_size, few_shot)
LLM_CONFIGS = {
    "llm_v1_zero_b20": ("primary", "v1", 20, False),
    "llm_v1_few_b20": ("primary", "v1", 20, True),
    "llm_v2_zero_b20": ("primary", "v2", 20, False),
    "llm_v2_few_b20": ("primary", "v2", 20, True),
    "llm_v2_zero_b1": ("primary", "v2", 1, False),
    "llm2_v2_zero_b20": ("secondary", "v2", 20, False),
    "llm2_v2_few_b20": ("secondary", "v2", 20, True),
}


def main() -> None:
    wanted = sys.argv[1:] or ["rules", *LLM_CONFIGS]
    dev = load_gold(path("annotations"), "dev")
    examples = llm_classify.pick_examples(dev)
    preds = pd.DataFrame({"id": dev["id"], "gold": dev["label"]})
    rows = []

    if "rules" in wanted:
        start = time.perf_counter()
        preds["rules"] = rules.classify_many(dev["text"])
        rows.append(
            {
                "method": "rules",
                "n": len(dev),
                "s_per_item": (time.perf_counter() - start) / len(dev),
            }
        )

    for name, (profile, version, batch_size, few) in LLM_CONFIGS.items():
        if name not in wanted:
            continue
        llm = LLM(profile)
        res = llm_classify.classify(
            llm,
            dev["text"],
            batch_size=batch_size,
            examples=examples if few else None,
            prompt=version,
            desc=name,
        )
        preds[name] = res.labels
        rows.append(
            {
                "method": name,
                "model": llm.model,
                "n": len(dev),
                "calls": res.n_calls,
                "failed_calls": res.n_failed,
                "tokens_per_item": (res.prompt_tokens + res.completion_tokens) / len(dev),
            }
        )

    table = []
    for row in rows:
        m = row["method"]
        scored = preds[preds[m].notna()]
        if "few" in m:
            scored = scored[~scored["id"].isin(examples["id"])]
        metrics = point_metrics(scored["gold"], scored[m])
        table.append({**row, "n_scored": len(scored), **metrics})
    table = pd.DataFrame(table).set_index("method")

    out_dir = path("processed")
    old = out_dir / "dev_preds.parquet"
    if old.exists():  # keep columns from configs not re-run this time
        prev = pd.read_parquet(old)
        for c in prev.columns.difference(preds.columns):
            preds[c] = prev[c].values
    preds.to_parquet(old, index=False)

    out = path("reports") / "tables" / "detect_dev.md"
    prev_table = (
        pd.read_csv(out.with_suffix(".csv"), index_col=0)
        if out.with_suffix(".csv").exists()
        else None
    )
    if prev_table is not None:
        table = pd.concat([prev_table.drop(table.index, errors="ignore"), table])
    table.to_csv(out.with_suffix(".csv"))
    with open(out, "w") as f:
        f.write("# Detection: development results on gold dev (n=100)\n\n")
        f.write(f"Few-shot rows exclude the {len(examples)} dev items used as examples.\n\n")
        f.write(table.round(3).to_markdown() + "\n")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
