"""Final detection experiment on the gold TEST split (RQ1 + RQ2) → reports/.

uv run python scripts/evaluate_detection.py [--no-setfit]

Every system is configured from dev results only; this script is the single place where
test labels are read. Outputs:
  reports/tables/detect_test.md       metrics with 95% bootstrap CIs, cost and latency
  reports/tables/detect_test.csv
  reports/tables/detect_test_per_class.md
  reports/figures/detect_confusion_*.png
  data/processed/test_preds.parquet
"""

import sys
import time

import pandas as pd

from frminer import viz
from frminer.annotate import load_gold
from frminer.config import path
from frminer.detect import llm_classify, ml, rules
from frminer.detect.evaluate import bootstrap_metrics, confusion, per_class
from frminer.llm import LLM

LLM_SYSTEMS = {
    # name: (profile, prompt, few_shot)
    "LLM zero-shot (prompt v1)": ("primary", "v1", False),
    "LLM zero-shot": ("primary", "v2", False),
    "LLM few-shot": ("primary", "v2", True),
    "LLM-2 zero-shot": ("secondary", "v2", False),
}


def main() -> None:
    ann = path("annotations")
    dev, test = load_gold(ann, "dev"), load_gold(ann, "test")
    silver = pd.read_parquet(path("processed") / "silver.parquet")
    examples = llm_classify.pick_examples(dev)
    preds = pd.DataFrame({"id": test["id"], "source": test["source"], "gold": test["label"]})
    info: dict[str, dict] = {}

    def timed(name: str, fn, **extra) -> None:
        start = time.perf_counter()
        preds[name] = fn()
        info[name] = {"s_per_item": (time.perf_counter() - start) / len(test), **extra}

    timed("Rules", lambda: rules.classify_many(test["text"]), train="none")

    for name, (profile, prompt, few) in LLM_SYSTEMS.items():
        llm = LLM(profile)
        res = llm_classify.classify(
            llm, test["text"], examples=examples if few else None, prompt=prompt, desc=name
        )
        preds[name] = res.labels
        info[name] = {
            "train": f"{len(examples)} dev examples" if few else "none",
            "model": llm.model,
            "s_per_item": res.latency_s / len(test),
            "tokens_per_item": (res.prompt_tokens + res.completion_tokens) / len(test),
            "failed_items": sum(lab is None for lab in res.labels),
        }

    for train_name, data in {"gold-dev": dev, "silver": silver}.items():
        trainers = {"TF-IDF+LR": lambda d: ml.train_tfidf_lr(d["text"], d["label"])}
        if "--no-setfit" not in sys.argv:
            trainers["SetFit"] = lambda d, n=train_name: ml.train_setfit(
                d["text"], d["label"], max_steps=SETFIT_STEPS[n]
            )
        for method, train in trainers.items():
            start = time.perf_counter()
            model = train(data)
            train_s = time.perf_counter() - start
            timed(
                f"{method} ({train_name})",
                lambda m=model: ml.predict(m, test["text"]),
                train=f"{len(data)} {train_name}",
                train_s=train_s,
            )

    systems = [c for c in preds.columns if c not in ("id", "source", "gold")]
    rows, classes = [], []
    for s in systems:
        scored = preds[preds[s].notna()]
        b = bootstrap_metrics(scored["gold"], scored[s])
        row = {"system": s, **info[s], "n": len(scored)}
        for metric in b.index:
            row[metric] = b.loc[metric, "value"]
            row[f"{metric}_ci"] = f"[{b.loc[metric, 'ci_low']:.2f}, {b.loc[metric, 'ci_high']:.2f}]"
        rows.append(row)
        classes.append(per_class(scored["gold"], scored[s])["f1"].rename(s))
    table = pd.DataFrame(rows).set_index("system")

    tables = path("reports") / "tables"
    table.to_csv(tables / "detect_test.csv")
    preds.to_parquet(path("processed") / "test_preds.parquet", index=False)
    show = [
        "train",
        "macro_f1",
        "macro_f1_ci",
        "accuracy",
        "kappa",
        "fr_precision",
        "fr_recall",
        "fr_f1",
        "fr_f1_ci",
        "s_per_item",
        "tokens_per_item",
        "train_s",
    ]
    with open(tables / "detect_test.md", "w") as f:
        f.write(f"# Detection results on gold test (n={len(test)})\n\n")
        f.write(
            "Point estimates with 95% bootstrap CIs (2,000 resamples). `kappa` = Cohen's κ "
            "against the human labels. `s_per_item`: for LLMs, model time per call divided by "
            "the 20 items in the call; for local models, inference time on the laptop "
            "(RTX 5050 GPU for SetFit). `train_s`: training time in seconds.\n\n"
        )
        f.write(table[[c for c in show if c in table]].round(3).to_markdown() + "\n")
    with open(tables / "detect_test_per_class.md", "w") as f:
        f.write("# Per-class F1 on gold test\n\n")
        f.write(pd.concat(classes, axis=1).round(2).to_markdown() + "\n")

    viz.apply_style()
    for s in systems:
        cm = confusion(preds["gold"], preds[s].fillna("other"))
        fig, ax = viz.plt.subplots(figsize=(5.2, 4.4))
        viz.confusion_heatmap(ax, cm, title=s)
        slug = s.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        viz.save(fig, path("reports") / "figures" / f"detect_confusion_{slug}.png")
    print(table[[c for c in show if c in table]].round(3).to_string())


SETFIT_STEPS = {"gold-dev": -1, "silver": 2000}

if __name__ == "__main__":
    main()
