"""Evaluate the rule and LLM extractors (RQ3a) → reports/tables/extract_eval.md.

uv run python scripts/evaluate_extraction.py

Two settings:
  1. Gold: the 35 gold feature requests, against the feature phrases written during
     annotation. Phrase-level precision/recall/F1 with embedding matching (paraphrases
     count; sensitivity to the threshold reported), best token-F1, and judge grades with
     the annotator's phrases as reference.
  2. Corpus: a random sample of corpus items the deployed classifier calls
     feature_request (gold excluded). No reference phrases: the judge grades from the
     text alone, so detection errors (items that ask for nothing) are part of the score.
The judge (configs: extract.judge) is a different model family from the extractor and
never sees which method produced a list. Outputs:
  reports/tables/extract_eval.md, reports/tables/extract_examples.md
  data/processed/extract_eval.parquet
"""

import pandas as pd

from frminer.annotate import load_gold
from frminer.config import load_config, path
from frminer.extract import judge, llm_extract, rules
from frminer.extract.evaluate import (
    best_token_f1,
    bootstrap_f1,
    item_counts,
    scores,
    sentence_embedder,
    split_gold,
)
from frminer.llm import LLM

METHODS = ("rules", "llm")
THRESHOLDS = (0.5, 0.6, 0.7)
N_EXAMPLES = 15


def run_extractors(texts: pd.Series, llm: LLM, batch_size: int, desc: str) -> pd.DataFrame:
    res = llm_extract.extract(llm, texts, batch_size=batch_size, desc=desc)
    out = pd.DataFrame(index=texts.index)
    out["rules"] = rules.extract_many(texts)
    out["llm"] = [e.features if e is not None else None for e in res.outputs]
    out["llm_competitor"] = [e.competitor if e is not None else None for e in res.outputs]
    out["rules_competitor"] = [", ".join(rules.competitors(t)) or None for t in texts]
    return out


def grade_blind(df: pd.DataFrame, judge_llm: LLM, batch_size: int, refs, desc: str) -> None:
    """Add <method>_grade columns. Both methods' lists are graded in one shuffled run."""
    rows = [
        (i, m, df.at[i, "text"], df.at[i, m], refs[i] if refs is not None else None)
        for i in df.index
        for m in METHODS
        if df.at[i, m] is not None
    ]
    rows = pd.DataFrame(rows, columns=["i", "method", "text", "extracted", "ref"])
    rows = rows.sample(frac=1, random_state=0)
    res = judge.grade(
        judge_llm,
        rows["text"].tolist(),
        rows["extracted"].tolist(),
        rows["ref"].tolist() if refs is not None else None,
        batch_size=batch_size,
        desc=desc,
    )
    rows["grade"] = res.outputs
    for m in METHODS:
        sub = rows[rows["method"] == m].set_index("i")["grade"]
        df[f"{m}_grade"] = df.index.map(sub)


def grade_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for m in METHODS:
        g = df[f"{m}_grade"].dropna()
        shares = g.value_counts(normalize=True)
        out[m] = {f"judge_{k}": float(shares.get(k, 0.0)) for k in judge.GRADES}
        out[m]["judge_n"] = len(g)
    return out


def main() -> None:
    cfg = load_config()["extract"]
    ecfg = cfg["eval"]
    llm, judge_llm = LLM(cfg["llm"]["profile"]), LLM(cfg["judge"]["profile"])
    embed = sentence_embedder(ecfg["embed_model"])

    # 1. Gold feature requests.
    gold = load_gold(path("annotations"))
    gold = gold[gold["label"] == "feature_request"].reset_index(drop=True)
    gold["reference"] = gold["features"].map(split_gold)
    gold = gold.join(run_extractors(gold["text"], llm, cfg["llm"]["batch_size"], "extract gold"))
    grade_blind(gold, judge_llm, cfg["judge"]["batch_size"], gold["reference"], "judge gold")

    rows = []
    for split, sub in [("all", gold), *gold.groupby("split")]:
        has_ref = sub[sub["reference"].str.len() > 0]
        for m in METHODS:
            preds = [p or [] for p in has_ref[m]]
            row = {"setting": f"gold {split}", "method": m, "items": len(has_ref)}
            for t in THRESHOLDS:
                counts = item_counts(preds, has_ref["reference"].tolist(), embed, t)
                s = scores(counts)
                if t == ecfg["threshold"]:
                    row.update(s)
                    lo, hi = bootstrap_f1(counts)
                    row["f1_ci"] = f"[{lo:.2f}, {hi:.2f}]"
                else:
                    row[f"f1@{t}"] = s["f1"]
            row["token_f1"] = best_token_f1(preds, has_ref["reference"].tolist())
            row.update(grade_summary(sub)[m])
            rows.append(row)

    # 2. Corpus sample of predicted feature requests.
    classified = pd.read_parquet(path("processed") / "classified.parquet")
    corpus = pd.read_parquet(path("interim") / "corpus.parquet")[["id", "source", "text"]]
    pool = corpus.merge(classified, on="id")
    pool = pool[(pool["label"] == "feature_request") & ~pool["id"].isin(gold["id"])]
    sample = pool.sample(n=ecfg["corpus_sample"], random_state=ecfg["seed"])
    sample = sample.reset_index(drop=True)
    sample = sample.join(
        run_extractors(sample["text"], llm, cfg["llm"]["batch_size"], "extract sample")
    )
    grade_blind(sample, judge_llm, cfg["judge"]["batch_size"], None, "judge sample")
    for m in METHODS:
        preds = sample[m].dropna()
        rows.append(
            {
                "setting": "corpus sample",
                "method": m,
                "items": len(sample),
                "empty": float((preds.str.len() == 0).mean()),
                "phrases_per_item": float(preds.str.len().mean()),
                **grade_summary(sample)[m],
            }
        )

    table = pd.DataFrame(rows)
    cols = ["setting", "method", "items", "precision", "recall", "f1", "f1_ci"]
    cols += [f"f1@{t}" for t in THRESHOLDS if t != ecfg["threshold"]]
    cols += ["token_f1", "item_hit", "empty", "phrases_per_item"]
    cols += [f"judge_{g}" for g in judge.GRADES]
    tables = path("reports") / "tables"
    with open(tables / "extract_eval.md", "w") as f:
        f.write("# Extraction evaluation\n\n")
        f.write(
            f"**Gold**: the {len(gold)} gold feature requests, against the annotator's "
            f"feature phrases ({int((gold['reference'].str.len() > 0).sum())} items have "
            f"one). Phrases match when the cosine similarity of their "
            f"`{ecfg['embed_model'].split('/')[-1]}` embeddings is ≥ {ecfg['threshold']} "
            "(one-to-one); `f1@x` shows other thresholds. `token_f1`: best lemma overlap "
            "per gold phrase. `item_hit`: items with at least one matched phrase. "
            "`empty`: items with no extracted phrase.\n\n"
            f"**Corpus sample**: {ecfg['corpus_sample']} random corpus items the deployed "
            "classifier labels feature_request (gold excluded), graded from the text alone, "
            "so classifier false positives count against a method unless it extracts "
            "nothing.\n\n"
            f"`judge_*`: share of items graded correct / partial / wrong by "
            f"{judge_llm.model} (blind to the method; for gold it also sees the annotator's "
            f"phrases). Extractor: {llm.model}.\n\n"
        )
        f.write(table[[c for c in cols if c in table]].round(3).to_markdown(index=False))
        f.write("\n")

    with open(tables / "extract_examples.md", "w") as f:
        f.write("# Extraction examples\n\n")
        for title, df, ref in [("Gold", gold, True), ("Corpus sample", sample, False)]:
            f.write(f"## {title}\n\n")
            for _, r in df.head(N_EXAMPLES).iterrows():
                text = " ".join(str(r["text"]).split())
                f.write(f"> {text[:300]}{'…' if len(text) > 300 else ''}\n\n")
                if ref:
                    f.write(f"- annotator: {r['reference']}\n")
                for m in METHODS:
                    comp = r[f"{m}_competitor"]
                    extra = f" (competitor: {comp})" if comp else ""
                    f.write(f"- {m} [{r[f'{m}_grade']}]: {r[m]}{extra}\n")
                f.write("\n")

    keep = ["id", "source", "text", *METHODS, "llm_competitor", "rules_competitor"]
    keep += [f"{m}_grade" for m in METHODS]
    out = pd.concat(
        [
            gold.assign(setting="gold")[["setting", *keep, "reference"]],
            sample.assign(setting="corpus")[["setting", *keep]],
        ]
    )
    out.to_parquet(path("processed") / "extract_eval.parquet", index=False)
    print(table[[c for c in cols if c in table]].round(3).to_string())


if __name__ == "__main__":
    main()
