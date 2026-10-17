"""Extract requested features from every predicted feature request in the corpus.

uv run python scripts/extract_corpus.py

Input: items the deployed classifier labels feature_request (data/processed/classified.parquet).
The LLM extractor is the pipeline method (rules are kept alongside for comparison). Runs in
chunks and stops cleanly at the daily quota; re-run to continue (finished chunks come
from the LLM cache). Outputs:
  data/processed/extractions.parquet  one row per item: llm/rule features, competitor
  data/processed/requests.parquet     one row per (item, LLM feature) with metadata
"""

import pandas as pd

from frminer.config import load_config, path
from frminer.extract import llm_extract, rules
from frminer.extract.normalize import normalize_many
from frminer.llm import LLM, QuotaExceeded

CHUNK = 1000


def main() -> None:
    cfg = load_config()["extract"]["llm"]
    corpus = pd.read_parquet(path("interim") / "corpus.parquet")
    classified = pd.read_parquet(path("processed") / "classified.parquet")
    items = corpus.merge(classified, on="id")
    items = items[items["label"] == "feature_request"].reset_index(drop=True)
    print(f"{len(items)} predicted feature requests")

    llm = LLM(cfg["profile"])
    feats: list = []
    comps: list = []
    for start in range(0, len(items), CHUNK):
        chunk = items["text"].iloc[start : start + CHUNK]
        try:
            desc = f"extract {start}-{start + len(chunk)}"
            res = llm_extract.extract(llm, chunk, batch_size=cfg["batch_size"], desc=desc)
        except QuotaExceeded as e:
            print(f"stopped at item {start}: {e}. Re-run tomorrow to continue.")
            break
        feats += [e.features if e is not None else None for e in res.outputs]
        comps += [e.competitor if e is not None else None for e in res.outputs]
    items = items.iloc[: len(feats)].copy()
    items["llm_features"] = feats
    items["competitor"] = [rules.canonical_competitor(c) for c in comps]
    items["rule_features"] = rules.extract_many(items["text"])
    items["rule_competitors"] = [rules.competitors(t) for t in items["text"]]

    meta = ["id", "source", "created_at", "rating", "engagement", "app_version", "confidence"]
    ext = items[[*meta, "llm_features", "competitor", "rule_features", "rule_competitors"]]
    ext.to_parquet(path("processed") / "extractions.parquet", index=False)

    req = ext.dropna(subset=["llm_features"]).explode("llm_features")
    req = req.dropna(subset=["llm_features"]).rename(columns={"llm_features": "feature"})
    req = req[[*meta, "feature", "competitor"]].reset_index(drop=True)
    req["feature_norm"] = normalize_many(req["feature"].tolist())
    req = req[req["feature_norm"] != ""]
    req.to_parquet(path("processed") / "requests.parquet", index=False)

    done = ext["llm_features"].notna()
    empty = done & (ext["llm_features"].str.len() == 0)
    n_fr = int((classified["label"] == "feature_request").sum())
    print(
        f"{len(ext)} / {n_fr} items extracted; "
        f"{int(empty.sum())} with no request ({empty.sum() / max(done.sum(), 1):.0%}); "
        f"{len(req)} feature mentions, {req['feature_norm'].nunique()} distinct"
    )
    print(req["feature_norm"].value_counts().head(25).to_string())


if __name__ == "__main__":
    main()
