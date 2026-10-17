"""Group extracted feature phrases into themes (RQ3b) → data/processed/themes*.parquet.

uv run python scripts/cluster_requests.py

Method A (pipeline): distinct phrases → embeddings → UMAP → HDBSCAN over a small grid;
the setting is chosen by a fixed rule (configs: cluster.select) on intrinsic quality, then
noise phrases near a centroid are attached and the LLM names every cluster.
Method B (comparison): BERTopic on the full texts of the items with a request.
Also draws the purity-check sample (annotations/purity_sample.csv): random clusters of
each method with random members, judged by the LLM judge and, for method A, by hand.
Outputs:
  data/processed/themes.parquet          one row per theme (name, description, size)
  data/processed/phrase_themes.parquet   feature phrase → theme
  data/processed/request_themes.parquet  requests.parquet + theme_id
  data/processed/bertopic_topics.parquet item → topic (method B)
  reports/tables/cluster_grid.md, reports/tables/cluster_compare.md
"""

import numpy as np
import pandas as pd

from frminer.cluster import merge, naming, phrases
from frminer.config import load_config, path
from frminer.llm import LLM


def select(grid: pd.DataFrame, rule: dict) -> pd.Series:
    ok = grid[
        (grid["outlier_mentions"] <= rule["max_outlier_mentions"])
        & grid["n_clusters"].between(rule["min_clusters"], rule["max_clusters"])
    ]
    if ok.empty:
        raise SystemExit("no grid setting satisfies cluster.select; widen the grid")
    return ok.sort_values("silhouette", ascending=False).iloc[0]


def bertopic_topics(texts: list[str], emb: np.ndarray, cfg: dict) -> np.ndarray:
    from bertopic import BERTopic
    from sklearn.cluster import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    model = BERTopic(
        umap_model=UMAP(
            n_components=cfg["umap"]["n_components"],
            n_neighbors=cfg["umap"]["n_neighbors"],
            min_dist=0.0,
            metric="cosine",
            random_state=cfg["seed"],
        ),
        hdbscan_model=HDBSCAN(min_cluster_size=cfg["bertopic"]["min_topic_size"]),
        # Stopword-free topic words (labels only; assignments come from UMAP + HDBSCAN).
        vectorizer_model=CountVectorizer(stop_words="english", min_df=2),
        calculate_probabilities=False,
    )
    topics, _ = model.fit_transform(texts, emb)
    return np.asarray(topics), model


def main() -> None:
    cfg = load_config()["cluster"]
    req = pd.read_parquet(path("processed") / "requests.parquet")
    req["phrase"] = req["feature"].str.strip().str.lower()
    table = (
        req.groupby("phrase")
        .agg(mentions=("id", "size"), items=("id", "nunique"))
        .reset_index()
        .sort_values("mentions", ascending=False, ignore_index=True)
    )
    weights = table["mentions"].to_numpy(float)
    print(f"{len(req)} mentions, {len(table)} distinct phrases")

    emb = phrases.embed(table["phrase"].tolist(), cfg["embed_model"])
    x = phrases.reduce(emb, cfg["umap"]["n_components"], cfg["umap"]["n_neighbors"], cfg["seed"])

    rows, runs = [], {}
    for mcs in cfg["grid"]["min_cluster_size"]:
        for ms in cfg["grid"]["min_samples"]:
            raw = phrases.hdbscan_labels(x, mcs, ms)
            labels = phrases.assign_outliers(emb, raw, cfg["outlier_threshold"])
            q = phrases.quality(emb, labels, weights)
            rows.append({"min_cluster_size": mcs, "min_samples": ms, **vars(q)})
            runs[(mcs, ms)] = labels
            print(rows[-1])
    grid = pd.DataFrame(rows)
    best = select(grid, cfg["select"])
    labels = runs[(int(best["min_cluster_size"]), int(best["min_samples"]))]
    print("selected", best.to_dict())

    # Consolidate clusters that are the same request in different words.
    table["pre_merge"] = labels
    pre = pd.DataFrame({"phrase": table["phrase"], "mentions": table["mentions"], "c": labels})
    pre = pre[pre["c"] >= 0].groupby("c")
    ids = list(pre.groups)
    descs = {
        c: naming.describe(g["phrase"].tolist(), g["mentions"].tolist(), top=6) for c, g in pre
    }
    sizes = {c: float(g["mentions"].sum()) for c, g in pre}
    proposed = merge.find_groups(LLM(cfg["naming"]["profile"]), ids, [descs[c] for c in ids])
    groups, checked = merge.verify_groups(LLM(cfg["judge"]["profile"]), proposed, descs, sizes)
    labels = merge.apply_groups(labels, groups)
    q_merged = phrases.quality(emb, labels, weights)
    print(f"merged {sum(len(g) for g in groups)} clusters into {len(groups)}: {vars(q_merged)}")

    # Renumber themes by mention volume (0 = biggest) and name them.
    table["cluster"] = labels
    vol = table[table["cluster"] >= 0].groupby("cluster")["mentions"].sum()
    order = {c: i for i, c in enumerate(vol.sort_values(ascending=False).index)}
    table["theme_id"] = table["cluster"].map(order).fillna(-1).astype(int)
    ids, cents = phrases.centroids(emb, table["theme_id"].to_numpy())
    cent_of = dict(zip(ids, cents, strict=True))
    table["sim"] = [
        float(e @ cent_of[t]) if t >= 0 else np.nan
        for e, t in zip(emb, table["theme_id"], strict=True)
    ]

    themes = (
        table[table["theme_id"] >= 0]
        .groupby("theme_id")
        .agg(n_phrases=("phrase", "size"), mentions=("mentions", "sum"))
        .reset_index()
    )
    members = table[table["theme_id"] >= 0].groupby("theme_id")
    themes["top_phrases"] = [
        naming.describe(g["phrase"].tolist(), g["mentions"].tolist(), top=8).split("; ")
        for _, g in members
    ]
    descriptions = [
        naming.describe(g["phrase"].tolist(), g["mentions"].tolist()) for _, g in members
    ]
    named = naming.name_clusters(LLM(cfg["naming"]["profile"]), descriptions)
    themes["name"] = [
        n["name"] if n else f"Theme {i}"
        for i, n in zip(themes["theme_id"], named.outputs, strict=True)
    ]
    themes["description"] = [n["description"] if n else "" for n in named.outputs]
    themed = req.merge(table[["phrase", "theme_id"]], on="phrase", how="left")
    themes["items"] = themes["theme_id"].map(themed.groupby("theme_id")["id"].nunique())

    proc = path("processed")
    themes.to_parquet(proc / "themes.parquet", index=False)
    table.drop(columns=["cluster", "pre_merge"]).to_parquet(
        proc / "phrase_themes.parquet", index=False
    )
    themed.drop(columns="phrase").to_parquet(proc / "request_themes.parquet", index=False)

    # Method B: BERTopic on the item texts.
    corpus = pd.read_parquet(path("interim") / "corpus.parquet")[["id", "text"]]
    items = corpus[corpus["id"].isin(req["id"])].reset_index(drop=True)
    item_emb = phrases.embed(items["text"].tolist(), cfg["embed_model"])
    topics, topic_model = bertopic_topics(items["text"].tolist(), item_emb, cfg)
    items["topic"] = topics
    items[["id", "topic"]].to_parquet(proc / "bertopic_topics.parquet", index=False)
    ones = np.ones(len(items))
    qa = phrases.quality(emb, table["theme_id"].to_numpy(), weights)
    qb = phrases.quality(item_emb, topics, ones)
    info = topic_model.get_topic_info()
    topic_words = {
        int(r.Topic): ", ".join(w for w, _ in topic_model.get_topic(r.Topic)[:6])
        for r in info.itertuples()
        if r.Topic >= 0
    }

    tables = path("reports") / "tables"
    with open(tables / "cluster_grid.md", "w") as f:
        f.write("# Phrase clustering: HDBSCAN grid\n\n")
        f.write(
            f"{len(table)} distinct phrases ({len(req)} mentions), `{cfg['embed_model']}` "
            f"embeddings, UMAP to {cfg['umap']['n_components']} dims. Noise phrases with "
            f"cosine ≥ {cfg['outlier_threshold']} to a centroid are attached after "
            "clustering. `outlier_mentions`: share of mentions left without a theme; "
            "`silhouette`: cosine, original embeddings. Selection rule: highest "
            f"silhouette with outlier_mentions ≤ {cfg['select']['max_outlier_mentions']} "
            f"and {cfg['select']['min_clusters']}–{cfg['select']['max_clusters']} "
            f"clusters → min_cluster_size={int(best['min_cluster_size'])}, "
            f"min_samples={int(best['min_samples'])}.\n\n"
        )
        f.write(grid.round(3).to_markdown(index=False) + "\n\n")
        n_ok = sum(ok for *_, ok in checked)
        f.write(
            f"LLM consolidation: {load_config()['models'][cfg['naming']['profile']]['model']} "
            f"proposed {len(proposed)} groups of same-request clusters "
            f"({len(checked)} merges); "
            f"{load_config()['models'][cfg['judge']['profile']]['model']} confirmed {n_ok} "
            f"of them as pairwise yes/no questions. Confirmed merges joined "
            f"{sum(len(g) for g in groups)} clusters into {len(groups)}, leaving "
            f"{q_merged.n_clusters} themes (silhouette {q_merged.silhouette:.3f}).\n\n"
            "Kept merges:\n\n"
        )
        for g in groups:
            names = [
                "; ".join(table[table["pre_merge"] == i].nlargest(2, "mentions")["phrase"])
                for i in g
            ]
            f.write("- " + " + ".join(f"[{n}]" for n in names) + "\n")
        f.write("\nRejected merges (largest cluster ← rejected member):\n\n")
        top2 = lambda i: "; ".join(  # noqa: E731
            table[table["pre_merge"] == i].nlargest(2, "mentions")["phrase"]
        )
        for a, b, ok in checked:
            if not ok:
                f.write(f"- [{top2(a)}] ← [{top2(b)}]\n")
    with open(tables / "cluster_compare.md", "w") as f:
        f.write("# Phrase clusters (A) vs BERTopic on item texts (B)\n\n")
        compare = pd.DataFrame(
            [
                {"method": "A: phrases + UMAP + HDBSCAN", "unit": "distinct phrase", **vars(qa)},
                {"method": "B: BERTopic on texts", "unit": "item", **vars(qb)},
            ]
        )
        f.write(
            "Silhouettes are computed in each method's own embedding space (phrases vs "
            "texts), so they are not directly comparable; purity (cluster_eval.md) is the "
            "shared measure.\n\n"
        )
        f.write(compare.round(3).to_markdown(index=False) + "\n\n## Top themes (A)\n\n")
        show = themes.head(30)[["theme_id", "name", "items", "mentions", "n_phrases"]]
        f.write(show.to_markdown(index=False) + "\n\n## Top topics (B)\n\n")
        sizes = items[items["topic"] >= 0]["topic"].value_counts().head(30)
        f.write(
            pd.DataFrame(
                {"topic": sizes.index, "items": sizes.values, "words": sizes.index.map(topic_words)}
            ).to_markdown(index=False)
            + "\n"
        )

    # Purity-check sample: random clusters, random members (uniform over distinct members).
    rng = np.random.default_rng(cfg["seed"])
    n_clusters, n_members = cfg["purity"]["clusters"], cfg["purity"]["members"]
    rows = []
    eligible = themes[themes["n_phrases"] >= n_members]["theme_id"].to_numpy()
    for rank, t in enumerate(
        rng.choice(eligible, size=min(n_clusters, len(eligible)), replace=False)
    ):
        th = themes.set_index("theme_id").loc[t]
        pool = table[table["theme_id"] == t]["phrase"].to_numpy()
        for m in rng.choice(pool, size=n_members, replace=False):
            rows.append(("A", rank, int(t), th["name"], th["description"], m))
    sizes = items[items["topic"] >= 0]["topic"].value_counts()
    eligible = sizes[sizes >= n_members].index.to_numpy()
    for rank, t in enumerate(
        rng.choice(eligible, size=min(n_clusters, len(eligible)), replace=False)
    ):
        pool = items[items["topic"] == t]["text"].to_numpy()
        for m in rng.choice(pool, size=n_members, replace=False):
            rows.append(
                ("B", rank, int(t), topic_words[int(t)], "", " ".join(str(m).split())[:400])
            )
    sample = pd.DataFrame(
        rows, columns=["method", "rank", "cluster", "name", "description", "member"]
    )
    sample.to_csv(path("annotations") / "purity_sample.csv", index=False)
    print(compare.round(3).to_string())
    print(themes.head(25)[["theme_id", "name", "items", "n_phrases"]].to_string())


if __name__ == "__main__":
    main()
