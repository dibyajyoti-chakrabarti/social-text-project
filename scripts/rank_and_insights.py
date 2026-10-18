"""Rank themes, find trends and compute insights (RQ4) → data/processed + reports/.

uv run python scripts/rank_and_insights.py

Inputs: themes.parquet and request_themes.parquet from cluster_requests.py.
Outputs:
  data/processed/theme_stats.parquet       per theme: volume, engagement, ratings, growth,
                                           priority, trend
  data/processed/theme_timeseries.parquet  theme × month: items and share of corpus
  data/processed/insights/{platform,ratings,competitors,existing}.parquet
  reports/tables/top_themes.md, trends.md, insights.md
  reports/figures/top_themes.png, trends_emerging.png
"""

import numpy as np
import pandas as pd

from frminer import viz
from frminer.cluster import phrases
from frminer.config import load_config, path
from frminer.insights import compare, existing
from frminer.llm import LLM
from frminer.rank import score, trends


def main() -> None:
    cfg = load_config()
    rcfg, icfg = cfg["rank"], cfg["insights"]
    proc = path("processed")
    themes = pd.read_parquet(proc / "themes.parquet")
    themed = pd.read_parquet(proc / "request_themes.parquet")
    corpus = pd.read_parquet(
        path("interim") / "corpus.parquet", columns=["source", "rating", "created_at"]
    )
    corpus_created = score.trend_items(corpus)["created_at"]  # trends ignore 5★ Play reviews

    # Ranking and trends.
    stats = score.theme_stats(
        themed, corpus_created, weeks=rcfg["window_weeks"], min_support=rcfg["min_support"]
    )
    stats["priority"] = score.priority(stats, rcfg["weights"])
    end = corpus_created.max() + pd.Timedelta(seconds=1)
    tot_prev, tot_recent = score.window_counts(
        corpus_created, score.windows(end, rcfg["window_weeks"])
    )
    stats = trends.classify_trends(stats, tot_prev, tot_recent, **rcfg["trend"])
    stats = themes[["theme_id", "name", "description"]].merge(stats, on="theme_id")
    stats = stats.sort_values("priority", ascending=False, ignore_index=True)
    stats["rank"] = np.arange(1, len(stats) + 1)
    items = score.item_table(themed)
    series = trends.monthly(score.trend_items(items), corpus_created)
    stats.to_parquet(proc / "theme_stats.parquet", index=False)
    series.to_parquet(proc / "theme_timeseries.parquet", index=False)

    # Insights.
    ins = proc / "insights"
    ins.mkdir(exist_ok=True)
    plat = compare.platform(items, icfg["min_items"])
    rat = compare.ratings(items, icfg["min_items"])
    comp = compare.competitors(items)
    for df, name in [(plat, "platform"), (rat, "ratings"), (comp, "competitors")]:
        df.merge(themes[["theme_id", "name"]], on="theme_id", how="left").to_parquet(
            ins / f"{name}.parquet", index=False
        )

    catalogue = existing.load_catalogue()
    model = cfg["cluster"]["embed_model"]
    theme_text = (themes["name"] + ". " + themes["description"]).tolist()
    cat_text = [f"{c['name']}. {c['description']}" for c in catalogue]
    t_emb, c_emb = phrases.embed(theme_text, model), phrases.embed(cat_text, model)
    cand_idx, cand_sim = existing.candidates(t_emb, c_emb, icfg["existing"]["candidates"])
    queries = [
        {
            "name": r.name,
            "description": r.description,
            "phrases": [p.rsplit(" (", 1)[0] for p in r.top_phrases[:6]],
            "candidates": [
                {k: catalogue[j][k] for k in ("name", "description", "availability")}
                for j in cand_idx[i]
            ],
        }
        for i, r in enumerate(themes.itertuples())
    ]
    res = existing.verify(LLM(icfg["existing"]["profile"]), queries)
    ex = themes[["theme_id", "name"]].copy()
    ex["status"] = [o["status"] if o else None for o in res.outputs]
    ex["feature"] = [
        catalogue[cand_idx[i][o["match"]]]["name"] if o and o["match"] is not None else None
        for i, o in enumerate(res.outputs)
    ]
    ex["availability"] = [
        catalogue[cand_idx[i][o["match"]]]["availability"] if o and o["match"] is not None else None
        for i, o in enumerate(res.outputs)
    ]
    ex["similarity"] = cand_sim[:, 0]
    ex = ex.merge(stats[["theme_id", "items", "rank"]], on="theme_id")
    ex.to_parquet(ins / "existing.parquet", index=False)

    write_reports(stats, series, plat, rat, comp, ex, themes, rcfg, tot_prev, tot_recent)


def write_reports(stats, series, plat, rat, comp, ex, themes, rcfg, tot_prev, tot_recent):
    tables, figs = path("reports") / "tables", path("reports") / "figures"
    w = rcfg["weights"]
    cols = ["rank", "name", "items", "engagement", "low_rating_share", "growth", "priority"]
    cols += ["trend"]
    with open(tables / "top_themes.md", "w") as f:
        f.write("# Top request themes by priority\n\n")
        f.write(
            f"priority = {w['volume']}·z(log items) + {w['engagement']}·z(engagement) + "
            f"{w['dissatisfaction']}·z(share ≤2★) + {w['growth']}·z(growth). Engagement: "
            "mean within-source percentile of thumbs-up/upvotes. Growth: log2 change in "
            f"the theme's share of corpus items, last {rcfg['window_weeks']} weeks vs the "
            f"{rcfg['window_weeks']} before, 5★ Play reviews left out (blank below "
            f"{rcfg['min_support']} items).\n\n"
        )
        f.write(stats.head(30)[cols].round(3).to_markdown(index=False) + "\n")
        by_vol = stats.sort_values("items", ascending=False).head(15)
        f.write("\n## Largest themes by volume\n\n")
        f.write(by_vol[["name", "items", "play_items", "reddit_items"]].to_markdown(index=False))
        f.write("\n")

    with open(tables / "trends.md", "w") as f:
        f.write("# Emerging and declining themes\n\n")
        f.write(
            f"Last {rcfg['window_weeks']} weeks ({tot_recent} corpus items) vs the "
            f"{rcfg['window_weeks']} weeks before ({tot_prev}). Two-proportion z-test on "
            f"the theme's share of corpus items, Benjamini–Hochberg q < "
            f"{rcfg['trend']['alpha']}, ratio ≥ {rcfg['trend']['min_ratio']} (or ≤ its "
            f"inverse), ≥ {rcfg['trend']['min_recent']} items in the larger window. "
            "5★ Play reviews are left out of all counts, because Google Play only returns them "
            "for the last weeks of the year.\n\n"
        )
        for trend in ("emerging", "declining"):
            sub = stats[stats["trend"] == trend].sort_values("ratio", ascending=trend != "emerging")
            f.write(f"## {trend.title()} ({len(sub)})\n\n")
            show = sub[["name", "prev_items", "recent_items", "ratio", "q"]]
            f.write((show.round(4).to_markdown(index=False) if len(sub) else "None.") + "\n\n")

    with open(tables / "insights.md", "w") as f:
        f.write("# Insights\n\n## Platform: most Reddit-leaning and Play-leaning themes\n\n")
        named = plat.merge(themes[["theme_id", "name"]], on="theme_id")
        cols_p = ["name", "play_items", "reddit_items", "play_share", "reddit_share", "log2_ratio"]
        f.write("Reddit-leaning:\n\n")
        f.write(named.nlargest(10, "log2_ratio")[cols_p].round(4).to_markdown(index=False))
        f.write("\n\nPlay-leaning:\n\n")
        f.write(named.nsmallest(10, "log2_ratio")[cols_p].round(4).to_markdown(index=False))
        r = rat.merge(themes[["theme_id", "name"]], on="theme_id")
        f.write("\n\n## Ratings: themes in the angriest and happiest Play reviews\n\n")
        cols_r = ["name", "n", "mean_rating", "low_share", "high_share"]
        f.write(r.nsmallest(10, "mean_rating")[cols_r].round(3).to_markdown(index=False))
        f.write("\n\n")
        f.write(r.nlargest(10, "mean_rating")[cols_r].round(3).to_markdown(index=False))
        f.write("\n\n## Competitors cited in requests\n\n")
        tot = comp.groupby("competitor")["items"].sum().sort_values(ascending=False)
        f.write(tot.rename("items").to_frame().to_markdown() + "\n\n")
        top = comp.merge(themes[["theme_id", "name"]], on="theme_id").head(15)
        f.write(top[["competitor", "name", "items"]].to_markdown(index=False) + "\n\n")
        f.write("## Requests for features that already exist\n\n")
        counts = ex["status"].value_counts()
        f.write(", ".join(f"{k}: {v} themes" for k, v in counts.items()) + f" (of {len(ex)}).\n\n")
        for status in ("exists", "premium_only", "partly"):
            sub = ex[ex["status"] == status].sort_values("items", ascending=False).head(12)
            f.write(f"### {status}\n\n")
            if len(sub):
                f.write(sub[["name", "feature", "availability", "items"]].to_markdown(index=False))
            f.write("\n\n")

    viz.apply_style()
    top = stats.head(15).iloc[::-1]
    fig, ax = viz.plt.subplots(figsize=(7, 5.5))
    ax.barh(top["name"], top["priority"], color=viz.SERIES[0])
    ax.set_xlabel("priority score")
    ax.set_title("Top 15 request themes by priority")
    viz.save(fig, figs / "top_themes.png")

    emerging = stats[stats["trend"] == "emerging"].nlargest(6, "ratio")
    if len(emerging):
        n = len(emerging)
        fig, axes = viz.plt.subplots(
            (n + 2) // 3, min(n, 3), figsize=(10, 2.8 * ((n + 2) // 3)), squeeze=False
        )
        for ax, (_, t) in zip(axes.flat, emerging.iterrows(), strict=False):
            s = series[series["theme_id"] == t["theme_id"]]
            ax.plot(s["month"], s["share"] * 1000, marker="o", ms=3)
            ax.set_title(t["name"], fontsize=9)
            ax.set_ylabel("per 1,000 items")
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle("Emerging themes: monthly share of all feedback")
        viz.save(fig, figs / "trends_emerging.png")
    print(stats.head(20)[cols].round(3).to_string())
    print(ex["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
