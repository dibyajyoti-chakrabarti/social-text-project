"""Cluster purity (RQ3b) → reports/tables/cluster_eval.md.

uv run python scripts/evaluate_clusters.py

Purity = share of sampled members that belong to their cluster, on the sample drawn by
cluster_requests.py (annotations/purity_sample.csv). The LLM judge (configs:
cluster.judge) rates every sampled cluster of both methods; the human check
(annotations/purity_labels.csv, from apps/purity.py) covers method A's first clusters
and is used to measure how far the judge can be trusted.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from frminer.cluster import purity
from frminer.config import load_config, path
from frminer.llm import LLM


def purity_ci(per_cluster: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of mean purity, resampling clusters."""
    vals = per_cluster.to_numpy(float)
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    cfg = load_config()["cluster"]
    ann = path("annotations")
    sample = pd.read_csv(ann / "purity_sample.csv")
    groups = list(sample.groupby(["method", "rank"], sort=True))
    labels = [
        f"{g['name'].iloc[0]}. {g['description'].iloc[0]}".strip(". ")
        if m == "A"
        else f"topic words: {g['name'].iloc[0]}"
        for (m, _), g in groups
    ]
    res = purity.judge(
        LLM(cfg["judge"]["profile"]),
        labels,
        [g["member"].tolist() for _, g in groups],
        desc="purity judge",
    )
    sample["judge"] = None
    for ((_m, _r), g), verdicts in zip(groups, res.outputs, strict=True):
        if verdicts is not None:
            sample.loc[g.index, "judge"] = verdicts
    sample["judge"] = sample["judge"].astype("boolean")

    human_path = ann / "purity_labels.csv"
    human = pd.read_csv(human_path) if human_path.exists() else pd.DataFrame()
    if len(human):
        human["human"] = ~human["intruder"].astype(bool)
        sample = sample.merge(
            human[["rank", "member", "human", "name_fit"]].assign(method="A"),
            on=["method", "rank", "member"],
            how="left",
        )

    rows = []
    for (method, who), col in [
        (("A", "LLM judge"), "judge"),
        (("B", "LLM judge"), "judge"),
        (("A", "human"), "human"),
        (("A", "LLM judge, human subset"), "judge"),
    ]:
        if col not in sample:
            continue
        sub = sample[sample["method"] == method]
        if who != "LLM judge":
            sub = sub[sub["human"].notna()] if "human" in sub else sub.iloc[:0]
        sub = sub[sub[col].notna()]
        if sub.empty:
            continue
        per_cluster = sub.groupby("rank")[col].mean()
        lo, hi = purity_ci(per_cluster)
        rows.append(
            {
                "method": {"A": "A: phrase clusters", "B": "B: BERTopic"}[method],
                "rater": who,
                "clusters": len(per_cluster),
                "members": len(sub),
                "purity": sub[col].astype(float).mean(),
                "ci": f"[{lo:.2f}, {hi:.2f}]",
                "fully_pure": float((per_cluster == 1).mean()),
            }
        )
    table = pd.DataFrame(rows)

    lines = [
        "# Cluster purity\n",
        f"Random sample of {cfg['purity']['clusters']} clusters per method × "
        f"{cfg['purity']['members']} random members. `purity`: share of members that "
        "belong (95% bootstrap CI over clusters); `fully_pure`: clusters with no intruder. "
        "Method A members are feature phrases, method B members are full posts/reviews.\n",
        table.round(3).to_markdown(index=False) + "\n",
    ]
    if "human" in sample and sample["human"].notna().any():
        both = sample[sample["human"].notna() & sample["judge"].notna()]
        h, j = both["human"].astype(bool), both["judge"].astype(bool)
        fit = sample.drop_duplicates("rank")["name_fit"].value_counts(normalize=True)
        lines.append(
            "\n## Judge vs human (method A)\n\n"
            f"{len(both)} members: agreement {float((h == j).mean()):.2f}, Cohen's κ "
            f"{cohen_kappa_score(h, j):.2f}; the judge calls {float((~j).mean()):.0%} "
            f"intruders, the human {float((~h).mean()):.0%}.\n\n"
            "Theme name fits its phrases (human): "
            + ", ".join(f"{k} {v:.0%}" for k, v in fit.items())
            + "\n"
        )
    intr = sample[(sample["method"] == "A") & (sample["judge"] == False)]  # noqa: E712
    lines.append("\n## Intruders flagged by the judge (method A, first 15)\n")
    for _, r in intr.head(15).iterrows():
        lines.append(f"- **{r['name']}**: {r['member']}")
    out = path("reports") / "tables" / "cluster_eval.md"
    out.write_text("\n".join(lines) + "\n")
    sample.to_csv(path("processed") / "purity_judged.csv", index=False)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
