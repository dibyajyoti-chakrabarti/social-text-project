"""Describe the labelled gold set → reports/tables/gold_summary.md.

uv run python scripts/gold_summary.py

Class distribution by split and source, the feature-phrase coverage, and intra-annotator
agreement against the pilot pass (annotations/gold_labels_pilot.csv: the first ~27 items,
labelled once, discarded, then relabelled from scratch in the final pass).
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from frminer.annotate import LABELS, LabelStore, load_gold
from frminer.config import path


def main() -> None:
    ann = path("annotations")
    gold = load_gold(ann)
    gold["label"] = pd.Categorical(gold["label"], categories=LABELS)

    by_split = pd.crosstab(gold["label"], gold["split"], margins=True, margins_name="total")
    by_split["share"] = (by_split["total"] / len(gold)).map("{:.1%}".format)
    by_source = pd.crosstab(gold["label"], gold["source"], margins=True, margins_name="total")
    by_stratum = pd.crosstab(gold["label"], gold["stratum"])

    fr = gold[gold["label"] == "feature_request"]
    n_phrases = fr["features"].str.split(";").explode().str.strip().replace("", pd.NA).count()

    pilot = LabelStore(ann / "gold_labels_pilot.csv").latest()
    both = pilot.join(gold.set_index("id")["label"].astype(str), lsuffix="_pilot", how="inner")
    agree = (both["label_pilot"] == both["label"]).mean()
    kappa = cohen_kappa_score(both["label_pilot"], both["label"])

    out = path("reports") / "tables" / "gold_summary.md"
    with open(out, "w") as f:
        f.write("# Gold set summary\n\n")
        f.write(f"{len(gold)} items, {gold['unsure'].sum()} flagged unsure.\n\n")
        f.write("## Labels by split\n\n" + by_split.to_markdown() + "\n\n")
        f.write("## Labels by source\n\n" + by_source.to_markdown() + "\n\n")
        f.write("## Labels by sampling stratum\n\n" + by_stratum.to_markdown() + "\n\n")
        f.write(
            f"## Feature phrases\n\n{len(fr)} feature requests, "
            f"{(fr['features'] != '').sum()} with at least one phrase, "
            f"{n_phrases} phrases in total.\n\n"
        )
        f.write(
            "## Intra-annotator agreement (pilot vs final pass)\n\n"
            f"{len(both)} items · raw agreement {agree:.1%} · Cohen's κ = {kappa:.2f}\n\n"
            + pd.crosstab(both["label_pilot"], both["label"]).to_markdown()
            + "\n"
        )
    print(out.read_text())


if __name__ == "__main__":
    main()
