"""Where do the detectors go wrong on test? → reports/tables/detect_errors.md.

uv run python scripts/error_analysis.py

Reads the predictions saved by evaluate_detection.py (nothing is re-run or tuned). For the
deployed model (SetFit trained on silver) and its teacher (Gemini zero-shot) it lists the
most frequent gold → predicted confusions, how often the two systems make the same
mistake (a hint that the item is ambiguous or that the annotator applied a convention the
models do not share), and short examples per confusion for the qualitative write-up.
"""

import pandas as pd

from frminer.annotate import load_gold
from frminer.config import path

DEPLOYED, TEACHER = "SetFit (silver)", "LLM zero-shot"
TOP_PAIRS = 6
EXAMPLES = 4
MAX_CHARS = 220


def _short(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


def main() -> None:
    preds = pd.read_parquet(path("processed") / "test_preds.parquet")
    test = load_gold(path("annotations"), "test")[["id", "text"]]
    preds = preds.merge(test, on="id", validate="one_to_one")

    lines = [
        "# Detection errors on gold test (n=300)\n",
        f"Deployed model: **{DEPLOYED}**; teacher: **{TEACHER}** (Gemini, prompt v2).\n",
    ]
    for system in (DEPLOYED, TEACHER):
        wrong = preds[preds[system].notna() & (preds[system] != preds["gold"])]
        pairs = wrong.groupby(["gold", system]).size().sort_values(ascending=False)
        lines.append(f"\n## {system}: {len(wrong)} errors\n")
        table = pairs.rename("n").reset_index().rename(columns={system: "predicted"})
        table["share_of_errors"] = (table["n"] / len(wrong)).map("{:.0%}".format)
        lines.append(table.head(10).to_markdown(index=False) + "\n")

    both = preds[(preds[DEPLOYED] != preds["gold"]) & (preds[TEACHER] != preds["gold"])]
    same = both[both[DEPLOYED] == both[TEACHER]]
    n_dep = int((preds[DEPLOYED] != preds["gold"]).sum())
    lines.append(
        "\n## Shared mistakes\n\n"
        f"Both systems are wrong on {len(both)} items; on {len(same)} of them they agree on "
        f"the same wrong label ({len(same) / n_dep:.0%} of the deployed model's errors). "
        "These are the candidates for genuinely ambiguous items or annotator conventions.\n"
    )
    lines.append(
        same.groupby(["gold", DEPLOYED])
        .size()
        .sort_values(ascending=False)
        .rename("n")
        .reset_index()
        .rename(columns={DEPLOYED: "both predicted"})
        .head(8)
        .to_markdown(index=False)
        + "\n"
    )

    wrong = preds[preds[DEPLOYED] != preds["gold"]]
    top = wrong.groupby(["gold", DEPLOYED]).size().sort_values(ascending=False).head(TOP_PAIRS)
    lines.append("\n## Examples of the deployed model's top confusions\n")
    for (gold, pred), n in top.items():
        lines.append(f"\n### gold `{gold}` → predicted `{pred}` ({n})\n")
        sample = wrong[(wrong["gold"] == gold) & (wrong[DEPLOYED] == pred)]
        for _, r in sample.sort_values("id").head(EXAMPLES).iterrows():
            lines.append(f"- *{r['source']}*, teacher said `{r[TEACHER]}`: {_short(r['text'])}")
        lines.append("")

    out = path("reports") / "tables" / "detect_errors.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
