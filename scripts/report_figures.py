"""Report figures drawn from the evaluation tables → reports/figures/report_*.png.

uv run python scripts/report_figures.py      (or: make report)

Reads only reports/tables (written by the pipeline scripts), so the figures always match
the numbers in the report.
"""

from pathlib import Path

import pandas as pd

from frminer import viz
from frminer.config import path

FAMILY = {
    "LLM": viz.SERIES[0],
    "Local, 12k silver labels": viz.SERIES[1],
    "Local, 100 gold labels": viz.SERIES[2],
    "Rules": viz.SERIES[3],
}


def family(system: str) -> str:
    if system == "Rules":
        return "Rules"
    if system.startswith("LLM"):
        return "LLM"
    return "Local, 12k silver labels" if "(silver)" in system else "Local, 100 gold labels"


def md_table(file: Path, n: int = 0) -> pd.DataFrame:
    """The n-th pipe table of a markdown file."""
    tables, block = [], []
    for line in [*file.read_text().splitlines(), ""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif block:
            rows = [r for r in block[1:] if not set("".join(r)) <= set(":-")]
            tables.append(pd.DataFrame(rows, columns=block[0]))
            block = []
    return tables[n]


def ci(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = s.str.strip("[]").str.split(",", expand=True).astype(float)
    return parts[0], parts[1]


def cost_accuracy(det: pd.DataFrame, out: Path) -> None:
    d = det[det["s_per_item"] > 0]
    fig, ax = viz.plt.subplots(figsize=(6.4, 4.0))
    for fam, color in FAMILY.items():
        f = d[d["system"].map(family) == fam]
        ax.scatter(
            f["s_per_item"],
            f["macro_f1"],
            s=46,
            color=color,
            label=fam,
            edgecolor=viz.SURFACE,
            linewidth=1.5,
            zorder=3,
        )
    for _, r in d.iterrows():
        if r["system"].startswith("LLM") and r["system"] != "LLM zero-shot":
            continue  # the LLM variants overlap; label the teacher only
        ax.annotate(
            r["system"],
            (r["s_per_item"], r["macro_f1"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=viz.INK_2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("seconds per item (log scale)")
    ax.set_ylabel("macro-F1 on gold test")
    ax.set_title("Detection: cost vs accuracy")
    ax.legend(fontsize=8, loc="lower right")
    viz.save(fig, out)


def distillation(det: pd.DataFrame, out: Path) -> None:
    d = det.set_index("system")
    models = ["TF-IDF+LR", "SetFit"]
    fig, ax = viz.plt.subplots(figsize=(5.6, 3.6))
    width = 0.34
    for k, (train, label, color) in enumerate(
        [
            ("gold-dev", "100 gold labels", viz.SERIES[2]),
            ("silver", "12k silver labels", viz.SERIES[1]),
        ]
    ):
        rows = d.loc[[f"{m} ({train})" for m in models]]
        lo, hi = ci(rows["macro_f1_ci"])
        x = [i + (k - 0.5) * width for i in range(len(models))]
        ax.bar(x, rows["macro_f1"], width=width * 0.92, color=color, label=label, zorder=3)
        ax.errorbar(
            x,
            rows["macro_f1"],
            yerr=[rows["macro_f1"] - lo.values, hi.values - rows["macro_f1"]],
            fmt="none",
            ecolor=viz.MUTED,
            elinewidth=1.2,
            capsize=3,
            zorder=4,
        )
    teacher = d.loc["LLM zero-shot", "macro_f1"]
    ax.axhline(teacher, color=viz.SERIES[0], linewidth=1.5, zorder=2)
    ax.text(
        -0.3,
        teacher + 0.015,
        f"teacher (Gemini zero-shot) {teacher:.2f}",
        fontsize=8,
        color=viz.INK_2,
    )
    ax.set_xticks(range(len(models)), models)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("macro-F1 on gold test")
    ax.set_title("Distillation: training labels for local models")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    viz.save(fig, out)


def purity(pur: pd.DataFrame, out: Path) -> None:
    pur = pur.copy()
    pur["purity"] = pur["purity"].astype(float)
    pur["lo"], pur["hi"] = ci(pur["ci"])
    pur["label"] = pur["method"].str.split(":").str[0] + ": " + pur["rater"]
    colors = {
        "LLM judge": viz.SERIES[0],
        "human": viz.SERIES[1],
        "LLM judge, human subset": viz.SERIES[4],
    }
    fig, ax = viz.plt.subplots(figsize=(6.0, 2.8))
    for i, r in enumerate(pur.itertuples()):
        y = len(pur) - 1 - i
        ax.errorbar(
            r.purity,
            y,
            xerr=[[r.purity - r.lo], [r.hi - r.purity]],
            fmt="o",
            color=colors[r.rater],
            elinewidth=2,
            markersize=7,
            markeredgecolor=viz.SURFACE,
        )
    ax.set_yticks(range(len(pur)), list(pur["label"])[::-1])
    ax.set_xlim(0.5, 1)
    ax.set_xlabel("purity: share of sampled members that belong (95% CI)")
    ax.set_title("Cluster purity (A: phrase clusters, B: BERTopic)")
    viz.save(fig, out)


def main() -> None:
    tables, figs = path("reports") / "tables", path("reports") / "figures"
    viz.apply_style()
    det = pd.read_csv(tables / "detect_test.csv")
    cost_accuracy(det, figs / "report_cost_accuracy.png")
    distillation(det, figs / "report_distillation.png")
    purity(md_table(tables / "cluster_eval.md"), figs / "report_purity.png")
    print("wrote", ", ".join(sorted(p.name for p in figs.glob("report_*.png"))))


if __name__ == "__main__":
    main()
