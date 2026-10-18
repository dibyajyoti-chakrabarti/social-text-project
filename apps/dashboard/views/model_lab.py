"""Model Lab: how well each pipeline stage works, straight from reports/tables and figures."""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders
from theme import BAR_GAP_LINE, CHART_CONFIG, INK_2, MUTED, SEQ_BLUE, SERIES, SURFACE

# Fixed slot per system family (colour follows the family, never its rank).
FAMILY_COLORS = {
    "LLM": SERIES[0],
    "Local, 12k silver labels": SERIES[1],
    "Local, 100 gold labels": SERIES[2],
    "Rules": SERIES[3],
}
DEPLOYED = "SetFit (silver)"
TEACHER = "LLM zero-shot"
# Labelled on the cost scatter; the other LLM variants sit on top of the teacher (hover them).
COST_LABELS = {
    DEPLOYED,
    TEACHER,
    "Rules",
    "TF-IDF+LR (silver)",
    "SetFit (gold-dev)",
    "TF-IDF+LR (gold-dev)",
}


def family(system: str) -> str:
    if system == "Rules":
        return "Rules"
    if system.startswith("LLM"):
        return "LLM"
    return "Local, 12k silver labels" if "(silver)" in system else "Local, 100 gold labels"


def ci_bounds(ci: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = ci.str.strip("[]").str.split(",", expand=True).astype(float)
    return parts[0], parts[1]


def confusion_png(system: str):
    slug = system.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    return loaders.FIGURES / f"detect_confusion_{slug}.png"


def section(md: str, heading: str) -> str:
    """Body of a '## heading' section of a markdown report."""
    m = re.search(rf"^## {re.escape(heading)}.*?\n(.*?)(?=^## |\Z)", md, flags=re.S | re.M)
    return m.group(1).strip() if m else ""


st.title("Model Lab")
st.caption(
    "Evaluation results as written by the pipeline scripts (reports/tables, reports/figures). "
    "Detection is scored on the 300-item gold test split, which nothing was tuned on."
)
det = loaders.detect_test().rename(columns={"system": "name"})
det["family"] = det["name"].map(family)
det["lo"], det["hi"] = ci_bounds(det["macro_f1_ci"])

detect_tab, distil_tab, errors_tab, extract_tab, cluster_tab = st.tabs(
    ["Detection", "Distillation", "Errors", "Extraction", "Clustering"]
)

with detect_tab:
    dep, teach = det.set_index("name").loc[DEPLOYED], det.set_index("name").loc[TEACHER]
    k = st.columns(4)
    k[0].metric(
        "Deployed: SetFit (silver)", f"{dep['macro_f1']:.2f} macro-F1", help=dep["macro_f1_ci"]
    )
    k[1].metric("Its teacher: Gemini zero-shot", f"{teach['macro_f1']:.2f} macro-F1")
    k[2].metric("Feature-request F1 (deployed)", f"{dep['fr_f1']:.2f}", help=dep["fr_f1_ci"])
    k[3].metric(
        "Speed-up over the LLM",
        f"{teach['s_per_item'] / dep['s_per_item']:.0f}×",
        help="Seconds per item, LLM call time vs local GPU inference",
    )

    left, right = st.columns(2)
    d = det.sort_values("macro_f1")
    fig = go.Figure()
    for fam, color in FAMILY_COLORS.items():
        f = d[d["family"] == fam]
        fig.add_scatter(
            x=f["macro_f1"],
            y=f["name"],
            mode="markers",
            name=fam,
            marker=dict(color=color, size=11, line=dict(color=SURFACE, width=2)),
            error_x=dict(
                type="data",
                symmetric=False,
                array=f["hi"] - f["macro_f1"],
                arrayminus=f["macro_f1"] - f["lo"],
                color=color,
                thickness=2,
                width=0,
            ),
            customdata=f[["macro_f1_ci", "accuracy", "kappa"]],
            hovertemplate=(
                "<b>%{y}</b><br>macro-F1 %{x:.3f} %{customdata[0]}"
                "<br>accuracy %{customdata[1]:.3f} · κ %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    fig.update_layout(
        height=420,
        xaxis=dict(title="macro-F1 on gold test (95% CI)", range=[0, 0.85]),
        yaxis=dict(categoryorder="array", categoryarray=d["name"].tolist(), automargin=True),
        legend=dict(traceorder="normal"),
        margin=dict(t=60),
    )
    with left:
        st.markdown("**Accuracy of every detection system**")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    c = det[det["s_per_item"] > 0]
    fig = go.Figure()
    for fam, color in FAMILY_COLORS.items():
        f = c[c["family"] == fam]
        fig.add_scatter(
            x=f["s_per_item"],
            y=f["macro_f1"],
            mode="markers+text",
            name=fam,
            text=[n if n in COST_LABELS else "" for n in f["name"]],
            customdata=f["name"],
            textposition="top center",
            textfont=dict(size=11, color=INK_2),
            marker=dict(color=color, size=11, line=dict(color=SURFACE, width=2)),
            hovertemplate=(
                "<b>%{customdata}</b><br>%{x:.4f} s/item · macro-F1 %{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    fig.update_layout(
        height=420,
        xaxis=dict(type="log", title="seconds per item (log scale)"),
        yaxis=dict(title="macro-F1", range=[0.2, 0.8]),
        margin=dict(t=60),
    )
    with right:
        st.markdown("**Cost vs accuracy**")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
        st.caption(
            "LLM times are per call ÷ 20 items; local models run on a laptop GPU. "
            "The four LLM variants overlap at the top right (hover for names)."
        )

    per_class = loaders.md_tables("detect_test_per_class.md")[0].set_index("")
    per_class = per_class.astype(float)
    fig = go.Figure(
        go.Heatmap(
            z=per_class.values,
            x=per_class.columns,
            y=[c.replace("_", " ") for c in per_class.index],
            colorscale=[[0, SURFACE]]
            + [[(i + 1) / len(SEQ_BLUE), s] for i, s in enumerate(SEQ_BLUE)],
            zmin=0,
            zmax=1,
            xgap=2,
            ygap=2,
            text=per_class.values,
            texttemplate="%{text:.2f}",
            colorbar=dict(title="F1", thickness=10),
            hovertemplate="%{x} · %{y}: F1 %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        xaxis=dict(side="top", tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=90),
    )
    st.markdown("**F1 per class**")
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    st.markdown("**Confusion matrix**")
    system = st.selectbox(
        "System", det["name"].tolist(), index=det["name"].tolist().index(DEPLOYED)
    )
    png = confusion_png(system)
    if png.exists():
        st.image(str(png), width=520)
    with st.expander("Full results table"):
        cols = [
            "name",
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
        st.dataframe(det[cols].round(3), hide_index=True, width="stretch")

with distil_tab:
    st.markdown(
        "**RQ2: can a small local model learn from LLM labels?** The same two local models "
        "trained on 100 hand-labelled gold-dev items vs 12k items labelled by Gemini (silver)."
    )
    rows = []
    for model in ["TF-IDF+LR", "SetFit"]:
        for train, label in [("gold-dev", "100 gold labels"), ("silver", "12k silver labels")]:
            r = det.set_index("name").loc[f"{model} ({train})"]
            rows.append({"model": model, "train": label, **r[["macro_f1", "lo", "hi"]].to_dict()})
    dist = pd.DataFrame(rows)
    fig = go.Figure()
    for label, color in [("100 gold labels", SERIES[2]), ("12k silver labels", SERIES[1])]:
        f = dist[dist["train"] == label]
        fig.add_bar(
            x=f["model"],
            y=f["macro_f1"],
            name=label,
            marker=dict(color=color, line=BAR_GAP_LINE),
            error_y=dict(
                type="data",
                symmetric=False,
                array=f["hi"] - f["macro_f1"],
                arrayminus=f["macro_f1"] - f["lo"],
                color=MUTED,
                thickness=1.5,
                width=6,
            ),
            hovertemplate=label + " · %{x}: macro-F1 %{y:.3f}<extra></extra>",
        )
    fig.add_hline(
        y=teach["macro_f1"],
        line=dict(color=SERIES[0], width=2),
        annotation_text=f"teacher (Gemini zero-shot) {teach['macro_f1']:.2f}",
        annotation_position="top left",
        annotation_font_color=INK_2,
    )
    fig.update_layout(
        barmode="group",
        height=380,
        yaxis=dict(title="macro-F1 on gold test", range=[0, 0.85]),
        legend=dict(traceorder="normal"),
        margin=dict(t=40),
    )
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        f"SetFit trained on silver labels matches its teacher ({dep['macro_f1']:.3f} vs "
        f"{teach['macro_f1']:.3f}) at {dep['s_per_item']:.4f} s/item and no API cost, so it "
        "classifies the full corpus."
    )

with errors_tab:
    st.markdown(loaders.md_text("detect_errors.md").replace("$", r"\$"))

with extract_tab:
    ext = loaders.md_tables("extract_eval.md")[0]
    for col in ext.columns[2:]:
        if col != "f1_ci":
            ext[col] = pd.to_numeric(ext[col], errors="coerce")
    st.markdown(
        "**RQ3a: extracting the requested feature.** Gemini (LLM) vs dependency-parse rules, "
        "graded by a different LLM family (Groq gpt-oss-20b) as correct / partial / wrong."
    )
    g = ext[ext["setting"].isin(["gold all", "corpus sample"])].copy()
    g["label"] = g["setting"] + " · " + g["method"]
    fig = go.Figure()
    for col, name, color in [
        ("judge_correct", "correct", SERIES[0]),
        ("judge_partial", "partial", SEQ_BLUE[1]),
        ("judge_wrong", "wrong", "#d6d5ce"),
    ]:
        fig.add_bar(
            x=g[col] * 100,
            y=g["label"],
            orientation="h",
            name=name,
            marker=dict(color=color, line=BAR_GAP_LINE),
            hovertemplate="%{y}: " + name + " %{x:.0f}%<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        barcornerradius=0,
        height=260,
        xaxis=dict(title="% of items", range=[0, 100]),
        yaxis=dict(autorange="reversed", automargin=True, gridcolor="rgba(0,0,0,0)"),
        legend=dict(traceorder="normal"),
        margin=dict(t=30),
    )
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.dataframe(
        ext[
            [
                "setting",
                "method",
                "items",
                "precision",
                "recall",
                "f1",
                "f1_ci",
                "item_hit",
                "empty",
                "judge_correct",
                "judge_partial",
                "judge_wrong",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    with st.expander("How this was measured"):
        intro = loaders.md_text("extract_eval.md").split("\n|", 1)[0]
        st.markdown(intro.split("\n", 1)[-1].replace("$", r"\$"))
    with st.expander("Extraction examples"):
        st.markdown(loaders.md_text("extract_examples.md").replace("$", r"\$"))

with cluster_tab:
    st.markdown("**RQ3b: grouping requests into themes.** Phrase clusters (A) vs BERTopic (B).")
    comp = loaders.md_tables("cluster_compare.md")[0]
    st.dataframe(comp, hide_index=True, width="stretch")
    pur = loaders.md_tables("cluster_eval.md")[0]
    pur["purity"] = pur["purity"].astype(float)
    pur["lo"], pur["hi"] = ci_bounds(pur["ci"])
    pur["label"] = pur["method"] + " · " + pur["rater"]
    rater_colors = {
        "LLM judge": SERIES[0],
        "human": SERIES[1],
        "LLM judge, human subset": SERIES[4],
    }
    fig = go.Figure()
    for rater, color in rater_colors.items():
        f = pur[pur["rater"] == rater]
        fig.add_scatter(
            x=f["purity"],
            y=f["label"],
            mode="markers",
            name=rater,
            marker=dict(color=color, size=11, line=dict(color=SURFACE, width=2)),
            error_x=dict(
                type="data",
                symmetric=False,
                array=f["hi"] - f["purity"],
                arrayminus=f["purity"] - f["lo"],
                color=color,
                thickness=2,
                width=0,
            ),
            customdata=f[["clusters", "members", "ci"]],
            hovertemplate=(
                "<b>%{y}</b><br>purity %{x:.3f} %{customdata[2]}"
                "<br>%{customdata[0]} clusters × members: %{customdata[1]}<extra></extra>"
            ),
        )
    fig.update_layout(
        height=300,
        xaxis=dict(title="purity: share of sampled members that belong (95% CI)", range=[0.5, 1]),
        yaxis=dict(autorange="reversed", automargin=True),
        legend=dict(traceorder="normal"),
        margin=dict(t=40),
    )
    st.markdown("**Cluster purity**")
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    md = loaders.md_text("cluster_eval.md")
    judge = section(md, "Judge vs human")
    if judge:
        st.info(judge.replace("\n\n", "  \n"))
    with st.expander("Intruders flagged by the judge"):
        st.markdown(section(md, "Intruders flagged").replace("$", r"\$"))
