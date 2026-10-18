"""Live Analyser: classify any text with the local SetFit model, extract the requested
features (rules by default, Gemini optionally) and find the nearest request theme.

Models load on the first analysis, not when the page opens; everything runs offline unless
the LLM toggle is switched on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frminer.config import ROOT

import loaders
from theme import BAR_GAP_LINE, CHART_CONFIG, SERIES

MODEL_DIR = ROOT / "models" / "setfit_silver"
CFG = loaders.config()
EMBED_MODEL = CFG["cluster"]["embed_model"]
THEME_MIN_SIM = CFG["cluster"]["outlier_threshold"]  # same bar used to attach phrases to themes
EXAMPLES = {
    "Feature request": "Love the app but please let me turn off the AI DJ completely, and a sleep "
    "timer for podcasts would be great. Apple Music already has this.",
    "Bug report": "Since the last update the app crashes every time I open my liked songs.",
    "Praise": "Best music app ever, the recommendations are spot on!",
}


@st.cache_resource(show_spinner="Loading the SetFit classifier…")
def classifier():
    from frminer.detect import ml

    return ml.load_setfit(MODEL_DIR)


@st.cache_resource(show_spinner="Loading the sentence encoder…")
def encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Building theme centroids (first run only)…")
def theme_centroids() -> tuple[np.ndarray, np.ndarray]:
    """Mention-weighted mean embedding of each theme's phrases, L2-normalised."""
    pt = pd.read_parquet(loaders.PROC / "phrase_themes.parquet")
    pt = pt[pt["theme_id"] >= 0]
    emb = encoder().encode(pt["phrase"].tolist(), normalize_embeddings=True, batch_size=256)
    ids = np.array(sorted(pt["theme_id"].unique()))
    weights = pt["mentions"].to_numpy(float)
    cents = np.stack(
        [
            np.average(
                emb[(pt["theme_id"] == t).to_numpy()],
                axis=0,
                weights=weights[(pt["theme_id"] == t).to_numpy()],
            )
            for t in ids
        ]
    )
    return ids, cents / np.linalg.norm(cents, axis=1, keepdims=True)


def nearest_themes(phrases: list[str], k: int = 3) -> list[pd.DataFrame]:
    from frminer.extract.normalize import normalize_many

    ids, cents = theme_centroids()
    keys = [n or p for n, p in zip(normalize_many(phrases), phrases, strict=True)]
    sims = encoder().encode(keys, normalize_embeddings=True) @ cents.T
    stats = loaders.theme_stats().set_index("theme_id")
    out = []
    for row in sims:
        top = np.argsort(-row)[:k]
        out.append(
            pd.DataFrame(
                {
                    "theme": stats.loc[ids[top], "name"].to_numpy(),
                    "similarity": row[top],
                    "rank": stats.loc[ids[top], "rank"].to_numpy(),
                    "items": stats.loc[ids[top], "items"].to_numpy(),
                }
            )
        )
    return out


def extractor_accuracy() -> dict[str, float]:
    """Share of gold feature requests the judge graded correct, per extraction method."""
    ext = loaders.md_tables("extract_eval.md")[0]
    ext = ext[ext["setting"] == "gold all"].set_index("method")
    return ext["judge_correct"].astype(float).to_dict()


st.title("Live Analyser")
st.caption(
    "Paste any review or post. Classification uses the local SetFit model "
    f"(gold-test macro-F1 {loaders.deployed_f1():.2f}); nothing leaves this machine unless "
    "LLM extraction is switched on."
)
if not (MODEL_DIR / "config.json").exists():
    st.error(
        f"No classifier at `{MODEL_DIR.relative_to(ROOT)}`: unpack the model bundle (PLAN.md §0)."
    )
    st.stop()

pick = st.segmented_control("Try an example", list(EXAMPLES), key="live_example")
if pick and st.session_state.get("_live_last_example") != pick:
    st.session_state["live_text"] = EXAMPLES[pick]
    st.session_state["_live_last_example"] = pick
text = st.text_area(
    "Text", key="live_text", height=130, placeholder="e.g. please add a sleep timer…"
)
use_llm = st.toggle(
    "Use LLM extraction (Gemini)",
    value=False,
    key="live_llm",
    help="Off: dependency-parse rules, fully offline. On: one Gemini request per new text "
    "(cached, so the same text is free the second time). Needs internet and GEMINI_KEY_* in .env.",
)
acc = extractor_accuracy()
st.caption(
    f"On the gold feature requests, rules extract a correct feature for {acc['rules']:.0%} of "
    f"items, Gemini for {acc['llm']:.0%} (extract_eval.md)."
)
if use_llm:
    st.warning(
        "LLM extraction is on: each new text costs 1 Gemini request from the free-tier quota."
    )

if not st.button("Analyse", type="primary", disabled=not text.strip()):
    st.stop()

# 1. Classification.
from frminer.detect import ml  # noqa: E402

proba, classes = ml.predict_proba(classifier(), [text])
probs = pd.Series(proba[0], index=classes).sort_values()
label = probs.idxmax()
left, right = st.columns([1, 2])
left.metric(
    "Predicted class",
    label.replace("_", " "),
    f"{probs.max():.0%} confidence",
    delta_color="off",
    delta_arrow="off",
)
fig = go.Figure(
    go.Bar(
        x=probs.values * 100,
        y=[c.replace("_", " ") for c in probs.index],
        orientation="h",
        marker=dict(
            color=[SERIES[0] if c == label else "#c3c2b7" for c in probs.index], line=BAR_GAP_LINE
        ),
        text=[f"{p:.0%}" for p in probs.values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    )
)
fig.update_layout(
    height=240, xaxis=dict(title="probability (%)", range=[0, 105]), margin=dict(t=10, r=40)
)
right.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

# 2. Feature extraction.
st.subheader("Requested features")
if use_llm:
    from frminer.extract import llm_extract
    from frminer.llm import LLM

    with st.spinner("Asking Gemini…"):
        try:
            res = llm_extract.extract(
                LLM(CFG["extract"]["llm"]["profile"]), [text], batch_size=1, workers=1
            )
            ext = res.outputs[0]
        except Exception as e:  # network down, quota spent, missing key
            ext = None
            st.error(f"LLM extraction failed ({type(e).__name__}); falling back to rules.")
    features = ext.features if ext else None
    competitor = ext.competitor if ext else None
    method = "Gemini"
if not use_llm or features is None:
    from frminer.extract import rules

    features = rules.extract(text)
    found = rules.competitors(text)
    competitor = found[0] if found else None
    method = "rules"

if label != "feature_request":
    st.caption(
        f"The classifier doesn't see a feature request here; extraction runs anyway ({method})."
    )
if not features:
    st.info(f"No requested feature found ({method}).")
    st.stop()
st.caption(
    f"Extracted by {method}" + (f" · competitor mentioned: **{competitor}**" if competitor else "")
)

# 3. Nearest themes.
for phrase, near in zip(features, nearest_themes(features), strict=True):
    best = near.iloc[0]
    with st.container(border=True):
        st.markdown(f"**“{phrase}”**")
        if best["similarity"] >= THEME_MIN_SIM:
            st.markdown(
                f"→ **{best['theme']}** (rank {int(best['rank'])}, {int(best['items']):,} items · "
                f"cosine {best['similarity']:.2f})"
            )
        else:
            st.markdown(
                f"→ no close theme (best: {best['theme']}, cosine {best['similarity']:.2f} "
                f"< {THEME_MIN_SIM}): possibly a new request"
            )
        st.dataframe(
            near,
            hide_index=True,
            width="stretch",
            column_config={
                "theme": st.column_config.TextColumn("Nearest themes", width="large"),
                "similarity": st.column_config.ProgressColumn(
                    "Cosine", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "rank": st.column_config.NumberColumn("Rank"),
                "items": st.column_config.NumberColumn("Items", format="%d"),
            },
        )
