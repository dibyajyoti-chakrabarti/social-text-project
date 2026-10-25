"""Live Analyser: classify any text with the local SetFit model, extract the requested
features (rules by default, Gemini optionally) and find the nearest request theme.

Models load on the first analysis, not when the page opens; everything runs offline unless
the LLM toggle is switched on.
"""

from __future__ import annotations

import streamlit as st

from frminer.config import ROOT

import inference
import loaders
from theme import CHART_CONFIG

EXAMPLES = {
    "Feature request": "Love the app but please let me turn off the AI DJ completely, and a sleep "
    "timer for podcasts would be great. Apple Music already has this.",
    "Bug report": "Since the last update the app crashes every time I open my liked songs.",
    "Praise": "Best music app ever, the recommendations are spot on!",
}


st.title("Live Analyser")
st.caption(
    "Paste any review or post. Classification uses the local SetFit model "
    f"(gold-test macro-F1 {loaders.deployed_f1():.2f}); nothing leaves this machine unless "
    "LLM extraction is switched on."
)
if not inference.model_ready():
    st.error(
        f"No classifier at `{inference.MODEL_DIR.relative_to(ROOT)}`: unpack the model bundle "
        "(PLAN.md §0)."
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
acc = loaders.extractor_accuracy()
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
probs = inference.classify(text)
label = probs.idxmax()
left, right = st.columns([1, 2])
left.metric(
    "Predicted class",
    label.replace("_", " "),
    f"{probs.max():.0%} confidence",
    delta_color="off",
    delta_arrow="off",
)
right.plotly_chart(inference.proba_chart(probs), width="stretch", config=CHART_CONFIG)

# 2. Feature extraction.
st.subheader("Requested features")
with st.spinner("Asking Gemini…" if use_llm else "Extracting…"):
    ext = inference.extract_features(text, use_llm)
if ext.error:
    st.error(f"LLM extraction failed ({ext.error}); falling back to rules.")
features, competitor, method = ext.features, ext.competitor, ext.method

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
for m in inference.match_themes(features):
    with st.container(border=True):
        st.markdown(f"**“{m.phrase}”**")
        if m.theme_id >= 0:
            th = loaders.theme_stats().set_index("theme_id").loc[m.theme_id]
            st.markdown(
                f"→ **{th['name']}** (rank {int(th['rank'])}, {int(th['items']):,} items · "
                f"cosine {m.similarity:.2f})"
            )
        else:
            st.markdown(
                f"→ no close theme (best: {m.nearest.iloc[0]['theme']}, cosine "
                f"{m.similarity:.2f} < {inference.THEME_MIN_SIM}): possibly a new request"
            )
        st.dataframe(
            m.nearest, hide_index=True, width="stretch", column_config=inference.NEAREST_COLUMNS
        )
