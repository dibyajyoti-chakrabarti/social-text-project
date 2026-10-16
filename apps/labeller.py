"""Keyboard-driven gold labelling app.

uv run streamlit run apps/labeller.py      (or: make label)

Keys: 1–6 pick a label · u toggles "unsure" · b / ← back · n / → next (no label)
For feature_request, a box appears for the requested feature(s); Enter saves.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from frminer.annotate import LABELS, LabelStore
from frminer.config import path

st.set_page_config(page_title="Gold labeller", page_icon="🏷️", layout="centered")

KEYS = {label: str(i + 1) for i, label in enumerate(LABELS)}
SHORT = {
    "feature_request": "feature request",
    "bug_report": "bug report",
    "complaint": "complaint",
    "praise": "praise",
    "question": "question",
    "other": "other",
}
SOURCE_NAMES = {
    "play": "Play Store review",
    "reddit_post": "Reddit post",
    "reddit_comment": "Reddit comment",
}


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(path("annotations") / "gold_sample.csv").sort_values("order")


def keyboard(mapping: dict[str, str], focus_key: str | None = None) -> None:
    """Map single keys to buttons (by widget key). Ignored while typing in a text field."""
    js = f"""
    <script>
    const P = window.parent, doc = P.document;
    P.__frmKeys = {json.dumps(mapping)};
    if (!P.__frmInit) {{
      P.__frmInit = true;
      doc.addEventListener('keydown', (e) => {{
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
        if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
        const key = P.__frmKeys[e.key.toLowerCase()];
        if (!key) return;
        const el = doc.querySelector(`.st-key-${{key}} button`);
        if (el) {{ e.preventDefault(); el.click(); }}
      }});
    }}
    const focusKey = {json.dumps(focus_key)};
    if (focusKey) setTimeout(() => {{
      const i = doc.querySelector(`.st-key-${{focusKey}} input`);
      if (i) i.focus();
    }}, 150);
    </script>
    """
    st.iframe(js, height=1)  # trusted, app-generated script only (invisible)


sample = load_sample()
store = LabelStore(path("annotations") / "gold_labels.csv")
labels = store.latest()
ids = sample["id"].tolist()
ss = st.session_state

if "idx" not in ss:
    unlabelled = [i for i, item_id in enumerate(ids) if item_id not in labels.index]
    ss.idx = unlabelled[0] if unlabelled else len(ids) - 1
    ss.pending_fr = False
    ss.is_unsure = False


def go(delta: int) -> None:
    ss.idx = min(max(ss.idx + delta, 0), len(ids) - 1)
    ss.pending_fr = False
    ss.is_unsure = bool(labels.loc[ids[ss.idx], "unsure"]) if ids[ss.idx] in labels.index else False


def choose(label: str) -> None:
    if label == "feature_request":
        ss.pending_fr = True
        return
    store.save(ids[ss.idx], label, unsure=ss.is_unsure)
    go(+1)


def save_fr() -> None:
    store.save(ids[ss.idx], "feature_request", features=ss.get("features", ""), unsure=ss.is_unsure)
    go(+1)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Decision rules (first match wins)")
    st.markdown(
        "1. Asks to **add / change / restore / remove** something → **feature request**\n"
        "2. Concrete **malfunction** → **bug report**\n"
        "3. Seeks **info / help** → **question**\n"
        "4. Negative opinion → **complaint**\n"
        "5. Positive opinion → **praise**\n"
        "6. Anything else → **other**"
    )
    st.caption("Full guidelines: `annotations/guidelines.md`")
    st.markdown("### Keys")
    st.markdown("`1–6` label · `u` unsure · `b`/`←` back · `n`/`→` next")
    if len(labels):
        st.markdown("### Your labels so far")
        st.dataframe(
            labels["label"].value_counts().rename("count"),
            width="stretch",
        )

# ---------------------------------------------------------------- progress
row = sample.iloc[ss.idx]
done = sample["id"].isin(labels.index)
dev = sample["split"] == "dev"
c1, c2 = st.columns(2)
c1.metric("Dev (labelled first)", f"{(done & dev).sum()} / {dev.sum()}")
c2.metric("Test", f"{(done & ~dev).sum()} / {(~dev).sum()}")
st.progress(done.mean())
if (done & dev).all() and not (done & ~dev).any():
    st.success("Dev split complete. Tell Claude: prompt development can start while you do test.")
if done.all():
    st.balloons()
    st.success("All items labelled. Thank you!")

# ---------------------------------------------------------------- item
existing = labels.loc[row["id"]] if row["id"] in labels.index else None
meta = f"#{ss.idx + 1} · {row['split']} · {SOURCE_NAMES[row['source']]}"
if existing is not None:
    meta += f" · labelled **{SHORT[existing['label']]}**"
    if existing["features"]:
        meta += f" ({existing['features']})"
st.caption(meta)
st.markdown(
    "<div style='white-space:pre-wrap;font-size:1.08rem;line-height:1.55;padding:1rem 1.1rem;"
    "border:1px solid rgba(128,128,128,.35);border-radius:10px;max-height:55vh;overflow:auto'>"
    f"{html.escape(str(row['text']))}</div>",
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------- controls
mapping: dict[str, str] = {
    "u": "unsure",
    "b": "back",
    "arrowleft": "back",
    "n": "next",
    "arrowright": "next",
}
focus = None
if ss.pending_fr:
    with st.form("fr_form", clear_on_submit=True, border=True):
        st.markdown(
            "**Feature request:** what is requested? (short phrases, `;`-separated, may be blank)"
        )
        st.text_input(
            "features",
            key="features",
            value=existing["features"] if existing is not None else "",
            label_visibility="collapsed",
            placeholder="e.g. sleep timer for podcasts; hide songs",
        )
        st.form_submit_button("Save ⏎", type="primary", on_click=save_fr)
    if st.button("Cancel (Esc)", key="cancel"):
        ss.pending_fr = False
        st.rerun()
    mapping["escape"] = "cancel"
    focus = "features"
else:
    cols = st.columns(3)
    for i, label in enumerate(LABELS):
        is_current = existing is not None and existing["label"] == label
        cols[i % 3].button(
            f"{KEYS[label]} · {SHORT[label]}",
            key=f"lbl_{label}",
            on_click=choose,
            args=(label,),
            width="stretch",
            type="primary" if is_current else "secondary",
        )
        mapping[KEYS[label]] = f"lbl_{label}"

n1, n2, n3 = st.columns(3)
n1.button("← back (b)", key="back", on_click=go, args=(-1,), width="stretch")
n2.button(
    f"unsure (u): {'ON' if ss.is_unsure else 'off'}",
    key="unsure",
    on_click=lambda: ss.update(is_unsure=not ss.is_unsure),
    width="stretch",
    type="primary" if ss.is_unsure else "secondary",
)
n3.button("next (n) →", key="next", on_click=go, args=(+1,), width="stretch")

keyboard(mapping, focus_key=focus)
