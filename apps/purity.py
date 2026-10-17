"""Cluster purity check: tick the phrases that do NOT belong to each theme.

uv run streamlit run apps/purity.py      (or: make purity)

Shows the first `cluster.purity.human` sampled themes of method A (annotations/
purity_sample.csv, drawn by scripts/cluster_requests.py), one per page, with 10 random
member phrases. Answers are saved to annotations/purity_labels.csv after every theme;
the app resumes at the first unanswered theme.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from frminer.config import load_config, path

st.set_page_config(page_title="Cluster purity", page_icon="🧩", layout="centered")

OUT = path("annotations") / "purity_labels.csv"
N_HUMAN = load_config()["cluster"]["purity"]["human"]
NAME_FIT = ["yes", "partly", "no"]


@st.cache_data
def load_sample() -> pd.DataFrame:
    s = pd.read_csv(path("annotations") / "purity_sample.csv")
    return s[(s["method"] == "A") & (s["rank"] < N_HUMAN)]


def load_answers() -> pd.DataFrame:
    if OUT.exists():
        return pd.read_csv(OUT)
    return pd.DataFrame(columns=["rank", "cluster", "member", "intruder", "name_fit", "ts"])


def save(rank: int, cluster: int, members: list[str], intruders: set[str], fit: str) -> None:
    ans = load_answers()
    ans = ans[ans["rank"] != rank]  # re-answering a theme replaces the old answer
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    new = pd.DataFrame(
        {
            "rank": rank,
            "cluster": cluster,
            "member": members,
            "intruder": [m in intruders for m in members],
            "name_fit": fit,
            "ts": ts,
        }
    )
    pd.concat([ans, new]).sort_values("rank").to_csv(OUT, index=False)


sample = load_sample()
ranks = sorted(sample["rank"].unique())
done = set(load_answers()["rank"])
if "pos" not in st.session_state:
    todo = [r for r in ranks if r not in done]
    st.session_state.pos = ranks.index(todo[0]) if todo else len(ranks)

st.progress(len(done) / len(ranks), text=f"{len(done)} / {len(ranks)} themes checked")

if st.session_state.pos >= len(ranks):
    st.success("All themes checked, thank you! You can close this tab.")
    if st.button("← Review the last theme"):
        st.session_state.pos = len(ranks) - 1
        st.rerun()
    st.stop()

rank = ranks[st.session_state.pos]
rows = sample[sample["rank"] == rank]
first = rows.iloc[0]
members = rows["member"].tolist()
prev = load_answers()
prev = prev[prev["rank"] == rank]
prev_intruders = set(prev.loc[prev["intruder"].astype(bool), "member"])

st.subheader(first["name"])
st.caption(first["description"])
st.markdown(
    "**Tick every phrase that does _not_ ask for the same thing as this theme.** "
    "Leave the rest unticked."
)
with st.form(f"theme-{rank}"):
    intruders = {
        m
        for i, m in enumerate(members)
        if st.checkbox(m, value=m in prev_intruders, key=f"{rank}-{i}")
    }
    fit = st.radio(
        "Does the theme name describe these phrases?",
        NAME_FIT,
        index=NAME_FIT.index(prev["name_fit"].iloc[0]) if len(prev) else 0,
        horizontal=True,
    )
    submitted = st.form_submit_button("Save and next →", type="primary")
if submitted:
    save(rank, int(first["cluster"]), members, intruders, fit)
    st.session_state.pos += 1
    st.rerun()
if st.session_state.pos > 0 and st.button("← Back"):
    st.session_state.pos -= 1
    st.rerun()
