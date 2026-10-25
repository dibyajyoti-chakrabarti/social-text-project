"""Pipeline Trace: write a comment about Spotify in a chat box and follow it through every
pipeline stage to the theme it ends up in.

Each step calls the code the batch pipeline ran on the corpus and names the script. Two
stages work on the whole corpus and cannot be rerun for one comment, so they are replayed:
- clustering (UMAP + HDBSCAN over all phrases): a phrase the corpus already contains keeps
  its stored theme; a new phrase joins the nearest theme centroid at cosine ≥
  cluster.outlier_threshold, the rule that attached noise phrases after clustering.
- ranking: one comment does not move a theme's rank, so the card shows the corpus figures.

Models load on the first message. Everything runs offline unless LLM extraction is on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from frminer.config import ROOT

import inference
import loaders
from theme import CHART_CONFIG, SOURCE_LABELS, STATUS_TEXT, md_escape

MIN_WORDS = loaders.config()["data"]["min_words"]
STAGES = ["Clean", "Detect", "Extract", "Normalise", "Theme", "Rank"]
EXAMPLES = {
    "Sleep timer": "Please add a sleep timer for podcasts, Apple Music already has one.",
    "Hide songs": "I wish I could hide songs from a playlist without deleting them.",
    "Something new": "Would love a karaoke mode that scores how well I sing along.",
    "Bug report": "The app keeps crashing every time I open my library since the last update.",
}
N_QUOTES = 2
QUOTE_WORDS = (8, 60)


@dataclass
class Trace:
    text: str
    cleaned: str = ""
    n_words: int = 0
    english: bool = True
    probs: pd.Series | None = None
    extraction: inference.Extraction | None = None
    norms: list[str] = field(default_factory=list)  # feature_norm of each extracted feature
    matches: list[inference.ThemeMatch] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # stage → why the pipeline drops it
    stopped: str | None = None  # stage the trace ended at, if it did not reach a theme

    def drop(self, stage: str, why: str, follow: bool) -> bool:
        """Record a pipeline filter; True when the trace should stop here."""
        self.dropped[stage] = why
        if not follow:
            self.stopped = stage
        return not follow


def run(text: str, use_llm: bool, follow: bool, status) -> Trace:
    from frminer.extract.normalize import normalize_many
    from frminer.ingest.clean import clean_text, english_mask

    t = Trace(text)
    status.update(label="1/5 Cleaning…")
    t.cleaned = clean_text(text)
    t.n_words = len(t.cleaned.split())
    t.english = bool(english_mask(pd.Series([t.cleaned])).iloc[0])
    why = []
    if t.n_words < MIN_WORDS:
        why.append(f"{t.n_words} words, under the {MIN_WORDS}-word minimum")
    if not t.english:
        why.append("not detected as English")
    if why and t.drop("Clean", " and ".join(why), follow):
        return t

    status.update(label="2/5 Classifying with SetFit…")
    t.probs = inference.classify(t.cleaned)
    label = t.probs.idxmax()
    if label != "feature_request" and t.drop(
        "Detect", f"classified as {label.replace('_', ' ')}, not a feature request", follow
    ):
        return t

    status.update(label="3/5 Asking Gemini…" if use_llm else "3/5 Extracting with rules…")
    t.extraction = inference.extract_features(t.cleaned, use_llm)
    if not t.extraction.features:
        t.drop("Extract", "no requested feature found", follow=False)
        return t

    status.update(label="4/5 Normalising…")
    t.norms = normalize_many(t.extraction.features)
    kept = [f for f, n in zip(t.extraction.features, t.norms, strict=True) if n]
    if not kept:
        t.drop("Normalise", "every phrase was only filler", follow=False)
        return t

    status.update(label="5/5 Matching themes…")
    t.matches = inference.match_themes(kept)
    if all(m.theme_id < 0 for m in t.matches):
        t.stopped = "Theme"
    return t


# Rendering.


def stage_state(t: Trace, stage: str) -> str:
    """done | warn (filtered, but followed) | stop | skipped."""
    order = STAGES.index(stage)
    if t.stopped and order > STAGES.index(t.stopped):
        return "skipped"
    if t.stopped == stage:
        return "stop"
    return "warn" if stage in t.dropped else "done"


ICONS = {
    "done": ":material/check_circle:",
    "warn": ":material/warning:",
    "stop": ":material/block:",
    "skipped": ":material/radio_button_unchecked:",
}
STRIP = {
    "done": ":green[:material/check_circle:]",
    "warn": ":orange[:material/warning:]",
    "stop": ":red[:material/block:]",
    "skipped": ":gray[:material/radio_button_unchecked:]",
}


def strip(t: Trace) -> str:
    return "  →  ".join(f"{STRIP[stage_state(t, s)]} {s}" for s in STAGES)


def step(t: Trace, n: int, stage: str, summary: str, expanded: bool):
    state = stage_state(t, stage)
    return st.expander(f"{n}. {stage}: {summary}", expanded=expanded, icon=ICONS[state])


def note_drop(t: Trace, stage: str) -> None:
    if stage not in t.dropped:
        return
    if t.stopped == stage:
        st.error(f"The pipeline drops the comment here: {t.dropped[stage]}.")
    else:
        st.warning(
            f"The pipeline would drop the comment here ({t.dropped[stage]}). "
            "Following it anyway to show the later stages."
        )


def show_clean(t: Trace, n: int, open_: bool) -> None:
    ok = "Clean" not in t.dropped
    summary = f"{t.n_words} words, {'English' if t.english else 'not English'}"
    with step(t, n, "Clean", summary + ("" if ok else ", filtered out"), open_):
        st.caption(
            "scripts/build_corpus.py · `frminer.ingest.clean`: the same cleaning and filters "
            "every Play review and Reddit post went through."
        )
        if t.cleaned != t.text.strip():
            st.markdown("**Cleaned text:** " + md_escape(t.cleaned))
        else:
            st.markdown("Nothing to clean: no user mentions, HTML entities or extra whitespace.")
        a, b = st.columns(2)
        a.metric("Words", t.n_words, f"minimum {MIN_WORDS}", delta_color="off", delta_arrow="off")
        b.metric("Language (lingua)", "English" if t.english else "Other")
        note_drop(t, "Clean")
        if t.stopped == "Clean":
            st.caption("Turn on *Follow past filters* to see what the later stages would do.")


def show_detect(t: Trace, n: int, i: int, open_: bool) -> None:
    probs = t.probs
    label = probs.idxmax()
    summary = f"{label.replace('_', ' ')} ({probs.max():.0%})"
    with step(t, n, "Detect", summary, open_):
        meta = loaders.corpus_meta()
        n_fr = int((meta["label"] == "feature_request").sum())
        st.caption(
            "scripts/classify_corpus.py · SetFit trained on 12k Gemini silver labels "
            f"(gold-test macro-F1 {loaders.deployed_f1():.2f}). It labelled all "
            f"{len(meta):,} corpus items; only the {n_fr:,} feature requests go on to extraction."
        )
        st.plotly_chart(
            inference.proba_chart(probs), width="stretch", config=CHART_CONFIG, key=f"trace_p{i}"
        )
        note_drop(t, "Detect")
        if t.stopped == "Detect":
            st.caption("Turn on *Follow past filters* to see what the later stages would do.")


def show_extract(t: Trace, n: int, open_: bool) -> None:
    ext = t.extraction
    found = len(ext.features)
    summary = f"{found} feature{'s' * (found != 1)} ({ext.method})"
    with step(t, n, "Extract", summary, open_):
        acc = loaders.extractor_accuracy()
        st.caption(
            "scripts/extract_corpus.py · the corpus was extracted with Gemini (a correct feature "
            f"for {acc['llm']:.0%} of gold requests); the offline rules get {acc['rules']:.0%}."
        )
        if ext.error:
            st.warning(f"LLM extraction failed ({ext.error}), so the rules stood in.")
        if ext.features:
            st.markdown("  \n".join(f"- `{f}`" for f in ext.features))
        st.markdown(f"**Competitor mentioned:** {ext.competitor or 'none'}")
        if t.stopped == "Extract":
            st.error("Nothing requested, so the comment never reaches a theme.")
            if ext.method == "rules":
                st.caption("The rules miss implicit requests; try the LLM toggle.")


def show_normalise(t: Trace, n: int, i: int, open_: bool) -> None:
    feats = t.extraction.features
    kept = sum(bool(x) for x in t.norms)
    with step(t, n, "Normalise", f"{kept} of {len(feats)} phrases kept", open_):
        st.caption(
            "scripts/extract_corpus.py · `frminer.extract.normalize`: strip filler (please, add, "
            "an option to) and lemmatise nouns into a counting key; phrases that are only filler "
            "are dropped. Clustering embeds the phrase itself, lowercased."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "extracted": feats,
                    "counting key": [x or "(dropped)" for x in t.norms],
                    "clustered as": [inference.cluster_key(f) for f in feats],
                }
            ),
            hide_index=True,
            width="stretch",
            key=f"trace_n{i}",
        )
        if t.stopped == "Normalise":
            st.error("Nothing left after normalising, so the comment never reaches a theme.")


def show_theme(t: Trace, n: int, i: int, open_: bool) -> None:
    names = loaders.theme_stats().set_index("theme_id")["name"]
    hits = sorted({names[m.theme_id] for m in t.matches if m.theme_id >= 0})
    summary = ", ".join(hits) if hits else "no close theme"
    with step(t, n, "Theme", summary, open_):
        known, _, _, agreement = inference.theme_index()
        st.caption(
            "scripts/cluster_requests.py · MiniLM embeddings → UMAP → HDBSCAN over all "
            f"{len(known):,} distinct corpus phrases, then outliers within cosine "
            f"{inference.THEME_MIN_SIM} of a theme centroid were attached to it. Clustering "
            "cannot be rerun for one comment, so a new phrase goes to its nearest centroid; for "
            f"the corpus phrases that rule agrees with clustering {agreement:.0%} of the time."
        )
        for j, m in enumerate(t.matches):
            with st.container(border=True):
                st.markdown(f"**“{md_escape(m.phrase)}”**")
                if m.mentions and m.theme_id >= 0:
                    st.markdown(
                        f"The corpus has this exact phrase {m.mentions:,} time"
                        f"{'s' * (m.mentions != 1)}, and clustering put it in "
                        f"**{names[m.theme_id]}** (cosine {m.similarity:.2f} to the centroid)."
                    )
                elif m.mentions:
                    st.markdown(
                        f"The corpus has this exact phrase {m.mentions:,} time"
                        f"{'s' * (m.mentions != 1)}, but clustering left it without a theme."
                    )
                elif m.theme_id >= 0:
                    st.markdown(
                        f"A new phrase. Its nearest centroid is **{names[m.theme_id]}** at "
                        f"cosine {m.similarity:.2f} ≥ {inference.THEME_MIN_SIM}, so it joins "
                        "that theme."
                    )
                else:
                    st.markdown(
                        f"A new phrase. Its nearest centroid is only at cosine {m.similarity:.2f} "
                        f"< {inference.THEME_MIN_SIM}, so it stays unassigned: possibly a "
                        "request nobody in the corpus made."
                    )
                st.dataframe(
                    m.nearest,
                    hide_index=True,
                    width="stretch",
                    column_config=inference.NEAREST_COLUMNS,
                    key=f"trace_t{i}_{j}",
                )


def quotes(theme_id: int, phrases: set[str]) -> pd.DataFrame:
    """Corpus comments in this theme, those with the same phrase first, then most confident."""
    rt = loaders.request_themes()
    rows = rt[rt["theme_id"] == theme_id].drop_duplicates("id")
    text = loaders.corpus_text()
    rows = rows.assign(
        body=rows["id"].map(text),
        same=rows["feature"].str.strip().str.lower().isin(phrases),
    )
    words = rows["body"].str.split().str.len()
    focused = words.between(*QUOTE_WORDS)
    rows = rows[focused] if focused.sum() >= N_QUOTES else rows
    return rows.sort_values(["same", "confidence"], ascending=False).head(N_QUOTES)


def show_rank(t: Trace, n: int, open_: bool) -> None:
    stats = loaders.theme_stats().set_index("theme_id")
    existing = loaders.insight("existing").set_index("theme_id")
    top = loaders.themes().set_index("theme_id")["top_phrases"]
    ids = list(dict.fromkeys(m.theme_id for m in t.matches if m.theme_id >= 0))
    summary = ", ".join(f"#{int(stats.loc[x, 'rank'])} of {len(stats)}" for x in ids)
    with step(t, n, "Rank", summary, open_):
        w = loaders.config()["rank"]["weights"]
        st.caption(
            f"scripts/rank_and_insights.py · priority = {w['volume']} volume + "
            f"{w['engagement']} engagement + {w['dissatisfaction']} share of ≤2★ reviews + "
            f"{w['growth']} growth (each as a z-score). One comment does not change these "
            "corpus figures."
        )
        for x in ids:
            th = stats.loc[x]
            with st.container(border=True):
                st.markdown(f"#### {th['name']}")
                st.caption(th["description"])
                a, b, c, d = st.columns(4)
                a.metric("Rank", f"{int(th['rank'])} of {len(stats)}")
                b.metric(
                    "Items",
                    f"{int(th['items']):,}",
                    f"{int(th['play_items']):,} Play · {int(th['reddit_items']):,} Reddit",
                    delta_color="off",
                    delta_arrow="off",
                )
                c.metric("Low-rating share", f"{th['low_rating_share']:.0%}")
                d.metric("Trend", th["trend"])
                if x in existing.index:
                    status = existing.loc[x, "status"]
                    title, meaning = STATUS_TEXT.get(status, (status, ""))
                    feature = existing.loc[x, "feature"]
                    match = f" (closest: {feature})" if status != "new" and feature else ""
                    st.markdown(f"**Already in Spotify?** {title}{match}. {meaning}")
                st.markdown("**Most asked as:** " + ", ".join(md_escape(p) for p in top[x][:5]))
                phrases = {m.key for m in t.matches if m.theme_id == x}
                for r in quotes(x, phrases).itertuples():
                    body = md_escape(" ".join(str(r.body).split()))
                    stars = f" · {int(r.rating)}★" if pd.notna(r.rating) else ""
                    st.markdown(f"> {body}")
                    st.caption(
                        f"{SOURCE_LABELS[r.source]} · {r.created_at:%d %b %Y}{stars} · "
                        f"extracted: {md_escape(r.feature)}"
                    )


def show_result(t: Trace) -> None:
    stats = loaders.theme_stats().set_index("theme_id")
    ids = list(dict.fromkeys(m.theme_id for m in t.matches if m.theme_id >= 0))
    if ids:
        lines = [f"**{stats.loc[x, 'name']}** (rank {int(stats.loc[x, 'rank'])})" for x in ids]
        st.success("**End theme:** " + "; ".join(lines), icon=":material/flag:")
    elif t.stopped == "Theme":
        st.info("**End theme:** none. No existing theme is close: a new request.")
    else:
        st.info(f"**End theme:** none. Dropped at *{t.stopped}*: {t.dropped[t.stopped]}.")


def render(t: Trace, i: int, latest: bool) -> None:
    st.markdown(strip(t))
    show_clean(t, 1, latest)
    if t.probs is not None:
        show_detect(t, 2, i, latest)
    if t.extraction is not None:
        show_extract(t, 3, latest)
    if t.norms:
        show_normalise(t, 4, i, latest)
    if t.matches:
        show_theme(t, 5, i, latest)
        if any(m.theme_id >= 0 for m in t.matches):
            show_rank(t, 6, latest)
    show_result(t)


# Page.

st.title("Pipeline Trace")
st.caption(
    "Write a comment about Spotify the way a user would, and follow it through each stage of "
    "the pipeline to the theme it ends up in. Classification, normalisation and theme matching "
    "run on this machine; nothing leaves it unless LLM extraction is switched on."
)
if not inference.model_ready():
    st.error(
        f"No classifier at `{inference.MODEL_DIR.relative_to(ROOT)}`: unpack the model bundle "
        "(PLAN.md §0)."
    )
    st.stop()

history: list[Trace] = st.session_state.setdefault("trace_history", [])
left, mid, right = st.columns([2, 2, 1], vertical_alignment="center")
use_llm = left.toggle(
    "Use LLM extraction (Gemini)",
    value=False,
    key="trace_llm",
    help="Off: dependency-parse rules, fully offline. On: the extractor the pipeline used, one "
    "Gemini request per new comment (cached). Needs internet and GEMINI_KEY_* in .env.",
)
follow = mid.toggle(
    "Follow past filters",
    value=False,
    key="trace_follow",
    help="Off: the trace stops where the pipeline would drop the comment. On: it notes the drop "
    "and carries on, to show what the later stages would do.",
)
if right.button("Clear chat", disabled=not history, width="stretch"):
    history.clear()
    st.rerun()

for i, t in enumerate(history):
    with st.chat_message("user"):
        st.text(t.text)
    with st.chat_message("assistant", avatar=":material/account_tree:"):
        render(t, i, latest=i == len(history) - 1)

if not history:
    with st.chat_message("assistant", avatar=":material/account_tree:"):
        st.markdown(
            "Type a comment below, or start from an example. I'll show what each stage does "
            "with it: " + " → ".join(STAGES) + "."
        )
picked = None
cols = st.columns(len(EXAMPLES))
for col, (name, example) in zip(cols, EXAMPLES.items(), strict=True):
    if col.button(name, key=f"trace_ex_{name}", width="stretch", help=example):
        picked = example

prompt = st.chat_input("Write a comment about Spotify…") or picked
if prompt:
    with st.chat_message("user"):
        st.text(prompt)
    with (
        st.chat_message("assistant", avatar=":material/account_tree:"),
        st.status("Running the pipeline…") as status,
    ):
        trace = run(prompt, use_llm, follow, status)
        status.update(label="Done", state="complete")
    history.append(trace)
    st.rerun()
