"""Models and theme matching shared by the Live Analyser and the Pipeline Trace.

Everything loads on first use (st.cache_resource), never when a page opens. Theme matching
follows scripts/cluster_requests.py: phrases are keyed by `feature.strip().lower()`, theme
centroids are the plain mean of their member phrases, and a phrase joins the nearest
theme only at cosine ≥ cluster.outlier_threshold (the bar used to attach noise phrases).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frminer.config import ROOT

import loaders
from theme import BAR_GAP_LINE, SERIES

MODEL_DIR = ROOT / "models" / "setfit_silver"
CFG = loaders.config()
EMBED_MODEL = CFG["cluster"]["embed_model"]
THEME_MIN_SIM = CFG["cluster"]["outlier_threshold"]


def model_ready() -> bool:
    return (MODEL_DIR / "config.json").exists()


@st.cache_resource(show_spinner="Loading the SetFit classifier…")
def classifier():
    from frminer.detect import ml

    return ml.load_setfit(MODEL_DIR)


@st.cache_resource(show_spinner="Loading the sentence encoder…")
def encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def classify(text: str) -> pd.Series:
    """Class probabilities for one text, ascending."""
    from frminer.detect import ml

    proba, classes = ml.predict_proba(classifier(), [text])
    return pd.Series(proba[0], index=classes).sort_values()


def proba_chart(probs: pd.Series) -> go.Figure:
    label = probs.idxmax()
    fig = go.Figure(
        go.Bar(
            x=probs.values * 100,
            y=[c.replace("_", " ") for c in probs.index],
            orientation="h",
            marker=dict(
                color=[SERIES[0] if c == label else "#c3c2b7" for c in probs.index],
                line=BAR_GAP_LINE,
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
    return fig


@dataclass
class Extraction:
    features: list[str]
    competitor: str | None
    method: str  # "Gemini" or "rules"
    error: str | None = None  # why the LLM failed, when it fell back to rules


def extract_features(text: str, use_llm: bool) -> Extraction:
    """The pipeline's LLM extractor (scripts/extract_corpus.py) or the offline rules; rules
    also stand in when the LLM call fails."""
    from frminer.extract import rules

    error = None
    if use_llm:
        from frminer.extract import llm_extract
        from frminer.llm import LLM

        try:
            res = llm_extract.extract(
                LLM(CFG["extract"]["llm"]["profile"]), [text], batch_size=1, workers=1
            )
            ext = res.outputs[0]
        except Exception as e:  # network down, quota spent, missing key
            ext, error = None, type(e).__name__
        if ext is not None:
            return Extraction(ext.features, rules.canonical_competitor(ext.competitor), "Gemini")
        error = error or "no answer"
    found = rules.competitors(text)
    return Extraction(rules.extract(text), found[0] if found else None, "rules", error)


def cluster_key(phrase: str) -> str:
    """The form a phrase was clustered in (scripts/cluster_requests.py)."""
    return str(phrase).strip().lower()


@st.cache_resource(show_spinner="Building theme centroids (first run only)…")
def theme_index() -> tuple[dict[str, tuple[int, float, int]], np.ndarray, np.ndarray, float]:
    """Every clustered phrase → (theme_id, cosine to its centroid, corpus mentions), the theme
    ids and their centroids, and how often the nearest centroid is a phrase's own theme (the
    accuracy of matching a new phrase by nearest centroid instead of clustering)."""
    from frminer.cluster import phrases

    pt = pd.read_parquet(loaders.PROC / "phrase_themes.parquet")
    known = {r.phrase: (int(r.theme_id), float(r.sim), int(r.mentions)) for r in pt.itertuples()}
    members = pt[pt["theme_id"] >= 0]
    emb = encoder().encode(members["phrase"].tolist(), normalize_embeddings=True, batch_size=256)
    labels = members["theme_id"].to_numpy()
    ids, cents = phrases.centroids(emb, labels)
    agreement = float((ids[(emb @ cents.T).argmax(axis=1)] == labels).mean())
    return known, ids, cents, agreement


@dataclass
class ThemeMatch:
    phrase: str
    key: str  # the clustered form of the phrase
    mentions: int  # times the corpus contains this exact phrase; > 0 means its theme is a lookup
    theme_id: int  # -1: no theme close enough
    similarity: float  # cosine to the assigned (or, without one, the nearest) centroid
    nearest: pd.DataFrame  # top-k themes by cosine


def match_themes(features: list[str], k: int = 3) -> list[ThemeMatch]:
    known, ids, cents, _ = theme_index()
    keys = [cluster_key(f) for f in features]
    sims = encoder().encode(keys, normalize_embeddings=True) @ cents.T
    stats = loaders.theme_stats().set_index("theme_id")
    out = []
    for phrase, key, row in zip(features, keys, sims, strict=True):
        top = np.argsort(-row)[:k]
        nearest = pd.DataFrame(
            {
                "theme": stats.loc[ids[top], "name"].to_numpy(),
                "similarity": row[top],
                "rank": stats.loc[ids[top], "rank"].to_numpy(),
                "items": stats.loc[ids[top], "items"].to_numpy(),
            }
        )
        best = float(row[top[0]])
        theme_id, sim, mentions = known.get(key, (-1, best, 0))
        if not mentions:
            theme_id = int(ids[top[0]]) if best >= THEME_MIN_SIM else -1
        sim = sim if theme_id >= 0 else best
        out.append(ThemeMatch(phrase, key, mentions, theme_id, sim, nearest))
    return out


NEAREST_COLUMNS = {
    "theme": st.column_config.TextColumn("Nearest themes", width="large"),
    "similarity": st.column_config.ProgressColumn(
        "Cosine", min_value=0.0, max_value=1.0, format="%.2f"
    ),
    "rank": st.column_config.NumberColumn("Rank"),
    "items": st.column_config.NumberColumn("Items", format="%d"),
}
