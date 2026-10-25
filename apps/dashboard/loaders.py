"""Cached readers for the precomputed pipeline outputs (PLAN.md §0 "Data for the dashboard").

Nothing here runs a model: every function reads a file written by a pipeline script.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frminer.config import ROOT, load_config
from frminer.rank import score

PROC = ROOT / "data" / "processed"
CORPUS = ROOT / "data" / "interim" / "corpus.parquet"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
REQUIRED = [PROC / "theme_stats.parquet", PROC / "request_themes.parquet", CORPUS]


def require_data() -> None:
    """Stop the page with setup instructions if the data bundle is not unpacked."""
    missing = [p.relative_to(ROOT) for p in REQUIRED if not p.exists()]
    if missing:
        st.error(
            "Missing pipeline outputs: "
            + ", ".join(f"`{m}`" for m in missing)
            + ". Unpack the data bundle at the repo root (PLAN.md §0, Getting set up)."
        )
        st.stop()


def config() -> dict:
    return load_config()


@st.cache_data(show_spinner=False)
def theme_stats() -> pd.DataFrame:
    return pd.read_parquet(PROC / "theme_stats.parquet")


@st.cache_data(show_spinner=False)
def themes() -> pd.DataFrame:
    return pd.read_parquet(PROC / "themes.parquet")


@st.cache_data(show_spinner=False)
def timeseries() -> pd.DataFrame:
    return pd.read_parquet(PROC / "theme_timeseries.parquet")


@st.cache_data(show_spinner=False)
def request_themes() -> pd.DataFrame:
    return pd.read_parquet(PROC / "request_themes.parquet")


@st.cache_data(show_spinner=False)
def theme_items() -> pd.DataFrame:
    """One row per (theme, item), as used for ranking (unassigned mentions dropped)."""
    return score.item_table(request_themes())


@st.cache_data(show_spinner=False)
def corpus_meta() -> pd.DataFrame:
    """Every corpus item with its source, date and SetFit label."""
    meta = pd.read_parquet(CORPUS, columns=["id", "source", "created_at", "rating"])
    labels = pd.read_parquet(PROC / "classified.parquet")
    return meta.merge(labels, on="id", how="left")


@st.cache_data(show_spinner=False)
def corpus_text() -> pd.Series:
    """Item text by id (for quotes)."""
    c = pd.read_parquet(CORPUS, columns=["id", "text"])
    return c.set_index("id")["text"]


@st.cache_data(show_spinner=False)
def insight(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / "insights" / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def detect_test() -> pd.DataFrame:
    """Gold-test detection results, one row per system (scripts/evaluate_detection.py)."""
    return pd.read_csv(TABLES / "detect_test.csv")


@st.cache_data(show_spinner=False)
def md_tables(name: str) -> list[pd.DataFrame]:
    """Every pipe table in reports/tables/<name>, in order, as string-typed frames."""
    tables, block = [], []
    for line in [*(TABLES / name).read_text().splitlines(), ""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif block:
            header, rows = block[0], [r for r in block[1:] if not set("".join(r)) <= set(":-")]
            tables.append(pd.DataFrame(rows, columns=header))
            block = []
    return tables


@st.cache_data(show_spinner=False)
def md_text(name: str) -> str:
    return (TABLES / name).read_text()


def deployed_f1() -> float:
    return float(detect_test().set_index("system").loc["SetFit (silver)", "macro_f1"])


def extractor_accuracy() -> dict[str, float]:
    """Share of gold feature requests the judge graded correct, per extraction method."""
    ext = md_tables("extract_eval.md")[0]
    ext = ext[ext["setting"] == "gold all"].set_index("method")
    return ext["judge_correct"].astype(float).to_dict()


def filter_window(
    df: pd.DataFrame, sources: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Rows from the chosen sources with created_at in [start, end)."""
    created = df["created_at"]
    return df[df["source"].isin(sources) & (created >= start) & (created < end)]
