"""Filter row shared by the pages: data sources and a month range."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

import loaders
from theme import SOURCE_LABELS, SOURCES


@dataclass(frozen=True)
class Window:
    sources: list[str]
    start: pd.Timestamp  # inclusive, UTC
    end: pd.Timestamp  # exclusive, UTC

    def label(self) -> str:
        last = self.end - pd.Timedelta(days=1)
        return f"{self.start:%b %Y} – {last:%b %Y}"


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    naive = ts.tz_convert("UTC").tz_localize(None)
    return naive.to_period("M").to_timestamp().tz_localize("UTC")


def _months() -> list[pd.Timestamp]:
    """Month starts covering the corpus, plus the start of the month after the last."""
    created = loaders.corpus_meta()["created_at"]
    first, last = _month_start(created.min()), _month_start(created.max())
    return list(pd.date_range(first, last + pd.offsets.MonthBegin(1), freq="MS"))


def full_window() -> Window:
    months = _months()
    return Window(SOURCES, months[0], months[-1])


def filter_row() -> Window:
    """Source and month-range controls in one row; values persist across pages.

    Streamlit drops a widget's state on pages that don't draw it, so the last values
    are kept under separate keys and used as the defaults when a page draws the row.
    """
    months = _months()
    left, right = st.columns([2, 3])
    sources = left.multiselect(
        "Sources",
        SOURCES,
        default=st.session_state.get("_f_sources", SOURCES),
        format_func=SOURCE_LABELS.get,
        key="f_sources",
    )
    start, last = right.select_slider(
        "Months",
        options=months[:-1],
        value=st.session_state.get("_f_months", (months[0], months[-2])),
        format_func=lambda m: f"{m:%b %Y}",
        key="f_months",
    )
    st.session_state["_f_sources"], st.session_state["_f_months"] = sources, (start, last)
    if not sources:
        st.warning("Pick at least one source.")
        st.stop()
    return Window(sources, start, last + pd.offsets.MonthBegin(1))
