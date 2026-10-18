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

    @property
    def is_full(self) -> bool:
        return self == full_window()

    def label(self) -> str:
        last = self.end - pd.Timedelta(days=1)
        return f"{self.start:%b %Y} – {last:%b %Y}"


def _months() -> list[pd.Timestamp]:
    """Month starts covering the corpus, plus the start of the month after the last."""
    created = loaders.corpus_meta()["created_at"]
    first = (
        created.min()
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
        .to_timestamp()
        .tz_localize("UTC")
    )
    last = (
        created.max()
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
        .to_timestamp()
        .tz_localize("UTC")
    )
    return list(pd.date_range(first, last + pd.offsets.MonthBegin(1), freq="MS"))


def full_window() -> Window:
    months = _months()
    return Window(SOURCES, months[0], months[-1])


def filter_row() -> Window:
    """Source and month-range controls in one row; values persist across pages."""
    months = _months()
    left, right = st.columns([2, 3])
    sources = left.multiselect(
        "Sources",
        SOURCES,
        default=SOURCES,
        format_func=SOURCE_LABELS.get,
        key="f_sources",
    )
    start, last = right.select_slider(
        "Months",
        options=months[:-1],
        value=(months[0], months[-2]),
        format_func=lambda m: f"{m:%b %Y}",
        key="f_months",
    )
    end = last + pd.offsets.MonthBegin(1)
    if not sources:
        st.warning("Pick at least one source.")
        st.stop()
    return Window(sources, start, end)
