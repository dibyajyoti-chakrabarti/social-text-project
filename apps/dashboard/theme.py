"""Plotly styling that matches the report figures (frminer.viz): same surface, ink and slots."""

from __future__ import annotations

import re

import plotly.graph_objects as go
import plotly.io as pio

from frminer.viz import AXIS, GRID, INK, INK_2, MUTED, SEQ_BLUE, SERIES, SOURCE_COLORS, SURFACE

SOURCE_LABELS = {
    "play": "Play Store",
    "reddit_post": "Reddit posts",
    "reddit_comment": "Reddit comments",
}
SOURCES = list(SOURCE_LABELS)

# Existing-feature status (insights/existing.parquet): short title, one-line meaning.
STATUS_TEXT = {
    "exists": (
        "Already exists",
        "Users ask for something everyone already has: a discoverability problem.",
    ),
    "premium_only": (
        "Premium only",
        "It exists, but only in Premium, and free users ask for it: a paywall.",
    ),
    "partly": ("Partly exists", "A limited version exists and users ask for more than it offers."),
    "new": ("Not in the catalogue", "No matching existing feature: a genuinely new request."),
}
MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~$<])")


def md_escape(text: str) -> str:
    """Show user text literally (no markdown, no $…$ maths)."""
    return MD_SPECIAL.sub(r"\\\1", text)


pio.templates["frminer"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="sans-serif", size=13, color=INK_2),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        colorway=SERIES,
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickcolor=AXIS, ticks=""),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickcolor=AXIS, ticks=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(color=INK, size=12)),
        margin=dict(l=8, r=8, t=40, b=8),
        barcornerradius=4,
        bargap=0.25,
    )
)
pio.templates.default = "frminer"

CHART_CONFIG = {"displayModeBar": False}

# 2px surface gap between adjacent / stacked fills.
BAR_GAP_LINE = dict(color=SURFACE, width=2)

__all__ = [
    "AXIS",
    "BAR_GAP_LINE",
    "CHART_CONFIG",
    "GRID",
    "INK",
    "INK_2",
    "MUTED",
    "SEQ_BLUE",
    "SERIES",
    "SOURCES",
    "SOURCE_COLORS",
    "SOURCE_LABELS",
    "STATUS_TEXT",
    "SURFACE",
    "md_escape",
]
