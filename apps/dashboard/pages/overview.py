"""Overview: headline numbers and the top request themes under adjustable priority weights."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from frminer.rank import score

import controls
import loaders
from theme import BAR_GAP_LINE, SOURCE_COLORS, SOURCE_LABELS, SOURCES

WEIGHT_HELP = {
    "volume": "log of the number of items asking for the theme",
    "engagement": "mean thumbs-up / upvote percentile (within each source)",
    "dissatisfaction": "share of the theme's Play reviews rated ≤ 2★",
    "growth": "change in share of all items, last 13 weeks vs the 13 before",
}

st.title("Feature Request Miner")
win = controls.filter_row()
st.caption(
    "Spotify feature requests mined from Google Play reviews and r/truespotify, "
    f"{controls.full_window().label()}."
)
meta = loaders.filter_window(loaders.corpus_meta(), win.sources, win.start, win.end)
items = loaders.filter_window(loaders.theme_items(), win.sources, win.start, win.end)
mentions = loaders.filter_window(loaders.request_themes(), win.sources, win.start, win.end)

# KPI row.
k = st.columns(4)
k[0].metric("Items analysed", f"{len(meta):,}")
k[1].metric(
    "Classified as feature requests",
    f"{(meta['label'] == 'feature_request').mean():.1%}" if len(meta) else "–",
    help="SetFit classifier trained on Gemini silver labels "
    f"(gold-test macro-F1 {loaders.deployed_f1():.2f}).",
)
k[2].metric(
    "Feature mentions", f"{len(mentions):,}", help="Requested features extracted by the LLM."
)
k[3].metric("Request themes", f"{items['theme_id'].nunique():,}")

# Priority weights.
defaults = loaders.config()["rank"]["weights"]
with st.expander("Priority weights", expanded=False):
    st.caption(
        "priority = Σ weight × z-score of each signal across themes. "
        "Signals use the full year, so source and month filters change the counts, not the ranking."
    )
    cols = st.columns(4)
    weights = {
        name: cols[i].slider(
            name.capitalize(),
            0.0,
            1.0,
            float(defaults[name]),
            0.05,
            key=f"w_{name}",
            help=WEIGHT_HELP[name],
        )
        for i, name in enumerate(defaults)
    }
    if st.button("Reset to defaults"):
        for name in defaults:
            st.session_state.pop(f"w_{name}", None)
        st.rerun()

stats = loaders.theme_stats().copy()
stats["default_rank"] = stats["rank"]
stats["priority"] = score.priority(stats, weights)
stats = stats.sort_values("priority", ascending=False, ignore_index=True)
stats["rank"] = range(1, len(stats) + 1)
stats["rank_change"] = stats["default_rank"] - stats["rank"]

counts = (
    items.groupby(["theme_id", "source"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=SOURCES, fill_value=0)
)
stats = stats.join(counts, on="theme_id").fillna({s: 0 for s in SOURCES})
stats["window_items"] = stats[SOURCES].sum(axis=1).astype(int)

# Top themes chart: ordered by priority, bar length = items in the filtered window.
st.subheader("Top request themes")
n = st.segmented_control("Show top", [10, 20, 30], default=20, key="top_n") or 20
st.caption(f"Ordered by priority; bar length = items asking for the theme, {win.label()}.")
top = stats.head(n).iloc[::-1]
labels = [f"{r} · {name}" for r, name in zip(top["rank"], top["name"], strict=True)]
fig = go.Figure()
for src in win.sources:
    fig.add_bar(
        y=labels,
        x=top[src],
        orientation="h",
        name=SOURCE_LABELS[src],
        marker=dict(color=SOURCE_COLORS[src], line=BAR_GAP_LINE),
        customdata=top[["priority", "low_rating_share", "trend", "window_items"]],
        hovertemplate=(
            "<b>%{y}</b><br>" + SOURCE_LABELS[src] + ": %{x:,} items"
            "<br>all selected sources: %{customdata[3]:,}"
            "<br>priority %{customdata[0]:.2f} · ≤2★ share %{customdata[1]:.0%}"
            " · %{customdata[2]}<extra></extra>"
        ),
    )
fig.update_layout(
    barmode="stack",
    legend=dict(traceorder="normal"),
    margin=dict(t=30),
    height=max(320, 26 * n + 90),
    xaxis_title="items",
    yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# Table view of the same ranking.
stats["low_rating_share"] = stats["low_rating_share"] * 100
table = stats[
    [
        "rank",
        "rank_change",
        "name",
        "window_items",
        "items",
        "priority",
        "low_rating_share",
        "growth",
        "trend",
        "description",
    ]
]
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "rank": st.column_config.NumberColumn("Rank", width="small"),
        "rank_change": st.column_config.NumberColumn(
            "Δ rank", help="Places gained vs the default weights", format="%+d", width="small"
        ),
        "name": st.column_config.TextColumn("Theme", width="medium"),
        "window_items": st.column_config.NumberColumn("Items (filtered)", format="%d"),
        "items": st.column_config.NumberColumn("Items (all)", format="%d"),
        "priority": st.column_config.NumberColumn("Priority", format="%.2f"),
        "low_rating_share": st.column_config.ProgressColumn(
            "≤2★ share",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            help="Share of the theme's Play reviews rated 1 or 2 stars",
        ),
        "growth": st.column_config.NumberColumn(
            "Growth (log₂)", format="%+.2f", help="Blank: too few items in the two windows"
        ),
        "trend": st.column_config.TextColumn("Trend", width="small"),
        "description": st.column_config.TextColumn("Description", width="large"),
    },
)
last_item = loaders.corpus_meta()["created_at"].max()
st.caption(
    f"{len(stats)} themes from {loaders.request_themes()['theme_id'].ge(0).sum():,} assigned "
    "mentions (clustered with MiniLM → UMAP → HDBSCAN, named by Gemini). "
    f"Unassigned mentions are left out. Latest item: {last_item:%d %b %Y}."
)
