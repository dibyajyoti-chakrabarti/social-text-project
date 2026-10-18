"""Trends: which themes grow or shrink, last quarter vs the one before, and monthly shapes."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import loaders
from theme import CHART_CONFIG, GRID, INK_2, MUTED, SERIES, SURFACE

# Emphasis: significant movers in colour, the rest recede (scatter) or stay slot 1 (lines).
TREND_COLORS = {"declining": SERIES[1], "emerging": SERIES[2], "stable": "#c3c2b7"}


def label_positions(d: pd.DataFrame) -> list[str]:
    """Stagger labels of points that sit close together (by growth) so they don't collide."""
    order = d["growth"].rank(method="first").astype(int) - 1
    cycle = ["bottom right", "top right", "middle right"]
    return [cycle[i % len(cycle)] for i in order]


cfg = loaders.config()["rank"]
weeks, tcfg = cfg["window_weeks"], cfg["trend"]
stats = loaders.theme_stats()
moving = stats[stats["trend"] != "stable"].sort_values("growth")
n_emerging, n_declining = (
    (stats["trend"] == "emerging").sum(),
    (stats["trend"] == "declining").sum(),
)

st.title("Trends")
st.markdown(
    f"Each theme's **share of all items** in the last {weeks} weeks vs the {weeks} weeks before. "
    f"A theme is *emerging* or *declining* when the share changes at least {tcfg['min_ratio']}× "
    f"with ≥ {tcfg['min_recent']} recent items and a two-proportion test is significant "
    f"(Benjamini–Hochberg q < {tcfg['alpha']}). 5★ Play reviews are left out of trends: "
    "Google Play only returns them for the last weeks of the year."
)
k = st.columns(3)
k[0].metric("Themes with enough data", f"{stats['growth'].notna().sum()} of {len(stats)}")
k[1].metric("Emerging", int(n_emerging))
k[2].metric("Declining", int(n_declining))
if n_emerging == 0:
    st.info(
        "No theme is significantly emerging in the last quarter"
        + ("; the significant moves are all declines." if n_declining else ".")
    )

# Growth vs size: every theme with a growth estimate, significant movers labelled.
data = stats.dropna(subset=["growth"])
fig = go.Figure()
for trend in ["stable", "declining", "emerging"]:
    d = data[data["trend"] == trend]
    if d.empty:
        continue
    fig.add_scatter(
        x=d["items"],
        y=d["growth"],
        mode="markers+text" if trend != "stable" else "markers",
        name=trend,
        text=d["name"] if trend != "stable" else None,
        textposition=label_positions(d) if trend != "stable" else None,
        textfont=dict(color=INK_2, size=12),
        marker=dict(
            color=TREND_COLORS[trend],
            size=11 if trend != "stable" else 8,
            line=dict(color=SURFACE, width=2),
        ),
        customdata=d[["name", "prev_items", "recent_items", "q"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{x:,} items · growth %{y:+.2f} log₂"
            "<br>previous %{customdata[1]:,} → recent %{customdata[2]:,} items"
            "<br>q = %{customdata[3]:.3f}<extra></extra>"
        ),
    )
fig.add_hline(y=0, line=dict(color=MUTED, width=1))
fig.update_layout(
    height=460,
    xaxis=dict(type="log", title="items (log scale)"),
    yaxis_title="growth: log₂ change in share",
    margin=dict(t=30),
)
st.subheader("Growth vs size")
st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
st.caption(
    "Above 0: a larger share of items in the recent quarter. "
    f"Themes with fewer than {cfg['min_support']} items in the two windows have no growth estimate."
)

# Movers table: significant ones, plus the largest non-significant rises for context.
st.subheader("Biggest movers")
rising = data[data["trend"] == "stable"].nlargest(5, "growth")
table = pd.concat([moving, rising])
cols = ["name", "trend", "prev_items", "recent_items", "ratio", "growth", "q"]
st.dataframe(
    table[cols],
    hide_index=True,
    width="stretch",
    column_config={
        "name": st.column_config.TextColumn("Theme", width="large"),
        "trend": "Trend",
        "prev_items": st.column_config.NumberColumn(f"Previous {weeks} wk", format="%d"),
        "recent_items": st.column_config.NumberColumn(f"Last {weeks} wk", format="%d"),
        "ratio": st.column_config.NumberColumn("Share ratio", format="%.2f×"),
        "growth": st.column_config.NumberColumn("Growth (log₂)", format="%+.2f"),
        "q": st.column_config.NumberColumn("q (BH)", format="%.3f"),
    },
)
st.caption("Significant movers first, then the five largest rises that are not significant.")

# Small multiples of monthly share.
st.subheader("Monthly share, theme by theme")
top = stats.nsmallest(8, "rank")
default = list(dict.fromkeys(moving["theme_id"].tolist() + top["theme_id"].tolist()))[:12]
names = dict(zip(stats["theme_id"], stats["name"], strict=True))
chosen = st.multiselect(
    "Themes",
    stats.sort_values("rank")["theme_id"].tolist(),
    default=default,
    format_func=names.get,
    key="trend_themes",
    help="Defaults: the significant movers and the top-ranked themes.",
)
if chosen:
    series = loaders.timeseries()
    series = series[series["theme_id"].isin(chosen)].assign(
        theme=lambda d: d["theme_id"].map(names), pct=lambda d: d["share"] * 100
    )
    trend_of = dict(zip(stats["theme_id"], stats["trend"], strict=True))
    color_map = {
        names[t]: SERIES[0] if trend_of[t] == "stable" else TREND_COLORS[trend_of[t]]
        for t in chosen
    }
    wrap = 4
    fig = px.line(
        series,
        x="month",
        y="pct",
        facet_col="theme",
        facet_col_wrap=wrap,
        facet_col_spacing=0.05,
        facet_row_spacing=0.12,
        color="theme",
        color_discrete_map=color_map,
        category_orders={"theme": [names[t] for t in chosen]},
        custom_data=["items"],
    )
    fig.update_traces(
        hovertemplate="%{x|%b %Y}: %{y:.2f}% (%{customdata[0]:,} items)<extra></extra>"
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1], font=dict(size=12)))
    fig.update_yaxes(
        matches=None, rangemode="tozero", title=None, showticklabels=True, gridcolor=GRID
    )
    fig.update_xaxes(title=None, tickformat="%b", dtick="M3")
    rows = -(-len(chosen) // wrap)
    fig.update_layout(height=200 * rows + 40, showlegend=False, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "% of all corpus items per month; each panel has its own y-scale. "
        "Orange: significantly declining; green: significantly emerging. "
        "First and last months are partial."
    )
