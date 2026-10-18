"""Insights (RQ4): platform differences, rating impact, competitors, and 'already exists'."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders
from theme import BAR_GAP_LINE, CHART_CONFIG, MUTED, SEQ_BLUE, SERIES, SOURCE_COLORS, SURFACE

N = 12
MIN_ITEMS = loaders.config()["insights"]["min_items"]
# Diverging star scale: red for 1★, neutral 3★, blue for 5★.
STAR_COLORS = {
    "r1": SERIES[7],
    "r2": "#f2a3a1",
    "r3": "#d6d5ce",
    "r4": SEQ_BLUE[1],
    "r5": SEQ_BLUE[4],
}
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

st.title("Insights")
platform_tab, ratings_tab, competitors_tab, existing_tab = st.tabs(
    ["Platform", "Ratings", "Competitors", "Already exists?"]
)

with platform_tab:
    plat = loaders.insight("platform")
    plat = plat[plat["play_items"] + plat["reddit_items"] >= MIN_ITEMS]
    picked = pd.concat([plat.nsmallest(N, "log2_ratio"), plat.nlargest(N, "log2_ratio")])
    picked = picked.sort_values("log2_ratio")
    play_lean = picked["log2_ratio"] < 0
    fig = go.Figure(
        go.Bar(
            x=picked["log2_ratio"],
            y=picked["name"],
            orientation="h",
            marker=dict(
                color=[
                    SOURCE_COLORS["play"] if p else SOURCE_COLORS["reddit_post"] for p in play_lean
                ],
                line=BAR_GAP_LINE,
            ),
            customdata=picked[["play_items", "reddit_items", "play_share", "reddit_share"]],
            hovertemplate=(
                "<b>%{y}</b><br>Play: %{customdata[0]:,} items (%{customdata[2]:.1%})"
                "<br>Reddit: %{customdata[1]:,} items (%{customdata[3]:.1%})"
                "<br>log₂ ratio %{x:+.2f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(
        height=26 * len(picked) + 80,
        xaxis_title="← more on Play Store · log₂(Reddit share / Play share) · more on Reddit →",
        yaxis=dict(automargin=True, gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=10),
    )
    st.markdown(
        f"The {N} most Play-leaning and {N} most Reddit-leaning themes. Shares are within each "
        "platform (a theme's items / all themed items from that platform), so the much larger "
        f"Play volume doesn't dominate. Themes with ≥ {MIN_ITEMS} items."
    )
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "Play reviewers ask about the free tier (ads, skips, shuffle, downloads); Reddit users "
        "about customisation and power-user features. "
        "Themes with 0 Reddit items sit at the left edge."
    )

with ratings_tab:
    rat = loaders.insight("ratings")
    rat = rat[rat["n"] >= MIN_ITEMS]
    angry = rat.nsmallest(N, "mean_rating")
    happy = rat.nlargest(N, "mean_rating")
    for title, d in [
        ("Themes in the angriest reviews", angry),
        ("Themes in the happiest reviews", happy),
    ]:
        d = d.iloc[::-1]
        fig = go.Figure()
        for col in ["r1", "r2", "r3", "r4", "r5"]:
            fig.add_bar(
                x=d[col] * 100,
                y=d["name"],
                orientation="h",
                name=f"{col[1]}★",
                marker=dict(color=STAR_COLORS[col], line=BAR_GAP_LINE),
                customdata=d[["n", "mean_rating"]],
                hovertemplate=(
                    "<b>%{y}</b><br>" + col[1] + "★: %{x:.0f}%"
                    "<br>%{customdata[0]:,} reviews · mean %{customdata[1]:.2f}★<extra></extra>"
                ),
            )
        fig.update_layout(
            barmode="stack",
            barcornerradius=0,
            height=26 * len(d) + 90,
            xaxis=dict(title="% of the theme's Play reviews", range=[0, 100]),
            yaxis=dict(automargin=True, gridcolor="rgba(0,0,0,0)"),
            legend=dict(traceorder="normal"),
            margin=dict(t=30),
        )
        st.markdown(f"**{title}** (by mean star rating)")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "Requests to switch things off (AI music, AI features, podcasts, playlist recommendations) "
        f"come in the lowest-rated reviews. Play reviews only; themes with ≥ {MIN_ITEMS} reviews."
    )
    with st.expander("All themes"):
        st.dataframe(
            rat.sort_values("mean_rating")[["name", "n", "mean_rating", "low_share", "high_share"]],
            hide_index=True,
            width="stretch",
            column_config={
                "name": st.column_config.TextColumn("Theme", width="large"),
                "n": st.column_config.NumberColumn("Reviews", format="%d"),
                "mean_rating": st.column_config.NumberColumn("Mean ★", format="%.2f"),
                "low_share": st.column_config.NumberColumn("≤2★", format="percent"),
                "high_share": st.column_config.NumberColumn("≥4★", format="percent"),
            },
        )

with competitors_tab:
    comp = loaders.insight("competitors")
    totals = comp.groupby("competitor")["items"].sum().sort_values(ascending=False)
    top_comp = totals.head(8).index.tolist()
    sub = comp[comp["competitor"].isin(top_comp)]
    top_themes = sub.groupby("name")["items"].sum().nlargest(N).index.tolist()
    grid = (
        sub[sub["name"].isin(top_themes)]
        .pivot_table(
            index="name", columns="competitor", values="items", aggfunc="sum", fill_value=0
        )
        .reindex(index=top_themes, columns=top_comp, fill_value=0)
    )
    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(
            go.Bar(
                x=totals.head(8).values,
                y=totals.head(8).index,
                orientation="h",
                marker=dict(color=SERIES[0], line=BAR_GAP_LINE),
                text=[f"{v:,}" for v in totals.head(8).values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:,} items<extra></extra>",
            )
        )
        fig.update_layout(
            height=360,
            xaxis_title="items mentioning it",
            yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, r=40),
        )
        st.markdown("**Competitors named in feature requests**")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    with right:
        fig = go.Figure(
            go.Heatmap(
                z=grid.values,
                x=grid.columns,
                y=grid.index,
                colorscale=[[0, SURFACE]]
                + [[(i + 1) / len(SEQ_BLUE), c] for i, c in enumerate(SEQ_BLUE)],
                xgap=2,
                ygap=2,
                text=[[f"{v}" if v else "" for v in row] for row in grid.values],
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(title="items", thickness=10),
                hovertemplate="%{y} · %{x}: %{z} items<extra></extra>",
            )
        )
        fig.update_layout(
            height=360,
            yaxis=dict(autorange="reversed", automargin=True, gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(side="top", gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=30),
        )
        st.markdown("**Which requests mention which competitor** (top themes)")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "Competitor names as extracted by the LLM and normalised by rules; a few variants "
        "(e.g. 'Apple' vs 'Apple Music') are not merged."
    )

with existing_tab:
    ex = loaders.insight("existing")
    counts = ex["status"].value_counts()
    k = st.columns(4)
    for col, status in zip(k, STATUS_TEXT, strict=True):
        col.metric(STATUS_TEXT[status][0], f"{int(counts.get(status, 0))} themes")
    st.caption(
        "Each theme was matched to its 3 nearest entries in a hand-written catalogue of existing "
        "Spotify features (annotations/existing_features.yaml) and an LLM decided the status. "
        "These are LLM judgments and not very stable (see PLAN.md §0 limitations)."
    )
    for status in ["exists", "premium_only", "partly"]:
        d = ex[ex["status"] == status].sort_values("items", ascending=False)
        title, explain = STATUS_TEXT[status]
        st.markdown(f"**{title}** · {explain}")
        st.dataframe(
            d[["rank", "name", "feature", "availability", "items"]],
            hide_index=True,
            width="stretch",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "name": st.column_config.TextColumn("Theme", width="large"),
                "feature": st.column_config.TextColumn("Existing feature", width="large"),
                "availability": "Availability",
                "items": st.column_config.NumberColumn("Items", format="%d"),
            },
        )
