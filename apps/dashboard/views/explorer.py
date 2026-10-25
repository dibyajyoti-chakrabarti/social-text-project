"""Request Explorer: one theme in depth — volume over time, who asks, ratings, quotes."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import controls
import loaders
from theme import BAR_GAP_LINE, CHART_CONFIG, SERIES, SOURCE_COLORS, SOURCE_LABELS, md_escape

N_QUOTES = 6
QUOTE_WORDS = (8, 80)  # prefer focused quotes over long rambles
st.title("Request Explorer")

stats = loaders.theme_stats().sort_values("rank")
names = dict(zip(stats["theme_id"], stats["name"], strict=True))
ranks = dict(zip(stats["theme_id"], stats["rank"], strict=True))
theme_id = st.selectbox(
    "Theme",
    stats["theme_id"].tolist(),
    format_func=lambda t: f"{ranks[t]} · {names[t]}",
    key="explorer_theme",
    help="Ordered by priority rank (default weights).",
)
win = controls.filter_row()

row = stats.set_index("theme_id").loc[theme_id]
theme = loaders.themes().set_index("theme_id").loc[theme_id]
items = loaders.filter_window(loaders.theme_items(), win.sources, win.start, win.end)
items = items[items["theme_id"] == theme_id]

st.subheader(row["name"])
st.write(row["description"])

k = st.columns(4)
k[0].metric("Priority rank", f"{int(row['rank'])} of {len(stats)}")
k[1].metric(
    "Items", f"{len(items):,}", help=f"In the filtered window; {int(row['items']):,} in all."
)
low = row["low_rating_share"]
k[2].metric("≤2★ share", "–" if pd.isna(low) else f"{low:.0%}", help="Of the theme's Play reviews")
growth = row["growth"]
k[3].metric(
    "Trend (last 13 weeks)",
    row["trend"],
    None if pd.isna(growth) else f"{growth:+.2f} log₂ share",
    delta_color="normal" if row["trend"] != "stable" else "off",
    help="Theme's share of all items (5★ Play reviews left out), last 13 weeks vs the 13 before; "
    "'declining'/'emerging' only when significant (BH-corrected).",
)

phrases = ", ".join(p.rsplit(" (", 1)[0] for p in theme["top_phrases"][:10])
st.caption(f"**Most common phrasings:** {phrases}")

if items.empty:
    st.info("No items for this theme in the selected sources and months.")
    st.stop()

left, right = st.columns([3, 2])

# Volume over time (full corpus months; the window filter shades the selection).
series = loaders.timeseries()
series = series[series["theme_id"] == theme_id].sort_values("month")
fig = go.Figure(
    go.Scatter(
        x=series["month"],
        y=series["share"] * 100,
        mode="lines+markers",
        line=dict(color=SERIES[0], width=2),
        marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
        customdata=series["items"],
        hovertemplate="%{x|%b %Y}: %{y:.2f}% of all items (%{customdata:,} items)<extra></extra>",
    )
)
full = controls.full_window()
if (win.start, win.end) != (full.start, full.end):
    fig.add_vrect(
        x0=win.start.tz_localize(None),
        x1=(win.end - pd.Timedelta(days=1)).tz_localize(None),
        fillcolor=SERIES[0],
        opacity=0.08,
        line_width=0,
    )
fig.update_layout(
    height=300,
    yaxis_title="% of all items",
    yaxis_rangemode="tozero",
    hovermode="x unified",
    margin=dict(t=10),
)
with left:
    st.markdown("**Share of all items per month** (all sources; 5★ Play reviews left out)")
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.caption("First and last months are partial. Shaded: selected months.")

# Who asks: items per source.
by_source = items["source"].value_counts().reindex(win.sources, fill_value=0)
fig = go.Figure(
    go.Bar(
        y=[SOURCE_LABELS[s] for s in by_source.index],
        x=by_source.values,
        orientation="h",
        marker=dict(color=[SOURCE_COLORS[s] for s in by_source.index], line=BAR_GAP_LINE),
        text=[f"{v:,}" for v in by_source.values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,} items<extra></extra>",
    )
)
fig.update_layout(
    height=170, xaxis_title="items", yaxis=dict(autorange="reversed"), margin=dict(t=10, r=40)
)
with right:
    st.markdown("**Who asks**")
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

# Ratings of Play reviews: theme vs all Play reviews in the window.
play = items[(items["source"] == "play") & items["rating"].notna()]
with right:
    st.markdown("**Star ratings** (Play reviews)")
    if play.empty:
        st.caption("No Play reviews in the selection.")
    else:
        corpus = loaders.filter_window(loaders.corpus_meta(), ["play"], win.start, win.end)
        stars = [1, 2, 3, 4, 5]
        theme_dist = (
            play["rating"].astype(int).value_counts(normalize=True).reindex(stars, fill_value=0)
        )
        all_dist = (
            corpus["rating"]
            .dropna()
            .astype(int)
            .value_counts(normalize=True)
            .reindex(stars, fill_value=0)
        )
        fig = go.Figure()
        for dist, label, color in [
            (theme_dist, "This theme", SERIES[0]),
            (all_dist, "All Play reviews", "#c3c2b7"),
        ]:
            fig.add_bar(
                x=[f"{s}★" for s in stars],
                y=dist.values * 100,
                name=label,
                marker=dict(color=color, line=BAR_GAP_LINE),
                hovertemplate=label + " · %{x}: %{y:.0f}%<extra></extra>",
            )
        fig.update_layout(
            barmode="group", height=230, yaxis_title="% of reviews", margin=dict(t=30)
        )
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

# Competitors mentioned alongside this theme.
comp = items["competitor"].dropna()
if len(comp):
    counts = comp.value_counts()
    listed = [f"{name} ({n})" for name, n in counts[counts >= 2].items()]
    if (counts < 2).any():
        listed.append(f"+{int((counts < 2).sum())} others mentioned once")
    st.markdown("**Competitors mentioned:** " + ", ".join(listed))

# Representative quotes.
st.subheader("What people say")
order = (
    st.segmented_control(
        "Pick quotes by",
        ["Most confident", "Most engagement"],
        default="Most confident",
        key="q_order",
    )
    or "Most confident"
)
cols = ["confidence", "engagement"] if order == "Most confident" else ["engagement", "confidence"]
mentions = loaders.filter_window(loaders.request_themes(), win.sources, win.start, win.end)
features = (
    mentions[mentions["theme_id"] == theme_id]
    .groupby("id")["feature"]
    .agg(lambda f: sorted(set(f)))
)
text = loaders.corpus_text()
quotes = items.assign(
    features=items["id"].map(features), words=items["id"].map(text).str.split().str.len()
)
focused = quotes["words"].between(*QUOTE_WORDS)
quotes = quotes[focused] if focused.sum() >= N_QUOTES else quotes
quotes = quotes.sort_values(cols, ascending=False)
for r in quotes.head(N_QUOTES).itertuples():
    body = text.get(r.id, "")
    body = body if len(body) <= 500 else body[:500].rsplit(" ", 1)[0] + " …"
    stars = f" · {int(r.rating)}★" if pd.notna(r.rating) else ""
    eng = "upvotes" if r.source != "play" else "thumbs-up"
    with st.container(border=True):
        st.markdown(md_escape(body).replace("\n", "  \n"))
        st.caption(
            f"{SOURCE_LABELS[r.source]} · {r.created_at:%d %b %Y}{stars} · "
            f"{0 if pd.isna(r.engagement) else int(r.engagement)} {eng} · "
            f"extracted: {'; '.join(r.features) if isinstance(r.features, list) else '–'}"
        )
