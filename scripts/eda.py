"""Exploratory analysis of the cleaned corpus → reports/figures/eda_*.png + summary table.

uv run python scripts/eda.py

Collection caveat handled here: Google Play pagination ran out early for the 5★ stream,
so 5★ reviews cover only the last ~6 weeks while 1–4★ cover the full year. Any
rating-mix statistic is therefore computed on the window where all ratings exist.
"""

import re

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from frminer import viz
from frminer.config import path

# A rough lexical preview of explicit request cues (NOT the Phase 3 rule classifier).
REQUEST_CUES = re.compile(
    r"\b(?:please add|pls add|add (?:a|an|the|back)|i wish|wish (?:it|there|they|you|spotify)"
    r"|would be (?:nice|great|cool|awesome)|should (?:add|have|let|be able)|bring back"
    r"|option to|ability to|let us|why can'?t|feature request|needs? (?:a|an) )",
    re.IGNORECASE,
)
PCT = PercentFormatter(1, decimals=0)


def full_rating_window_start(df: pd.DataFrame) -> pd.Timestamp:
    """First day from which every star rating was collected (the 5★ stream's start)."""
    play = df[df["source"] == "play"]
    return play.loc[play["rating"] == 5, "created_at"].min().normalize() + pd.Timedelta(days=1)


def weekly(df: pd.DataFrame) -> pd.Series:
    # Drop the partial first and last weeks.
    return df.set_index("created_at").resample("W")["id"].count().iloc[1:-1]


def _xy(s: pd.Series):
    return s.index, s.values


def weekly_volume(df: pd.DataFrame, figs) -> None:
    sources = [s for s in viz.SOURCE_COLORS if s in set(df["source"])]
    fig, axes = viz.plt.subplots(len(sources), 1, figsize=(8, 2 * len(sources)), sharex=True)
    for ax, src in zip(np.atleast_1d(axes), sources, strict=True):
        sub = df[df["source"] == src]
        color = viz.SOURCE_COLORS[src]
        if src == "play":
            start = full_rating_window_start(df)
            ax.plot(*_xy(weekly(sub[sub["rating"] <= 4])), color=color, label="1–4★ (full year)")
            five = weekly(sub[(sub["rating"] == 5) & (sub["created_at"] >= start)])
            ax.plot(*_xy(five), color=color, linestyle="--", label=f"5★ (from {start:%b %d})")
            ax.legend(loc="lower left", fontsize=8, ncols=2)
        else:
            ax.plot(*_xy(weekly(sub)), color=color)
        ax.set_title(viz.SOURCE_LABELS[src], fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("items / week")
    fig.suptitle("Weekly volume by source", x=0.01, ha="left", fontweight="bold")
    figs.append(viz.save(fig, path("reports") / "figures" / "eda_weekly_volume.png"))


def length_ecdf(df: pd.DataFrame, figs) -> None:
    fig, ax = viz.plt.subplots(figsize=(7, 3.6))
    for src, color in viz.SOURCE_COLORS.items():
        words = np.sort(df.loc[df["source"] == src, "n_words"].to_numpy())
        if len(words) == 0:
            continue
        y = np.arange(1, len(words) + 1) / len(words)
        label = f"{viz.SOURCE_LABELS[src]} (median {np.median(words):.0f})"
        ax.plot(words, y, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("words per item (log scale)")
    ax.set_ylabel("share of items ≤ x")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Text length distribution (Play reviews are capped at 500 characters)")
    ax.legend(loc="lower right")
    figs.append(viz.save(fig, path("reports") / "figures" / "eda_length_ecdf.png"))


def rating_distribution(df: pd.DataFrame, figs) -> None:
    start = full_rating_window_start(df)
    play = df[(df["source"] == "play") & (df["created_at"] >= start)]
    counts = play["rating"].value_counts(normalize=True).sort_index()
    fig, ax = viz.plt.subplots(figsize=(5.5, 3))
    bars = ax.bar(counts.index.astype(str), counts.values, color=viz.SERIES[0], width=0.6)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in counts.values], color=viz.INK_2, fontsize=8)
    ax.set_xlabel("star rating")
    ax.set_ylabel("share of Play reviews")
    ax.yaxis.set_major_formatter(PCT)
    end = play["created_at"].max()
    ax.set_title(f"Play Store ratings, {start:%b %d} – {end:%b %d %Y} (all ratings collected)")
    figs.append(viz.save(fig, path("reports") / "figures" / "eda_ratings.png"))


def flair_distribution(df: pd.DataFrame, figs) -> None:
    posts = df[df["source"] == "reddit_post"]
    if posts.empty:
        return
    counts = posts["flair"].fillna("(no flair)").value_counts().head(12).iloc[::-1]
    fig, ax = viz.plt.subplots(figsize=(6, 3.8))
    bars = ax.barh(counts.index, counts.values, color=viz.SOURCE_COLORS["reddit_post"], height=0.6)
    ax.bar_label(bars, padding=3, color=viz.INK_2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("posts")
    ax.set_title("r/truespotify post flairs")
    figs.append(viz.save(fig, path("reports") / "figures" / "eda_flairs.png"))


def request_cue_rates(df: pd.DataFrame, figs) -> pd.DataFrame:
    df = df.assign(cue=df["text"].str.contains(REQUEST_CUES))
    by_source = df.groupby("source")["cue"].mean()
    by_rating = df[df["source"] == "play"].groupby("rating")["cue"].mean()
    fig, (a1, a2) = viz.plt.subplots(1, 2, figsize=(9, 3), width_ratios=[1, 1.3])

    labels = [viz.SOURCE_LABELS[s] for s in by_source.index]
    colors = [viz.SOURCE_COLORS[s] for s in by_source.index]
    b1 = a1.barh(labels, by_source.values, color=colors, height=0.6)
    a1.bar_label(b1, labels=[f"{v:.1%}" for v in by_source.values], padding=3, fontsize=8)
    a1.set_title("By source")
    a1.grid(axis="y", visible=False)
    a1.xaxis.set_major_formatter(PCT)

    b2 = a2.bar(by_rating.index.astype(str), by_rating.values, color=viz.SERIES[0], width=0.6)
    a2.bar_label(b2, labels=[f"{v:.1%}" for v in by_rating.values], fontsize=8, color=viz.INK_2)
    a2.set_title("Play Store, by star rating")
    a2.set_xlabel("star rating")
    a2.yaxis.set_major_formatter(PCT)

    fig.suptitle(
        "Share of items containing an explicit request cue (lexical preview)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    figs.append(viz.save(fig, path("reports") / "figures" / "eda_request_cues.png"))
    return df


def summary_table(df: pd.DataFrame) -> str:
    rows = []
    for src, g in df.groupby("source"):
        rows.append(
            {
                "source": viz.SOURCE_LABELS[src],
                "items": f"{len(g):,}",
                "from": g["created_at"].min().date(),
                "to": g["created_at"].max().date(),
                "median words": int(g["n_words"].median()),
                "engagement > 0": f"{(g['engagement'] > 0).mean():.0%}",
                "explicit request cue": f"{g['cue'].mean():.1%}",
            }
        )
    start = full_rating_window_start(df)
    note = (
        f"\nNote: Play 5★ reviews were only retrievable from {start.date()} "
        "(Google Play pagination limit); 1–4★ reviews cover the full window.\n"
    )
    return pd.DataFrame(rows).to_markdown(index=False) + "\n" + note


def main() -> None:
    viz.apply_style()
    df = pd.read_parquet(path("interim") / "corpus.parquet")
    figs: list = []
    weekly_volume(df, figs)
    length_ecdf(df, figs)
    rating_distribution(df, figs)
    flair_distribution(df, figs)
    df = request_cue_rates(df, figs)
    table = summary_table(df)
    out = path("reports") / "tables" / "corpus_summary.md"
    out.write_text(table)
    print(table)
    for f in figs:
        print("figure:", f)


if __name__ == "__main__":
    main()
