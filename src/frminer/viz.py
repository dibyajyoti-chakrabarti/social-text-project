"""Shared matplotlib styling for report figures (light mode, validated palette).

Categorical slots are assigned in fixed order and follow the entity, never its rank:
SOURCE_COLORS maps each data source to the same slot in every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots (light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SOURCE_COLORS = {"play": SERIES[0], "reddit_post": SERIES[1], "reddit_comment": SERIES[2]}
SOURCE_LABELS = {
    "play": "Play Store reviews",
    "reddit_post": "Reddit posts",
    "reddit_comment": "Reddit comments",
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def confusion_heatmap(ax: plt.Axes, cm, title: str = "") -> None:
    """Row-normalised confusion matrix (rows = gold), annotated with raw counts."""
    from matplotlib.colors import LinearSegmentedColormap

    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, *SEQ_BLUE])
    ax.imshow(norm.values, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    short = [c.replace("_", " ").replace("feature request", "feat. req.") for c in cm.columns]
    ax.set_xticks(range(len(short)), short, rotation=40, ha="right")
    ax.set_yticks(range(len(short)), short)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.iat[i, j]
            if v:
                color = SURFACE if norm.iat[i, j] > 0.55 else INK
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color=color)
    if title:
        ax.set_title(title)
