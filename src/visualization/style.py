"""Shared plotting style so every figure in the project looks consistent."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import config

# Categorical slots, used in this fixed order (validated colorblind-safe for <= 3 series).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9893"
SURFACE = "#fcfcfb"

# Direction classes keep the same color in every chart.
CLASS_COLORS = {"down": ORANGE, "flat": NEUTRAL, "up": BLUE}

# Diverging map for correlations: red (-1) -> gray (0) -> blue (+1).
DIVERGING = LinearSegmentedColormap.from_list(
    "corr", ["#e34948", "#f0efec", BLUE], N=256
)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (11, 4.5),
        "figure.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure to reports/figures/<name>.png for use in the report."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
