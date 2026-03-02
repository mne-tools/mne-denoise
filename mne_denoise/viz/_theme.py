"""Shared publication-quality theme for mne-denoise figures.

Provides a colorblind-safe color palette, font-size constants, and
helper functions that any plotting function can use to produce
consistent, journal-ready figures.

Usage inside a plotting function::

    from ._theme import COLORS, FONTS, style_axes, pub_figure, pub_legend

Usage from a notebook (applies globally to subsequent plt calls)::

    from mne_denoise.viz import set_pub_style

    set_pub_style()  # optional, sets rcParams once for the session

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# =====================================================================
# Colorblind-safe Wong palette (Nature Methods, 2011)
# =====================================================================
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#BBBBBB",
    "light_gray": "#DDDDDD",
    "dark": "#333333",
}

# Semantic aliases ??? use these in plotting code
COLORS["primary"] = COLORS["blue"]
COLORS["secondary"] = COLORS["orange"]
COLORS["accent"] = COLORS["red"]
COLORS["success"] = COLORS["green"]
COLORS["muted"] = COLORS["gray"]
COLORS["text"] = COLORS["dark"]
COLORS["before"] = COLORS["dark"]  # PSD / signal before cleaning
COLORS["after"] = COLORS["green"]  # PSD / signal after cleaning
COLORS["line_marker"] = COLORS["red"]
COLORS["no_artifact"] = COLORS["gray"]

# =====================================================================
# Font sizes (pt) ??? tuned for single-column journal figures (~3.5 in)
# and two-column figures (~7 in).
# =====================================================================
FONTS = {
    "suptitle": 13,
    "title": 10,
    "label": 9,
    "tick": 8,
    "legend": 7.5,
    "annotation": 7.5,
}

# =====================================================================
# Default figure parameters
# =====================================================================
DEFAULT_DPI = 200
DEFAULT_FIGSIZE = (11, 8.5)  # landscape letter


# =====================================================================
# Axes styling
# =====================================================================
def style_axes(ax, grid=False):
    """Apply publication styling to a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style.
    grid : bool
        If True, add a subtle background grid.
    """
    ax.tick_params(
        labelsize=FONTS["tick"],
        direction="out",
        length=3,
        width=0.5,
        color="#999999",
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(0.5)
        ax.spines[sp].set_color("#666666")
    if grid:
        ax.grid(True, alpha=0.12, linewidth=0.4, color="#888888")
        ax.set_axisbelow(True)


# =====================================================================
# Figure factory
# =====================================================================
def pub_figure(nrows=1, ncols=1, figsize=None, dpi=None, gridspec_kw=None, **kwargs):
    """Create a figure + axes with publication defaults.

    Thin wrapper around ``plt.subplots`` that applies the theme's
    default DPI, white background, and calls :func:`style_axes` on
    every axes.

    Parameters
    ----------
    nrows, ncols : int
        Subplot grid dimensions.
    figsize : tuple | None
        (width, height) in inches.  Defaults to :data:`DEFAULT_FIGSIZE`.
    dpi : int | None
        Resolution. Defaults to :data:`DEFAULT_DPI`.
    gridspec_kw : dict | None
        Forwarded to ``plt.subplots``.
    **kwargs
        Additional keyword arguments forwarded to ``plt.subplots``.

    Returns
    -------
    fig : Figure
    axes : Axes or ndarray of Axes
    """
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    if dpi is None:
        dpi = DEFAULT_DPI

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw=gridspec_kw,
        **kwargs,
    )
    fig.set_facecolor("white")
    return fig, axes


# =====================================================================
# Legend helper
# =====================================================================
def pub_legend(ax, **kwargs):
    """Add a clean, minimal legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    **kwargs
        Overrides forwarded to ``ax.legend()``.

    Returns
    -------
    legend : Legend
    """
    defaults = {
        "fontsize": FONTS["legend"],
        "frameon": True,
        "fancybox": False,
        "edgecolor": "#cccccc",
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


# =====================================================================
# Global rcParams setter (opt-in, for notebooks)
# =====================================================================
def set_pub_style():
    """Apply the mne-denoise publication theme to matplotlib rcParams.

    Call this once at the top of a notebook to ensure *all* subsequent
    figures ??? including those from MNE-Python's own plotting functions ???
    share the same clean look.

    This is **optional**: the individual ``plot_*`` functions in this
    package already apply the theme locally via :func:`style_axes`.
    """
    plt.rcParams.update(
        {
            # Font sizes
            "font.size": FONTS["label"],
            "axes.titlesize": FONTS["title"],
            "axes.labelsize": FONTS["label"],
            "xtick.labelsize": FONTS["tick"],
            "ytick.labelsize": FONTS["tick"],
            "legend.fontsize": FONTS["legend"],
            "figure.titlesize": FONTS["suptitle"],
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "axes.edgecolor": "#666666",
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.color": "#999999",
            "ytick.color": "#999999",
            # Grid
            "axes.grid": False,
            "grid.alpha": 0.12,
            "grid.linewidth": 0.4,
            "grid.color": "#888888",
            # Figure
            "figure.facecolor": "white",
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Legend
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "#cccccc",
            # Lines
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
        }
    )
