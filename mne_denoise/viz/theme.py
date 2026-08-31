"""Visualization theme helpers."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import matplotlib as mpl
import matplotlib.pyplot as plt

# =====================================================================
# Colorblind-safe Wong palette (Nature Methods, 2011)
# =====================================================================
_BASE_COLORS = {
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

_SEMANTIC_COLORS = {
    "primary": _BASE_COLORS["blue"],
    "secondary": _BASE_COLORS["orange"],
    "accent": _BASE_COLORS["red"],
    "success": _BASE_COLORS["green"],
    "muted": _BASE_COLORS["gray"],
    "text": _BASE_COLORS["dark"],
    "before": _BASE_COLORS["dark"],  # PSD / signal before cleaning
    "after": _BASE_COLORS["green"],  # PSD / signal after cleaning
    "excluded": _BASE_COLORS["red"],  # component excluded from the clean output
    "kept": _BASE_COLORS["dark"],  # component kept in the clean output
    "line_marker": _BASE_COLORS["red"],
    "no_artifact": _BASE_COLORS["gray"],
    "edge": _BASE_COLORS["dark"],  # bar / scatter edge color
    "placeholder": "#999999",  # "no data" text colour
    "separator": "#e0e0e0",  # subtle separator lines
    "label_secondary": "#555555",  # secondary stat labels
    "highlight": _BASE_COLORS["yellow"],  # best-method star etc.
    "stat_mean": _BASE_COLORS["blue"],  # group mean trend/marker
    "stat_subject": _BASE_COLORS["gray"],  # paired subject trajectories
    "stat_reference": _BASE_COLORS["dark"],  # reference thresholds / baselines
    "stat_ci": _BASE_COLORS["light_gray"],  # confidence / interval shading
    "stat_highlight": _BASE_COLORS["yellow"],  # best metric marker
}

BASE_COLORS = MappingProxyType(_BASE_COLORS)
SEMANTIC_COLORS = MappingProxyType(_SEMANTIC_COLORS)
COLORS = MappingProxyType({**_BASE_COLORS, **_SEMANTIC_COLORS})

# `METHOD_COLORS` names broad denoising concepts used across the package.
METHOD_COLORS = MappingProxyType(
    {
        "original": COLORS["dark"],
        "before": COLORS["dark"],
        "after": COLORS["green"],
        "dss": COLORS["blue"],
        "zapline": COLORS["orange"],
        "dss_smooth": COLORS["cyan"],
        "dss_segment": COLORS["purple"],
        "clean": COLORS["green"],
    }
)

# Generic palette and colormaps for reusable spectral series/time-frequency plots.
SERIES_COLORS = (
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent"],
    COLORS["purple"],
    COLORS["success"],
    COLORS["cyan"],
)
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"


AXIS_COLOR = "#666666"
TICK_COLOR = "#999999"
GRID_COLOR = "#888888"
GRID_ALPHA = 0.12
GRID_LINEWIDTH = 0.4
LEGEND_EDGE_COLOR = "#cccccc"
SAVEFIG_DPI = 300
SAVEFIG_BBOX = "tight"
SAVEFIG_PAD_INCHES = 0.05


def get_color(key, fallback=None):
    """Return a color from the shared viz palettes by key.

    Parameters
    ----------
    key : str
        A key into :data:`COLORS` or :data:`METHOD_COLORS`.
    fallback : str | None
        Returned if *key* is not found in any shared palette.
        Defaults to ``COLORS["dark"]``.

    Returns
    -------
    str
        Hex color string.
    """
    if key in COLORS:
        return COLORS[key]
    if key in METHOD_COLORS:
        return METHOD_COLORS[key]
    return fallback if fallback is not None else COLORS["dark"]


def get_series_color(index, colors=None):
    """Return a color from the shared spectral series palette.

    Parameters
    ----------
    index : int
        Series index in plotting order.
    colors : sequence of str | None
        Optional palette override. If provided, this sequence is cycled
        instead of :data:`SERIES_COLORS`.

    Returns
    -------
    str
        Hex color string.
    """
    palette = SERIES_COLORS if colors is None else tuple(colors)
    if len(palette) == 0:
        raise ValueError("colors must contain at least one color.")
    return palette[index % len(palette)]


# =====================================================================
# Font sizes (pt) — tuned for single-column journal figures (~3.5 in)
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
    """Apply per-axes theme overrides.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style.
    grid : bool, default=False
        If True, add a subtle background grid.
    """
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(0.5)
        ax.spines[sp].set_color(AXIS_COLOR)
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
        ax.set_axisbelow(True)


# =====================================================================
# Figure factory
# =====================================================================
def themed_figure(
    nrows=1,
    ncols=1,
    figsize=None,
    dpi=None,
    gridspec_kw=None,
    rc=None,
    **kwargs,
):
    """Create a figure + axes with the package plotting defaults.

    Parameters
    ----------
    nrows : int, default=1
        Number of subplot rows.
    ncols : int, default=1
        Number of subplot columns.
    figsize : tuple | None
        (width, height) in inches.  Defaults to :data:`DEFAULT_FIGSIZE`.
    dpi : int | None
        Resolution. Defaults to :data:`DEFAULT_DPI`.
    gridspec_kw : dict | None
        Forwarded to ``plt.subplots``.
    rc : dict | None
        Optional matplotlib rcParams overrides merged into the shared
        theme defaults for this figure only.
    **kwargs
        Additional keyword arguments forwarded to ``plt.subplots``.

    Returns
    -------
    fig : Figure
        Matplotlib figure created by ``plt.subplots``.
    axes : Axes or ndarray of Axes
        Axes object returned by ``plt.subplots``.
    """
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    if dpi is None:
        dpi = DEFAULT_DPI

    with use_theme(rc=rc):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            dpi=dpi,
            gridspec_kw=gridspec_kw,
            **kwargs,
        )

    return fig, axes


# =====================================================================
# Legend helper
# =====================================================================
def themed_legend(ax, **kwargs):
    """Add a clean, minimal legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    **kwargs
        Overrides forwarded to ``ax.legend()``.

    Returns
    -------
    legend : Legend
        The legend instance returned by ``ax.legend()``.
    """
    defaults = {
        "fontsize": FONTS["legend"],
        "frameon": True,
        "fancybox": False,
        "edgecolor": LEGEND_EDGE_COLOR,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def _finalize_fig(fig, show=True, fname=None, tight=True):
    """Finalize a figure by applying layout, saving, and/or showing it."""
    if tight:
        with contextlib.suppress(Exception):
            fig.tight_layout()
    if fname is not None:
        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            fname,
            dpi=SAVEFIG_DPI,
            bbox_inches=SAVEFIG_BBOX,
            pad_inches=SAVEFIG_PAD_INCHES,
        )
    if show:
        plt.show()
    elif fname is not None:
        # Figure was saved to disk; close to free memory.
        plt.close(fig)
    # If show=False and fname=None the caller owns the figure.
    return fig


# =====================================================================
# rcParams dict (shared between set_theme and use_theme)
# =====================================================================
_THEME_RC = {
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
    "axes.edgecolor": AXIS_COLOR,
    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.color": TICK_COLOR,
    "ytick.color": TICK_COLOR,
    # Grid
    "axes.grid": False,
    "grid.alpha": GRID_ALPHA,
    "grid.linewidth": GRID_LINEWIDTH,
    "grid.color": GRID_COLOR,
    # Figure
    "figure.facecolor": "white",
    "figure.dpi": DEFAULT_DPI,
    "savefig.dpi": SAVEFIG_DPI,
    "savefig.bbox": SAVEFIG_BBOX,
    "savefig.pad_inches": SAVEFIG_PAD_INCHES,
    # Legend
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": LEGEND_EDGE_COLOR,
    # Lines
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
}


# =====================================================================
# rcParams helpers
# =====================================================================
def get_theme_rc(rc: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return the theme rcParams, optionally merged with overrides.

    Parameters
    ----------
    rc : mapping | None
        Optional matplotlib rcParams overrides. Only the provided keys
        are changed; all other theme defaults are preserved.

    Returns
    -------
    rc_out : dict
        Copy of the theme rcParams with overrides applied.
    """
    rc_out = dict(_THEME_RC)
    if rc is not None:
        rc_out.update(rc)
    return rc_out


# =====================================================================
# Context-manager style application (recommended for library use)
# =====================================================================
@contextlib.contextmanager
def use_theme(name="default", rc: Mapping[str, object] | None = None):
    """Context manager that temporarily applies the mne-denoise theme.

    Parameters
    ----------
    name : str, default='default'
        Currently only ``"default"`` is supported.
    rc : mapping | None
        Optional matplotlib rcParams overrides merged into the shared
        theme defaults for the duration of the context.
    """
    if name != "default":
        raise ValueError(f"Unknown theme {name!r}; only 'default' is supported.")
    with mpl.rc_context(rc=get_theme_rc(rc)):
        yield


# =====================================================================
# Global rcParams setter (opt-in, for notebooks only)
# =====================================================================
def set_theme(rc: Mapping[str, object] | None = None):
    """Apply the mne-denoise plotting theme to matplotlib rcParams.

    Parameters
    ----------
    rc : mapping | None
        Optional matplotlib rcParams overrides merged into the shared
        theme defaults before applying them globally.
    """
    plt.rcParams.update(get_theme_rc(rc))


__all__ = [
    "COLORS",
    "METHOD_COLORS",
    "SERIES_COLORS",
    "SEQUENTIAL_CMAP",
    "DIVERGING_CMAP",
    "FONTS",
    "DEFAULT_DPI",
    "DEFAULT_FIGSIZE",
    "get_color",
    "get_series_color",
    "style_axes",
    "themed_figure",
    "themed_legend",
    "get_theme_rc",
    "use_theme",
    "set_theme",
]
