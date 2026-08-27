"""ASR-specific visualization diagnostics.

This module provides plotting helpers for diagnostics that are intrinsic to
Artifact Subspace Reconstruction and have no generic equivalent:

- :func:`plot_asr_repair_timeline` -- per-window count of reconstructed
  components over time.
- :func:`plot_asr_calibration_fraction` -- bar chart comparing the fraction
  of data retained as clean calibration across one or more estimators.
- :func:`plot_asr_component_reconstruction` -- heatmap of per-window
  component variance relative to the rejection threshold.
- :func:`plot_guided_asr_weights` -- heatmap of the experimental GuidedASR
  soft component weights.

For before/after signal overlays, PSD comparisons, per-channel power-ratio
topographies, grand averages, and metric scatters, use the generic
:mod:`mne_denoise.viz` helpers which work on any denoiser's input/output.
"""

from __future__ import annotations

import numpy as np

from .theme import (
    COLORS,
    get_series_color,
    style_axes,
    themed_figure,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

__all__ = [
    "plot_asr_repair_timeline",
    "plot_asr_calibration_fraction",
    "plot_asr_component_reconstruction",
    "plot_guided_asr_weights",
]


# ---------------------------------------------------------------------------
# Shared output helper
# ---------------------------------------------------------------------------


def _finish(fig, ax, *, show: bool, fname: str | None):
    """Apply the save-and-show convention shared by every plot in this module.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save/show.
    ax : matplotlib.axes.Axes
        The axes object to return.
    show : bool
        If True, call ``plt.show()``.
    fname : str | None
        If not None, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if fname is not None:
        fig.savefig(fname, dpi=fig.dpi, bbox_inches="tight")
    if show and plt is not None:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Repair timeline
# ---------------------------------------------------------------------------


def plot_asr_repair_timeline(
    estimator,
    *,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Plot the per-window count of reconstructed components over time.

    Visualizes whether ASR surgically repaired isolated bursts (sparse
    activity is good) or modified a large portion of the recording
    (potential over-cleaning). The data is read from the fitted
    estimator's ``diagnostics_`` attribute.

    Parameters
    ----------
    estimator : ASR | AdaptiveASR | JugglerASR
        A fitted estimator that has been used to ``transform`` data, so that
        ``diagnostics_`` is populated.
    title : str | None, default=None
        Custom title for the plot. If None, an auto-generated title showing
        the percentage of modified windows is used.
    ax : matplotlib.axes.Axes | None, default=None
        Axes to plot into. If None, a new themed figure is created.
    show : bool, default=True
        Whether to call ``plt.show()`` after plotting.
    fname : str | None, default=None
        If provided, the figure is saved to this file path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes with the repair timeline.

    Raises
    ------
    ValueError
        If the estimator has no transform diagnostics or if diagnostics
        contain no processing windows.
    """
    diag = getattr(estimator, "diagnostics_", None)
    sfreq = float(getattr(estimator, "sfreq_", 0.0) or 0.0)
    if not diag or sfreq <= 0:
        raise ValueError("estimator has no transform diagnostics; run transform first.")
    starts = np.asarray(diag.get("window_starts", []), dtype=float)
    stops = np.asarray(diag.get("window_stops", []), dtype=float)
    counts = np.asarray(diag.get("n_components_reconstructed", []), dtype=float)
    if starts.size == 0:
        raise ValueError("diagnostics contain no processing windows.")
    centers = (starts + stops) / 2.0 / sfreq

    if ax is None:
        fig, ax = themed_figure(figsize=(11, 3.2))
    else:
        fig = ax.figure
    ax.fill_between(centers, counts, step="mid", color=COLORS["primary"], alpha=0.5)
    ax.plot(centers, counts, color=COLORS["primary"], lw=0.8, drawstyle="steps-mid")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Components reconstructed")
    frac = float(np.mean(counts > 0)) * 100 if counts.size else 0.0
    ax.set_title(title or f"ASR repair timeline ({frac:.0f}% of windows modified)")
    style_axes(ax, grid=True)
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


# ---------------------------------------------------------------------------
# Calibration / reference fraction
# ---------------------------------------------------------------------------


def plot_asr_calibration_fraction(
    estimators,
    *,
    labels=None,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Bar chart of the clean calibration fraction for one or more estimators.

    Compares how much of the candidate calibration data was retained as
    clean reference material. For standard ASR this is the fraction of
    clean *windows*; for JugglerASR it is the fraction of clean *samples*
    selected by DBSCAN or GEV. Too small a fraction suggests the cutoff
    or data quality needs adjustment.

    Parameters
    ----------
    estimators : fitted estimator | list of fitted estimators
        One or more fitted ASR-family estimators. Each contributes one bar
        extracted from its ``calibration_info_`` attribute.
    labels : list of str | None, default=None
        Bar labels. If None, defaults to the class name of each estimator.
    title : str | None, default=None
        Custom title for the plot. If None, uses a sensible default.
    ax : matplotlib.axes.Axes | None, default=None
        Axes to plot into. If None, a new themed figure is created.
    show : bool, default=True
        Whether to call ``plt.show()`` after plotting.
    fname : str | None, default=None
        If provided, the figure is saved to this file path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes with the bar chart.
    """
    if not isinstance(estimators, list | tuple):
        estimators = [estimators]
    if labels is None:
        labels = [type(e).__name__ for e in estimators]

    fracs = []
    for e in estimators:
        info = getattr(e, "calibration_info_", {}) or {}
        n_sel = info.get("reference_selected_samples")
        n_cand = info.get("reference_candidate_samples")
        if n_sel is not None and n_cand:
            fracs.append(100.0 * n_sel / n_cand)
            continue
        n_clean = info.get("n_clean_windows")
        n_tot = info.get("n_calibration_windows")
        if n_clean is not None and n_tot:
            fracs.append(100.0 * n_clean / n_tot)
        else:
            fracs.append(np.nan)

    if ax is None:
        fig, ax = themed_figure(figsize=(1.6 * len(labels) + 2, 4.2))
    else:
        fig = ax.figure
    x = np.arange(len(labels))
    ax.bar(x, fracs, color=[get_series_color(i) for i in range(len(labels))])
    for xi, f in zip(x, fracs):
        if np.isfinite(f):
            ax.text(xi, f, f"{f:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Calibration fraction (%)")
    ax.set_title(title or "ASR calibration / reference fraction")
    style_axes(ax, grid=True)
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


# ---------------------------------------------------------------------------
# Component-reconstruction map
# ---------------------------------------------------------------------------


def plot_asr_component_reconstruction(
    estimator,
    *,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Heatmap of per-window component variance relative to rejection thresholds.

    Displays a 2D image where each column is a processing window and each
    row is a principal component. The color encodes the ratio of the
    component's variance to the rejection threshold: values above 1.0
    indicate components that ASR reconstructed in that window.

    Parameters
    ----------
    estimator : ASR | AdaptiveASR | JugglerASR
        A fitted estimator that has been used to ``transform`` data, so that
        ``diagnostics_`` is populated.
    title : str | None, default=None
        Custom title for the plot. If None, uses a sensible default.
    ax : matplotlib.axes.Axes | None, default=None
        Axes to plot into. If None, a new themed figure is created.
    show : bool, default=True
        Whether to call ``plt.show()`` after plotting.
    fname : str | None, default=None
        If provided, the figure is saved to this file path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes with the heatmap.

    Raises
    ------
    ValueError
        If the estimator has no transform diagnostics or if diagnostics
        lack per-window ``component_variances``.
    """
    diag = getattr(estimator, "diagnostics_", None)
    sfreq = float(getattr(estimator, "sfreq_", 0.0) or 0.0)
    if not diag:
        raise ValueError("estimator has no transform diagnostics; run transform first.")
    cv = np.asarray(diag.get("component_variances", []), dtype=float)
    ct = np.asarray(diag.get("component_thresholds", []), dtype=float)
    if cv.ndim != 2 or cv.size == 0:
        raise ValueError("diagnostics lack per-window component_variances.")
    # ratio > 1 → component exceeded threshold → reconstructed
    ratio = cv / np.maximum(ct, np.finfo(float).eps)
    starts = np.asarray(diag.get("window_starts", []), dtype=float)
    extent = None
    if starts.size == cv.shape[0] and sfreq > 0:
        extent = [starts[0] / sfreq, starts[-1] / sfreq, 0, cv.shape[1]]

    if ax is None:
        fig, ax = themed_figure(figsize=(11, 3.6))
    else:
        fig = ax.figure
    im = ax.imshow(
        ratio.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=2.0,
        extent=extent,
    )
    fig.colorbar(im, ax=ax, label="variance / threshold")
    ax.set_xlabel("Time (s)" if extent else "Window index")
    ax.set_ylabel("Component")
    ax.set_title(title or "ASR component reconstruction map")
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


def plot_guided_asr_weights(
    estimator,
    *,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Plot experimental GuidedASR soft weights by window and component.

    Parameters
    ----------
    estimator : GuidedASR
        A fitted estimator that has run ``transform`` and whose
        ``diagnostics_`` contains ``soft_weights``.
    title : str | None, default=None
        Custom figure title.
    ax : matplotlib.axes.Axes | None, default=None
        Axes to plot into. If None, create a themed figure.
    show : bool, default=True
        Whether to call ``plt.show()``.
    fname : str | None, default=None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the heatmap.
    ax : matplotlib.axes.Axes
        Axes containing the heatmap.

    Notes
    -----
    GuidedASR is an unpublished, unvalidated research prototype. This plot is
    a diagnostic aid and does not establish that the selected weights separate
    neural activity from artifacts correctly.
    """
    diag = getattr(estimator, "diagnostics_", None)
    if not diag or "soft_weights" not in diag:
        raise ValueError(
            "estimator has no soft_weights; fit + transform a GuidedASR first."
        )
    weights = np.asarray(diag["soft_weights"], dtype=float)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError(
            "soft_weights must be a non-empty (n_windows, n_components) array."
        )

    sfreq = float(getattr(estimator, "sfreq_", 0.0) or 0.0)
    starts = np.asarray(diag.get("window_starts", []), dtype=float)
    stops = np.asarray(diag.get("window_stops", []), dtype=float)
    extent = None
    if starts.size == weights.shape[0] and sfreq > 0:
        stop = stops[-1] if stops.size == starts.size else starts[-1] + 1.0
        extent = [starts[0] / sfreq, stop / sfreq, 0, weights.shape[1]]

    if ax is None:
        fig, ax = themed_figure(figsize=(11, 3.6))
    else:
        fig = ax.figure
    image = ax.imshow(
        weights.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    fig.colorbar(image, ax=ax, label="soft weight (1 = keep, 0 = suppress)")
    ax.set_xlabel("Time (s)" if extent else "Window index")
    ax.set_ylabel("Component")
    ax.set_title(title or f"GuidedASR soft weights (mean = {weights.mean():.3f})")
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)
