"""Shared summary panel painters.

Internal panel-level building blocks used by
the public ``mne_denoise.viz`` summary composers.

Design principles
-----------------
1. Plotting-only helpers: no estimator fitting and no side effects.
2. Study-agnostic panel defaults.
3. Theme-first styling through :mod:`mne_denoise.viz.theme`.
4. Explicit inputs, with limited fallback placeholders for missing panels.

This module is private and not part of the public API surface.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .theme import (
    COLORS,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    FONTS,
    style_axes,
    themed_legend,
    use_theme,
)


def _new_summary_figure(figsize=None, dpi=None, constrained_layout=False):
    """Create a themed figure for summary composers.

    Parameters
    ----------
    figsize : tuple[float, float] | None
        Figure size. Falls back to theme default when ``None``.
    dpi : int | None
        Figure resolution. Falls back to theme default when ``None``.
    constrained_layout : bool
        Forwarded to ``matplotlib.figure.Figure``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Initialized themed figure.
    """
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    if dpi is None:
        dpi = DEFAULT_DPI
    with use_theme():
        fig = plt.figure(
            figsize=figsize,
            dpi=dpi,
            constrained_layout=constrained_layout,
        )
    return fig


def _new_summary_grid(figsize=None, dpi=None, hspace=0.42, wspace=0.30):
    """Create a canonical 3x2 summary layout.

    Parameters
    ----------
    figsize : tuple[float, float] | None
        Figure size. Falls back to theme default when ``None``.
    dpi : int | None
        Figure resolution. Falls back to theme default when ``None``.
    hspace : float
        Vertical spacing between rows.
    wspace : float
        Horizontal spacing between columns.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    gs : matplotlib.gridspec.GridSpec
        Grid spec with fixed outer margins used by summary dashboards.
    """
    from matplotlib.gridspec import GridSpec

    fig = _new_summary_figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=hspace,
        wspace=wspace,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )
    return fig, gs


def _draw_summary_table(ax, rows, title="(f)  Summary"):
    """Render a compact key/value summary table panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used for rendering.
    rows : sequence[tuple[object, object]]
        Table rows as ``(label, value)`` pairs in display order.
    title : str
        Panel title.
    """
    ax.axis("off")

    n_rows = len(rows)
    row_h = min(0.085, 0.90 / max(n_rows, 1))
    y_top = 0.94

    for idx, (label, value) in enumerate(rows):
        y = y_top - idx * row_h
        ax.text(
            0.03,
            y,
            str(label),
            transform=ax.transAxes,
            fontsize=FONTS["label"] - 0.5,
            color=COLORS["label_secondary"],
            va="top",
        )
        ax.text(
            0.97,
            y,
            str(value),
            transform=ax.transAxes,
            fontsize=FONTS["label"] - 0.5,
            color=COLORS["text"],
            va="top",
            ha="right",
            fontfamily="monospace",
            fontweight="bold",
        )
        y_line = y - row_h * 0.35
        ax.plot(
            [0.03, 0.97],
            [y_line, y_line],
            color=COLORS["separator"],
            lw=0.4,
            transform=ax.transAxes,
            clip_on=False,
        )

    ax.set_title(
        title,
        fontsize=FONTS["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )


def _plot_selection_scores_panel(
    ax,
    scores,
    selected_count,
    selected_label="Selected",
    title="(a)  Component scores",
):
    """Plot component scores with selected/retained coloring.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used for rendering.
    scores : array-like | None
        Score vector. If empty or ``None``, a placeholder message is shown.
    selected_count : int
        Number of leading components marked as selected.
    selected_label : str
        Legend label for selected components.
    title : str
        Panel title.
    """
    style_axes(ax, grid=True)
    scores = None if scores is None else np.asarray(scores, dtype=float).ravel()
    if scores is None or scores.size == 0:
        ax.text(
            0.5,
            0.5,
            "No scores available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
    else:
        x_vals = np.arange(scores.size)
        bar_colors = [
            COLORS["accent"] if i < selected_count else COLORS["primary"]
            for i in range(scores.size)
        ]
        ax.bar(x_vals, scores, color=bar_colors, edgecolor="none", width=0.75)
        ax.set_xlim(-0.6, scores.size - 0.4)
        themed_legend(
            ax,
            handles=[
                plt.Rectangle((0, 0), 1, 1, fc=COLORS["accent"], label=selected_label),
                plt.Rectangle((0, 0), 1, 1, fc=COLORS["primary"], label="Retained"),
            ],
            loc="upper right",
        )
    ax.set_xlabel("Component", fontsize=FONTS["label"])
    ax.set_ylabel("Score", fontsize=FONTS["label"])
    ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")


def _plot_patterns_panel(
    fig,
    subplot_spec,
    patterns,
    info=None,
    channel_names=None,
    max_components=4,
    title="(b)  Spatial patterns",
):
    """Plot component patterns with topomap or line fallback.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Parent figure.
    subplot_spec : matplotlib.gridspec.SubplotSpec
        Grid location where the panel should be drawn.
    patterns : array-like | None
        Pattern matrix with shape ``(n_channels, n_components)``.
    info : mne.Info | None
        Optional MNE info. When provided, topomaps are drawn.
    channel_names : sequence[str] | None
        Optional x-axis labels for non-topomap fallback.
    max_components : int
        Maximum number of components to display.
    title : str
        Panel title.

    Raises
    ------
    ValueError
        If ``patterns`` is provided with invalid dimensionality.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    has_patterns = patterns is not None and np.size(patterns) > 0
    if not has_patterns:
        ax = fig.add_subplot(subplot_spec)
        ax.text(
            0.5,
            0.5,
            "No patterns available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
        style_axes(ax)
        ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")
        return

    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim != 2:
        raise ValueError("patterns must be 2D with shape (n_channels, n_components).")

    if info is not None:
        import mne

        n_show = min(patterns.shape[1], max_components)
        gs_inner = GridSpecFromSubplotSpec(
            1, n_show, subplot_spec=subplot_spec, wspace=0.25
        )
        for comp_idx in range(n_show):
            ax = fig.add_subplot(gs_inner[0, comp_idx])
            mne.viz.plot_topomap(
                patterns[:, comp_idx], info, axes=ax, show=False, contours=4
            )
            ax.set_title(f"C{comp_idx + 1}", fontsize=FONTS["tick"], pad=2)
        anchor = fig.add_subplot(subplot_spec)
        pos = anchor.get_position()
        anchor.set_visible(False)
        fig.text(
            pos.x0, pos.y1 + 0.01, title, fontsize=FONTS["title"], fontweight="semibold"
        )
        return

    ax = fig.add_subplot(subplot_spec)
    style_axes(ax, grid=True)
    n_channels, n_components = patterns.shape
    n_show = min(n_components, max_components)
    x = np.arange(n_channels)
    for comp_idx in range(n_show):
        color = COLORS["primary"] if comp_idx % 2 == 0 else COLORS["secondary"]
        ax.plot(
            x,
            patterns[:, comp_idx],
            lw=1.0,
            color=color,
            alpha=0.8,
            label=f"C{comp_idx + 1}",
        )

    if channel_names is not None and len(channel_names) == n_channels:
        step = max(1, n_channels // 12)
        ticks = x[::step]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [channel_names[i] for i in ticks],
            rotation=45,
            ha="right",
            fontsize=FONTS["tick"] - 0.5,
        )
    else:
        step = max(1, n_channels // 12)
        ax.set_xticks(x[::step])

    themed_legend(ax, loc="upper right")
    ax.set_xlabel("Channel", fontsize=FONTS["label"])
    ax.set_ylabel("Weight", fontsize=FONTS["label"])
    ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")


def _plot_removed_power_panel(
    fig,
    ax,
    removed,
    info=None,
    channel_names=None,
    title="(c)  Removed signal power",
    no_data_text="No removal data",
):
    """Plot per-channel RMS power of removed signal.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Parent figure used for optional colorbar creation.
    ax : matplotlib.axes.Axes
        Axis used for rendering.
    removed : array-like | None
        Removed data as ``(n_channels, n_times)`` or
        ``(n_epochs, n_channels, n_times)``.
    info : mne.Info | None
        Optional MNE info for topomap rendering.
    channel_names : sequence[str] | None
        Optional x-axis labels for bar fallback.
    title : str
        Panel title.
    no_data_text : str
        Placeholder message when no removed data is available.

    Raises
    ------
    ValueError
        If ``removed`` has unsupported dimensionality.
    """
    removed = None if removed is None else np.asarray(removed, dtype=float)
    has_removed = removed is not None and removed.size > 0
    if not has_removed:
        style_axes(ax)
        ax.text(
            0.5,
            0.5,
            no_data_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
        ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")
        return

    if removed.ndim == 3:
        removed = removed.mean(axis=0)
    if removed.ndim != 2:
        raise ValueError("removed must be 2D or 3D with channels on axis -2.")

    removed_rms = np.sqrt(np.mean(removed**2, axis=1))
    if info is not None:
        import mne

        im, _ = mne.viz.plot_topomap(removed_rms, info, axes=ax, show=False, contours=4)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("RMS", fontsize=FONTS["tick"])
    else:
        style_axes(ax, grid=True)
        x = np.arange(removed_rms.size)
        ax.bar(x, removed_rms, color=COLORS["accent"], edgecolor="none", width=0.75)
        if channel_names is not None and len(channel_names) == removed_rms.size:
            step = max(1, removed_rms.size // 12)
            ticks = x[::step]
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [channel_names[i] for i in ticks],
                rotation=45,
                ha="right",
                fontsize=FONTS["tick"] - 0.5,
            )
        ax.set_xlabel("Channel", fontsize=FONTS["label"])
        ax.set_ylabel("RMS", fontsize=FONTS["label"])
    ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")


def _plot_source_trace_panel(
    ax,
    sources,
    sfreq=None,
    title="(d)  Component traces",
    empty_text="No source data",
):
    """Plot a short component-trace window with stacked offsets.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used for rendering.
    sources : array-like | None
        Source data as ``(n_components, n_times)`` or
        ``(n_epochs, n_components, n_times)``.
    sfreq : float | None
        Sampling frequency used to build the time axis.
    title : str
        Panel title.
    empty_text : str
        Placeholder message when no source data is provided.

    Notes
    -----
    The panel displays up to four components over a deterministic
    two-second (or shorter) window starting at sample zero.

    Raises
    ------
    ValueError
        If ``sources`` has unsupported dimensionality.
    """
    style_axes(ax)
    if sources is None:
        ax.text(
            0.5,
            0.5,
            empty_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
        ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")
        return

    sources = np.asarray(sources, dtype=float)
    if sources.ndim == 3:
        sources = sources.mean(axis=0)
    if sources.ndim != 2:
        raise ValueError("sources must be 2D or 3D with components on axis -2.")
    if sfreq is None:
        ax.text(
            0.5,
            0.5,
            "sfreq required for time axis",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
        ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")
        return

    n_samples = sources.shape[1]
    win_samples = min(n_samples, int(min(2.0, n_samples / sfreq) * sfreq))
    start = 0 if n_samples <= win_samples else 0
    sl = slice(start, start + win_samples)
    t = np.arange(win_samples) / sfreq
    scale = np.std(sources[0, sl]) or 1.0

    n_show = min(4, sources.shape[0])
    for idx in range(n_show):
        offset = -idx * 3.5 * scale
        color = COLORS["primary"] if idx % 2 == 0 else COLORS["secondary"]
        ax.plot(
            t,
            sources[idx, sl] + offset,
            color=color,
            lw=0.8,
            alpha=0.85,
            label=f"C{idx + 1}",
        )

    ax.set_xlabel("Time (s)", fontsize=FONTS["label"])
    ax.set_ylabel("Amplitude (a.u.)", fontsize=FONTS["label"])
    ax.set_yticks([])
    themed_legend(ax, loc="upper right")
    ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")


def _plot_before_after_psd_panel(
    ax,
    freqs=None,
    psd_before=None,
    psd_after=None,
    line_freq=None,
    fmax=100.0,
    title="(e)  Power spectral density",
    before_label="Before",
    after_label="After",
):
    """Plot a before/after PSD comparison panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used for rendering.
    freqs : array-like | None
        Frequency axis.
    psd_before, psd_after : array-like | None
        PSD values with frequency on the last axis.
    line_freq : float | None
        Optional vertical marker (for example line frequency).
    fmax : float
        Upper x-axis bound.
    title : str
        Panel title.
    before_label, after_label : str
        Legend labels for the two curves.
    """
    style_axes(ax, grid=True)
    if freqs is None or psd_before is None or psd_after is None:
        ax.text(
            0.5,
            0.5,
            "No PSD data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
        ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")
        return

    freqs = np.asarray(freqs, dtype=float)
    psd_before = np.asarray(psd_before, dtype=float)
    psd_after = np.asarray(psd_after, dtype=float)
    mean_before = psd_before.reshape(-1, psd_before.shape[-1]).mean(axis=0)
    mean_after = psd_after.reshape(-1, psd_after.shape[-1]).mean(axis=0)
    ax.semilogy(freqs, mean_before, color=COLORS["before"], lw=1.8, label=before_label)
    ax.semilogy(freqs, mean_after, color=COLORS["after"], lw=1.2, label=after_label)
    if line_freq is not None:
        ax.axvline(line_freq, color=COLORS["line_marker"], ls=":", lw=0.8, alpha=0.8)
    ax.set_xlim(0.0, fmax)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("PSD", fontsize=FONTS["label"])
    themed_legend(ax, loc="upper right")
    ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")


def _plot_segment_counts_panel(ax, segment_info, title="(a)  Segment counts"):
    """Plot per-segment count bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used for rendering.
    segment_info : sequence[dict]
        Segment metadata. The panel reads ``count`` and falls back to
        ``n_selected`` then ``n_removed``.
    title : str
        Panel title.
    """
    style_axes(ax, grid=True)
    counts = []
    for segment in segment_info:
        count = segment.get("count")
        if count is None:
            count = segment.get("n_selected", segment.get("n_removed", 0))
        counts.append(int(count))
    x = np.arange(len(counts))
    ax.bar(x, counts, color=COLORS["primary"], edgecolor="none", width=0.75)
    ax.axhline(np.mean(counts), color=COLORS["accent"], ls="--", lw=0.9)
    ax.set_xlabel("Segment index", fontsize=FONTS["label"])
    ax.set_ylabel("Count", fontsize=FONTS["label"])
    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")


def _plot_segment_metric_panel(
    ax,
    segment_info,
    title="(b)  Segment metric",
):
    """Plot optional per-segment scalar metric.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used for rendering.
    segment_info : sequence[dict]
        Segment metadata. The panel reads ``metric`` and falls back to
        ``fine_freq`` then ``frequency``.
    title : str
        Panel title.
    """
    style_axes(ax, grid=True)
    values = []
    for segment in segment_info:
        if "metric" in segment:
            values.append(float(segment["metric"]))
        elif "fine_freq" in segment:
            values.append(float(segment["fine_freq"]))
        elif "frequency" in segment:
            values.append(float(segment["frequency"]))
    if not values:
        ax.text(
            0.5,
            0.5,
            "No metric in segment_info",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
    else:
        x = np.arange(len(values))
        ax.plot(x, values, "o-", color=COLORS["primary"], lw=1.0, markersize=3)
        ax.set_xlabel("Segment index", fontsize=FONTS["label"])
        ax.set_ylabel("Value", fontsize=FONTS["label"])
        ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")


def _plot_segment_boundaries_panel(
    ax,
    segment_info,
    sfreq=None,
    title="(c)  Segment boundaries",
):
    """Plot segment boundaries as a horizontal timeline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used for rendering.
    segment_info : sequence[dict]
        Segment metadata with sample-based ``start``/``end`` fields.
    sfreq : float | None
        Sampling frequency used to convert boundaries into seconds.
    title : str
        Panel title.
    """
    style_axes(ax)
    if sfreq is None or sfreq <= 0:
        ax.text(
            0.5,
            0.5,
            "sfreq required",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
        ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")
        return

    starts = [float(segment.get("start", 0.0)) / sfreq for segment in segment_info]
    ends = [float(segment.get("end", 0.0)) / sfreq for segment in segment_info]
    for idx, (start, end) in enumerate(zip(starts, ends)):
        width = max(0.0, end - start)
        ax.barh(
            0,
            width,
            left=start,
            height=0.7,
            color=COLORS["primary"],
            edgecolor="white",
            linewidth=0.25,
        )
        count = segment_info[idx].get(
            "count",
            segment_info[idx].get("n_selected", segment_info[idx].get("n_removed", 0)),
        )
        ax.text(
            (start + end) / 2,
            0,
            str(count),
            ha="center",
            va="center",
            fontsize=5,
            color="white",
            fontweight="bold",
        )
    ax.set_xlabel("Time (s)", fontsize=FONTS["label"])
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(title, fontsize=FONTS["title"], fontweight="semibold", loc="left")
