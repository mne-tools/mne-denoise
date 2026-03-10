"""Generic multi-panel summary composers.

This module provides method-agnostic summary figures that compose
precomputed inputs only. Estimator fitting and benchmark-specific defaults
are intentionally out of scope for this plotting layer.

This module contains:
1. Dashboard-style denoising summaries.
2. Metric trade-off summary composers.
3. Generic component/signal/interaction/endpoint summary composers.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from collections.abc import Mapping

import mne
import numpy as np

from ._summary_panels import (
    _draw_summary_table,
    _new_summary_figure,
    _new_summary_grid,
    _plot_before_after_psd_panel,
    _plot_patterns_panel,
    _plot_removed_power_panel,
    _plot_segment_boundaries_panel,
    _plot_segment_counts_panel,
    _plot_segment_metric_panel,
    _plot_selection_scores_panel,
    _plot_source_trace_panel,
)
from ._utils import _compute_gfp
from .signals import plot_power_ratio_map
from .spectra import plot_psd_comparison
from .stats import plot_metric_comparison, plot_tradeoff_scatter
from .theme import (
    COLORS,
    FONTS,
    _finalize_fig,
    get_series_color,
    style_axes,
    themed_figure,
    themed_legend,
)


def plot_denoising_summary(
    inst_before,
    inst_after,
    *,
    info,
    times,
    title="Denoising Summary",
    show=True,
    fname=None,
):
    """Plot a generic denoising diagnostics dashboard.

    Parameters
    ----------
    inst_before, inst_after : MNE object | ndarray
        Signals before and after denoising. Supported array shapes are
        ``(n_channels, n_times)`` and ``(n_epochs, n_channels, n_times)``.
        MNE Raw/Epochs/Evoked inputs are also accepted.
    info : mne.Info
        Channel info used for the power-ratio map panel.
    times : array-like of shape (n_times,)
        Explicit time axis for GFP traces.
    title : str
        Figure title.
    show : bool
        Whether to display the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If ``times`` is not 1D or does not match the GFP length.

    Notes
    -----
    The figure is composed of three panels:
    1. Per-channel power ratio map (before/after).
    2. PSD comparison (before/after).
    3. GFP overlay with difference shading.

    Examples
    --------
    >>> from mne_denoise.viz import plot_denoising_summary
    >>> fig = plot_denoising_summary(
    ...     before,
    ...     after,
    ...     info=info,
    ...     times=times,
    ...     show=False,
    ... )
    """
    from matplotlib.gridspec import GridSpec

    fig = _new_summary_figure(figsize=(12, 10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    ax_map = fig.add_subplot(gs[0, 0])
    plot_power_ratio_map(inst_before, inst_after, info=info, ax=ax_map, show=False)

    ax_psd = fig.add_subplot(gs[0, 1])
    plot_psd_comparison(inst_before, inst_after, ax=ax_psd, show=False)

    ax_gfp = fig.add_subplot(gs[1, :])

    times = np.asarray(times, dtype=float)
    gfp_before = _compute_gfp(inst_before)
    gfp_after = _compute_gfp(inst_after)
    if times.ndim != 1 or times.size != gfp_before.size:
        raise ValueError("times must be 1D and match the GFP vector length.")

    ax_gfp.plot(times, gfp_before, label="Before", color=COLORS["before"], alpha=0.7)
    ax_gfp.plot(times, gfp_after, label="After", color=COLORS["after"], alpha=0.7)
    ax_gfp.fill_between(
        times,
        gfp_before,
        gfp_after,
        color=COLORS["muted"],
        alpha=0.2,
        label="Difference",
    )
    themed_legend(ax_gfp, loc="best")
    ax_gfp.set_xlabel("Time", fontsize=FONTS["label"])
    ax_gfp.set_ylabel("Global Field Power", fontsize=FONTS["label"])
    ax_gfp.set_title("Temporal Signal Comparison (GFP)", fontsize=FONTS["title"])
    style_axes(ax_gfp, grid=True)

    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_metric_tradeoff_summary(
    data,
    *,
    group_col,
    subject_col,
    x_col,
    y_col,
    metric_col,
    x_label=None,
    y_label=None,
    metric_label=None,
    group_order=None,
    group_colors=None,
    group_labels=None,
    tradeoff_title="Metric Trade-off",
    metric_title="Metric Comparison",
    reference_x=None,
    reference_y=None,
    reference_value=None,
    reference_label="Reference",
    suptitle="Trade-off Analysis",
    show=True,
    fname=None,
):
    """Plot a two-panel metric trade-off summary.

    Parameters
    ----------
    data : mapping of str to array-like
        Columnar data mapping with aligned 1D columns.
    group_col : str
        Group column name.
    subject_col : str
        Subject column name used for paired metric comparison.
    x_col, y_col : str
        Metric columns used for trade-off scatter axes.
    metric_col : str
        Metric column used for the right-panel comparison.
    x_label, y_label, metric_label : str | None
        Optional axis labels.
    group_order : sequence of str | None
        Optional group display order.
    group_colors, group_labels : mapping | None
        Optional color/label overrides keyed by group.
    tradeoff_title, metric_title, suptitle : str
        Panel and figure titles.
    reference_x, reference_y, reference_value : float | None
        Optional reference lines.
    reference_label : str
        Label for the metric reference line.
    show : bool
        Whether to display the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Notes
    -----
    This is a composition wrapper around:
    - :func:`mne_denoise.viz.plot_tradeoff_scatter`
    - :func:`mne_denoise.viz.plot_metric_comparison`
    """
    columns = {name: np.asarray(values) for name, values in data.items()}
    fig, axes = themed_figure(1, 2, figsize=(16, 6))

    plot_tradeoff_scatter(
        columns,
        group_col=group_col,
        group_order=group_order,
        group_colors=group_colors,
        group_labels=group_labels,
        x_col=x_col,
        y_col=y_col,
        x_label=x_label,
        y_label=y_label,
        title=tradeoff_title,
        reference_x=reference_x,
        reference_y=reference_y,
        ax=axes[0],
        show=False,
    )

    plot_metric_comparison(
        columns,
        metric_col=metric_col,
        metric_label=metric_label,
        group_col=group_col,
        subject_col=subject_col,
        group_order=group_order,
        group_colors=group_colors,
        group_labels=group_labels,
        title=metric_title,
        reference_value=reference_value,
        reference_label=reference_label,
        ax=axes[1],
        show=False,
    )

    fig.suptitle(suptitle, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_cleaning_summary(
    scores=None,
    selected_count=0,
    patterns=None,
    info=None,
    channel_names=None,
    removed=None,
    sources=None,
    sfreq=None,
    freqs=None,
    psd_before=None,
    psd_after=None,
    line_freq=None,
    fmax=100.0,
    segment_info=None,
    summary_rows=None,
    title="Component Cleaning Summary",
    figsize=None,
    dpi=200,
    show=True,
    fname=None,
):
    """Plot a generic component-cleaning dashboard.

    Parameters
    ----------
    scores : array-like | None
        Component score vector.
    selected_count : int
        Number of selected/removed components.
    patterns : array-like | None
        Component patterns with shape ``(n_channels, n_components)``.
    info : mne.Info | None
        Optional MNE info for topomap rendering.
    channel_names : sequence of str | None
        Channel labels used by non-topomap fallback panels.
    removed : array-like | None
        Removed signal with shape ``(n_channels, n_times)`` or
        ``(n_epochs, n_channels, n_times)``.
    sources : array-like | None
        Component source traces with shape ``(n_components, n_times)`` or
        ``(n_epochs, n_components, n_times)``.
    sfreq : float | None
        Sampling frequency for source/segment time axes.
    freqs : array-like | None
        Frequency axis for PSD panel.
    psd_before, psd_after : array-like | None
        PSD values for before/after comparison.
    line_freq : float | None
        Optional line frequency marker in PSD panel.
    fmax : float
        Upper frequency bound for PSD panel.
    segment_info : list[dict] | None
        Optional segmented metadata. If provided, segmented panels replace
        score/removed/source panels.
    summary_rows : list[tuple[str, object]] | None
        Optional explicit table rows.
    title : str
        Figure title.
    figsize : tuple[float, float] | None
        Figure size.
    dpi : int
        Figure resolution.
    show : bool
        Whether to show the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If panel inputs are incompatible with expected dimensions or metadata
        requirements (for example invalid source/pattern/segment shapes).

    Notes
    -----
    This function is a thin composer around internal panel painters in
    :mod:`mne_denoise.viz._summary_panels`. It does not run fitting or
    denoising; all inputs are expected to be precomputed by the caller.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_component_cleaning_summary
    >>> rng = np.random.default_rng(0)
    >>> freqs = np.linspace(0, 80, 161)
    >>> fig = plot_component_cleaning_summary(
    ...     scores=np.array([2.0, 1.2, 0.7]),
    ...     selected_count=1,
    ...     patterns=rng.standard_normal((5, 3)),
    ...     removed=rng.standard_normal((5, 200)),
    ...     sources=rng.standard_normal((3, 200)),
    ...     sfreq=200.0,
    ...     freqs=freqs,
    ...     psd_before=rng.random((5, freqs.size)),
    ...     psd_after=rng.random((5, freqs.size)),
    ...     show=False,
    ... )
    """
    fig, gs = _new_summary_grid(figsize=figsize, dpi=dpi, hspace=0.40)

    if segment_info:
        ax_a = fig.add_subplot(gs[0, 0])
        _plot_segment_counts_panel(
            ax_a, segment_info, title="(a)  Segment selection counts"
        )

        ax_b = fig.add_subplot(gs[0, 1])
        _plot_segment_metric_panel(ax_b, segment_info, title="(b)  Segment metric")

        ax_c = fig.add_subplot(gs[1, 0])
        _plot_segment_boundaries_panel(
            ax_c,
            segment_info,
            sfreq=sfreq,
            title="(c)  Segment boundaries",
        )

        _plot_patterns_panel(
            fig,
            gs[1, 1],
            patterns,
            info=info,
            channel_names=channel_names,
            title="(d)  Component patterns",
        )
    else:
        ax_a = fig.add_subplot(gs[0, 0])
        _plot_selection_scores_panel(
            ax_a,
            scores,
            int(selected_count),
            selected_label=f"Selected ({int(selected_count)})",
            title="(a)  Component scores",
        )

        _plot_patterns_panel(
            fig,
            gs[0, 1],
            patterns,
            info=info,
            channel_names=channel_names,
            title="(b)  Component patterns",
        )

        ax_c = fig.add_subplot(gs[1, 0])
        _plot_removed_power_panel(
            fig,
            ax_c,
            removed,
            info=info,
            channel_names=channel_names,
            title="(c)  Removed signal power",
        )

        ax_d = fig.add_subplot(gs[1, 1])
        _plot_source_trace_panel(
            ax_d,
            sources,
            sfreq=sfreq,
            title="(d)  Component traces",
        )

    ax_e = fig.add_subplot(gs[2, 0])
    _plot_before_after_psd_panel(
        ax_e,
        freqs=freqs,
        psd_before=psd_before,
        psd_after=psd_after,
        line_freq=line_freq,
        fmax=fmax,
        title="(e)  Power spectral density",
    )

    ax_f = fig.add_subplot(gs[2, 1])
    rows = list(summary_rows) if summary_rows is not None else []
    if not rows:
        if scores is not None:
            rows.append(("n_components", np.asarray(scores).size))
        if selected_count:
            rows.append(("selected_count", int(selected_count)))
        if segment_info:
            rows.append(("n_segments", len(segment_info)))
        if line_freq is not None:
            rows.append(("line_freq_hz", f"{float(line_freq):.3g}"))
        if sfreq is not None:
            rows.append(("sfreq_hz", f"{float(sfreq):.6g}"))
    if not rows:
        rows = [("status", "No metadata provided")]
    _draw_summary_table(ax_f, rows)

    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_signal_diagnostics_summary(
    signals,
    *,
    channel,
    times,
    group_order,
    reference_group,
    group_colors,
    group_labels,
    channel_names=None,
    channel_label=None,
    windows=None,
    title="Signal Diagnostics Summary",
    show=True,
    fname=None,
):
    """Plot grouped time-domain signal diagnostics.

    Parameters
    ----------
    signals : mapping[str, array-like | MNE object]
        Mapping from group name to signal data. Each value must be 2D
        ``(n_channels, n_times)`` or 3D ``(n_epochs, n_channels, n_times)``.
    channel : int | str
        Channel selector. String selectors require explicit ``channel_names``.
    times : array-like of shape (n_times,)
        Explicit time axis.
    channel_names : sequence of str | None
        Explicit channel names used for string ``channel`` selectors and
        channel labeling. The function does not infer names from MNE inputs.
    channel_label : str | None
        Explicit channel label when ``channel_names`` is not provided.
    group_order : sequence of str
        Explicit group plotting order.
    reference_group : str
        Explicit reference group used for the difference panel.
    group_colors, group_labels : mapping
        Explicit color and display-label mappings keyed by group.
    windows : sequence of tuple[float, float, str] | None
        Optional highlighted windows. Each entry must be
        ``(start, stop, label)``.
    title : str
        Figure title.
    show : bool
        Whether to show the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If ``signals`` is empty, ``times`` is shape-incompatible, channel
        metadata is insufficient for the requested selector, or group ordering
        is invalid.

    Notes
    -----
    This is a strict explicit API: ``group_order``, ``reference_group``,
    ``group_colors``, and ``group_labels`` are caller-owned inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_signal_diagnostics_summary
    >>> rng = np.random.default_rng(0)
    >>> n_times = 200
    >>> times = np.arange(n_times) / 200.0
    >>> signals = {
    ...     "before": rng.standard_normal((3, n_times)),
    ...     "after": rng.standard_normal((3, n_times)),
    ... }
    >>> fig = plot_signal_diagnostics_summary(
    ...     signals,
    ...     channel=1,
    ...     channel_label="C2",
    ...     times=times,
    ...     group_order=["before", "after"],
    ...     reference_group="before",
    ...     group_colors={"before": "#4C72B0", "after": "#55A868"},
    ...     group_labels={"before": "Before", "after": "After"},
    ...     windows=[(0.08, 0.14, "early")],
    ...     show=False,
    ... )
    """
    if not isinstance(signals, Mapping) or len(signals) == 0:
        raise ValueError("signals must be a non-empty mapping.")

    group_order = list(group_order)
    if len(group_order) == 0:
        raise ValueError("group_order must contain at least one group name.")
    if reference_group not in group_order:
        raise ValueError("reference_group must be present in group_order.")

    prepared = {}
    n_channels = None
    n_times = None
    for name in group_order:
        data_in = signals[name]
        if isinstance(data_in, (mne.io.BaseRaw, mne.Evoked, mne.BaseEpochs)):
            arr = np.asarray(data_in.get_data(), dtype=float)
        else:
            arr = np.asarray(data_in, dtype=float)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        if arr.ndim != 2:
            raise ValueError(
                "Each signal must be 2D (n_channels, n_times) or 3D "
                "(n_epochs, n_channels, n_times)."
            )
        if n_channels is None:
            n_channels = arr.shape[0]
            n_times = arr.shape[1]
        elif arr.shape != (n_channels, n_times):
            raise ValueError("All signals must have matching channel/time dimensions.")
        prepared[name] = arr

    names = None
    if channel_names is not None:
        names = list(channel_names)
        if len(names) != n_channels:
            raise ValueError(
                "channel_names length must match the number of signal channels."
            )

    if isinstance(channel, str):
        if names is None:
            raise ValueError("String channel selectors require channel names.")
        channel_idx = names.index(channel)
    else:
        channel_idx = int(channel)

    if channel_label is not None:
        channel_label = str(channel_label)
    elif names is not None:
        channel_label = str(names[channel_idx])
    else:
        raise ValueError(
            "channel_label must be provided when channel_names is not provided."
        )

    time_axis = np.asarray(times, dtype=float)
    if time_axis.ndim != 1 or time_axis.size != n_times:
        raise ValueError("times must be 1D and match n_times.")

    window_specs = [] if windows is None else list(windows)

    fig, axes = themed_figure(2, 2, figsize=(14, 8))
    if isinstance(axes, np.ndarray):
        ax_trace, ax_gfp, ax_diff, ax_table = axes.ravel()
    else:
        ax_trace = ax_gfp = ax_diff = ax_table = axes

    ref_name = reference_group
    ref_data = prepared[ref_name]

    for group in group_order:
        color = group_colors[group]
        label = group_labels[group]
        data = prepared[group]
        ax_trace.plot(
            time_axis,
            data[channel_idx],
            color=color,
            lw=1.3,
            alpha=0.9,
            label=label,
        )
        ax_gfp.plot(
            time_axis,
            _compute_gfp(data),
            color=color,
            lw=1.3,
            alpha=0.9,
            label=label,
        )
        if group != ref_name:
            ax_diff.plot(
                time_axis,
                data[channel_idx] - ref_data[channel_idx],
                color=color,
                lw=1.1,
                alpha=0.85,
                label=f"{label} - {ref_name}",
            )

    for win_idx, (start, stop, _) in enumerate(window_specs):
        shade = get_series_color(win_idx)
        for panel in (ax_trace, ax_gfp, ax_diff):
            panel.axvspan(start, stop, color=shade, alpha=0.10, lw=0)

    ax_trace.set_xlabel("Time", fontsize=FONTS["label"])
    ax_trace.set_ylabel("Amplitude", fontsize=FONTS["label"])
    ax_trace.set_title(
        f"(a)  Channel overlay ({channel_label})", fontsize=FONTS["title"]
    )
    style_axes(ax_trace, grid=True)
    themed_legend(ax_trace)

    ax_gfp.set_xlabel("Time", fontsize=FONTS["label"])
    ax_gfp.set_ylabel("Global Field Power", fontsize=FONTS["label"])
    ax_gfp.set_title("(b)  GFP comparison", fontsize=FONTS["title"])
    style_axes(ax_gfp, grid=True)
    themed_legend(ax_gfp)

    ax_diff.axhline(0.0, color=COLORS["stat_reference"], ls="--", lw=0.8, alpha=0.5)
    ax_diff.set_xlabel("Time", fontsize=FONTS["label"])
    ax_diff.set_ylabel("Difference", fontsize=FONTS["label"])
    ax_diff.set_title(f"(c)  Difference vs {ref_name}", fontsize=FONTS["title"])
    style_axes(ax_diff, grid=True)
    themed_legend(ax_diff)

    rows = [("reference_group", ref_name), ("channel", channel_label)]
    for group in group_order:
        rms = np.sqrt(np.mean(prepared[group][channel_idx] ** 2))
        rows.append((f"{group} RMS", f"{rms:.4g}"))
    for start, stop, label in window_specs:
        mask = (time_axis >= start) & (time_axis <= stop)
        if not mask.any():
            continue
        ref_mean = np.mean(ref_data[channel_idx, mask])
        rows.append((f"{label} mean ({ref_name})", f"{ref_mean:.4g}"))
    _draw_summary_table(ax_table, rows, title="(d)  Summary")

    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_condition_interaction_summary(
    traces,
    times,
    errors=None,
    condition_order=None,
    group_order=None,
    group_colors=None,
    group_labels=None,
    windows=None,
    title="Condition Interaction Summary",
    show=True,
    fname=None,
):
    """Plot condition-by-group interaction traces.

    Parameters
    ----------
    traces : mapping[str, mapping[str, array-like]]
        Nested mapping ``condition -> group -> trace``.
    times : array-like of shape (n_times,)
        Explicit time axis.
    errors : mapping | None
        Optional nested mapping ``condition -> group -> standard error``.
    condition_order : sequence of str | None
        Optional condition order. Defaults to first-seen order.
    group_order : sequence of str | None
        Optional group order. Defaults to first-seen order across conditions.
    group_colors, group_labels : mapping | None
        Optional color/label overrides keyed by group.
    windows : sequence of tuple[float, float, str] | None
        Optional highlighted windows as explicit ``(start, stop, label)`` tuples.
    title : str
        Figure title.
    show : bool
        Whether to show the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If ``traces`` is empty, ``times`` is not 1D, or trace/error arrays do
        not match the expected time dimension.

    Notes
    -----
    Conditions are rendered as separate panels and groups as overlaid lines
    within each panel.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_condition_interaction_summary
    >>> times = np.linspace(-0.2, 0.5, 300)
    >>> traces = {
    ...     "A": {"before": np.sin(times), "after": 0.7 * np.sin(times)},
    ...     "B": {"before": np.cos(times), "after": 0.6 * np.cos(times)},
    ... }
    >>> fig = plot_condition_interaction_summary(
    ...     traces,
    ...     times=times,
    ...     windows=[(0.08, 0.14, "w1")],
    ...     show=False,
    ... )
    """
    if not isinstance(traces, Mapping) or len(traces) == 0:
        raise ValueError("traces must be a non-empty mapping of conditions.")

    if condition_order is None:
        condition_order = list(traces.keys())
    else:
        condition_order = list(condition_order)

    if group_order is None:
        seen = []
        for condition in condition_order:
            for group in traces[condition]:
                if group not in seen:
                    seen.append(group)
        group_order = seen
    else:
        group_order = list(group_order)

    time_axis = np.asarray(times, dtype=float)
    if time_axis.ndim != 1:
        raise ValueError("times must be 1D.")

    fig, axes = themed_figure(
        1, len(condition_order), figsize=(6 * len(condition_order), 5), squeeze=False
    )
    window_specs = [] if windows is None else list(windows)

    for cond_idx, condition in enumerate(condition_order):
        ax = axes[0, cond_idx]
        for win_idx, (start, stop, _) in enumerate(window_specs):
            ax.axvspan(start, stop, color=get_series_color(win_idx), alpha=0.10, lw=0)

        for group_idx, group in enumerate(group_order):
            if group not in traces[condition]:
                continue
            trace = np.asarray(traces[condition][group], dtype=float)
            if trace.ndim != 1 or trace.size != time_axis.size:
                raise ValueError("All traces must be 1D and match times length.")
            color = (
                group_colors[group]
                if group_colors is not None and group in group_colors
                else get_series_color(group_idx)
            )
            label = (
                group_labels[group]
                if group_labels is not None and group in group_labels
                else group
            )
            ax.plot(time_axis, trace, color=color, lw=1.4, alpha=0.9, label=label)
            if (
                errors is not None
                and condition in errors
                and group in errors[condition]
            ):
                se = np.asarray(errors[condition][group], dtype=float)
                if se.shape != trace.shape:
                    raise ValueError(
                        "errors traces must match corresponding trace shape."
                    )
                ax.fill_between(
                    time_axis, trace - se, trace + se, color=color, alpha=0.15, lw=0
                )

        ax.axhline(0.0, color=COLORS["stat_reference"], ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Time", fontsize=FONTS["label"])
        if cond_idx == 0:
            ax.set_ylabel("Amplitude", fontsize=FONTS["label"])
        ax.set_title(str(condition), fontsize=FONTS["title"])
        style_axes(ax, grid=True)
        themed_legend(ax)

    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_group_condition_interaction_summary(
    traces,
    times,
    errors=None,
    group_order=None,
    condition_order=None,
    condition_labels=None,
    windows=None,
    title="Group Condition Interaction Summary",
    show=True,
    fname=None,
):
    """Plot group-wise condition interaction traces.

    Parameters
    ----------
    traces : mapping[str, mapping[str, array-like]]
        Nested mapping ``group -> condition -> trace``.
    times : array-like of shape (n_times,)
        Explicit time axis.
    errors : mapping | None
        Optional nested mapping ``group -> condition -> standard error``.
    group_order : sequence of str | None
        Optional group order. Defaults to first-seen order.
    condition_order : sequence of str | None
        Optional condition order. Defaults to first-seen across groups.
    condition_labels : mapping | None
        Optional label overrides keyed by condition.
    windows : sequence of tuple[float, float, str] | None
        Optional highlighted windows as explicit ``(start, stop, label)`` tuples.
    title : str
        Figure title.
    show : bool
        Whether to show the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If ``traces`` is empty, ``times`` is not 1D, or trace/error arrays do
        not match the expected time dimension.

    Notes
    -----
    Groups are rendered as separate panels and conditions as overlaid lines
    within each panel.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_group_condition_interaction_summary
    >>> times = np.linspace(-0.2, 0.5, 300)
    >>> traces = {
    ...     "before": {"A": np.sin(times), "B": np.cos(times)},
    ...     "after": {"A": 0.7 * np.sin(times), "B": 0.6 * np.cos(times)},
    ... }
    >>> fig = plot_group_condition_interaction_summary(
    ...     traces,
    ...     times=times,
    ...     windows=[(0.08, 0.14, "w1")],
    ...     show=False,
    ... )
    """
    if not isinstance(traces, Mapping) or len(traces) == 0:
        raise ValueError("traces must be a non-empty mapping of groups.")

    if group_order is None:
        group_order = list(traces.keys())
    else:
        group_order = list(group_order)

    if condition_order is None:
        seen = []
        for group in group_order:
            for condition in traces[group]:
                if condition not in seen:
                    seen.append(condition)
        condition_order = seen
    else:
        condition_order = list(condition_order)

    time_axis = np.asarray(times, dtype=float)
    if time_axis.ndim != 1:
        raise ValueError("times must be 1D.")

    window_specs = [] if windows is None else list(windows)
    fig, axes = themed_figure(
        1, len(group_order), figsize=(6 * len(group_order), 5), squeeze=False
    )

    for group_idx, group in enumerate(group_order):
        ax = axes[0, group_idx]
        for win_idx, (start, stop, _) in enumerate(window_specs):
            ax.axvspan(start, stop, color=get_series_color(win_idx), alpha=0.10, lw=0)

        for cond_idx, condition in enumerate(condition_order):
            if condition not in traces[group]:
                continue
            trace = np.asarray(traces[group][condition], dtype=float)
            if trace.ndim != 1 or trace.size != time_axis.size:
                raise ValueError("All traces must be 1D and match times length.")
            label = (
                condition_labels[condition]
                if condition_labels is not None and condition in condition_labels
                else condition
            )
            color = get_series_color(cond_idx)
            ax.plot(time_axis, trace, color=color, lw=1.4, alpha=0.9, label=label)
            if errors is not None and group in errors and condition in errors[group]:
                se = np.asarray(errors[group][condition], dtype=float)
                if se.shape != trace.shape:
                    raise ValueError(
                        "errors traces must match corresponding trace shape."
                    )
                ax.fill_between(
                    time_axis, trace - se, trace + se, color=color, alpha=0.15, lw=0
                )

        ax.axhline(0.0, color=COLORS["stat_reference"], ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Time", fontsize=FONTS["label"])
        if group_idx == 0:
            ax.set_ylabel("Amplitude", fontsize=FONTS["label"])
        ax.set_title(str(group), fontsize=FONTS["title"])
        style_axes(ax, grid=True)
        themed_legend(ax)

    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_endpoint_metrics_summary(
    data,
    metric_col,
    group_col="group",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    reference_value=None,
    reference_label="Reference",
    null_distribution=None,
    observed_value=None,
    title="Endpoint Metrics Summary",
    show=True,
    fname=None,
):
    """Plot a generic endpoint-metric storyboard.

    Parameters
    ----------
    data : mapping[str, array-like]
        Columnar mapping with at least ``metric_col``, ``group_col``, and
        ``subject_col``.
    metric_col : str
        Metric column to summarize.
    group_col : str
        Group identifier column.
    subject_col : str
        Subject identifier column.
    group_order : sequence of str | None
        Optional group order. Defaults to first-seen order.
    group_colors, group_labels : mapping | None
        Optional color/label overrides keyed by group.
    reference_value : float | None
        Optional horizontal reference line for metric panels.
    reference_label : str
        Label for ``reference_value``.
    null_distribution : array-like | None
        Optional null distribution for the null panel.
    observed_value : float | None
        Observed statistic for the null panel.
    title : str
        Figure title.
    show : bool
        Whether to show the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    KeyError
        If required columns are missing from ``data``.
    ValueError
        If numeric conversion or plotting operations receive incompatible
        shapes.

    Notes
    -----
    The output is a 2x2 storyboard combining grouped means, paired subject
    trajectories, per-group distributions, and optional null-distribution
    diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_endpoint_metrics_summary
    >>> data = {
    ...     "subject": np.array(["s1", "s1", "s2", "s2"]),
    ...     "group": np.array(["A", "B", "A", "B"]),
    ...     "score": np.array([1.2, 0.9, 1.1, 0.8]),
    ... }
    >>> fig = plot_endpoint_metrics_summary(
    ...     data,
    ...     metric_col="score",
    ...     show=False,
    ... )
    """
    columns = {name: np.asarray(values) for name, values in data.items()}
    metric = np.asarray(columns[metric_col], dtype=float)
    groups = np.asarray(columns[group_col], dtype=object)
    subjects = np.asarray(columns[subject_col], dtype=object)

    if group_order is None:
        group_order = list(dict.fromkeys(groups.tolist()))
    else:
        group_order = list(group_order)

    fig, axes = themed_figure(2, 2, figsize=(14, 9))
    if isinstance(axes, np.ndarray):
        ax_a, ax_b, ax_c, ax_d = axes.ravel()
    else:
        ax_a = ax_b = ax_c = ax_d = axes

    means = []
    sems = []
    for group in group_order:
        vals = metric[groups == group]
        vals = vals[np.isfinite(vals)]
        means.append(vals.mean() if vals.size else np.nan)
        if vals.size > 1:
            sems.append(vals.std(ddof=1) / np.sqrt(vals.size))
        else:
            sems.append(0.0)
    x = np.arange(len(group_order))
    bar_colors = [
        group_colors[group]
        if group_colors is not None and group in group_colors
        else get_series_color(i)
        for i, group in enumerate(group_order)
    ]
    ax_a.bar(
        x,
        means,
        yerr=sems,
        color=bar_colors,
        edgecolor=COLORS["edge"],
        linewidth=0.6,
        alpha=0.85,
    )
    if reference_value is not None:
        ax_a.axhline(
            reference_value,
            color=COLORS["stat_reference"],
            ls="--",
            lw=0.8,
            alpha=0.6,
            label=reference_label,
        )
        themed_legend(ax_a)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [
            group_labels[group]
            if group_labels is not None and group in group_labels
            else group
            for group in group_order
        ],
        fontsize=FONTS["tick"],
    )
    ax_a.set_ylabel(metric_col, fontsize=FONTS["label"])
    ax_a.set_title("(a)  Group mean ± SEM", fontsize=FONTS["title"])
    style_axes(ax_a, grid=True)

    subj_order = list(dict.fromkeys(subjects.tolist()))
    traces = np.full((len(subj_order), len(group_order)), np.nan, dtype=float)
    for subj_idx, subject in enumerate(subj_order):
        mask_subject = subjects == subject
        for grp_idx, group in enumerate(group_order):
            vals = metric[mask_subject & (groups == group)]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                traces[subj_idx, grp_idx] = vals[0]
        ax_b.plot(
            range(len(group_order)),
            traces[subj_idx],
            "o-",
            color=COLORS["stat_subject"],
            alpha=0.35,
            markersize=4,
            lw=0.8,
        )
    mean_trace = np.nanmean(traces, axis=0)
    ax_b.plot(
        range(len(group_order)),
        mean_trace,
        "s-",
        color=COLORS["stat_mean"],
        markersize=6,
        lw=1.6,
        label="Mean",
    )
    if reference_value is not None:
        ax_b.axhline(
            reference_value,
            color=COLORS["stat_reference"],
            ls="--",
            lw=0.8,
            alpha=0.6,
            label=reference_label,
        )
    ax_b.set_xticks(range(len(group_order)))
    ax_b.set_xticklabels(
        [
            group_labels[group]
            if group_labels is not None and group in group_labels
            else group
            for group in group_order
        ],
        fontsize=FONTS["tick"],
    )
    ax_b.set_ylabel(metric_col, fontsize=FONTS["label"])
    ax_b.set_title("(b)  Paired subject trajectories", fontsize=FONTS["title"])
    style_axes(ax_b, grid=True)
    themed_legend(ax_b)

    bp = ax_c.boxplot(
        [metric[(groups == group) & np.isfinite(metric)] for group in group_order],
        labels=[
            group_labels[group]
            if group_labels is not None and group in group_labels
            else group
            for group in group_order
        ],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(COLORS["edge"])
    if reference_value is not None:
        ax_c.axhline(
            reference_value, color=COLORS["stat_reference"], ls="--", lw=0.8, alpha=0.6
        )
    ax_c.set_ylabel(metric_col, fontsize=FONTS["label"])
    ax_c.set_title("(c)  Distribution by group", fontsize=FONTS["title"])
    style_axes(ax_c, grid=True)

    if null_distribution is not None:
        null_distribution = np.asarray(null_distribution, dtype=float)
        null_distribution = null_distribution[np.isfinite(null_distribution)]
        ax_d.hist(
            null_distribution,
            bins=30,
            color=COLORS["muted"],
            alpha=0.65,
            edgecolor=COLORS["edge"],
            linewidth=0.4,
            label="Null",
        )
        if observed_value is not None:
            ax_d.axvline(
                observed_value,
                color=COLORS["accent"],
                lw=1.2,
                ls="--",
                label="Observed",
            )
            if null_distribution.size:
                p_val = np.mean(np.abs(null_distribution) >= abs(observed_value))
                ax_d.text(
                    0.98,
                    0.95,
                    f"p={p_val:.4g}",
                    transform=ax_d.transAxes,
                    ha="right",
                    va="top",
                    fontsize=FONTS["annotation"],
                )
        ax_d.set_xlabel(metric_col, fontsize=FONTS["label"])
        ax_d.set_title("(d)  Null distribution", fontsize=FONTS["title"])
        themed_legend(ax_d)
        style_axes(ax_d, grid=True)
    else:
        rows = [
            ("metric", metric_col),
            ("n_subjects", len(subj_order)),
            ("n_groups", len(group_order)),
            ("reference", reference_value if reference_value is not None else "None"),
        ]
        _draw_summary_table(ax_d, rows, title="(d)  Summary")

    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)
