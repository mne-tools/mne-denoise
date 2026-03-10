"""Signal-domain visualization primitives.

This module contains reusable, method-agnostic plots for time-domain
comparisons between original and denoised signals.

This module contains:
1. Global field power (GFP) comparisons for 2D/3D arrays or MNE objects.
2. Channel-level time-course overlays with explicit channel selection.
3. Topographic power-ratio maps from per-channel variances.
4. Single-trace overlays for reconstruction checks.
5. Grouped grand-average evoked comparisons.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import mne
import numpy as np

from ._utils import _compute_gfp
from .theme import (
    COLORS,
    FONTS,
    SEQUENTIAL_CMAP,
    _finalize_fig,
    get_series_color,
    style_axes,
    themed_figure,
    themed_legend,
)


def _as_signal_array(inst):
    """Return signal data as a float array with shape (C, T) or (E, C, T)."""
    if isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
        data = np.asarray(inst.get_data(), dtype=float)
    else:
        data = np.asarray(inst, dtype=float)

    if data.ndim not in (2, 3):
        raise ValueError(
            "Input must be 2D (n_channels, n_times) or 3D "
            "(n_epochs, n_channels, n_times)."
        )
    return data


def _bootstrap_gfp(data, ci, n_boot):
    """Bootstrap confidence interval for GFP from epoched data."""
    n_epochs = data.shape[0]
    rng = np.random.default_rng(42)

    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n_epochs, n_epochs, replace=True)
        evoked_boot = data[idx].mean(axis=0)
        boots.append(np.sqrt(np.mean(evoked_boot**2, axis=0)))

    boots = np.asarray(boots)
    alpha = (1 - ci) / 2
    return np.percentile(boots, [100 * alpha, 100 * (1 - alpha)], axis=0)


def _variance_per_channel(data):
    """Compute per-channel variance from 2D/3D signal arrays."""
    if data.ndim == 2:
        return np.var(data, axis=1)
    return np.mean(np.var(data, axis=2), axis=0)


def _as_channel_variance(inst_or_var):
    """Return a per-channel variance vector."""
    if isinstance(inst_or_var, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
        return _variance_per_channel(_as_signal_array(inst_or_var))

    arr = np.asarray(inst_or_var, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim in (2, 3):
        return _variance_per_channel(arr)
    raise ValueError(
        "Input must be 1D variances or 2D/3D signal data for variance estimation."
    )


def _extract_overlay_trace(inst, pick):
    """Extract a single trace for overlay plotting from 2D/3D inputs."""
    if isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
        data = _as_signal_array(inst)
    else:
        data = np.asarray(inst, dtype=float)
        if data.ndim == 1:
            return data
        if data.ndim not in (2, 3):
            raise ValueError(
                "Input must be 1D, 2D (n_channels, n_times), or 3D "
                "(n_epochs, n_channels, n_times)."
            )

    if data.ndim == 3:
        data = data.mean(axis=0)

    ch_names = list(inst.ch_names) if hasattr(inst, "ch_names") else None
    n_channels = data.shape[0]

    if n_channels > 1 and pick is None:
        raise ValueError("pick must be provided when overlaying multi-channel data.")

    if pick is None:
        pick_idx = 0
    elif isinstance(pick, str):
        if ch_names is None:
            raise ValueError(
                "String picks require channel names; pass MNE input or integer picks."
            )
        if pick not in ch_names:
            raise ValueError(f"Unknown channel name: {pick}")
        pick_idx = ch_names.index(pick)
    else:
        pick_idx = int(pick)
        if pick_idx < 0 or pick_idx >= n_channels:
            raise ValueError(
                f"Channel index {pick_idx} is out of range for {n_channels} channels."
            )
    return data[pick_idx]


def plot_evoked_gfp_comparison(
    inst_before,
    inst_after,
    times,
    ci=0.95,
    n_boot=1000,
    colors=(COLORS["before"], COLORS["after"]),
    linestyles=("-", "-"),
    labels=("Before", "After"),
    x_label="Time",
    y_label="Global Field Power",
    title="Evoked GFP Comparison",
    show=True,
    ax=None,
    fname=None,
):
    """Plot GFP comparison for before/after signals.

    Parameters
    ----------
    inst_before, inst_after : MNE object | ndarray
        Signal inputs to compare. Supported array shapes are
        ``(n_channels, n_times)`` and ``(n_epochs, n_channels, n_times)``.
    times : array-like of shape (n_times,)
        Explicit time axis.
    ci : float | None
        Confidence level for bootstrap bands. If None, no interval is drawn.
        Bootstrap intervals are only computed for 3D epoched inputs.
    n_boot : int
        Number of bootstrap resamples when ``ci`` is not None.
    colors : tuple[str, str]
        Colors for before/after curves.
    linestyles : tuple[str, str]
        Linestyles for before/after curves.
    labels : tuple[str, str]
        Legend labels for before/after curves.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    title : str
        Panel title.
    show : bool
        If True, display the figure.
    ax : matplotlib.axes.Axes | None
        Target axes. If None, create a new figure and axes.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If input shapes are invalid, if time lengths differ between inputs,
        or if ``times`` length does not match ``n_times``.

    Notes
    -----
    GFP is computed as RMS across channels. For 3D epoched inputs,
    epochs are averaged first.

    Examples
    --------
    >>> from mne_denoise.viz import plot_evoked_gfp_comparison
    >>> fig = plot_evoked_gfp_comparison(
    ...     before_array, after_array, times=np.arange(500) / 250.0, show=False
    ... )
    """
    data_before = _as_signal_array(inst_before)
    data_after = _as_signal_array(inst_after)
    if data_before.shape[-1] != data_after.shape[-1]:
        raise ValueError("inst_before and inst_after must share the same n_times.")

    time_axis = np.asarray(times, dtype=float)
    if time_axis.ndim != 1 or time_axis.size != data_before.shape[-1]:
        raise ValueError("times must be a 1D array with length equal to n_times.")

    if ax is None:
        fig, ax = themed_figure(figsize=(10, 6))
    else:
        fig = ax.figure

    gfp_before = _compute_gfp(data_before)
    gfp_after = _compute_gfp(data_after)

    ax.plot(
        time_axis,
        gfp_before,
        color=colors[0],
        linestyle=linestyles[0],
        label=labels[0],
        linewidth=1.5,
    )
    ax.plot(
        time_axis,
        gfp_after,
        color=colors[1],
        linestyle=linestyles[1],
        label=labels[1],
        linewidth=1.5,
    )

    if ci is not None and data_before.ndim == 3:
        ci_low, ci_high = _bootstrap_gfp(data_before, ci=ci, n_boot=n_boot)
        ax.fill_between(
            time_axis, ci_low, ci_high, color=colors[0], alpha=0.2, linewidth=0
        )
    if ci is not None and data_after.ndim == 3:
        ci_low, ci_high = _bootstrap_gfp(data_after, ci=ci, n_boot=n_boot)
        ax.fill_between(
            time_axis, ci_low, ci_high, color=colors[1], alpha=0.2, linewidth=0
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_channel_time_course_comparison(
    inst_before,
    inst_after,
    picks,
    times,
    start=0,
    stop=None,
    before_label="Before",
    after_label="After",
    x_label="Time",
    show=True,
    fname=None,
):
    """Plot before/after channel time courses for explicit channel picks.

    Parameters
    ----------
    inst_before, inst_after : MNE object | ndarray
        Inputs to compare with shape ``(n_channels, n_times)`` or
        ``(n_epochs, n_channels, n_times)``.
    picks : sequence of int | sequence of str
        Channels to display. String picks require MNE inputs with ``ch_names``.
    start, stop : int | None
        Optional sample-index bounds applied after resolving ``times``.
    times : array-like of shape (n_times,)
        Explicit time axis.
    before_label : str
        Legend label for the first input.
    after_label : str
        Legend label for the second input.
    x_label : str
        X-axis label.
    show : bool
        If True, display the figure.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If shapes are invalid or inconsistent, picks are invalid, or
        ``times`` length does not match ``n_times``.

    Examples
    --------
    >>> from mne_denoise.viz import plot_channel_time_course_comparison
    >>> fig = plot_channel_time_course_comparison(
    ...     before_array,
    ...     after_array,
    ...     picks=[0, 2],
    ...     times=np.arange(1000) / 250.0,
    ...     show=False,
    ... )
    """
    data_before = _as_signal_array(inst_before)
    data_after = _as_signal_array(inst_after)
    if data_before.shape[-2:] != data_after.shape[-2:]:
        raise ValueError(
            "inst_before and inst_after must share the same channel/time dimensions."
        )

    if picks is None:
        raise ValueError("picks must be provided explicitly.")
    picks = list(picks)
    if len(picks) == 0:
        raise ValueError("picks cannot be empty.")

    ch_names = list(inst_before.ch_names) if hasattr(inst_before, "ch_names") else None
    n_channels = data_before.shape[-2]
    resolved = []
    labels = []
    for pick in picks:
        if isinstance(pick, str):
            if ch_names is None:
                raise ValueError(
                    "String picks require channel names; pass MNE input or integer picks."
                )
            if pick not in ch_names:
                raise ValueError(f"Unknown channel name: {pick}")
            idx = ch_names.index(pick)
            label = pick
        else:
            idx = int(pick)
            if idx < 0 or idx >= n_channels:
                raise ValueError(
                    f"Channel index {idx} is out of range for {n_channels} channels."
                )
            label = ch_names[idx] if ch_names is not None else f"ch{idx}"
        resolved.append(idx)
        labels.append(label)

    time_axis = np.asarray(times, dtype=float)
    if time_axis.ndim != 1 or time_axis.size != data_before.shape[-1]:
        raise ValueError("times must be a 1D array with length equal to n_times.")
    time_axis = time_axis[slice(start, stop)]
    data_before = data_before[..., slice(start, stop)]
    data_after = data_after[..., slice(start, stop)]

    fig, axes = themed_figure(
        len(resolved),
        1,
        sharex=True,
        figsize=(10, 2 * len(resolved)),
    )
    axes = np.atleast_1d(axes)

    for row_idx, (pick_idx, label) in enumerate(zip(resolved, labels)):
        ax = axes[row_idx]
        if data_before.ndim == 3:
            y_before = data_before[:, pick_idx, :].mean(axis=0)
            y_after = data_after[:, pick_idx, :].mean(axis=0)
        else:
            y_before = data_before[pick_idx]
            y_after = data_after[pick_idx]

        ax.plot(
            time_axis,
            y_before,
            label=before_label,
            color=COLORS["before"],
            alpha=0.7,
        )
        ax.plot(
            time_axis,
            y_after,
            label=after_label,
            color=COLORS["after"],
            alpha=0.7,
        )
        ax.set_ylabel(label)
        style_axes(ax, grid=True)
        if row_idx == 0:
            themed_legend(ax, loc="best")

    axes[-1].set_xlabel(x_label)
    return _finalize_fig(fig, show=show, fname=fname)


def plot_power_ratio_map(
    inst_before,
    inst_after,
    info,
    vlim=(None, None),
    cmap=SEQUENTIAL_CMAP,
    colorbar_label="Power Ratio (After / Before)",
    title="Power Ratio Map",
    show=True,
    ax=None,
    fname=None,
):
    """Plot a topomap of preserved power ratio after denoising.

    Parameters
    ----------
    inst_before, inst_after : MNE object | ndarray
        Inputs used to estimate per-channel variance. Accepted forms:
        1D channel variances, 2D channel-by-time arrays, 3D epoch arrays,
        or MNE Raw/Epochs/Evoked objects.
    info : mne.Info
        Sensor metadata used by ``mne.viz.plot_topomap``.
    vlim : tuple[float | None, float | None]
        Lower and upper limits passed to ``mne.viz.plot_topomap``.
    cmap : str | matplotlib.colors.Colormap
        Colormap passed to ``mne.viz.plot_topomap``.
    colorbar_label : str
        Colorbar label.
    title : str
        Panel title.
    show : bool
        If True, display the figure.
    ax : matplotlib.axes.Axes | None
        Target axes. If None, create a new figure and axes.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If ``info`` is missing or if channel counts do not match ``info``.

    Notes
    -----
    Ratio values are computed as ``var_after / var_before`` channel-wise.

    Examples
    --------
    >>> from mne_denoise.viz import plot_power_ratio_map
    >>> fig = plot_power_ratio_map(
    ...     before_array,
    ...     after_array,
    ...     info=info,
    ...     show=False,
    ... )
    """
    if info is None:
        raise ValueError("info must be provided explicitly.")

    var_before = _as_channel_variance(inst_before)
    var_after = _as_channel_variance(inst_after)
    if var_before.shape != var_after.shape:
        raise ValueError("inst_before and inst_after must provide matching channels.")
    if var_before.shape[0] != len(info["ch_names"]):
        raise ValueError("Variance length must match info channel count.")

    ratio = np.divide(
        var_after,
        var_before,
        out=np.full_like(var_after, np.nan, dtype=float),
        where=var_before != 0,
    )

    if ax is None:
        fig, ax = themed_figure(figsize=(5, 5))
    else:
        fig = ax.figure

    im, _ = mne.viz.plot_topomap(
        ratio,
        info,
        axes=ax,
        show=False,
        vlim=vlim,
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_signal_overlay(
    inst_before,
    inst_after,
    times,
    pick=None,
    start=None,
    stop=None,
    scale_after=True,
    before_label="Before",
    after_label="After",
    x_label="Time",
    y_label="Amplitude",
    title=None,
    show=True,
    fname=None,
):
    """Overlay one before/after trace to inspect reconstruction quality.

    Parameters
    ----------
    inst_before, inst_after : MNE object | ndarray
        Inputs to compare. Accepted array signal shapes are
        ``(n_times,)``, ``(n_channels, n_times)``, and
        ``(n_epochs, n_channels, n_times)``.
    times : array-like of shape (n_times,)
        Explicit time axis for both traces after length alignment.
    pick : int | str | None
        Channel to display. Required when input has more than one channel.
        String picks require MNE channel names.
    start, stop : float | None
        Optional lower/upper bounds applied on the time axis.
    scale_after : bool
        If True, scale the after-trace to the before-trace standard deviation.
    before_label : str
        Legend label for the first input.
    after_label : str
        Legend label for the second input.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    title : str | None
        Optional custom title.
    show : bool
        If True, display the figure.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If input shapes are invalid, multi-channel data is used without
        ``pick``, or ``times`` length is inconsistent.

    Notes
    -----
    If traces have different lengths, both are trimmed to the common prefix
    before any time-window filtering.

    Examples
    --------
    >>> from mne_denoise.viz import plot_signal_overlay
    >>> fig = plot_signal_overlay(
    ...     before_array,
    ...     after_array,
    ...     pick=0,
    ...     times=np.arange(1000) / 250.0,
    ...     show=False,
    ... )
    """
    data_before = _extract_overlay_trace(inst_before, pick=pick)
    data_after = _extract_overlay_trace(inst_after, pick=pick)

    n_samples = min(data_before.size, data_after.size)
    data_before = data_before[:n_samples]
    data_after = data_after[:n_samples]
    time_axis = np.asarray(times, dtype=float)
    if time_axis.ndim != 1 or time_axis.size != n_samples:
        raise ValueError("times must be a 1D array with length equal to n_times.")

    mask = np.ones(n_samples, dtype=bool)
    if start is not None:
        mask &= time_axis >= start
    if stop is not None:
        mask &= time_axis <= stop

    time_axis = time_axis[mask]
    data_before = data_before[mask]
    data_after = data_after[mask]

    if scale_after:
        scaler = np.std(data_before) / (np.std(data_after) + 1e-9)
        data_after = data_after * scaler

    fig, ax = themed_figure(figsize=(12, 4))
    ax.plot(
        time_axis,
        data_before,
        color=COLORS["before"],
        label=before_label,
        alpha=0.5,
        linewidth=1,
    )
    ax.plot(
        time_axis,
        data_after,
        color=COLORS["after"],
        linestyle="--",
        label=after_label,
        alpha=0.85,
        linewidth=1.2,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or "Signal Overlay Comparison")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_grand_average_evokeds(
    all_evokeds,
    channels,
    time_windows=None,
    suptitle=None,
    group_order=None,
    group_colors=None,
    group_labels=None,
    amplitude_scale=1.0,
    y_label="Amplitude",
    x_label="Time",
    time_window_colors=None,
    time_window_alpha=0.06,
    panel_title_template="Grand Average at {channel}",
    figsize=None,
    fname=None,
    show=True,
):
    """Plot group-mean evoked responses with optional SEM bands.

    Parameters
    ----------
    all_evokeds : mapping[str, sequence[mne.Evoked]]
        Mapping from group key to subject-level evoked list.
    channels : sequence[str]
        Channel names to plot.
    time_windows : mapping[str, tuple[float, float]] | None
        Optional named windows to shade on each axis.
    suptitle : str | None
        Optional figure-level title.
    group_order : sequence[str] | None
        Explicit plotting order. If None, first-seen mapping order is used.
    group_colors : mapping[str, str] | None
        Optional colors by group key.
    group_labels : mapping[str, str] | None
        Optional display labels by group key.
    amplitude_scale : float
        Multiplicative factor applied to evoked amplitudes before plotting.
    y_label : str
        Y-axis label used for all panels.
    x_label : str
        X-axis label used for all panels.
    time_window_colors : mapping[str, str] | None
        Optional colors for named ``time_windows`` entries.
    time_window_alpha : float
        Alpha used for shaded ``time_windows``.
    panel_title_template : str
        Format string for panel titles. Must support ``{channel}``.
    figsize : tuple[float, float] | None
        Figure size in inches.
    fname : path-like | None
        Optional output path used to save the figure.
    show : bool
        If True, display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If required groups/channels are missing or lists are empty.

    Notes
    -----
    This function is MNE-evoked oriented and expects ``mne.Evoked`` inputs.

    Examples
    --------
    >>> from mne_denoise.viz import plot_grand_average_evokeds
    >>> fig = plot_grand_average_evokeds(
    ...     all_evokeds,
    ...     channels=("Cz", "Pz"),
    ...     amplitude_scale=1.0,
    ...     y_label="Amplitude",
    ...     show=False,
    ... )
    """
    if group_order is None:
        group_order = list(all_evokeds.keys())
    else:
        group_order = list(group_order)

    channels = tuple(channels)
    if len(channels) == 0:
        raise ValueError("channels cannot be empty.")

    n_channels = len(channels)
    if figsize is None:
        figsize = (6 * n_channels, 4.5)

    fig, axes = themed_figure(1, n_channels, figsize=figsize)
    axes = np.atleast_1d(axes)

    for col_idx, ch_name in enumerate(channels):
        ax = axes.flat[col_idx]
        for group_idx, group in enumerate(group_order):
            if group not in all_evokeds:
                raise ValueError(f"Group '{group}' was not found in all_evokeds.")
            evoked_list = list(all_evokeds[group])
            if len(evoked_list) == 0:
                raise ValueError(f"Group '{group}' has no evoked entries.")

            if ch_name not in evoked_list[0].ch_names:
                raise ValueError(
                    f"Channel '{ch_name}' was not found in group '{group}'."
                )
            ch_idx = evoked_list[0].ch_names.index(ch_name)
            time_axis = np.asarray(evoked_list[0].times, dtype=float)
            stacked = np.array(
                [ev.data[ch_idx] * amplitude_scale for ev in evoked_list],
                dtype=float,
            )

            grand_mean = stacked.mean(axis=0)
            n_sub = stacked.shape[0]
            grand_sem = (
                stacked.std(axis=0, ddof=1) / np.sqrt(n_sub)
                if n_sub > 1
                else np.zeros_like(grand_mean)
            )

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

            ax.plot(
                time_axis,
                grand_mean,
                color=color,
                lw=1.8,
                alpha=0.85,
                label=label,
            )
            if n_sub > 1:
                ax.fill_between(
                    time_axis,
                    grand_mean - grand_sem,
                    grand_mean + grand_sem,
                    color=color,
                    alpha=0.15,
                    lw=0,
                )

        ax.axvline(0, color=COLORS["gray"], ls="--", alpha=0.5)
        ax.axhline(0, color=COLORS["gray"], alpha=0.3)
        if time_windows:
            for window_name, (t0, t1) in time_windows.items():
                color = (
                    time_window_colors.get(window_name, COLORS["gray"])
                    if time_window_colors is not None
                    else COLORS["gray"]
                )
                ax.axvspan(t0, t1, alpha=time_window_alpha, color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label, fontsize=FONTS["label"])
        ax.set_title(
            panel_title_template.format(channel=ch_name),
            fontsize=FONTS["title"],
        )
        themed_legend(ax)
        style_axes(ax)

    title = suptitle or "Grand-Average Evoked ± SEM"
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)
