"""Component visualization functions."""

from __future__ import annotations

import numpy as np
from matplotlib.gridspec import GridSpec

from .. import _mne
from ._utils import _get_components, _get_info, _get_patterns, _get_scores
from .theme import (
    COLORS,
    DIVERGING_CMAP,
    FONTS,
    SEQUENTIAL_CMAP,
    _finalize_fig,
    get_series_color,
    style_axes,
    themed_figure,
    themed_legend,
)


def _resolve_component_indices(
    n_components,
    n_available,
    default_max,
):
    """Normalize component selection to an explicit list of indices."""
    if n_components is None:
        return list(range(min(default_max, n_available)))

    if isinstance(n_components, int):
        return list(range(min(n_components, n_available)))

    indices = [int(idx) for idx in n_components]
    invalid = [idx for idx in indices if idx < 0 or idx >= n_available]
    if invalid:
        raise ValueError(f"Component indices out of range: {invalid}")
    return indices


def plot_component_score_curve(
    estimator,
    mode="raw",
    ax=None,
    show=True,
    fname=None,
):
    """Plot a 1D component score curve for a fitted estimator.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing ``eigenvalues_`` or ``scores_``.
    mode : {'raw', 'cumulative', 'ratio'}
        Score display mode:

        - ``'raw'``: raw score/eigenvalue per component.
        - ``'cumulative'``: normalized cumulative sum.
        - ``'ratio'``: same values as ``'raw'`` but labeled as a ratio view.
    ax : matplotlib.axes.Axes | None
        Target axes. If None, a new themed figure is created.
    show : bool, default=True
        If True, show the figure.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If ``mode`` is invalid, or if scores are missing/invalid.
    """
    valid_modes = {"raw", "cumulative", "ratio"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")

    scores = _get_scores(estimator)
    if scores is None:
        raise ValueError("Estimator does not expose component scores.")

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Component scores must be a non-empty 1D array.")

    if ax is None:
        fig, ax = themed_figure(figsize=(7, 4))
    else:
        fig = ax.figure

    x = np.arange(1, scores.size + 1)
    if mode == "cumulative":
        y = np.cumsum(scores)
        y = y / y[-1]
        ylabel = "Cumulative Score (Normalized)"
    elif mode == "ratio":
        y = scores
        ylabel = "Power Ratio"
    else:
        y = scores
        ylabel = "Score / Eigenvalue"

    ax.plot(
        x,
        y,
        ".-",
        color=COLORS["primary"],
        linewidth=1.6,
        markersize=5,
        label="Scores",
    )

    if mode != "cumulative":
        mean_score = np.mean(scores)
        ax.axhline(
            mean_score,
            color=COLORS["muted"],
            linestyle="--",
            linewidth=0.9,
            label=f"Mean ({mean_score:.3g})",
        )

        n_selected = getattr(estimator, "n_selected_", None)
        if n_selected is None:
            n_selected = getattr(estimator, "n_removed_", None)
        if n_selected is not None and 0 < n_selected < scores.size:
            ax.axvline(
                n_selected + 0.5,
                color=COLORS["accent"],
                linestyle="--",
                linewidth=1.0,
                label=f"Cutoff ({n_selected})",
            )

        themed_legend(ax, loc="best")

    ax.set_xlabel("Component")
    ax.set_ylabel(ylabel)
    ax.set_title("Component Scores")
    style_axes(ax, grid=True)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_window_score_traces(
    scores,
    threshold=None,
    ax=None,
    show=True,
    fname=None,
):
    """Plot per-window score traces from a 2D score matrix.

    Parameters
    ----------
    scores : array-like of shape (n_windows, n_scores)
        Score matrix to display.
    threshold : float | None
        Optional horizontal threshold line.
    ax : matplotlib.axes.Axes | None
        Target axes. If None, a new themed figure is created.
    show : bool, default=True
        If True, display the figure.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("scores must be a non-empty 2D array.")

    if ax is None:
        fig, ax = themed_figure(figsize=(9, 4))
    else:
        fig = ax.figure

    n_windows, n_scores = scores.shape
    for idx in range(n_scores):
        vals = scores[:, idx]
        valid = np.isfinite(vals)
        if not np.any(valid):
            continue
        ax.plot(
            np.where(valid)[0],
            vals[valid],
            color=get_series_color(idx),
            linewidth=1.2,
            alpha=0.85,
            label=f"Score {idx + 1}",
        )

    if threshold is not None:
        ax.axhline(
            float(threshold),
            color=COLORS["accent"],
            linestyle="--",
            linewidth=1.0,
            label=f"Threshold ({float(threshold):.3g})",
        )

    ax.set_xlabel("Window")
    ax.set_ylabel("Score")
    ax.set_title("Window Score Traces")
    style_axes(ax, grid=True)
    if n_scores <= 10 or threshold is not None:
        themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_patterns(
    estimator,
    info=None,
    picks=None,
    n_components=None,
    ax=None,
    show=True,
    fname=None,
):
    """Plot spatial component patterns.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing ``patterns_``.
    info : mne.Info | None
        Measurement info used for topomap rendering.
    picks : array-like of int | None
        Channel indices used for topomap rendering. If None, no topomap is
        attempted and the function falls back to channel-weight line plots.
    n_components : int | sequence of int | None
        Components to plot. If an int, plot the first ``n_components``.
    ax : matplotlib.axes.Axes | None
        Optional target axes. Supported only for the line-plot fallback or
        when rendering a single topomap.
    show : bool, default=True
        If True, show the figure.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If patterns are not 2D, if no components are selected, or when
        ``ax`` is passed while requesting multiple topomaps. Also raised when
        ``picks`` is provided without valid ``info``.
    """
    patterns = np.asarray(_get_patterns(estimator))
    if patterns.ndim != 2:
        raise ValueError(
            "patterns_ must be a 2D array of shape (n_channels, n_components)."
        )

    indices = _resolve_component_indices(
        n_components,
        patterns.shape[1],
        default_max=6,
    )
    if not indices:
        raise ValueError("No components selected for plotting.")

    if picks is not None and info is None:
        raise ValueError("info is required when picks is provided.")
    if picks is not None:
        _mne.require_mne("component pattern topomap visualization")
        topo_info = _mne.mne.pick_info(info, picks)
        if ax is not None:
            if len(indices) != 1:
                raise ValueError("ax can only be used when plotting a single topomap.")
            fig = ax.figure
            _mne.mne.viz.plot_topomap(
                patterns[picks, indices[0]],
                topo_info,
                axes=ax,
                show=False,
                contours=4,
            )
            ax.set_title(f"Comp {indices[0]}")
            return _finalize_fig(fig, show=show, fname=fname)

        n_show = len(indices)
        n_cols = min(4, n_show)
        n_rows = int(np.ceil(n_show / n_cols))
        fig, axes = themed_figure(
            n_rows,
            n_cols,
            figsize=(3 * n_cols, 3 * n_rows),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for i, (plot_ax, comp_idx) in enumerate(zip(flat_axes, indices)):
            _mne.mne.viz.plot_topomap(
                patterns[picks, comp_idx],
                topo_info,
                axes=plot_ax,
                show=False,
                contours=4,
            )
            plot_ax.set_title(
                f"Comp {comp_idx}",
                fontsize=FONTS["tick"],
                color=get_series_color(i),
            )

        for plot_ax in flat_axes[len(indices) :]:
            plot_ax.axis("off")

        fig.suptitle(
            "Component Patterns", fontsize=FONTS["title"], fontweight="semibold"
        )
        return _finalize_fig(fig, show=show, fname=fname)

    if ax is None:
        fig, ax = themed_figure(figsize=(8, 4.5))
    else:
        fig = ax.figure

    for i, comp_idx in enumerate(indices):
        ax.plot(
            patterns[:, comp_idx],
            marker="o",
            markersize=4,
            linewidth=1.3,
            alpha=0.85,
            color=get_series_color(i),
            label=f"Comp {comp_idx}",
        )

    ax.axhline(0, color=COLORS["muted"], linestyle="-", alpha=0.35)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Pattern Weight")
    ax.set_title("Component Patterns")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_summary(
    estimator,
    data=None,
    info=None,
    picks=None,
    times=None,
    sfreq=None,
    n_components=None,
    psd_fmax=None,
    show=True,
    plot_ci=True,
    fname=None,
):
    """Plot a compact per-component summary dashboard.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing component patterns and a transform/source API.
    data : mne.io.BaseRaw | mne.BaseEpochs | ndarray | None
        Input data used to compute component sources when they are not cached.
    info : mne.Info | None
        Sensor metadata for topomap rendering.
    picks : array-like of int | None
        Channel indices used for topomap rendering. If None, the pattern panel
        uses a text placeholder instead of topomaps.
    times : array-like of shape (n_times,) | None
        Explicit time coordinates for source time-course panels. If None,
        sample indices are used.
    sfreq : float | None
        Sampling frequency used for PSD computation when ``info`` is not
        available. Required if ``info`` cannot be resolved.
    n_components : int | sequence of int | None
        Components to plot. If None, plot up to five components.
    psd_fmax : float | None
        Maximum frequency (Hz) shown in the PSD column. If None, defaults to
        ``min(100, sfreq / 2)`` to preserve previous behavior.
    show : bool, default=True
        If True, show the figure.
    plot_ci : bool, default=True
        If True and sources are epoched, overlay a 95% CI band based on SEM.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If no components are selected, if ``psd_fmax`` is not positive, if
        ``times`` length mismatches source length, or if ``picks`` is provided
        without valid ``info``. Also raised when neither ``info`` nor ``sfreq``
        is provided.
    """
    if picks is not None and info is None:
        raise ValueError("info is required when picks is provided.")
    info = _get_info(estimator, info)
    patterns = np.asarray(_get_patterns(estimator))
    sources = _get_components(estimator, data)

    indices = _resolve_component_indices(
        n_components,
        patterns.shape[1],
        default_max=5,
    )
    if not indices:
        raise ValueError("No components selected for plotting.")

    fig, root_ax = themed_figure(figsize=(12, 3 * len(indices)))
    root_ax.remove()
    gs = GridSpec(len(indices), 3, figure=fig, width_ratios=[1, 2, 1])
    if info is not None:
        sfreq_eff = float(info["sfreq"])
    elif sfreq is not None:
        sfreq_eff = float(sfreq)
    else:
        raise ValueError("sfreq is required when info is not available.")
    if sfreq_eff <= 0:
        raise ValueError("sfreq must be strictly positive.")
    if times is None:
        times_template = np.arange(sources.shape[1])
        time_label = "Time (samples)"
    else:
        times_template = np.asarray(times)
        if times_template.shape[0] != sources.shape[1]:
            raise ValueError("times must have length equal to source n_times.")
        time_label = "Time"
    if psd_fmax is None:
        psd_fmax = min(100.0, sfreq_eff / 2.0)
    psd_fmax = float(psd_fmax)
    if psd_fmax <= 0:
        raise ValueError("psd_fmax must be strictly positive.")
    psd_fmax = min(psd_fmax, sfreq_eff / 2.0)

    for row_idx, comp_idx in enumerate(indices):
        ax_topo = fig.add_subplot(gs[row_idx, 0])
        if picks is not None:
            _mne.require_mne("component pattern topomap visualization")
            topo_info = _mne.mne.pick_info(info, picks)
            # If the estimator was fitted on the exact subset of channels specified by picks,
            # patterns is already the correct size.
            if patterns.shape[0] == len(picks):
                topo_data = patterns[:, comp_idx]
            else:
                topo_data = patterns[picks, comp_idx]
            _mne.mne.viz.plot_topomap(topo_data, topo_info, axes=ax_topo, show=False)
            ax_topo.set_title(f"Comp {comp_idx} Pattern")
        else:
            ax_topo.text(0.5, 0.5, "No topomap info", ha="center", va="center")
            ax_topo.set_axis_off()

        ax_time = fig.add_subplot(gs[row_idx, 1])
        if sources.ndim == 3:
            comp_data = sources[comp_idx]
            mean_tc = comp_data.mean(axis=1)
            ax_time.plot(times_template, mean_tc, label="Mean", color=COLORS["before"])

            if plot_ci:
                std_tc = comp_data.std(axis=1) / np.sqrt(comp_data.shape[1])
                ax_time.fill_between(
                    times_template,
                    mean_tc - 2 * std_tc,
                    mean_tc + 2 * std_tc,
                    color=COLORS["muted"],
                    alpha=0.3,
                    label="95% CI (SEM)",
                )
                themed_legend(ax_time, loc="best")
        else:
            comp_data = sources[comp_idx]
            ax_time.plot(times_template, comp_data, color=COLORS["before"])

        ax_time.set_title(f"Comp {comp_idx} Time Course")
        ax_time.set_xlabel(time_label)
        style_axes(ax_time, grid=True)

        ax_psd = fig.add_subplot(gs[row_idx, 2])
        if sources.ndim == 3:
            d_flat = sources[comp_idx].T
        else:
            d_flat = sources[comp_idx][np.newaxis, :]

        _mne.require_mne("component PSD visualization")
        from mne.time_frequency import psd_array_welch

        psd_spec, freqs = psd_array_welch(
            d_flat,
            sfreq=sfreq_eff,
            fmin=0,
            fmax=psd_fmax,
            n_fft=min(2048, d_flat.shape[-1]),
            verbose=False,
        )
        mean_psd = np.mean(psd_spec, axis=0)
        mean_psd = np.clip(mean_psd, a_min=1e-30, a_max=None)

        ax_psd.semilogy(freqs, mean_psd, color=COLORS["primary"])
        ax_psd.set_title("PSD")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_xlim(0, psd_fmax)
        style_axes(ax_psd, grid=True)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_epochs_image(
    estimator,
    data=None,
    n_components=None,
    show=True,
    fname=None,
):
    """Plot component activity as an epoch-by-time image.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing component sources via cache or transform.
    data : mne.io.BaseRaw | mne.BaseEpochs | ndarray | None
        Input data used to compute sources when they are not cached.
    n_components : int | sequence of int | None
        Components to plot. If None, plot up to five components.
    show : bool, default=True
        If True, show the figure.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If sources are not 2D/3D, or if no components are selected.
    """
    sources = _get_components(estimator, data)
    if sources.ndim == 2:
        sources = sources[:, :, np.newaxis]
    if sources.ndim != 3:
        raise ValueError("Component sources must be 2D or 3D.")

    indices = _resolve_component_indices(
        n_components,
        sources.shape[0],
        default_max=5,
    )
    if not indices:
        raise ValueError("No components selected for plotting.")

    fig, axes = themed_figure(
        len(indices),
        1,
        figsize=(8, 2 * len(indices)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, comp_idx in zip(axes, indices):
        img = sources[comp_idx].T
        ax.imshow(img, aspect="auto", origin="lower", cmap=DIVERGING_CMAP)
        ax.set_title(f"Comp {comp_idx}")
        ax.set_ylabel("Epochs")

    axes[-1].set_xlabel("Time (samples)")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_time_series(
    estimator,
    data=None,
    n_components=None,
    times=None,
    show=True,
    ax=None,
    fname=None,
):
    """Plot stacked component time series with fixed vertical offsets.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing component sources via cache or transform.
    data : mne.io.BaseRaw | mne.BaseEpochs | ndarray | None
        Input data used to compute sources when they are not cached.
    n_components : int | sequence of int | None
        Components to plot. If None, plot up to twenty components.
    times : array-like of shape (n_times,) | None
        Explicit time coordinates. If None, sample indices are used.
    show : bool, default=True
        If True, show the figure.
    ax : matplotlib.axes.Axes | None
        Optional target axes. If None, a new themed figure is created.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If no components are selected or if ``times`` length mismatches source
        length.
    """
    sources = _get_components(estimator, data)
    scores = _get_scores(estimator)

    if sources.ndim == 3:
        sources = sources.mean(axis=2)

    indices = _resolve_component_indices(
        n_components,
        sources.shape[0],
        default_max=20,
    )
    if not indices:
        raise ValueError("No components selected for plotting.")

    if ax is None:
        fig, ax = themed_figure(figsize=(10, max(4.0, len(indices) * 0.5)))
    else:
        fig = ax.figure

    if times is None:
        time_axis = np.arange(sources.shape[1])
        time_label = "Time (samples)"
    else:
        time_axis = np.asarray(times)
        if time_axis.shape[0] != sources.shape[1]:
            raise ValueError("times must have length equal to source n_times.")
        time_label = "Time"
    x_min = float(time_axis[0])
    x_max = float(time_axis[-1])
    x_pad = 0.03 * (x_max - x_min if x_max != x_min else 1.0)
    label_x = x_max + x_pad * 0.25
    offset_step = 3.0

    for row_idx, comp_idx in enumerate(indices):
        comp = sources[comp_idx]
        std = np.std(comp)
        if std < 1e-15:
            std = 1.0
        comp_norm = comp / std
        offset = -row_idx * offset_step
        color = get_series_color(row_idx)

        ax.plot(time_axis, comp_norm + offset, color=color, linewidth=1.5)

        label = f"Comp {comp_idx}"
        if scores is not None and comp_idx < len(scores):
            label += f" (λ={scores[comp_idx]:.2f})"
        ax.text(
            label_x, offset, label, va="center", fontsize=FONTS["tick"], color=color
        )

    ax.set_xlim(x_min, x_max + x_pad)
    ax.set_yticks([])
    ax.set_xlabel(time_label)
    ax.set_title("Component Time Series")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_spectrogram(
    component_data,
    sfreq,
    freqs=None,
    fmax=50.0,
    n_cycles=None,
    title="Component Spectrogram",
    ax=None,
    show=True,
    fname=None,
):
    """Plot a time-frequency power view for one component.

    Parameters
    ----------
    component_data : ndarray, shape (n_times,) or (n_epochs, n_times)
        Single-component time series or repeated epochs of one component.
    sfreq : float
        Sampling frequency.
    freqs : ndarray | None
        Frequencies to compute. If None, frequencies are generated from
        1 Hz to ``fmax`` (capped at Nyquist).
    fmax : float | None
        Upper frequency bound used when ``freqs`` is None.
        Defaults to 50 Hz to preserve prior behavior.
    n_cycles : float | ndarray | None
        Number of cycles for multitaper estimation.
    title : str
        Axes title.
    ax : matplotlib.axes.Axes | None
        Optional target axes. If None, a new themed figure is created.
    show : bool, default=True
        If True, show the figure.
    fname : path-like | None
        Optional output path used to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If ``component_data`` is not 1D/2D, or if ``fmax`` is not positive
        when ``freqs`` is None.
    """
    _mne.require_mne("component spectrogram visualization")
    component_data = np.asarray(component_data)
    if component_data.ndim == 1:
        data = component_data[np.newaxis, np.newaxis, :]
    elif component_data.ndim == 2:
        data = component_data[:, np.newaxis, :]
    else:
        raise ValueError("component_data must be 1D or 2D.")

    if freqs is None:
        if fmax is None:
            upper = sfreq / 2.0
        else:
            upper = min(float(fmax), sfreq / 2.0)
        if upper <= 0:
            raise ValueError("fmax must be strictly positive when freqs is None.")
        upper = max(2.0, upper)
        freqs = np.arange(1.0, np.floor(upper) + 1.0, 1.0)
    else:
        freqs = np.asarray(freqs, dtype=float)
    if n_cycles is None:
        n_cycles = freqs / 4.0

    from mne.time_frequency import tfr_array_multitaper

    tfr = tfr_array_multitaper(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        verbose=False,
    )
    power = tfr[:, 0].mean(axis=0)
    times = np.arange(power.shape[1]) / sfreq

    if ax is None:
        fig, ax = themed_figure(figsize=(10, 5))
    else:
        fig = ax.figure

    im = ax.pcolormesh(times, freqs, power, shading="gouraud", cmap=SEQUENTIAL_CMAP)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Power")
    style_axes(ax, grid=False)

    return _finalize_fig(fig, show=show, fname=fname)
