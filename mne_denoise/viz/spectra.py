"""Spectral and time-frequency visualization primitives.

This module contains reusable, method-agnostic plots focused on
frequency-domain and time-frequency diagnostics.

This module contains:
1. PSD comparisons for before/after denoising outputs.
2. Component-spectrum comparisons for extracted sources.
3. Spectrogram and time-frequency mask visualizations.
4. Narrowband scan summaries for spectral sweeps.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import mne
import numpy as np
from scipy import signal

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


def _compute_array_psd(data, sfreq, fmin, fmax):
    """Compute PSDs for array-like inputs using Welch's method."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim > 2:
        data = data.reshape(-1, data.shape[-1])

    nperseg = min(data.shape[-1], int(sfreq * 2))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    keep = (freqs >= fmin) & (freqs <= fmax)
    return freqs[keep], psd[..., keep]


def _compute_psd_matrix(inst, sfreq, fmin, fmax):
    """Return PSD matrix with shape ``(n_series, n_freqs)``."""
    if isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
        spectrum = inst.compute_psd(fmin=fmin, fmax=fmax)
        freqs = np.asarray(spectrum.freqs, dtype=float)
        psd = np.asarray(spectrum.get_data(return_freqs=False), dtype=float)
    else:
        if sfreq is None:
            raise ValueError("sfreq must be provided when plotting PSDs from arrays.")
        freqs, psd = _compute_array_psd(inst, sfreq=sfreq, fmin=fmin, fmax=fmax)
        freqs = np.asarray(freqs, dtype=float)
        psd = np.asarray(psd, dtype=float)

    return freqs, psd.reshape(-1, psd.shape[-1])


def _as_component_data(components):
    """Normalize component inputs to canonical 2D shape ``(n_components, n_times)``."""
    if isinstance(components, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
        data = np.asarray(components.get_data(), dtype=float)
    else:
        data = np.asarray(components, dtype=float)

    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data.mean(axis=0)
    raise ValueError(
        "components must be 1D, 2D, or 3D data, or an MNE Raw/Epochs/Evoked object."
    )


def _compute_array_spectrogram(data, picks, sfreq, fmin, fmax, n_freqs):
    """Compute a mean channel spectrogram for canonical array inputs."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        selected = data[picks, :]
    elif data.ndim == 3:
        selected = data[:, picks, :].reshape(-1, data.shape[-1])
    else:
        raise ValueError(
            "Array spectrogram inputs must be 2D (n_channels, n_times) "
            "or 3D (n_epochs, n_channels, n_times)."
        )

    nperseg = min(selected.shape[-1], int(sfreq * 2))
    noverlap = max(0, nperseg // 2)
    freqs, _, spec = signal.spectrogram(
        selected, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1
    )
    spec = np.asarray(spec, dtype=float)
    spec = spec.mean(axis=0)

    keep = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[keep]
    spec = spec[keep]
    if freqs.size < 2:
        raise ValueError("Could not compute a valid frequency grid in [fmin, fmax].")

    target_freqs = np.linspace(fmin, fmax, n_freqs, dtype=float)
    interp_spec = np.empty((target_freqs.size, spec.shape[1]), dtype=float)
    for time_idx in range(spec.shape[1]):
        interp_spec[:, time_idx] = np.interp(
            target_freqs,
            freqs,
            spec[:, time_idx],
        )
    return target_freqs, interp_spec


def _add_colorbar(fig, ax, image, label):
    """Add a lightly styled colorbar to an axis."""
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(label, fontsize=FONTS["label"])
    colorbar.ax.tick_params(labelsize=FONTS["tick"])
    colorbar.outline.set_edgecolor(COLORS["edge"])
    colorbar.outline.set_linewidth(0.5)
    return colorbar


def _add_line_markers(ax, line_freq, fmax):
    """Add line-frequency markers and visible harmonics to an axis."""
    if line_freq is None:
        return

    ax.axvline(
        line_freq,
        color=COLORS["line_marker"],
        linestyle="--",
        alpha=0.7,
        label=f"{line_freq:g} Hz",
    )

    harmonic = 2
    while line_freq * harmonic <= fmax:
        ax.axvline(
            line_freq * harmonic,
            color=COLORS["line_marker"],
            linestyle="--",
            alpha=0.3,
        )
        harmonic += 1


def plot_narrowband_score_scan(
    frequencies,
    eigenvalues,
    peak_freq=None,
    true_freqs=None,
    ax=None,
    show=True,
    fname=None,
):
    """Plot score/eigenvalue profiles from a narrowband scan.

    Parameters
    ----------
    frequencies : array-like of shape (n_freqs,)
        Frequency grid used in the scan.
    eigenvalues : array-like of shape (n_freqs,) | (n_freqs, n_components)
        Scan scores. For 2D inputs, the first column is treated as dominant.
    peak_freq : float | None
        Optional frequency to highlight with a marker and vertical line.
    true_freqs : sequence of float | None
        Optional reference frequencies to mark.
    ax : matplotlib.axes.Axes | None
        Target axes. If None, create a new figure and axes.
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
        If ``frequencies`` is not 1D, if ``eigenvalues`` is not 1D/2D,
        or if their first dimensions do not match.

    Notes
    -----
    This function is plotting-only and does not run frequency estimation.
    ``peak_freq`` and ``true_freqs`` are optional annotations supplied
    directly by the caller.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_narrowband_score_scan
    >>> freqs = np.linspace(6, 40, 50)
    >>> scores = np.exp(-0.5 * ((freqs - 12.0) / 1.5) ** 2)
    >>> fig = plot_narrowband_score_scan(
    ...     freqs, scores, peak_freq=12.0, true_freqs=[12.0, 24.0], show=False
    ... )
    """
    frequencies = np.asarray(frequencies, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if frequencies.ndim != 1:
        raise ValueError("frequencies must be a 1D array.")
    if eigenvalues.ndim not in (1, 2):
        raise ValueError("eigenvalues must be a 1D or 2D array.")
    if eigenvalues.shape[0] != frequencies.shape[0]:
        raise ValueError(
            "frequencies and eigenvalues must have matching first dimensions."
        )

    if ax is None:
        fig, ax = themed_figure(figsize=(10, 4))
    else:
        fig = ax.figure

    dominant = eigenvalues[:, 0] if eigenvalues.ndim == 2 else eigenvalues
    if eigenvalues.ndim == 2 and eigenvalues.shape[1] > 1:
        ax.plot(
            frequencies,
            eigenvalues[:, 1:],
            color=COLORS["muted"],
            linestyle="-",
            alpha=0.6,
            linewidth=1.2,
        )

    ax.plot(
        frequencies,
        dominant,
        color=COLORS["primary"],
        marker="o",
        linestyle="-",
        markersize=4,
        linewidth=1.8,
        label="Dominant component",
    )

    if peak_freq is not None:
        peak_idx = np.argmin(np.abs(frequencies - peak_freq))
        ax.plot(
            peak_freq,
            dominant[peak_idx],
            color=COLORS["accent"],
            marker="*",
            linestyle="none",
            markersize=14,
            label=f"Peak: {peak_freq:.1f} Hz",
        )
        ax.axvline(peak_freq, color=COLORS["accent"], linestyle="--", alpha=0.5)

    if true_freqs is not None:
        palette = [
            COLORS["accent"],
            COLORS["success"],
            COLORS["secondary"],
            COLORS["purple"],
        ]
        for idx, freq in enumerate(true_freqs):
            ax.axvline(
                freq,
                color=palette[idx % len(palette)],
                linestyle="--",
                alpha=0.5,
                label=f"True: {freq:g} Hz",
            )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Score / Eigenvalue", fontsize=FONTS["label"])
    ax.set_title("Narrowband Score Scan", fontsize=FONTS["title"])
    style_axes(ax, grid=True)

    if peak_freq is not None or true_freqs is not None:
        themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname, tight=False)


def plot_psd_comparison(
    inst_before,
    inst_after,
    fmin=0,
    fmax=np.inf,
    sfreq=None,
    line_freq=None,
    show=True,
    average=True,
    ax=None,
    fname=None,
):
    """Plot PSD comparison for original and denoised data.

    Parameters
    ----------
    inst_before, inst_after : MNE object | ndarray
        Inputs to compare. Supported MNE inputs are Raw, Epochs, and Evoked.
        Array inputs are interpreted with the last axis as time. When either
        input is an array, ``sfreq`` must be provided.
    fmin, fmax : float
        Frequency bounds to display.
    sfreq : float | None
        Sampling frequency for array inputs.
    line_freq : float | None
        Optional line frequency to mark, along with visible harmonics.
    show : bool
        Whether to display the figure.
    average : bool
        If True, average PSDs across non-frequency axes.
    ax : Axes | None
        Optional axis to draw into.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If array inputs are used without ``sfreq``.

    Notes
    -----
    PSD backend is selected by input type:
    - MNE inputs use ``compute_psd``.
    - Array inputs use SciPy Welch PSD.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_psd_comparison
    >>> before = np.random.randn(8, 2000)
    >>> after = before * 0.8
    >>> fig = plot_psd_comparison(before, after, sfreq=250.0, show=False)
    """
    if ax is None:
        fig, ax = themed_figure(figsize=(8, 4))
    else:
        fig = ax.figure

    for inst, label, color in [
        (inst_before, "Before", COLORS["before"]),
        (inst_after, "After", COLORS["after"]),
    ]:
        freqs, psd = _compute_psd_matrix(inst, sfreq=sfreq, fmin=fmin, fmax=fmax)
        if average:
            axis = tuple(range(psd.ndim - 1))
            psd_mean = np.mean(psd, axis=axis)
            ax.semilogy(freqs, psd_mean, label=label, color=color)
        else:
            psd = psd.reshape(-1, psd.shape[-1])
            ax.semilogy(freqs, psd.T, color=color, alpha=0.2)
            ax.plot([], [], color=color, label=label)

    display_fmax = float(np.max(freqs)) if np.isinf(fmax) else fmax
    _add_line_markers(ax, line_freq=line_freq, fmax=display_fmax)

    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    ax.set_title("PSD Comparison", fontsize=FONTS["title"])
    ax.set_xlim(left=max(0.0, fmin), right=display_fmax)
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname, tight=False)


def plot_psd_zoom_comparison(
    freqs_before,
    psd_before,
    freqs_after,
    psd_after,
    series_name="",
    title="",
    zoom_freqs=None,
    zoom_annotations=None,
    fmax=125.0,
    zoom_half_width_hz=8.0,
    series_colors=None,
    series_labels=None,
    fname=None,
    show=True,
):
    """Plot a PSD comparison plus zoomed panels around selected frequencies.

    Parameters
    ----------
    freqs_before, freqs_after : array-like of shape (n_freqs,)
        Frequency vectors for the before/after PSD curves.
    psd_before, psd_after : array-like of shape (n_freqs,)
        PSD vectors aligned with ``freqs_before`` and ``freqs_after``.
    series_name : str
        Series key used for optional color/label mapping.
    title : str
        Optional figure suptitle.
    zoom_freqs : array-like of shape (n_zoom,)
        Frequency centers for zoom panels.
    zoom_annotations : sequence[str] | None
        Optional annotation text per zoom panel.
    fmax : float
        Max frequency on the full-spectrum panel.
    zoom_half_width_hz : float
        Half-width (Hz) around each ``zoom_freq``.
    series_colors : mapping[str, str] | None
        Optional color overrides by series name.
    series_labels : mapping[str, str] | None
        Optional display label overrides by series name.
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
        If ``zoom_freqs`` is empty/non-1D or if ``zoom_half_width_hz <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_psd_zoom_comparison
    >>> freqs = np.linspace(0, 120, 512)
    >>> before = np.exp(-freqs / 40)
    >>> after = before * 0.7
    >>> fig = plot_psd_zoom_comparison(
    ...     freqs, before, freqs, after, zoom_freqs=[50.0], show=False
    ... )
    """
    freqs_before = np.asarray(freqs_before, dtype=float)
    psd_before = np.asarray(psd_before, dtype=float)
    freqs_after = np.asarray(freqs_after, dtype=float)
    psd_after = np.asarray(psd_after, dtype=float)
    zoom_freqs = np.asarray(zoom_freqs, dtype=float)
    if zoom_freqs.ndim != 1 or zoom_freqs.size == 0:
        raise ValueError("zoom_freqs must be a non-empty 1D array-like.")
    zoom_half_width_hz = float(zoom_half_width_hz)
    if zoom_half_width_hz <= 0:
        raise ValueError("zoom_half_width_hz must be positive.")

    n_zoom = len(zoom_freqs)
    fig, axes = themed_figure(1, 1 + n_zoom, figsize=(4 * (1 + n_zoom), 4))
    axes = np.atleast_1d(axes)

    if series_colors and series_name in series_colors:
        series_color = series_colors[series_name]
    else:
        series_color = COLORS["after"]

    if series_name:
        if series_labels and series_name in series_labels:
            series_label = series_labels[series_name]
        else:
            series_label = series_name
    else:
        series_label = "After"

    ax = axes[0]
    ax.semilogy(
        freqs_before,
        psd_before,
        color=COLORS["before"],
        alpha=0.5,
        lw=1,
        label="Before",
    )
    ax.semilogy(
        freqs_after,
        psd_after,
        color=series_color,
        lw=1.5,
        label=series_label,
    )
    for freq in zoom_freqs:
        ax.axvline(freq, color=COLORS["line_marker"], ls="--", alpha=0.2)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    ax.set_title("PSD Comparison", fontsize=FONTS["title"])
    ax.set_xlim(0, fmax)
    themed_legend(ax)
    style_axes(ax, grid=True)

    for idx, freq in enumerate(zoom_freqs):
        ax = axes[1 + idx]
        zoom = zoom_half_width_hz
        before_mask = (freqs_before >= freq - zoom) & (freqs_before <= freq + zoom)
        after_mask = (freqs_after >= freq - zoom) & (freqs_after <= freq + zoom)
        ax.semilogy(
            freqs_before[before_mask],
            psd_before[before_mask],
            color=COLORS["before"],
            alpha=0.5,
            lw=1,
        )
        ax.semilogy(
            freqs_after[after_mask],
            psd_after[after_mask],
            color=series_color,
            lw=1.5,
        )
        ax.axvline(freq, color=COLORS["line_marker"], ls="--", alpha=0.4)

        panel_title = f"{freq:.0f} Hz"
        if zoom_annotations is not None and idx < len(zoom_annotations):
            panel_title += f"\n{zoom_annotations[idx]}"
        ax.set_title(panel_title, fontsize=FONTS["tick"])
        ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
        if idx == 0:
            ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
        style_axes(ax, grid=True)

    if title:
        fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_psd_gallery(
    freqs_reference,
    psd_reference,
    series_psds,
    zoom_freqs,
    fmax=125.0,
    zoom_half_width_hz=8.0,
    title="",
    series_order=None,
    series_colors=None,
    series_labels=None,
    fname=None,
    show=True,
):
    """Plot full-spectrum and zoomed PSD panels across multiple series.

    Parameters
    ----------
    freqs_reference : array-like of shape (n_freqs,)
        Frequency vector for the reference PSD.
    psd_reference : array-like of shape (n_freqs,)
        Reference PSD values.
    series_psds : mapping[str, tuple[array-like, array-like]]
        Mapping from series name to ``(freqs, psd)`` arrays.
    zoom_freqs : array-like of shape (n_zoom,)
        Frequency centers for zoom panels.
    fmax : float
        Max frequency on the full-spectrum panels.
    zoom_half_width_hz : float
        Half-width (Hz) around each zoom center.
    title : str
        Optional figure suptitle.
    series_order : sequence[str] | None
        Optional plotting order. Missing names are shown as empty placeholders.
    series_colors : mapping[str, str] | None
        Optional color overrides by series name.
    series_labels : mapping[str, str] | None
        Optional display label overrides by series name.
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
        If ``zoom_freqs`` is empty/non-1D or if ``zoom_half_width_hz <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_psd_gallery
    >>> freqs = np.linspace(0, 120, 512)
    >>> before = np.exp(-freqs / 40)
    >>> series = {"A": (freqs, before * 0.8), "B": (freqs, before * 0.6)}
    >>> fig = plot_psd_gallery(freqs, before, series, zoom_freqs=[50.0], show=False)
    """
    freqs_reference = np.asarray(freqs_reference, dtype=float)
    psd_reference = np.asarray(psd_reference, dtype=float)
    zoom_freqs = np.asarray(zoom_freqs, dtype=float)
    if zoom_freqs.ndim != 1 or zoom_freqs.size == 0:
        raise ValueError("zoom_freqs must be a non-empty 1D array-like.")
    zoom_half_width_hz = float(zoom_half_width_hz)
    if zoom_half_width_hz <= 0:
        raise ValueError("zoom_half_width_hz must be positive.")
    if series_order is None:
        series_order = list(series_psds.keys())

    n_rows = len(series_order)
    n_cols = 1 + len(zoom_freqs)
    fig, axes = themed_figure(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.asarray(axes, dtype=object)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, series_name in enumerate(series_order):
        if series_name not in series_psds:
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.text(
                    0.5,
                    0.5,
                    f"{series_name}\nno data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=FONTS["label"],
                    color=COLORS["placeholder"],
                )
                ax.axis("off")
            continue

        freqs_series, psd_series = series_psds[series_name]
        freqs_series = np.asarray(freqs_series, dtype=float)
        psd_series = np.asarray(psd_series, dtype=float)
        if series_colors and series_name in series_colors:
            color = series_colors[series_name]
        else:
            color = get_series_color(row_idx)
        if series_labels and series_name in series_labels:
            label = series_labels[series_name]
        else:
            label = series_name

        ax = axes[row_idx, 0]
        ax.semilogy(
            freqs_reference,
            psd_reference,
            color=COLORS["before"],
            alpha=0.4,
            lw=0.8,
            label="Before",
        )
        ax.semilogy(freqs_series, psd_series, color=color, lw=1.2, label=label)
        for freq in zoom_freqs:
            ax.axvline(freq, color=COLORS["line_marker"], ls="--", alpha=0.15)
        ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
        if row_idx == 0:
            ax.set_title("Full PSD", fontsize=FONTS["title"])
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONTS["annotation"],
            color=color,
        )
        ax.set_xlim(0, fmax)
        themed_legend(ax, fontsize=6)
        style_axes(ax, grid=True)

        for col_idx, freq in enumerate(zoom_freqs):
            ax = axes[row_idx, 1 + col_idx]
            zoom = zoom_half_width_hz
            before_mask = (freqs_reference >= freq - zoom) & (
                freqs_reference <= freq + zoom
            )
            series_mask = (freqs_series >= freq - zoom) & (freqs_series <= freq + zoom)
            ax.semilogy(
                freqs_reference[before_mask],
                psd_reference[before_mask],
                color=COLORS["before"],
                alpha=0.4,
                lw=0.8,
            )
            ax.semilogy(
                freqs_series[series_mask],
                psd_series[series_mask],
                color=color,
                lw=1.2,
            )
            ax.axvline(freq, color=COLORS["line_marker"], ls="--", alpha=0.3)
            if row_idx == 0:
                ax.set_title(f"{freq:.0f} Hz", fontsize=FONTS["title"])
            if col_idx == 0:
                ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
            ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
            style_axes(ax, grid=True)

    if title:
        fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold", y=1.01)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_psd_overlay(
    freqs_reference,
    psd_reference,
    series_psds,
    focus_freq,
    fmax=125.0,
    focus_half_width_hz=10.0,
    n_harmonics=3,
    title="",
    series_order=None,
    series_colors=None,
    series_labels=None,
    fname=None,
    show=True,
):
    """Plot full-spectrum and focused PSD overlays across multiple series.

    Parameters
    ----------
    freqs_reference : array-like of shape (n_freqs,)
        Frequency vector for the reference PSD.
    psd_reference : array-like of shape (n_freqs,)
        Reference PSD values.
    series_psds : mapping[str, tuple[array-like, array-like]]
        Mapping from series name to ``(freqs, psd)`` arrays.
    focus_freq : float
        Center frequency for the zoomed overlay panel.
    fmax : float
        Max frequency shown in the full-spectrum panel.
    focus_half_width_hz : float
        Half-width (Hz) used for the zoomed panel around ``focus_freq``.
    n_harmonics : int
        Number of harmonics to mark on the full-spectrum panel.
    title : str
        Optional title for the full-spectrum panel.
    series_order : sequence[str] | None
        Optional plotting order.
    series_colors : mapping[str, str] | None
        Optional color overrides by series name.
    series_labels : mapping[str, str] | None
        Optional label overrides by series name.
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
        If ``focus_half_width_hz <= 0``.

    Notes
    -----
    When ``series_order`` is not provided, overlay order follows the
    insertion order of ``series_psds``.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_psd_overlay
    >>> freqs = np.linspace(0, 120, 512)
    >>> before = np.exp(-freqs / 40)
    >>> series = {"A": (freqs, before * 0.8), "B": (freqs, before * 0.6)}
    >>> fig = plot_psd_overlay(freqs, before, series, focus_freq=50.0, show=False)
    """
    freqs_reference = np.asarray(freqs_reference, dtype=float)
    psd_reference = np.asarray(psd_reference, dtype=float)
    focus_half_width_hz = float(focus_half_width_hz)
    if focus_half_width_hz <= 0:
        raise ValueError("focus_half_width_hz must be positive.")

    if series_order is None:
        series_order = list(series_psds.keys())

    fig, axes = themed_figure(1, 2, figsize=(16, 5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.semilogy(
        freqs_reference,
        psd_reference,
        color=COLORS["before"],
        alpha=0.4,
        lw=1,
        label="Before",
    )
    for idx, series_name in enumerate(series_order):
        if series_name not in series_psds:
            continue
        freqs_series, psd_series = series_psds[series_name]
        if series_colors and series_name in series_colors:
            color = series_colors[series_name]
        else:
            color = get_series_color(idx)
        if series_labels and series_name in series_labels:
            label = series_labels[series_name]
        else:
            label = series_name
        ax.semilogy(
            freqs_series,
            psd_series,
            color=color,
            lw=1.2,
            label=label,
        )
    for harmonic_idx in range(1, n_harmonics + 2):
        harmonic = focus_freq * harmonic_idx
        if harmonic < fmax:
            ax.axvline(harmonic, color=COLORS["line_marker"], ls="--", alpha=0.15)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    ax.set_title(title or "Full Spectrum Comparison", fontsize=FONTS["title"])
    ax.set_xlim(0, fmax)
    themed_legend(ax)
    style_axes(ax, grid=True)

    ax = axes[1]
    zoom = focus_half_width_hz
    before_mask = (freqs_reference >= focus_freq - zoom) & (
        freqs_reference <= focus_freq + zoom
    )
    ax.semilogy(
        freqs_reference[before_mask],
        psd_reference[before_mask],
        color=COLORS["before"],
        alpha=0.4,
        lw=1,
        label="Before",
    )
    for idx, series_name in enumerate(series_order):
        if series_name not in series_psds:
            continue
        freqs_series, psd_series = series_psds[series_name]
        series_mask = (freqs_series >= focus_freq - zoom) & (
            freqs_series <= focus_freq + zoom
        )
        if series_colors and series_name in series_colors:
            color = series_colors[series_name]
        else:
            color = get_series_color(idx)
        if series_labels and series_name in series_labels:
            label = series_labels[series_name]
        else:
            label = series_name
        ax.semilogy(
            freqs_series[series_mask],
            psd_series[series_mask],
            color=color,
            lw=1.5,
            label=label,
        )
    ax.axvline(focus_freq, color=COLORS["line_marker"], ls="--", alpha=0.4)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_title(f"Zoom at {focus_freq:.0f} Hz", fontsize=FONTS["title"])
    themed_legend(ax)
    style_axes(ax, grid=True)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_psd_comparison(
    inst_before,
    components,
    component_indices,
    sfreq=None,
    peak_freq=None,
    fmin=1,
    fmax=40,
    show=True,
    fname=None,
):
    """Plot input PSD next to PSDs of selected components.

    Parameters
    ----------
    inst_before : MNE object | ndarray
        Baseline signal used for the reference PSD.
    components : MNE object | ndarray
        Component signals with canonical shape ``(n_components, n_times)``,
        or ``(n_epochs, n_components, n_times)``.
    component_indices : int | sequence of int
        Explicit component index/indices to include in the component PSD panel.
    sfreq : float | None
        Sampling frequency for array inputs. If ``components`` is an MNE object
        and ``sfreq`` is None, ``components.info['sfreq']`` is used.
    peak_freq : float | None
        Optional frequency marker shown on both panels.
    fmin, fmax : float
        Frequency bounds for PSD computation.
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
        If ``component_indices`` is empty/out of range or if array inputs
        are provided without ``sfreq``.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_component_psd_comparison
    >>> signal = np.random.randn(8, 2000)
    >>> sources = np.random.randn(4, 2000)
    >>> fig = plot_component_psd_comparison(
    ...     signal,
    ...     sources,
    ...     component_indices=[0, 1],
    ...     sfreq=250.0,
    ...     show=False,
    ... )
    """
    fig, axes = themed_figure(1, 2, figsize=(12, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    freqs_before, psd_before = _compute_psd_matrix(
        inst_before, sfreq=sfreq, fmin=fmin, fmax=fmax
    )
    axes[0].semilogy(
        freqs_before,
        psd_before.mean(axis=0),
        color=COLORS["before"],
        label="Before",
    )
    axes[0].set_title("Original Data PSD", fontsize=FONTS["title"])
    axes[0].set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    axes[0].set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    style_axes(axes[0], grid=True)

    component_data = _as_component_data(components)
    if np.isscalar(component_indices):
        indices = [int(component_indices)]
    else:
        indices = [int(idx) for idx in component_indices]
    if len(indices) == 0:
        raise ValueError("component_indices cannot be empty.")
    invalid = [idx for idx in indices if idx < 0 or idx >= component_data.shape[0]]
    if invalid:
        raise ValueError(f"Component indices out of range: {invalid}")

    component_sfreq = sfreq
    if component_sfreq is None and isinstance(
        components, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)
    ):
        component_sfreq = float(components.info["sfreq"])
    if component_sfreq is None:
        raise ValueError("sfreq must be provided when components are arrays.")

    freqs_components, psd_components = _compute_psd_matrix(
        component_data[indices], sfreq=component_sfreq, fmin=fmin, fmax=fmax
    )
    for idx, component_psd in enumerate(psd_components):
        comp_idx = indices[idx]
        axes[1].semilogy(
            freqs_components,
            component_psd,
            color=get_series_color(idx),
            label=f"Component {comp_idx}",
        )

    axes[1].set_title("Component PSD", fontsize=FONTS["title"])
    axes[1].set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    axes[1].set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    style_axes(axes[1], grid=True)

    if peak_freq is not None:
        for axis in axes:
            axis.axvline(
                peak_freq,
                color=COLORS["line_marker"],
                linestyle="--",
                alpha=0.7,
            )

    themed_legend(axes[0], loc="best")
    themed_legend(axes[1], loc="best")

    return _finalize_fig(fig, show=show, fname=fname, tight=False)


def plot_spectrogram_comparison(
    inst_before,
    inst_after,
    picks,
    times,
    sfreq=None,
    fmin=1,
    fmax=40,
    n_freqs=20,
    show=True,
    fname=None,
):
    """Compare before/after spectrograms averaged across selected channels.

    Parameters
    ----------
    inst_before, inst_after : MNE object | ndarray
        Inputs to compare. Either both MNE objects or both arrays.
        Array inputs must be 2D ``(n_channels, n_times)`` or
        3D ``(n_epochs, n_channels, n_times)``.
    picks : sequence of int
        Explicit channel picks used for averaging.
    times : array-like of shape (n_times,)
        Explicit time vector used on x-axis.
    sfreq : float | None
        Sampling frequency for array inputs.
    fmin, fmax : float
        Frequency bounds for the spectrogram.
    n_freqs : int
        Number of frequencies in the display grid.
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
        If ``picks``/``times`` are invalid, if input types are mixed,
        if array inputs are missing ``sfreq``, or if shape constraints fail.

    Notes
    -----
    This function enforces an explicit ``times`` input for both MNE and
    NumPy inputs to avoid hidden axis inference.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_spectrogram_comparison
    >>> before = np.random.randn(8, 2000)
    >>> after = before * 0.8
    >>> t = np.arange(before.shape[-1]) / 250.0
    >>> fig = plot_spectrogram_comparison(
    ...     before, after, picks=[0, 1], times=t, sfreq=250.0, show=False
    ... )
    """
    if n_freqs < 2:
        raise ValueError("n_freqs must be at least 2.")
    if fmax <= fmin:
        raise ValueError("fmax must be greater than fmin.")

    if picks is None:
        raise ValueError("picks must be provided explicitly.")
    picks = list(picks)
    if len(picks) == 0:
        raise ValueError("picks cannot be empty.")

    is_mne_before = isinstance(
        inst_before, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)
    )
    is_mne_after = isinstance(inst_after, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked))
    if is_mne_before != is_mne_after:
        raise ValueError("inst_before and inst_after must be both MNE or both arrays.")

    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array.")

    freqs = np.linspace(fmin, fmax, n_freqs, dtype=float)
    if is_mne_before:
        n_times_before = inst_before.get_data().shape[-1]
        n_times_after = inst_after.get_data().shape[-1]
        if n_times_before != n_times_after:
            raise ValueError("inst_before and inst_after must share the same n_times.")
        if times.size != n_times_before:
            raise ValueError("times must match the signal n_times.")

        tfr_before = inst_before.compute_tfr(
            method="multitaper",
            freqs=freqs,
            n_cycles=freqs / 2.0,
            picks=picks,
        )
        tfr_after = inst_after.compute_tfr(
            method="multitaper",
            freqs=freqs,
            n_cycles=freqs / 2.0,
            picks=picks,
        )
        data_before = np.asarray(tfr_before.data, dtype=float)
        data_after = np.asarray(tfr_after.data, dtype=float)
        mean_axes = tuple(range(data_before.ndim - 2))
        if mean_axes:
            data_before = data_before.mean(axis=mean_axes)
            data_after = data_after.mean(axis=mean_axes)
    else:
        if sfreq is None:
            raise ValueError("sfreq must be provided when inputs are arrays.")
        data_before_arr = np.asarray(inst_before, dtype=float)
        data_after_arr = np.asarray(inst_after, dtype=float)
        if data_before_arr.shape[-1] != data_after_arr.shape[-1]:
            raise ValueError("inst_before and inst_after must share the same n_times.")
        if times.size != data_before_arr.shape[-1]:
            raise ValueError("times must match the signal n_times.")

        if data_before_arr.ndim not in (2, 3) or data_after_arr.ndim not in (2, 3):
            raise ValueError(
                "Array spectrogram inputs must be 2D or 3D with time as last axis."
            )
        n_channels = data_before_arr.shape[-2]
        pick_indices = [int(pick) for pick in picks]
        invalid = [idx for idx in pick_indices if idx < 0 or idx >= n_channels]
        if invalid:
            raise ValueError(f"Channel picks out of range: {invalid}")

        freqs, data_before = _compute_array_spectrogram(
            data_before_arr,
            picks=pick_indices,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_freqs=n_freqs,
        )
        _, data_after = _compute_array_spectrogram(
            data_after_arr,
            picks=pick_indices,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_freqs=n_freqs,
        )

    diff = data_before - data_after

    fig, axes = themed_figure(
        1, 3, figsize=(15, 4), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    vmax = float(max(np.max(data_before), np.max(data_after)))
    diff_limit = float(np.max(np.abs(diff)))

    def _plot_im(ax, data, title, cmap, vlims, colorbar_label):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=vlims[0],
            vmax=vlims[1],
        )
        ax.set_title(title, fontsize=FONTS["title"])
        ax.set_xlabel("Time (s)", fontsize=FONTS["label"])
        ax.set_ylabel("Frequency (Hz)", fontsize=FONTS["label"])
        style_axes(ax, grid=False)
        _add_colorbar(fig, ax, im, colorbar_label)

    _plot_im(
        axes[0],
        data_before,
        "Before",
        cmap=SEQUENTIAL_CMAP,
        vlims=(0.0, vmax),
        colorbar_label="Power",
    )
    _plot_im(
        axes[1],
        data_after,
        "After",
        cmap=SEQUENTIAL_CMAP,
        vlims=(0.0, vmax),
        colorbar_label="Power",
    )
    _plot_im(
        axes[2],
        diff,
        "Before - After",
        cmap=DIVERGING_CMAP,
        vlims=(-diff_limit, diff_limit),
        colorbar_label="Power Difference",
    )

    return _finalize_fig(fig, show=show, fname=fname, tight=False)


def plot_time_frequency_mask(
    mask,
    times,
    freqs,
    title="Time-Frequency Mask",
    ax=None,
    show=True,
    fname=None,
):
    """Visualize a time-frequency mask matrix.

    Parameters
    ----------
    mask : array-like of shape (n_freqs, n_times)
        Time-frequency weights.
    times : array-like of shape (n_times,)
        Time axis coordinates.
    freqs : array-like of shape (n_freqs,)
        Frequency axis coordinates.
    title : str
        Panel title.
    ax : matplotlib.axes.Axes | None
        Target axes. If None, create a new figure and axes.
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
        If dimensions of ``mask``, ``times``, and ``freqs`` are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_time_frequency_mask
    >>> mask = np.random.rand(20, 100)
    >>> times = np.linspace(0, 2.0, 100)
    >>> freqs = np.linspace(1.0, 40.0, 20)
    >>> fig = plot_time_frequency_mask(mask, times, freqs, show=False)
    """
    mask = np.asarray(mask)
    times = np.asarray(times)
    freqs = np.asarray(freqs)

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    if times.ndim != 1 or freqs.ndim != 1:
        raise ValueError("times and freqs must be 1D arrays.")
    if mask.shape != (len(freqs), len(times)):
        raise ValueError("mask shape must match (len(freqs), len(times)).")

    if ax is None:
        fig, ax = themed_figure(figsize=(10, 5))
    else:
        fig = ax.figure

    im = ax.pcolormesh(
        times,
        freqs,
        mask,
        shading="auto",
        cmap=SEQUENTIAL_CMAP,
        vmin=0,
        vmax=1,
    )
    ax.set_ylabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_xlabel("Time (s)", fontsize=FONTS["label"])
    ax.set_title(title, fontsize=FONTS["title"])
    style_axes(ax, grid=False)
    _add_colorbar(fig, ax, im, "Mask Weight")

    return _finalize_fig(fig, show=show, fname=fname)
