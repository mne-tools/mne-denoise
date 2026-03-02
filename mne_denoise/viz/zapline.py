"""ZapLine-specific visualization functions.

Provides reusable plotting utilities for ZapLine analysis and results.

Functions
---------
plot_psd_comparison
    Compare power spectral density before and after cleaning.
plot_component_scores
    Visualize DSS component eigenvalues with removal threshold.
plot_spatial_patterns
    Display spatial patterns of noise components.
plot_cleaning_summary
    Combined multi-panel summary figure.
plot_zapline_analytics
    Legacy analytics dashboard.
plot_adaptive_summary
    Comprehensive adaptive ZapLine-plus diagnostics dashboard.

Authors Sina Esmaeili (sina.esmaeili@umontreal.ca)
        Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ._theme import COLORS, FONTS, pub_legend, style_axes

if TYPE_CHECKING:
    pass


def plot_psd_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    sfreq: float,
    line_freq: float | None = None,
    fmax: float = 100.0,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """Compare power spectral density before and after cleaning.

    Parameters
    ----------
    data_before : ndarray, shape (n_channels, n_times)
        Original data before cleaning.
    data_after : ndarray, shape (n_channels, n_times)
        Cleaned data after ZapLine.
    sfreq : float
        Sampling frequency in Hz.
    line_freq : float | None
        Line noise frequency to mark. If None, no vertical line is drawn.
    fmax : float
        Maximum frequency to display.
    ax : Axes | None
        Matplotlib axes. If None, creates new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.

    Examples
    --------
    >>> from mne_denoise.viz import plot_psd_comparison
    >>> plot_psd_comparison(data, cleaned, sfreq=1000, line_freq=50)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    nperseg = min(data_before.shape[1], int(sfreq * 2))
    freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
    _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)

    ax.semilogy(freqs, np.mean(psd_before, axis=0), "b-", alpha=0.5, label="Before")
    ax.semilogy(freqs, np.mean(psd_after, axis=0), "g-", label="After")

    if line_freq is not None:
        ax.axvline(
            line_freq, color="r", linestyle="--", alpha=0.7, label=f"{line_freq} Hz"
        )
        # Mark harmonics if within range
        for h in range(2, 5):
            if line_freq * h < fmax:
                ax.axvline(line_freq * h, color="r", linestyle="--", alpha=0.3)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density: Before vs After")
    ax.set_xlim(0, fmax)
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_component_scores(
    estimator,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """Visualize DSS component eigenvalues with removal threshold.

    Parameters
    ----------
    estimator : ZapLine
        Fitted ZapLine estimator with ``eigenvalues_`` attribute.
    ax : Axes | None
        Matplotlib axes. If None, creates new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.

    Examples
    --------
    >>> from mne_denoise.viz import plot_component_scores
    >>> zapline = ZapLine(sfreq=1000, line_freq=50).fit(data)
    >>> plot_component_scores(zapline)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    scores = getattr(estimator, "eigenvalues_", None)
    if scores is None or len(scores) == 0:
        ax.text(
            0.5,
            0.5,
            "No scores available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return ax

    ax.bar(range(len(scores)), scores, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axhline(
        np.mean(scores), color="red", linestyle="--", linewidth=1.5, label="Mean"
    )

    n_removed = getattr(estimator, "n_removed_", 0)
    if n_removed > 0:
        ax.axvline(
            n_removed - 0.5,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Removed: {n_removed}",
        )

    ax.set_xlabel("Component")
    ax.set_ylabel("Score (eigenvalue)")
    ax.set_title("Component Scores")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_spatial_patterns(
    estimator,
    n_patterns: int = 3,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """Display spatial patterns of noise components.

    Parameters
    ----------
    estimator : ZapLine
        Fitted ZapLine estimator with ``patterns_`` attribute.
    n_patterns : int
        Number of top patterns to display.
    ax : Axes | None
        Matplotlib axes. If None, creates new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.

    Examples
    --------
    >>> from mne_denoise.viz import plot_spatial_patterns
    >>> zapline = ZapLine(sfreq=1000, line_freq=50).fit(data)
    >>> plot_spatial_patterns(zapline, n_patterns=3)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    patterns = getattr(estimator, "patterns_", None)
    if patterns is None or patterns.size == 0:
        ax.text(
            0.5,
            0.5,
            "No patterns available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return ax

    n_show = min(n_patterns, patterns.shape[1])
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))

    for i in range(n_show):
        ax.plot(
            patterns[:, i],
            label=f"Component {i}",
            marker="o",
            markersize=4,
            alpha=0.8,
            color=colors[i],
        )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Pattern weight")
    ax.set_title(f"Spatial Patterns (Top {n_show} Components)")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_cleaning_summary(
    data_before: np.ndarray,
    data_after: np.ndarray,
    estimator,
    sfreq: float,
    line_freq: float | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create a combined multi-panel cleaning summary.

    Parameters
    ----------
    data_before : ndarray, shape (n_channels, n_times)
        Original data before cleaning.
    data_after : ndarray, shape (n_channels, n_times)
        Cleaned data after ZapLine.
    estimator : ZapLine
        Fitted ZapLine estimator.
    sfreq : float
        Sampling frequency in Hz.
    line_freq : float | None
        Line noise frequency.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : Figure
        The matplotlib figure.

    Examples
    --------
    >>> from mne_denoise.viz import plot_cleaning_summary
    >>> zapline = ZapLine(sfreq=1000, line_freq=50)
    >>> cleaned = zapline.fit_transform(data)
    >>> plot_cleaning_summary(data, cleaned, zapline, sfreq=1000, line_freq=50)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # PSD comparison
    plot_psd_comparison(
        data_before, data_after, sfreq, line_freq=line_freq, ax=axes[0, 0], show=False
    )

    # Component scores
    plot_component_scores(estimator, ax=axes[0, 1], show=False)

    # Spatial patterns
    plot_spatial_patterns(estimator, ax=axes[1, 0], show=False)

    # Statistics text
    ax = axes[1, 1]
    ax.axis("off")

    stats = []
    if line_freq is not None:
        stats.append(f"Line Frequency: {line_freq:.1f} Hz")

    n_removed = getattr(estimator, "n_removed_", 0)
    stats.append(f"Components Removed: {n_removed}")

    n_harmonics = getattr(estimator, "n_harmonics_", None)
    if n_harmonics is not None:
        stats.append(f"Harmonics: {n_harmonics}")

    # Compute power reduction at line frequency
    if line_freq is not None:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)
        idx = np.argmin(np.abs(freqs - line_freq))
        power_before = np.mean(psd_before[:, idx])
        power_after = np.mean(psd_after[:, idx])
        if power_after > 0:
            reduction_db = 10 * np.log10(power_before / power_after)
            stats.append(f"Power Reduction: {reduction_db:.1f} dB")

    ax.text(
        0.1,
        0.8,
        "\n".join(stats),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.5},
    )
    ax.set_title("Summary Statistics")

    plt.suptitle("ZapLine Cleaning Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_zapline_analytics(result, sfreq=None, show=True):
    """Plot ZapLine cleaning analytics (legacy function).

    Parameters
    ----------
    result : ZapLine | dict
        Result from ZapLine estimator or result dictionary.
    sfreq : float | None
        Sampling frequency (unused, kept for compatibility).
    show : bool
        Whether to show the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure handle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot 1: Component scores
    ax = axes[0]
    scores = None
    if hasattr(result, "eigenvalues_"):
        scores = result.eigenvalues_
    elif hasattr(result, "dss_eigenvalues"):
        scores = result.dss_eigenvalues
    elif isinstance(result, dict) and "eigenvalues" in result:
        scores = result["eigenvalues"]

    if scores is not None and isinstance(scores, np.ndarray) and scores.size > 0:
        ax.bar(range(len(scores)), scores, color="steelblue")
        ax.axhline(np.mean(scores), color="red", linestyle="--", label="Mean")

        n_rem = _get_n_removed(result)
        is_adaptive = _is_adaptive(result)

        if is_adaptive:
            # For adaptive mode, show per-chunk average instead of misleading sum
            chunk_info = _get_chunk_info(result)
            if chunk_info:
                avg_rem = np.mean([c.get("n_removed", 0) for c in chunk_info])
                display_n_rem = min(int(round(avg_rem)), len(scores))
                label = f"Avg/chunk: {avg_rem:.1f} (total: {n_rem})"
            else:
                display_n_rem = min(n_rem, len(scores))
                label = f"Removed: {n_rem}"
            if display_n_rem > 0:
                ax.axvline(
                    display_n_rem - 0.5,
                    color="green",
                    linestyle="--",
                    label=label,
                )
        else:
            if n_rem > 0:
                display_n_rem = min(n_rem, len(scores))
                ax.axvline(
                    display_n_rem - 0.5,
                    color="green",
                    linestyle="--",
                    label=f"Removed: {n_rem}",
                )
        ax.set_xlabel("Component")
        ax.set_ylabel("Score (eigenvalue)")
        ax.set_title("Component Scores")
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No scores available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Plot 2: Removed signal power
    ax = axes[1]
    removed = _get_removed(result)
    if removed is not None and np.any(removed != 0):
        removed_power = np.mean(removed**2, axis=1)
        ax.bar(range(len(removed_power)), removed_power, color="salmon")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Mean Squared Amplitude")
        ax.set_title("Removed Power per Channel")
    else:
        ax.text(
            0.5,
            0.5,
            "No removal data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Plot 3: Summary stats
    ax = axes[2]
    ax.axis("off")
    stats_text = []
    if hasattr(result, "line_freq") and result.line_freq is not None:
        stats_text.append(f"Line Frequency: {result.line_freq:.1f} Hz")
    if hasattr(result, "n_harmonics") and result.n_harmonics is not None:
        stats_text.append(f"Harmonics: {result.n_harmonics}")

    n_rem = _get_n_removed(result)
    is_adaptive = _is_adaptive(result)

    if is_adaptive:
        chunk_info = _get_chunk_info(result)
        n_chunks = len(chunk_info) if chunk_info else 0
        stats_text.append(f"Mode: Adaptive ({n_chunks} chunks)")
        stats_text.append(f"Total Components Removed: {n_rem}")
        if chunk_info:
            per_chunk = [c.get("n_removed", 0) for c in chunk_info]
            stats_text.append(
                f"Per-chunk: min={min(per_chunk)}, "
                f"max={max(per_chunk)}, avg={np.mean(per_chunk):.1f}"
            )
    else:
        stats_text.append(f"Components Removed: {n_rem}")

    removed = _get_removed(result)
    cleaned = _get_cleaned(result)
    if cleaned is not None and removed is not None:
        total_var = np.var(cleaned + removed)
        removed_var = np.var(removed)
        if total_var > 0:
            pct_removed = 100 * removed_var / total_var
            stats_text.append(f"Variance Removed: {pct_removed:.2f}%")

    ax.text(
        0.1,
        0.8,
        "\n".join(stats_text),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax.set_title("Summary")

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_adaptive_summary(
    result,
    data_before=None,
    data_after=None,
    sfreq=None,
    channel_names=None,
    info=None,
    title=None,
    figsize=None,
    dpi=200,
    show=True,
):
    """Comprehensive adaptive ZapLine-plus diagnostics dashboard.

    Creates a publication-quality multi-panel figure showing all available
    adaptive mode results: per-chunk component removal, fine-frequency
    variation, segment timeline, artifact power, PSD comparison, and
    summary statistics.

    Parameters
    ----------
    result : ZapLine
        Fitted ZapLine estimator with ``adaptive=True`` (must have
        ``adaptive_results_`` attribute populated).
    data_before : ndarray, shape (n_channels, n_times) | None
        Original data before cleaning, for PSD comparison. If None, PSD
        panel is skipped.
    data_after : ndarray, shape (n_channels, n_times) | None
        Cleaned data after ZapLine, for PSD comparison. If None, PSD
        panel is skipped.
    sfreq : float | None
        Sampling frequency. If None, extracted from ``result.sfreq``.
    channel_names : list of str | None
        Channel names for axis labels. If None, uses numeric indices.
    info : mne.Info | None
        MNE channel info for topomap rendering.  When provided,
        panel (d) uses ``mne.viz.plot_topomap`` instead of a bar chart.
    title : str | None
        Overall figure title. If None, uses a default.
    figsize : tuple | None
        Figure size in inches (width, height). If None, defaults to
        (11, 8.5) for landscape letter.
    dpi : int
        Figure resolution. Default 200.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : Figure
        The matplotlib figure.

    Examples
    --------
    >>> from mne_denoise.viz import plot_adaptive_summary
    >>> zapline = ZapLine(sfreq=1000, line_freq=50, adaptive=True)
    >>> raw_clean = zapline.fit_transform(raw)
    >>> plot_adaptive_summary(zapline)
    """
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    chunk_info = _get_chunk_info(result)
    if not chunk_info:
        return plot_zapline_analytics(result, sfreq=sfreq, show=show)

    if sfreq is None:
        sfreq = getattr(result, "sfreq", None)

    removed = _get_removed(result)
    n_chunks = len(chunk_info)
    has_psd = data_before is not None and data_after is not None and sfreq is not None
    line_freq = getattr(result, "line_freq", None)

    per_chunk_removed = [c.get("n_removed", 0) for c in chunk_info]
    fine_freqs = [c.get("fine_freq", c.get("frequency", 0)) for c in chunk_info]
    coarse_freqs = [c.get("frequency", 0) for c in chunk_info]
    artifact_present = [c.get("artifact_present", True) for c in chunk_info]

    # Pre-compute PSD early so summary panel can reference it
    freqs = psd_before = psd_after = None
    if has_psd:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)

    # =====================================================================
    # Shared theme
    # =====================================================================
    C = COLORS
    F = FONTS

    if figsize is None:
        from ._theme import DEFAULT_FIGSIZE

        figsize = DEFAULT_FIGSIZE

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.40,
        wspace=0.30,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )

    # =================================================================
    # (a) Components removed per segment
    # =================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    style_axes(ax_a, grid=True)

    bar_colors = [C["primary"] if ap else C["muted"] for ap in artifact_present]
    ax_a.bar(
        range(n_chunks),
        per_chunk_removed,
        color=bar_colors,
        edgecolor="none",
        width=0.75,
    )
    mean_val = np.mean(per_chunk_removed)
    ax_a.axhline(mean_val, color=C["accent"], ls="--", lw=0.9, zorder=4)

    ax_a.set_xlabel("Segment index", fontsize=F["label"])
    ax_a.set_ylabel("Components removed", fontsize=F["label"])
    ax_a.set_title(
        "(a)  Components removed per segment",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )
    ax_a.set_xlim(-0.6, n_chunks - 0.4)
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    pub_legend(
        ax_a,
        loc="upper left",
        handles=[
            Patch(fc=C["primary"], label="Artifact present"),
            Patch(fc=C["muted"], label="No artifact"),
            plt.Line2D(
                [],
                [],
                color=C["accent"],
                ls="--",
                lw=0.9,
                label=f"Mean ({mean_val:.1f})",
            ),
        ],
    )

    # =================================================================
    # (b) Fine frequency variation
    # =================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    style_axes(ax_b, grid=True)

    ax_b.plot(
        range(n_chunks),
        fine_freqs,
        color=C["primary"],
        marker="o",
        markersize=2.2,
        linewidth=0.7,
        label="Fine-tuned",
        zorder=3,
    )
    if any(abs(f - ff) > 1e-6 for f, ff in zip(coarse_freqs, fine_freqs)):
        ax_b.axhline(
            coarse_freqs[0],
            color=C["secondary"],
            ls="--",
            lw=1.0,
            label=f"Nominal ({coarse_freqs[0]:.0f} Hz)",
            zorder=2,
        )

    ax_b.set_xlabel("Segment index", fontsize=F["label"])
    ax_b.set_ylabel("Frequency (Hz)", fontsize=F["label"])
    ax_b.set_title(
        "(b)  Detected peak frequency",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )
    ax_b.set_xlim(-0.6, n_chunks - 0.4)

    freq_range = max(fine_freqs) - min(fine_freqs)
    freq_pad = max(0.02, freq_range * 0.20)
    ax_b.set_ylim(min(fine_freqs) - freq_pad, max(fine_freqs) + freq_pad)
    pub_legend(ax_b, loc="upper left")

    # =================================================================
    # (c) Segment timeline
    # =================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    style_axes(ax_c)

    if sfreq is not None and sfreq > 0:
        total_dur = chunk_info[-1]["end"] / sfreq
        for _i, c in enumerate(chunk_info):
            s_t = c["start"] / sfreq
            e_t = c["end"] / sfreq
            ap = c.get("artifact_present", True)
            col = C["primary"] if ap else C["muted"]
            ax_c.barh(
                0,
                e_t - s_t,
                left=s_t,
                height=0.7,
                color=col,
                edgecolor="white",
                linewidth=0.25,
            )
            # Label only segments wide enough to fit text
            seg_frac = (e_t - s_t) / total_dur
            if seg_frac > 0.018:
                n_r = c.get("n_removed", 0)
                tc = "white" if ap else C["text"]
                ax_c.text(
                    (s_t + e_t) / 2,
                    0,
                    str(n_r),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=tc,
                    fontweight="bold",
                )

        ax_c.set_xlabel("Time (s)", fontsize=F["label"])
        ax_c.set_yticks([])
        ax_c.set_ylim(-0.5, 0.5)
        ax_c.set_title(
            "(c)  Segment boundaries",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )
        pub_legend(
            ax_c,
            loc="upper right",
            handles=[
                Patch(fc=C["primary"], label="Artifact present"),
                Patch(fc=C["muted"], label="No artifact"),
            ],
        )
    else:
        ax_c.text(
            0.5,
            0.5,
            "sfreq required",
            ha="center",
            va="center",
            transform=ax_c.transAxes,
            fontsize=F["label"],
            color="#999999",
        )
        ax_c.set_title(
            "(c)  Segment boundaries",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )

    # =================================================================
    # (d) Removed artifact power  (topomap if info, else bar chart)
    # =================================================================
    _has_removed = removed is not None and np.any(removed != 0)

    if _has_removed and info is not None:
        # --- topomap of removed artifact RMS ----------------------------
        import mne

        removed_rms = np.sqrt(np.mean(removed**2, axis=1))

        # Auto-scale
        if np.max(removed_rms) < 1e-3:
            scale, unit = 1e6, "\u00b5V"
        elif np.max(removed_rms) < 1.0:
            scale, unit = 1e3, "mV"
        else:
            scale, unit = 1.0, "V"

        ax_d = fig.add_subplot(gs[1, 1])
        im, _ = mne.viz.plot_topomap(
            removed_rms,
            info,
            axes=ax_d,
            show=False,
            contours=4,
        )
        cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
        cbar.set_label(f"RMS ({unit})", fontsize=F["tick"])
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _, s=scale: f"{x * s:.1f}")
        )
        cbar.ax.tick_params(labelsize=F["tick"] - 1)
        ax_d.set_title(
            "(d)  Removed artifact power",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )
    else:
        # --- fallback: bar chart ----------------------------------------
        ax_d = fig.add_subplot(gs[1, 1])
        style_axes(ax_d, grid=True)

        if _has_removed:
            removed_rms = np.sqrt(np.mean(removed**2, axis=1))
            n_ch = len(removed_rms)
            x_pos = np.arange(n_ch)

            scale, unit = 1.0, "V"
            if np.max(removed_rms) < 1e-3:
                scale, unit = 1e6, "\u00b5V"
            elif np.max(removed_rms) < 1.0:
                scale, unit = 1e3, "mV"

            ax_d.bar(
                x_pos,
                removed_rms * scale,
                color=C["primary"],
                edgecolor="none",
                width=0.75,
                alpha=0.85,
            )

            if channel_names is not None and len(channel_names) == n_ch:
                step = max(1, n_ch // 16) if n_ch > 20 else 1
                ax_d.set_xticks(x_pos[::step])
                ax_d.set_xticklabels(
                    [channel_names[i] for i in range(0, n_ch, step)],
                    rotation=45,
                    ha="right",
                    fontsize=F["tick"] - 0.5,
                )
            else:
                step = max(1, n_ch // 12)
                ax_d.set_xticks(x_pos[::step])

            ax_d.set_xlim(-0.6, n_ch - 0.4)
            ax_d.set_ylabel(f"RMS ({unit})", fontsize=F["label"])
        else:
            ax_d.text(
                0.5,
                0.5,
                "No removal data",
                ha="center",
                va="center",
                transform=ax_d.transAxes,
                fontsize=F["label"],
                color="#999999",
            )

        ax_d.set_xlabel("Channel", fontsize=F["label"])
        ax_d.set_title(
            "(d)  Removed artifact power",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )

    # =================================================================
    # (e) PSD before / after
    # =================================================================
    ax_e = fig.add_subplot(gs[2, 0])
    style_axes(ax_e, grid=True)

    if has_psd:
        fmax = 100.0
        mean_before = np.mean(psd_before, axis=0)
        mean_after = np.mean(psd_after, axis=0)

        ax_e.semilogy(
            freqs,
            mean_before,
            color="#333333",
            linewidth=2.0,
            linestyle="-",
            alpha=1.0,
            label="Before",
            zorder=2,
        )
        ax_e.semilogy(
            freqs,
            mean_after,
            color=C["after"],
            linewidth=1.2,
            linestyle="-",
            alpha=0.9,
            label="After",
            zorder=3,
        )

        if line_freq is not None:
            ax_e.axvline(
                line_freq,
                color=C["accent"],
                ls=":",
                lw=0.8,
                alpha=0.8,
                label=f"{line_freq:.0f} Hz",
                zorder=4,
            )
            for h in range(2, 5):
                if line_freq * h < fmax:
                    ax_e.axvline(
                        line_freq * h,
                        color=C["accent"],
                        ls=":",
                        lw=0.5,
                        alpha=0.25,
                    )

        ax_e.set_xlabel("Frequency (Hz)", fontsize=F["label"])
        ax_e.set_ylabel(r"PSD (V$^2\!$/Hz)", fontsize=F["label"])
        ax_e.set_xlim(0, fmax)
        ax_e.set_title(
            "(e)  Power spectral density",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )
        pub_legend(ax_e, loc="upper right")
    else:
        scores = getattr(result, "eigenvalues_", None)
        if scores is not None and isinstance(scores, np.ndarray) and scores.size:
            ax_e.bar(range(len(scores)), scores, color=C["primary"], edgecolor="none")
            ax_e.axhline(np.mean(scores), color=C["accent"], ls="--", lw=0.9)
            ax_e.set_xlabel("Component", fontsize=F["label"])
            ax_e.set_ylabel("Eigenvalue", fontsize=F["label"])
            ax_e.set_title(
                "(e)  DSS eigenvalues (last segment)",
                fontsize=F["title"],
                fontweight="semibold",
                loc="left",
                pad=6,
            )
        else:
            ax_e.text(
                0.5,
                0.5,
                "Provide data_before / data_after\nfor PSD comparison",
                ha="center",
                va="center",
                transform=ax_e.transAxes,
                fontsize=F["label"],
                color="#999999",
            )
            ax_e.set_title(
                "(e)  Power spectral density",
                fontsize=F["title"],
                fontweight="semibold",
                loc="left",
                pad=6,
            )

    # =================================================================
    # (f) Summary statistics
    # =================================================================
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.axis("off")

    total_removed = _get_n_removed(result)
    n_present = sum(1 for ap in artifact_present if ap)
    cleaned = _get_cleaned(result)

    rows = [
        ("Line frequency", f"{line_freq:.1f} Hz" if line_freq else "auto"),
        ("Harmonics", str(getattr(result, "n_harmonics", "None"))),
        ("Number of segments", str(n_chunks)),
        ("Segments with artifact", f"{n_present} / {n_chunks}"),
        ("Total components removed", str(total_removed)),
        (
            "Per-segment min / max",
            f"{min(per_chunk_removed)} / {max(per_chunk_removed)}",
        ),
        (
            "Per-segment mean \u00b1 std",
            f"{np.mean(per_chunk_removed):.1f} \u00b1 {np.std(per_chunk_removed):.1f}",
        ),
        ("Fine freq range", f"{min(fine_freqs):.2f} \u2013 {max(fine_freqs):.2f} Hz"),
    ]

    # Power reduction at line frequency
    if has_psd and line_freq is not None and freqs is not None:
        idx = np.argmin(np.abs(freqs - line_freq))
        pb = np.mean(psd_before[:, idx])
        pa = np.mean(psd_after[:, idx])
        if pa > 0 and pb > 0:
            reduction_db = 10 * np.log10(pb / pa)
            rows.append(("Power reduction at line freq", f"{reduction_db:.1f} dB"))

    # Variance explained by removed signal
    if cleaned is not None and removed is not None and np.any(removed != 0):
        total_var = np.var(cleaned + removed)
        removed_var = np.var(removed)
        if total_var > 0:
            pct = 100 * removed_var / total_var
            rows.append(("Variance removed", f"{pct:.4f} %"))

    # Render as aligned two-column table
    n_rows = len(rows)
    row_h = min(0.085, 0.90 / max(n_rows, 1))
    y_top = 0.94

    for i, (label, value) in enumerate(rows):
        y = y_top - i * row_h
        # Label (left-aligned)
        ax_f.text(
            0.03,
            y,
            label,
            transform=ax_f.transAxes,
            fontsize=F["label"] - 0.5,
            color="#555555",
            va="top",
            fontfamily="sans-serif",
        )
        # Value (right-aligned, bold monospace)
        ax_f.text(
            0.97,
            y,
            value,
            transform=ax_f.transAxes,
            fontsize=F["label"] - 0.5,
            color=C["text"],
            va="top",
            ha="right",
            fontfamily="monospace",
            fontweight="bold",
        )
        # Separator line
        y_line = y - row_h * 0.35
        ax_f.plot(
            [0.03, 0.97],
            [y_line, y_line],
            color="#e0e0e0",
            lw=0.4,
            transform=ax_f.transAxes,
            clip_on=False,
        )

    ax_f.set_title(
        "(f)  Summary",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )

    # =================================================================
    # Suptitle
    # =================================================================
    if title is None:
        title = "Adaptive ZapLine+ Diagnostics"
    fig.suptitle(title, fontsize=F["suptitle"], fontweight="bold", color=C["text"])

    if show:
        plt.show()

    return fig


def plot_zapline_summary(
    result,
    data_before=None,
    data_after=None,
    sfreq=None,
    channel_names=None,
    info=None,
    max_components=4,
    title=None,
    figsize=None,
    dpi=200,
    show=True,
):
    """Publication-quality diagnostics dashboard for standard ZapLine.

    Creates a 3??2 panel figure that summarises every stage of the
    standard (non-adaptive) ZapLine pipeline:

    +-----------------------------------+-----------------------------------+
    | (a) DSS eigenvalues + threshold   | (b) Spatial patterns (topomaps)   |
    +-----------------------------------+-----------------------------------+
    | (c) Removed artifact (topomap)    | (d) Artifact source time series   |
    +-----------------------------------+-----------------------------------+
    | (e) PSD before / after            | (f) Summary statistics table      |
    +-----------------------------------+-----------------------------------+

    When ``info`` (an MNE ``Info`` object) is provided, panels (b) and (c)
    are rendered as topographic maps.  Otherwise they fall back to bar / stem
    plots.

    Parameters
    ----------
    result : ZapLine
        Fitted ZapLine estimator (standard mode).
    data_before : ndarray, shape (n_channels, n_times) | None
        Original data before cleaning (enables PSD panel).
    data_after : ndarray, shape (n_channels, n_times) | None
        Cleaned data after ZapLine (enables PSD panel).
    sfreq : float | None
        Sampling frequency.  Defaults to ``result.sfreq``.
    channel_names : list of str | None
        Channel names for axis labels (used when ``info`` is ``None``).
    info : mne.Info | None
        MNE channel info for topomap rendering.  When provided,
        panels (b) and (c) use ``mne.viz.plot_topomap`` instead of
        bar / stem plots.
    max_components : int
        Maximum number of spatial-pattern topomaps to show in panel (b).
        Capped to the actual number of removed components.
    title : str | None
        Figure super-title. Defaults to ``"ZapLine Diagnostics"``.
    figsize : tuple | None
        ``(width, height)`` in inches. Defaults to the theme default.
    dpi : int
        Figure resolution.
    show : bool
        Whether to call ``plt.show()`` at the end.

    Returns
    -------
    fig : Figure
        The matplotlib figure.
    """
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Patch

    # ---- Adaptive guard: delegate to the adaptive dashboard ----
    if _is_adaptive(result):
        return plot_adaptive_summary(
            result,
            data_before=data_before,
            data_after=data_after,
            sfreq=sfreq,
            channel_names=channel_names,
            title=title,
            figsize=figsize,
            dpi=dpi,
            show=show,
        )

    # ---- Resolve basic attributes ----
    if sfreq is None:
        sfreq = getattr(result, "sfreq", None)
    line_freq = getattr(result, "line_freq", None)
    eigenvalues = getattr(result, "eigenvalues_", None)
    patterns = getattr(result, "patterns_", None)
    n_removed = _get_n_removed(result)
    removed = _get_removed(result)
    sources = getattr(result, "sources_", None)
    has_psd = data_before is not None and data_after is not None and sfreq is not None

    # Pre-compute PSD
    freqs = psd_before = psd_after = None
    if has_psd:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)

    # ---- Theme ----
    C = COLORS
    F = FONTS
    if figsize is None:
        from ._theme import DEFAULT_FIGSIZE

        figsize = DEFAULT_FIGSIZE

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.42,
        wspace=0.30,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )

    # =================================================================
    # (a) DSS eigenvalues with threshold
    # =================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    style_axes(ax_a, grid=True)

    if eigenvalues is not None and eigenvalues.size > 0:
        n_eig = len(eigenvalues)
        x_eig = np.arange(n_eig)

        bar_colors = [
            C["accent"] if i < n_removed else C["primary"] for i in range(n_eig)
        ]
        ax_a.bar(x_eig, eigenvalues, color=bar_colors, edgecolor="none", width=0.75)

        mean_eig = np.mean(eigenvalues)
        ax_a.axhline(mean_eig, color=C["muted"], ls="--", lw=0.8, zorder=3)

        if n_removed > 0 and n_removed < n_eig:
            ax_a.axvline(
                n_removed - 0.5,
                color=C["accent"],
                ls="-",
                lw=1.0,
                zorder=4,
            )

        pub_legend(
            ax_a,
            loc="upper right",
            handles=[
                Patch(fc=C["accent"], label=f"Removed ({n_removed})"),
                Patch(fc=C["primary"], label="Retained"),
                plt.Line2D(
                    [],
                    [],
                    color=C["muted"],
                    ls="--",
                    lw=0.8,
                    label=f"Mean ({mean_eig:.3f})",
                ),
            ],
        )
        ax_a.set_xlim(-0.6, n_eig - 0.4)
    else:
        ax_a.text(
            0.5,
            0.5,
            "No eigenvalues available",
            ha="center",
            va="center",
            transform=ax_a.transAxes,
            fontsize=F["label"],
            color="#999999",
        )

    ax_a.set_xlabel("Component", fontsize=F["label"])
    ax_a.set_ylabel("Eigenvalue", fontsize=F["label"])
    ax_a.set_title(
        "(a)  DSS eigenvalues",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )

    # =================================================================
    # (b) Spatial mixing patterns  (topomaps if info, else stem plot)
    # =================================================================
    _has_patterns = patterns is not None and patterns.size > 0

    if _has_patterns and info is not None:
        # --- topomap mode: subdivide gs[0, 1] into 1??N sub-axes --------
        import mne

        n_ch, n_pat = patterns.shape
        n_show = min(n_pat, max_components)
        gs_b = GridSpecFromSubplotSpec(
            1,
            n_show,
            subplot_spec=gs[0, 1],
            wspace=0.25,
        )
        palette = [C["accent"], C["primary"], C["secondary"], C["success"]]
        for k in range(n_show):
            ax_k = fig.add_subplot(gs_b[0, k])
            mne.viz.plot_topomap(
                patterns[:, k],
                info,
                axes=ax_k,
                show=False,
                contours=4,
            )
            ax_k.set_title(
                f"Comp {k + 1}",
                fontsize=F["tick"],
                pad=3,
                color=palette[k % len(palette)],
                fontweight="semibold",
            )
        # Panel title anchored to the first topomap sub-axis
        pos = fig.add_subplot(gs[0, 1]).get_position()
        fig.axes[-1].set_visible(False)  # hide the dummy axes
        fig.text(
            pos.x0,
            pos.y1 + 0.015,
            "(b)  Spatial mixing patterns",
            fontsize=F["title"],
            fontweight="semibold",
            ha="left",
            va="bottom",
        )
    else:
        # --- fallback: stem / bar plot (NOT connected lines) ------------
        ax_b = fig.add_subplot(gs[0, 1])
        style_axes(ax_b, grid=True)

        if _has_patterns:
            n_ch, n_pat = patterns.shape
            n_show = min(n_pat, max_components)
            x_ch = np.arange(n_ch)

            palette = [C["accent"], C["primary"], C["secondary"], C["success"]]
            for k in range(n_show):
                markerline, stemlines, baseline = ax_b.stem(
                    x_ch,
                    patterns[:, k],
                    linefmt="-",
                    markerfmt="o",
                    basefmt="",
                    label=f"Comp {k + 1}",
                )
                color = palette[k % len(palette)]
                markerline.set(color=color, markersize=3)
                stemlines.set(color=color, linewidth=0.7, alpha=0.6)

            if channel_names is not None and len(channel_names) == n_ch:
                if n_ch <= 20:
                    ax_b.set_xticks(x_ch)
                    ax_b.set_xticklabels(
                        channel_names,
                        rotation=45,
                        ha="right",
                        fontsize=F["tick"] - 0.5,
                    )
                else:
                    step = max(1, n_ch // 12)
                    ax_b.set_xticks(x_ch[::step])
                    ax_b.set_xticklabels(
                        [channel_names[i] for i in range(0, n_ch, step)],
                        rotation=45,
                        ha="right",
                        fontsize=F["tick"] - 0.5,
                    )
            else:
                step = max(1, n_ch // 12)
                ax_b.set_xticks(x_ch[::step])

            pub_legend(ax_b, loc="upper right")
        else:
            ax_b.text(
                0.5,
                0.5,
                "No patterns available",
                ha="center",
                va="center",
                transform=ax_b.transAxes,
                fontsize=F["label"],
                color="#999999",
            )

        ax_b.set_xlabel("Channel", fontsize=F["label"])
        ax_b.set_ylabel("Weight", fontsize=F["label"])
        ax_b.set_title(
            "(b)  Spatial mixing patterns",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )

    # =================================================================
    # (c) Removed artifact power  (topomap if info, else bar chart)
    # =================================================================
    _has_removed = removed is not None and np.any(removed != 0)

    if _has_removed and info is not None:
        # --- topomap of removed artifact RMS ----------------------------
        import mne

        removed_rms = np.sqrt(np.mean(removed**2, axis=1))

        # Auto-scale
        if np.max(removed_rms) < 1e-3:
            scale, unit = 1e6, "\u00b5V"
        elif np.max(removed_rms) < 1.0:
            scale, unit = 1e3, "mV"
        else:
            scale, unit = 1.0, "V"

        ax_c = fig.add_subplot(gs[1, 0])
        im, _ = mne.viz.plot_topomap(
            removed_rms,
            info,
            axes=ax_c,
            show=False,
            contours=4,
        )
        cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.set_label(f"RMS ({unit})", fontsize=F["tick"])
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _, s=scale: f"{x * s:.1f}")
        )
        cbar.ax.tick_params(labelsize=F["tick"] - 1)
        ax_c.set_title(
            "(c)  Removed artifact power",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )
    else:
        # --- fallback: bar chart ----------------------------------------
        ax_c = fig.add_subplot(gs[1, 0])
        style_axes(ax_c, grid=True)

        if _has_removed:
            removed_rms = np.sqrt(np.mean(removed**2, axis=1))
            n_ch = len(removed_rms)
            x_pos = np.arange(n_ch)

            scale, unit = 1.0, "V"
            if np.max(removed_rms) < 1e-3:
                scale, unit = 1e6, "\u00b5V"
            elif np.max(removed_rms) < 1.0:
                scale, unit = 1e3, "mV"

            ax_c.bar(
                x_pos,
                removed_rms * scale,
                color=C["accent"],
                edgecolor="none",
                width=0.75,
                alpha=0.85,
            )

            if channel_names is not None and len(channel_names) == n_ch:
                step = max(1, n_ch // 16) if n_ch > 20 else 1
                ax_c.set_xticks(x_pos[::step])
                ax_c.set_xticklabels(
                    [channel_names[i] for i in range(0, n_ch, step)],
                    rotation=45,
                    ha="right",
                    fontsize=F["tick"] - 0.5,
                )
            else:
                step = max(1, n_ch // 12)
                ax_c.set_xticks(x_pos[::step])

            ax_c.set_xlim(-0.6, n_ch - 0.4)
            ax_c.set_ylabel(f"RMS ({unit})", fontsize=F["label"])
        else:
            ax_c.text(
                0.5,
                0.5,
                "No removal data\n(run fit_transform first)",
                ha="center",
                va="center",
                transform=ax_c.transAxes,
                fontsize=F["label"],
                color="#999999",
            )

        ax_c.set_xlabel("Channel", fontsize=F["label"])
        ax_c.set_title(
            "(c)  Removed artifact power",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )

    # =================================================================
    # (d) Artifact source time series
    # =================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    style_axes(ax_d)

    if sources is not None and sources.size > 0 and sfreq is not None:
        n_src = sources.shape[0]
        n_show = min(n_src, 4)
        n_samples = sources.shape[1]
        t_full = np.arange(n_samples) / sfreq

        # Pick a random 2-second window
        show_dur = min(2.0, n_samples / sfreq)
        win_samples = int(show_dur * sfreq)

        if n_samples > win_samples:
            rng = np.random.RandomState(42)
            win_start = rng.randint(0, n_samples - win_samples)
        else:
            win_start = 0
        win_end = win_start + win_samples
        sl = slice(win_start, win_end)
        t_win = t_full[sl] - t_full[win_start]  # zero-based

        palette = [C["accent"], C["primary"], C["secondary"], C["success"]]
        offsets = []
        for k in range(n_show):
            sig = sources[k, sl]
            offset = -k * np.std(sources[0, sl]) * 3.5
            offsets.append(offset)
            ax_d.plot(
                t_win,
                sig + offset,
                color=palette[k % len(palette)],
                linewidth=0.5,
                alpha=0.85,
                label=f"Comp {k + 1}",
            )

        ax_d.set_xlabel("Time (s)", fontsize=F["label"])
        ax_d.set_ylabel("Amplitude (a.u.)", fontsize=F["label"])
        ax_d.set_yticks([])
        pub_legend(ax_d, loc="upper right")
    elif sources is not None and sources.size > 0:
        ax_d.text(
            0.5,
            0.5,
            "sfreq required for time axis",
            ha="center",
            va="center",
            transform=ax_d.transAxes,
            fontsize=F["label"],
            color="#999999",
        )
    else:
        ax_d.text(
            0.5,
            0.5,
            "No source data\n(run fit_transform first)",
            ha="center",
            va="center",
            transform=ax_d.transAxes,
            fontsize=F["label"],
            color="#999999",
        )

    ax_d.set_title(
        "(d)  Artifact sources (2 s)",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )

    # =================================================================
    # (e) PSD before / after
    # =================================================================
    ax_e = fig.add_subplot(gs[2, 0])
    style_axes(ax_e, grid=True)

    if has_psd:
        fmax = 100.0
        mean_before = np.mean(psd_before, axis=0)
        mean_after = np.mean(psd_after, axis=0)

        ax_e.semilogy(
            freqs,
            mean_before,
            color="#333333",
            linewidth=2.0,
            linestyle="-",
            alpha=1.0,
            label="Before",
            zorder=2,
        )
        ax_e.semilogy(
            freqs,
            mean_after,
            color=C["after"],
            linewidth=1.2,
            linestyle="-",
            alpha=0.9,
            label="After",
            zorder=3,
        )

        if line_freq is not None:
            ax_e.axvline(
                line_freq,
                color=C["accent"],
                ls=":",
                lw=0.8,
                alpha=0.8,
                label=f"{line_freq:.0f} Hz",
                zorder=4,
            )
            n_harmonics = getattr(result, "n_harmonics_", None) or 0
            for h in range(2, n_harmonics + 2):
                if line_freq * h < fmax:
                    ax_e.axvline(
                        line_freq * h,
                        color=C["accent"],
                        ls=":",
                        lw=0.5,
                        alpha=0.25,
                    )

        ax_e.set_xlabel("Frequency (Hz)", fontsize=F["label"])
        ax_e.set_ylabel(r"PSD (V$^2\!$/Hz)", fontsize=F["label"])
        ax_e.set_xlim(0, fmax)
        pub_legend(ax_e, loc="upper right")
    else:
        ax_e.text(
            0.5,
            0.5,
            "Provide data_before / data_after\nfor PSD comparison",
            ha="center",
            va="center",
            transform=ax_e.transAxes,
            fontsize=F["label"],
            color="#999999",
        )

    ax_e.set_title(
        "(e)  Power spectral density",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )

    # =================================================================
    # (f) Summary statistics
    # =================================================================
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.axis("off")

    _get_cleaned(result)
    rows = [
        ("Mode", "Standard"),
        ("Line frequency", f"{line_freq:.1f} Hz" if line_freq else "???"),
        ("Harmonics", str(getattr(result, "n_harmonics_", "???"))),
        ("Components removed", str(n_removed)),
        ("n_remove setting", str(getattr(result, "n_remove", "???"))),
        ("Threshold (\u03c3)", f"{getattr(result, 'threshold', '???')}"),
        (
            "nkeep / rank",
            f"{getattr(result, 'nkeep', '???')} / {getattr(result, 'rank', '???')}",
        ),
    ]

    # Eigenvalue statistics
    if eigenvalues is not None and eigenvalues.size > 0:
        rows.append(("Max eigenvalue", f"{eigenvalues[0]:.4f}"))
        if n_removed > 0 and n_removed < len(eigenvalues):
            rows.append(
                (
                    "Eigen gap (removed/next)",
                    f"{eigenvalues[n_removed - 1]:.4f} / {eigenvalues[n_removed]:.4f}",
                )
            )

    # Power reduction
    if has_psd and line_freq is not None and freqs is not None:
        idx = np.argmin(np.abs(freqs - line_freq))
        pb = np.mean(psd_before[:, idx])
        pa = np.mean(psd_after[:, idx])
        if pa > 0 and pb > 0:
            reduction_db = 10 * np.log10(pb / pa)
            rows.append(("Power reduction at line freq", f"{reduction_db:.1f} dB"))

    # Variance explained
    if removed is not None and np.any(removed != 0) and data_before is not None:
        total_var = np.var(data_before)
        removed_var = np.var(removed)
        if total_var > 0:
            pct = 100 * removed_var / total_var
            rows.append(("Variance removed", f"{pct:.4f} %"))

    # Render table
    n_rows = len(rows)
    row_h = min(0.085, 0.90 / max(n_rows, 1))
    y_top = 0.94

    for i, (label, value) in enumerate(rows):
        y = y_top - i * row_h
        ax_f.text(
            0.03,
            y,
            label,
            transform=ax_f.transAxes,
            fontsize=F["label"] - 0.5,
            color="#555555",
            va="top",
            fontfamily="sans-serif",
        )
        ax_f.text(
            0.97,
            y,
            value,
            transform=ax_f.transAxes,
            fontsize=F["label"] - 0.5,
            color=C["text"],
            va="top",
            ha="right",
            fontfamily="monospace",
            fontweight="bold",
        )
        y_line = y - row_h * 0.35
        ax_f.plot(
            [0.03, 0.97],
            [y_line, y_line],
            color="#e0e0e0",
            lw=0.4,
            transform=ax_f.transAxes,
            clip_on=False,
        )

    ax_f.set_title(
        "(f)  Summary",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )

    # =================================================================
    # Suptitle
    # =================================================================
    if title is None:
        title = "ZapLine Diagnostics"
    fig.suptitle(title, fontsize=F["suptitle"], fontweight="bold", color=C["text"])

    if show:
        plt.show()

    return fig


def _get_n_removed(result):
    """Extract n_removed from a ZapLine result, handling both modes."""
    n_rem = 0
    if hasattr(result, "n_removed_") and result.n_removed_ is not None:
        n_rem = result.n_removed_
    elif hasattr(result, "n_removed") and result.n_removed is not None:
        n_rem = result.n_removed
    elif isinstance(result, dict):
        n_rem = result.get("n_removed", 0) or 0
    return n_rem


def _get_removed(result):
    """Extract removed data from a ZapLine result, handling both modes."""
    # Direct attribute (standard mode, or adaptive after fix)
    if hasattr(result, "removed") and result.removed is not None:
        return result.removed
    # Adaptive results dict fallback
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("removed")
    # Dict input
    if isinstance(result, dict):
        return result.get("removed")
    return None


def _get_cleaned(result):
    """Extract cleaned data from a ZapLine result, handling both modes."""
    if hasattr(result, "cleaned") and result.cleaned is not None:
        return result.cleaned
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("cleaned")
    if isinstance(result, dict):
        return result.get("cleaned")
    return None


def _is_adaptive(result):
    """Check whether the result is from adaptive mode."""
    if hasattr(result, "adaptive") and result.adaptive:
        return True
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return True
    return bool(isinstance(result, dict) and "chunk_info" in result)


def _get_chunk_info(result):
    """Extract per-chunk info from an adaptive ZapLine result."""
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("chunk_info", [])
    if isinstance(result, dict):
        return result.get("chunk_info", [])
    return []
