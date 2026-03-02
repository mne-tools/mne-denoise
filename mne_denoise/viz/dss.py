"""Generalized DSS diagnostics dashboards.

Publication-quality visualizations that work for **any** fitted
:class:`~mne_denoise.dss.DSS` instance ??? regardless of bias function,
smoothing mode, or segmented mode.  Mirrors the panel structure of
:mod:`~mne_denoise.viz.zapline` but removes all ZapLine / line-noise
specific assumptions.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ._theme import (
    COLORS,
    DEFAULT_FIGSIZE,
    FONTS,
    pub_legend,
    style_axes,
)
from ._utils import _get_info, _get_scores

# =====================================================================
# Helper: universal attribute accessors for DSS estimators
# =====================================================================


def _get_n_selected(estimator):
    """Return the number of components auto-selected (``n_selected_``)."""
    if hasattr(estimator, "n_selected_") and estimator.n_selected_ is not None:
        return estimator.n_selected_
    if hasattr(estimator, "n_removed_") and estimator.n_removed_ is not None:
        return estimator.n_removed_
    return 0


def _get_eigenvalues(estimator):
    """Eigenvalue vector, or *None*."""
    return _get_scores(estimator)


def _get_segment_results(estimator):
    """Return ``segment_results_`` list-of-dict or ``[]``."""
    if hasattr(estimator, "segment_results_") and estimator.segment_results_:
        return estimator.segment_results_
    # ZapLine adaptive fallback
    if hasattr(estimator, "adaptive_results_") and estimator.adaptive_results_:
        return estimator.adaptive_results_.get("chunk_info", [])
    return []


def _is_segmented(estimator):
    """Check whether the result is from segmented / adaptive mode."""
    if hasattr(estimator, "segmented") and estimator.segmented:
        return True
    return bool(_get_segment_results(estimator))


def _get_removed(estimator):
    """Return the removed artifact array (n_channels, n_times) or *None*."""
    if hasattr(estimator, "removed") and estimator.removed is not None:
        return estimator.removed
    if hasattr(estimator, "removed_") and estimator.removed_ is not None:
        return estimator.removed_
    return None


def _get_bias_name(estimator):
    """Human-readable bias function name."""
    bias = getattr(estimator, "bias", None)
    if bias is None:
        return "Unknown"
    name = type(bias).__name__
    # Pretty-print common names
    rename = {
        "BandpassBias": "Bandpass",
        "TrialAverageBias": "Trial Average",
        "SmoothingBias": "Smoothing",
        "PeakFilterBias": "Peak Filter",
        "CombFilterBias": "Comb Filter",
        "NotchBias": "Notch",
    }
    return rename.get(name, name)


# =====================================================================
# 1. Standard (non-segmented) DSS diagnostics
# =====================================================================


def plot_dss_summary(
    estimator,
    data_before=None,
    data_after=None,
    sfreq=None,
    channel_names=None,
    info=None,
    max_components=4,
    fmax=None,
    title=None,
    figsize=None,
    dpi=200,
    show=True,
):
    """Publication-quality 3??2 diagnostics dashboard for any fitted DSS.

    +-----------------------------------+-----------------------------------+
    | (a) DSS eigenvalues + threshold   | (b) Spatial patterns (topomaps)   |
    +-----------------------------------+-----------------------------------+
    | (c) Removed artifact power        | (d) Artifact source time-series   |
    +-----------------------------------+-----------------------------------+
    | (e) PSD before / after            | (f) Summary statistics table      |
    +-----------------------------------+-----------------------------------+

    Automatically delegates to :func:`plot_dss_segmented_summary` when the
    estimator was fitted in segmented mode.

    Parameters
    ----------
    estimator : DSS
        Fitted DSS instance (any bias function).
    data_before : ndarray, shape (n_channels, n_times) | None
        Original data before cleaning, for PSD panel.
    data_after : ndarray, shape (n_channels, n_times) | None
        Cleaned data after DSS, for PSD panel.
    sfreq : float | None
        Sampling frequency.  If ``None``, extracted from the bias or estimator.
    channel_names : list of str | None
        Channel names for axis labels.
    info : mne.Info | None
        MNE channel info for topomap rendering.
    max_components : int
        Maximum topomaps to show in panel (b). Default 4.
    fmax : float | None
        Maximum frequency for PSD panel. Default 100 Hz (or Nyquist/2).
    title : str | None
        Super-title. Defaults to ``"DSS Diagnostics (<bias>)"``.
    figsize : tuple | None
        Figure size in inches.
    dpi : int
        Figure resolution. Default 200.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
    """
    # Delegate to segmented dashboard when appropriate
    if _is_segmented(estimator):
        return plot_dss_segmented_summary(
            estimator,
            data_before=data_before,
            data_after=data_after,
            sfreq=sfreq,
            channel_names=channel_names,
            info=info,
            title=title,
            figsize=figsize,
            dpi=dpi,
            show=show,
        )

    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Patch

    # ---- Resolve attributes ----
    if sfreq is None:
        sfreq = getattr(estimator, "sfreq", None)
        if sfreq is None:
            bias = getattr(estimator, "bias", None)
            sfreq = getattr(bias, "sfreq", None)

    if info is None:
        info = _get_info(estimator)

    eigenvalues = _get_eigenvalues(estimator)
    patterns = getattr(estimator, "patterns_", None)
    n_selected = _get_n_selected(estimator)
    removed = _get_removed(estimator)
    sources = getattr(estimator, "sources_", None)
    has_psd = data_before is not None and data_after is not None and sfreq is not None

    # Compute removed artifact from data_before/after if not cached
    if removed is None and data_before is not None and data_after is not None:
        removed = data_before - data_after

    # Compute sources from estimator if not cached
    if sources is None and data_before is not None:
        try:
            old_rt = getattr(estimator, "return_type", None)
            if old_rt is not None:
                estimator.return_type = "sources"
            sources = estimator.transform(data_before)
            if old_rt is not None:
                estimator.return_type = old_rt
        except Exception:
            sources = None

    if fmax is None:
        fmax = min(100.0, sfreq / 2) if sfreq else 100.0

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
            C["accent"] if i < n_selected else C["primary"] for i in range(n_eig)
        ]
        ax_a.bar(
            x_eig,
            eigenvalues,
            color=bar_colors,
            edgecolor="none",
            width=0.75,
        )

        mean_eig = np.mean(eigenvalues)
        ax_a.axhline(mean_eig, color=C["muted"], ls="--", lw=0.8, zorder=3)

        if 0 < n_selected < n_eig:
            ax_a.axvline(
                n_selected - 0.5,
                color=C["accent"],
                ls="-",
                lw=1.0,
                zorder=4,
            )

        pub_legend(
            ax_a,
            loc="upper right",
            handles=[
                Patch(fc=C["accent"], label=f"Selected ({n_selected})"),
                Patch(fc=C["primary"], label="Retained"),
                plt.Line2D(
                    [],
                    [],
                    color=C["muted"],
                    ls="--",
                    lw=0.8,
                    label=f"Mean ({mean_eig:.4f})",
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
        # Panel title
        pos = fig.add_subplot(gs[0, 1]).get_position()
        fig.axes[-1].set_visible(False)
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
        import mne

        removed_rms = np.sqrt(np.mean(removed**2, axis=1))

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

        # Pick a representative 2-second window
        show_dur = min(2.0, n_samples / sfreq)
        win_samples = int(show_dur * sfreq)

        if n_samples > win_samples:
            rng = np.random.RandomState(42)
            win_start = rng.randint(0, n_samples - win_samples)
        else:
            win_start = 0
        win_end = win_start + win_samples
        sl = slice(win_start, win_end)
        t_win = t_full[sl] - t_full[win_start]

        palette = [C["accent"], C["primary"], C["secondary"], C["success"]]
        for k in range(n_show):
            sig = sources[k, sl]
            offset = -k * np.std(sources[0, sl]) * 3.5
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
        "(d)  DSS sources (2 s)",
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

        # Mark bias frequency if available (bandpass, notch, etc.)
        bias = getattr(estimator, "bias", None)
        bias_freq = getattr(bias, "freq", None) or getattr(bias, "line_freq", None)
        if bias_freq is not None:
            ax_e.axvline(
                bias_freq,
                color=C["accent"],
                ls=":",
                lw=0.8,
                alpha=0.8,
                label=f"{bias_freq:.0f} Hz",
                zorder=4,
            )

        ax_e.set_xlabel("Frequency (Hz)", fontsize=F["label"])
        ax_e.set_ylabel(r"PSD (V$^2\!$/Hz)", fontsize=F["label"])
        ax_e.set_xlim(0, fmax)
        pub_legend(ax_e, loc="upper right")
    else:
        # Fallback: show eigenvalue bar chart if no PSD data
        if eigenvalues is not None and eigenvalues.size > 0:
            ax_e.bar(
                range(len(eigenvalues)),
                eigenvalues,
                color=C["primary"],
                edgecolor="none",
            )
            ax_e.axhline(
                np.mean(eigenvalues),
                color=C["accent"],
                ls="--",
                lw=0.9,
            )
            ax_e.set_xlabel("Component", fontsize=F["label"])
            ax_e.set_ylabel("Eigenvalue", fontsize=F["label"])
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

    bias_name = _get_bias_name(estimator)
    rows = [
        ("Bias function", bias_name),
        ("Mode", "Standard"),
        ("n_components", str(getattr(estimator, "n_components", "???"))),
        ("Components selected", str(n_selected) if n_selected else "???"),
        ("Selection method", str(getattr(estimator, "selection_method", "???"))),
    ]

    # Smoothing
    smooth = getattr(estimator, "smooth", None)
    if smooth is not None:
        if isinstance(smooth, int):
            rows.append(("Smoothing", f"window={smooth} samples"))
        else:
            rows.append(("Smoothing", type(smooth).__name__))

    # Eigenvalue statistics
    if eigenvalues is not None and eigenvalues.size > 0:
        rows.append(("Max eigenvalue", f"{eigenvalues[0]:.4f}"))
        if n_selected > 0 and n_selected < len(eigenvalues):
            rows.append(
                (
                    "Eigen gap (sel/next)",
                    f"{eigenvalues[n_selected - 1]:.4f} / {eigenvalues[n_selected]:.4f}",
                )
            )

    # Power reduction at bias frequency
    if has_psd and freqs is not None:
        bias = getattr(estimator, "bias", None)
        bias_freq = getattr(bias, "freq", None) or getattr(bias, "line_freq", None)
        if bias_freq is not None:
            idx = np.argmin(np.abs(freqs - bias_freq))
            pb = np.mean(psd_before[:, idx])
            pa = np.mean(psd_after[:, idx])
            if pa > 0 and pb > 0:
                reduction_db = 10 * np.log10(pb / pa)
                rows.append(
                    (
                        f"Power reduction @ {bias_freq:.0f} Hz",
                        f"{reduction_db:.1f} dB",
                    )
                )

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
        title = f"DSS Diagnostics ({bias_name})"
    fig.suptitle(
        title,
        fontsize=F["suptitle"],
        fontweight="bold",
        color=C["text"],
    )

    if show:
        plt.show()

    return fig


# =====================================================================
# 2. Segmented DSS diagnostics
# =====================================================================


def plot_dss_segmented_summary(
    estimator,
    data_before=None,
    data_after=None,
    sfreq=None,
    channel_names=None,
    info=None,
    fmax=None,
    title=None,
    figsize=None,
    dpi=200,
    show=True,
):
    """Publication-quality diagnostics for segmented DSS.

    Creates a 3??2 multi-panel figure:

    +---------------------------------------------+---------------------------+
    | (a) Components selected per segment         | (b) Eigenvalue heatmap    |
    +---------------------------------------------+---------------------------+
    | (c) Segment timeline                        | (d) Removed artifact RMS  |
    +---------------------------------------------+---------------------------+
    | (e) PSD before / after                      | (f) Summary statistics    |
    +---------------------------------------------+---------------------------+

    Parameters
    ----------
    estimator : DSS
        Fitted DSS instance with ``segmented=True`` (must have
        ``segment_results_``).
    data_before : ndarray, shape (n_channels, n_times) | None
        Original data before cleaning, for PSD panel.
    data_after : ndarray, shape (n_channels, n_times) | None
        Cleaned data after DSS, for PSD panel.
    sfreq : float | None
        Sampling frequency.
    channel_names : list of str | None
        Channel names for axis labels.
    info : mne.Info | None
        MNE channel info for topomap rendering.
    fmax : float | None
        Maximum frequency for PSD panel.
    title : str | None
        Super-title.
    figsize : tuple | None
        Figure size in inches.
    dpi : int
        Figure resolution. Default 200.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    seg_results = _get_segment_results(estimator)
    if not seg_results:
        # Fall back to standard dashboard
        return plot_dss_summary(
            estimator,
            data_before=data_before,
            data_after=data_after,
            sfreq=sfreq,
            channel_names=channel_names,
            info=info,
            title=title,
            figsize=figsize,
            dpi=dpi,
            show=show,
        )

    # ---- Resolve attributes ----
    if sfreq is None:
        sfreq = getattr(estimator, "sfreq", None)
        if sfreq is None:
            bias = getattr(estimator, "bias", None)
            sfreq = getattr(bias, "sfreq", None)

    if info is None:
        info = _get_info(estimator)

    removed = _get_removed(estimator)
    n_segments = len(seg_results)
    has_psd = data_before is not None and data_after is not None and sfreq is not None

    # Compute removed artifact from data_before/after if not cached
    if removed is None and data_before is not None and data_after is not None:
        removed = data_before - data_after

    if fmax is None:
        fmax = min(100.0, sfreq / 2) if sfreq else 100.0

    # Extract per-segment stats
    per_seg_selected = [s.get("n_selected", s.get("n_removed", 0)) for s in seg_results]
    per_seg_eigenvalues = [s.get("eigenvalues", np.array([])) for s in seg_results]

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
    # (a) Components selected per segment
    # =================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    style_axes(ax_a, grid=True)

    ax_a.bar(
        range(n_segments),
        per_seg_selected,
        color=C["primary"],
        edgecolor="none",
        width=0.75,
    )
    mean_sel = np.mean(per_seg_selected)
    ax_a.axhline(mean_sel, color=C["accent"], ls="--", lw=0.9, zorder=4)

    ax_a.set_xlabel("Segment index", fontsize=F["label"])
    ax_a.set_ylabel("Components selected", fontsize=F["label"])
    ax_a.set_title(
        "(a)  Components selected per segment",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )
    ax_a.set_xlim(-0.6, n_segments - 0.4)
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    pub_legend(
        ax_a,
        loc="upper left",
        handles=[
            Patch(fc=C["primary"], label="Selected"),
            plt.Line2D(
                [],
                [],
                color=C["accent"],
                ls="--",
                lw=0.9,
                label=f"Mean ({mean_sel:.1f})",
            ),
        ],
    )

    # =================================================================
    # (b) Eigenvalue heatmap across segments
    # =================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    style_axes(ax_b)

    # Build a matrix: segments ?? max_components
    max_n_eig = max((len(e) for e in per_seg_eigenvalues if len(e) > 0), default=0)

    if max_n_eig > 0:
        # Limit display to first 10 components for readability
        n_show_eig = min(max_n_eig, 10)
        eig_matrix = np.full((n_segments, n_show_eig), np.nan)
        for i, ev in enumerate(per_seg_eigenvalues):
            n = min(len(ev), n_show_eig)
            eig_matrix[i, :n] = ev[:n]

        im = ax_b.imshow(
            eig_matrix.T,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            origin="lower",
        )
        cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
        cbar.set_label("Eigenvalue", fontsize=F["tick"])
        cbar.ax.tick_params(labelsize=F["tick"] - 1)

        ax_b.set_xlabel("Segment index", fontsize=F["label"])
        ax_b.set_ylabel("Component", fontsize=F["label"])

        # Mark n_selected boundary per segment
        for i, ns in enumerate(per_seg_selected):
            if 0 < ns <= n_show_eig:
                ax_b.plot(
                    i,
                    ns - 0.5,
                    "w_",
                    markersize=6,
                    markeredgewidth=1.5,
                )
    else:
        ax_b.text(
            0.5,
            0.5,
            "No eigenvalues available",
            ha="center",
            va="center",
            transform=ax_b.transAxes,
            fontsize=F["label"],
            color="#999999",
        )

    ax_b.set_title(
        "(b)  Eigenvalue heatmap",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )

    # =================================================================
    # (c) Segment timeline
    # =================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    style_axes(ax_c)

    if sfreq is not None and sfreq > 0:
        total_dur = (
            seg_results[-1].get("end", seg_results[-1].get("end_sample", 0)) / sfreq
        )

        for _i, s in enumerate(seg_results):
            s_t = s.get("start", s.get("start_sample", 0)) / sfreq
            e_t = s.get("end", s.get("end_sample", 0)) / sfreq
            ns = s.get("n_selected", s.get("n_removed", 0))

            # Color intensity based on n_selected
            col = C["primary"] if ns > 0 else C["muted"]
            ax_c.barh(
                0,
                e_t - s_t,
                left=s_t,
                height=0.7,
                color=col,
                edgecolor="white",
                linewidth=0.25,
            )

            # Label segments wide enough
            seg_frac = (e_t - s_t) / max(total_dur, 1e-9)
            if seg_frac > 0.018:
                tc = "white" if ns > 0 else C["text"]
                ax_c.text(
                    (s_t + e_t) / 2,
                    0,
                    str(ns),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=tc,
                    fontweight="bold",
                )

        ax_c.set_xlabel("Time (s)", fontsize=F["label"])
        ax_c.set_yticks([])
        ax_c.set_ylim(-0.5, 0.5)
        pub_legend(
            ax_c,
            loc="upper right",
            handles=[
                Patch(fc=C["primary"], label="Artifact cleaned"),
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
        import mne

        removed_rms = np.sqrt(np.mean(removed**2, axis=1))

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

        # Mark bias frequency
        bias = getattr(estimator, "bias", None)
        bias_freq = getattr(bias, "freq", None) or getattr(bias, "line_freq", None)
        if bias_freq is not None:
            ax_e.axvline(
                bias_freq,
                color=C["accent"],
                ls=":",
                lw=0.8,
                alpha=0.8,
                label=f"{bias_freq:.0f} Hz",
                zorder=4,
            )

        ax_e.set_xlabel("Frequency (Hz)", fontsize=F["label"])
        ax_e.set_ylabel(r"PSD (V$^2\!$/Hz)", fontsize=F["label"])
        ax_e.set_xlim(0, fmax)
        pub_legend(ax_e, loc="upper right")
    else:
        eigenvalues = _get_eigenvalues(estimator)
        if eigenvalues is not None and eigenvalues.size > 0:
            ax_e.bar(
                range(len(eigenvalues)),
                eigenvalues,
                color=C["primary"],
                edgecolor="none",
            )
            ax_e.axhline(
                np.mean(eigenvalues),
                color=C["accent"],
                ls="--",
                lw=0.9,
            )
            ax_e.set_xlabel("Component", fontsize=F["label"])
            ax_e.set_ylabel("Eigenvalue", fontsize=F["label"])
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

    bias_name = _get_bias_name(estimator)
    total_selected = sum(per_seg_selected)

    rows = [
        ("Bias function", bias_name),
        ("Mode", "Segmented"),
        ("Number of segments", str(n_segments)),
        ("Total components selected", str(total_selected)),
        (
            "Per-segment min / max",
            f"{min(per_seg_selected)} / {max(per_seg_selected)}",
        ),
        (
            "Per-segment mean \u00b1 std",
            f"{np.mean(per_seg_selected):.1f} \u00b1 "
            f"{np.std(per_seg_selected):.1f}",
        ),
    ]

    # Eigenvalue range across segments
    all_max = [ev[0] for ev in per_seg_eigenvalues if len(ev) > 0]
    if all_max:
        rows.append(
            (
                "Max eigenvalue range",
                f"{min(all_max):.3f} \u2013 {max(all_max):.3f}",
            )
        )

    # Smoothing
    smooth = getattr(estimator, "smooth", None)
    if smooth is not None:
        if isinstance(smooth, int):
            rows.append(("Smoothing", f"window={smooth} samples"))
        else:
            rows.append(("Smoothing", type(smooth).__name__))

    # Power reduction
    if has_psd and freqs is not None:
        bias = getattr(estimator, "bias", None)
        bias_freq = getattr(bias, "freq", None) or getattr(bias, "line_freq", None)
        if bias_freq is not None:
            idx = np.argmin(np.abs(freqs - bias_freq))
            pb = np.mean(psd_before[:, idx])
            pa = np.mean(psd_after[:, idx])
            if pa > 0 and pb > 0:
                reduction_db = 10 * np.log10(pb / pa)
                rows.append(
                    (
                        f"Power reduction @ {bias_freq:.0f} Hz",
                        f"{reduction_db:.1f} dB",
                    )
                )

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
        title = f"Segmented DSS Diagnostics ({bias_name})"
    fig.suptitle(
        title,
        fontsize=F["suptitle"],
        fontweight="bold",
        color=C["text"],
    )

    if show:
        plt.show()

    return fig


# =====================================================================
# 3. Standalone panel functions
# =====================================================================


def plot_dss_eigenvalues(
    estimator,
    ax=None,
    show=True,
):
    """Bar chart of DSS eigenvalues with auto-selection threshold.

    Parameters
    ----------
    estimator : DSS
        Fitted DSS instance.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. If ``None``, creates a new figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
    """
    from matplotlib.patches import Patch

    C = COLORS
    F = FONTS
    eigenvalues = _get_eigenvalues(estimator)
    n_selected = _get_n_selected(estimator)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=200)
        fig.set_facecolor("white")
    else:
        fig = ax.figure

    style_axes(ax, grid=True)

    if eigenvalues is not None and eigenvalues.size > 0:
        n_eig = len(eigenvalues)
        x_eig = np.arange(n_eig)

        bar_colors = [
            C["accent"] if i < n_selected else C["primary"] for i in range(n_eig)
        ]
        ax.bar(
            x_eig,
            eigenvalues,
            color=bar_colors,
            edgecolor="none",
            width=0.75,
        )

        mean_eig = np.mean(eigenvalues)
        ax.axhline(mean_eig, color=C["muted"], ls="--", lw=0.8, zorder=3)

        if 0 < n_selected < n_eig:
            ax.axvline(
                n_selected - 0.5,
                color=C["accent"],
                ls="-",
                lw=1.0,
                zorder=4,
            )

        pub_legend(
            ax,
            loc="upper right",
            handles=[
                Patch(fc=C["accent"], label=f"Selected ({n_selected})"),
                Patch(fc=C["primary"], label="Retained"),
                plt.Line2D(
                    [],
                    [],
                    color=C["muted"],
                    ls="--",
                    lw=0.8,
                    label=f"Mean ({mean_eig:.4f})",
                ),
            ],
        )
        ax.set_xlim(-0.6, n_eig - 0.4)
    else:
        ax.text(
            0.5,
            0.5,
            "No eigenvalues available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=F["label"],
            color="#999999",
        )

    ax.set_xlabel("Component", fontsize=F["label"])
    ax.set_ylabel("Eigenvalue", fontsize=F["label"])
    ax.set_title("DSS Eigenvalues", fontsize=F["title"], fontweight="semibold")

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_dss_patterns(
    estimator,
    info=None,
    max_components=6,
    ax=None,
    show=True,
):
    """Spatial mixing patterns as topomaps (or stem plot fallback).

    Parameters
    ----------
    estimator : DSS
        Fitted DSS instance.
    info : mne.Info | None
        MNE channel info for topomap rendering.
    max_components : int
        Maximum number of patterns to show.
    ax : matplotlib.axes.Axes | None
        Axes to plot on (only used for stem fallback; topomaps
        generate their own sub-axes).
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
    """
    C = COLORS
    F = FONTS

    if info is None:
        info = _get_info(estimator)

    patterns = getattr(estimator, "patterns_", None)
    _has_patterns = patterns is not None and patterns.size > 0

    if _has_patterns and info is not None:
        import mne

        n_ch, n_pat = patterns.shape
        n_show = min(n_pat, max_components)

        fig, axes = plt.subplots(
            1,
            n_show,
            figsize=(2.5 * n_show, 3),
            dpi=200,
        )
        fig.set_facecolor("white")
        if n_show == 1:
            axes = [axes]

        palette = [
            C["accent"],
            C["primary"],
            C["secondary"],
            C["success"],
            C["purple"],
            C["cyan"],
        ]
        for k, ax_k in enumerate(axes):
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

        fig.suptitle(
            "DSS Spatial Patterns",
            fontsize=F["title"],
            fontweight="semibold",
        )
        plt.tight_layout()
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=200)
            fig.set_facecolor("white")
        else:
            fig = ax.figure

        style_axes(ax, grid=True)

        if _has_patterns:
            n_ch, n_pat = patterns.shape
            n_show = min(n_pat, max_components)
            x_ch = np.arange(n_ch)

            palette = [C["accent"], C["primary"], C["secondary"], C["success"]]
            for k in range(n_show):
                markerline, stemlines, baseline = ax.stem(
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

            pub_legend(ax, loc="upper right")
        else:
            ax.text(
                0.5,
                0.5,
                "No patterns available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=F["label"],
                color="#999999",
            )

        ax.set_xlabel("Channel", fontsize=F["label"])
        ax.set_ylabel("Weight", fontsize=F["label"])
        ax.set_title(
            "DSS Spatial Patterns",
            fontsize=F["title"],
            fontweight="semibold",
        )
        plt.tight_layout()

    if show:
        plt.show()

    return fig


# =====================================================================
# 5. Three-way DSS mode comparison
# =====================================================================


def _suppression_ratio(f, psd_before, psd_after, freq, bw=2.0):
    """Suppression ratio (dB) at a single target frequency."""
    mask = (f >= freq - bw) & (f <= freq + bw)
    pb = psd_before.mean(0)[mask].mean()
    pa = psd_after.mean(0)[mask].mean()
    if pa <= 0:
        return np.inf
    return 10 * np.log10(pb / pa)


def _spectral_distortion(f, psd_before, psd_after, line_freq=50.0, n_harm=3, bw=2.0):
    """Spectral distortion (dB RMS) at non-harmonic frequencies."""
    safe = np.ones(len(f), dtype=bool)
    for k in range(1, n_harm + 1):
        safe &= ~((f >= line_freq * k - bw * 2) & (f <= line_freq * k + bw * 2))
    safe &= (f >= 2) & (f <= 160)
    if not safe.any():
        return 0.0
    ratio = psd_after.mean(0)[safe] / psd_before.mean(0)[safe]
    return np.sqrt(np.mean((10 * np.log10(ratio)) ** 2))


def _variance_removed(data_before, data_after):
    """Percentage of total variance removed."""
    return 100 * (1 - np.var(data_after) / np.var(data_before))


def plot_dss_comparison(
    bias,
    data,
    *,
    n_components=20,
    n_select="auto",
    selection_method="combined",
    smooth=None,
    max_prop_remove=0.2,
    min_select=1,
    line_freq=50.0,
    n_harmonics=3,
    title=None,
    figsize=(16, 5),
    show=True,
):
    """Run and compare DSS in three modes: plain, smoothed, segmented.

    Fits three :class:`~mne_denoise.dss.DSS` instances with the same
    ``bias`` on the same ``data``:

    * **(A) Plain** ??? no smoothing, no segmentation.
    * **(B) + Smoothing** ??? eigenvalue decomposition smoothed.
    * **(C) + Smoothing + Segmentation** ??? smoothed, with adaptive
      per-segment component selection.

    Prints a comparison table (SR at harmonics, spectral distortion,
    variance removed) and returns a 1??2 PSD overlay figure.

    Parameters
    ----------
    bias : BaseBias
        Bias function instance (e.g. ``CombFilterBias``, ``PeakFilterBias``).
    data : mne.io.Raw
        Continuous raw data to denoise.
    n_components : int
        Number of DSS components to compute.
    n_select : int | "auto"
        Component selection strategy.
    selection_method : str
        Auto-selection method (``"combined"``, ``"ratio"``, etc.).
    smooth : int | None
        Smoothing window in samples.  If *None*, defaults to
        ``int(sfreq / line_freq)``.
    max_prop_remove : float
        Maximum proportion of components to remove (segmented mode).
    min_select : int
        Minimum components to remove per segment.
    line_freq : float
        Fundamental frequency for suppression-ratio computation.
    n_harmonics : int
        Number of harmonics to evaluate (for SR and distortion).
    title : str | None
        Figure super-title.  Auto-generated from bias name if *None*.
    figsize : tuple
        Figure size ``(width, height)``.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The PSD comparison figure.
    results : dict
        Dict with keys ``"plain"``, ``"smooth"``, ``"segmented"``, each
        containing ``{"estimator", "cleaned_raw", "data", "metrics"}``.
    """
    from ..dss.linear import DSS  # local import to avoid circular

    sfreq = data.info["sfreq"]
    data_orig = data.get_data()

    if smooth is None:
        smooth = int(sfreq / line_freq)

    bias_name = type(bias).__name__

    print(f"DSS + {bias_name} ??? 3-Way Comparison")
    print("=" * 65)

    # ?????? A) Plain DSS ??????
    dss_plain = DSS(
        bias=bias,
        n_components=n_components,
        n_select=n_select,
        selection_method=selection_method,
        return_type="raw",
    )
    raw_plain = dss_plain.fit_transform(data)
    data_plain = raw_plain.get_data()
    ev_plain = dss_plain.eigenvalues_
    n_plain = dss_plain.n_selected_
    print(
        f"  A) Plain DSS:          {n_plain} comp(s) | "
        f"max eigenvalue = {ev_plain[0]:.6f}"
    )

    # ?????? B) DSS + smoothing ??????
    dss_smooth = DSS(
        bias=bias,
        n_components=n_components,
        n_select=n_select,
        selection_method=selection_method,
        smooth=smooth,
        return_type="raw",
    )
    raw_smooth = dss_smooth.fit_transform(data)
    data_smooth = raw_smooth.get_data()
    ev_smooth = dss_smooth.eigenvalues_
    n_smooth = dss_smooth.n_selected_
    boost = ev_smooth[0] / ev_plain[0] if ev_plain[0] > 0 else float("inf")
    print(
        f"  B) + Smoothing:        {n_smooth} comp(s) | "
        f"max eigenvalue = {ev_smooth[0]:.6f} ({boost:.1f}?? boost)"
    )

    # ?????? C) DSS + smoothing + segmentation ??????
    dss_seg = DSS(
        bias=bias,
        n_components=n_components,
        n_select=n_select,
        selection_method=selection_method,
        smooth=smooth,
        segmented=True,
        max_prop_remove=max_prop_remove,
        min_select=min_select,
    )
    raw_seg = dss_seg.fit_transform(data)
    data_seg = raw_seg.get_data()
    n_segments = len(dss_seg.segment_results_)
    seg_n_sel = [s["n_selected"] for s in dss_seg.segment_results_]
    seg_evals = [s["eigenvalues"][0] for s in dss_seg.segment_results_]
    total_removed = sum(seg_n_sel)
    print(
        f"  C) + Segmentation:     {n_segments} seg(s) | "
        f"n_removed: {min(seg_n_sel)}???{max(seg_n_sel)} "
        f"(total {total_removed}) | "
        f"eigenvalue range: {min(seg_evals):.4f}???{max(seg_evals):.4f}"
    )

    # ?????? PSD computation ??????
    nperseg = int(sfreq * 4)
    f_psd, psd_orig = signal.welch(data_orig, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_A = signal.welch(data_plain, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_B = signal.welch(data_smooth, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_C = signal.welch(data_seg, fs=sfreq, nperseg=nperseg, axis=-1)

    # ?????? Metrics table ??????
    labels = ["(A) Plain", "(B) + Smooth", "(C) + Smooth + Seg"]
    cleaned_list = [data_plain, data_smooth, data_seg]
    psd_list = [psd_A, psd_B, psd_C]

    harmonics = [line_freq * k for k in range(1, n_harmonics + 1)]
    harm_labels = [f"SR@{int(h)}" for h in harmonics]

    header = f"{'Method':<22} " + "  ".join(f"{h:>7}" for h in harm_labels)
    header += f" {'Mean SR':>8} {'SD':>5} {'Var%':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    all_metrics = []
    for label, cleaned, psd_after in zip(labels, cleaned_list, psd_list):
        srs = [_suppression_ratio(f_psd, psd_orig, psd_after, h) for h in harmonics]
        sr_mean = np.mean(srs)
        sd = _spectral_distortion(
            f_psd, psd_orig, psd_after, line_freq=line_freq, n_harm=n_harmonics
        )
        var_pct = _variance_removed(data_orig, cleaned)
        metrics = {
            "sr_harmonics": srs,
            "sr_mean": sr_mean,
            "sd": sd,
            "var_pct": var_pct,
        }
        all_metrics.append(metrics)

        sr_str = "  ".join(f"{s:>7.1f}" for s in srs)
        print(f"{label:<22} {sr_str}  {sr_mean:>7.1f}  {sd:>4.2f}  " f"{var_pct:>5.3f}")

    # ?????? PSD overlay plot ??????
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: full PSD
    ax = axes[0]
    fmask = f_psd <= 160
    ax.semilogy(
        f_psd[fmask], psd_orig.mean(0)[fmask], "k", lw=2, alpha=0.4, label="Original"
    )
    ax.semilogy(
        f_psd[fmask], psd_A.mean(0)[fmask], "C0", lw=1.2, alpha=0.7, label="(A) Plain"
    )
    ax.semilogy(
        f_psd[fmask],
        psd_B.mean(0)[fmask],
        "C1",
        lw=1.2,
        alpha=0.7,
        label="(B) + Smooth",
    )
    ax.semilogy(
        f_psd[fmask],
        psd_C.mean(0)[fmask],
        "C3",
        lw=1.5,
        alpha=0.9,
        label="(C) + Smooth + Seg",
    )
    for k in range(1, n_harmonics + 1):
        ax.axvline(line_freq * k, color="r", ls="--", alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V??/Hz)")
    ax.set_title(f"PSD: {bias_name} ??? 3 Modes")
    ax.legend(fontsize=9)

    # Right: zoom on fundamental
    ax = axes[1]
    zoom = (f_psd >= line_freq - 5) & (f_psd <= line_freq + 5)
    ax.semilogy(
        f_psd[zoom], psd_orig.mean(0)[zoom], "k", lw=2, alpha=0.4, label="Original"
    )
    ax.semilogy(
        f_psd[zoom], psd_A.mean(0)[zoom], "C0", lw=1.5, alpha=0.7, label="(A) Plain"
    )
    ax.semilogy(
        f_psd[zoom], psd_B.mean(0)[zoom], "C1", lw=1.5, alpha=0.7, label="(B) + Smooth"
    )
    ax.semilogy(
        f_psd[zoom],
        psd_C.mean(0)[zoom],
        "C3",
        lw=2,
        alpha=0.9,
        label="(C) + Smooth + Seg",
    )
    ax.axvline(line_freq, color="r", ls="--", alpha=0.5, label=f"{int(line_freq)} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V??/Hz)")
    ax.set_title(f"Zoom: {int(line_freq - 5)}???{int(line_freq + 5)} Hz")
    ax.legend(fontsize=9)

    if title is None:
        title = f"{bias_name} ??? Plain vs Smoothing vs Segmented"
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if show:
        plt.show()

    # ?????? Build results dict ??????
    results = {
        "plain": {
            "estimator": dss_plain,
            "cleaned_raw": raw_plain,
            "data": data_plain,
            "metrics": all_metrics[0],
        },
        "smooth": {
            "estimator": dss_smooth,
            "cleaned_raw": raw_smooth,
            "data": data_smooth,
            "metrics": all_metrics[1],
        },
        "segmented": {
            "estimator": dss_seg,
            "cleaned_raw": raw_seg,
            "data": data_seg,
            "metrics": all_metrics[2],
        },
    }
    return fig, results
