"""Benchmark visualizations for line-noise removal comparisons.

Publication-quality figures that summarise a multi-method, multi-subject
benchmark.  Every ``plot_*`` function returns a ``matplotlib.figure.Figure``
and accepts an optional ``save_path`` for disk persistence.

The functions use the shared theme from ``mne_denoise.viz._theme`` so
they match the rest of the package's visual style.

Functions
---------
plot_psd_gallery
    Grid of PSD panels (methods ?? harmonics) for one subject.
plot_subject_psd_overlay
    Full-spectrum + fundamental-zoom PSD comparison for one subject.
plot_metric_bars
    Group-mean (?? SEM) bar chart across methods for each QA metric.
plot_tradeoff_scatter
    Attenuation-vs-distortion scatter with group-mean stars.
plot_r_comparison
    R(f???) bar / paired-dot plot.
plot_harmonic_attenuation
    Per-harmonic attenuation grouped-bar chart.
plot_paired_metrics
    Paired dot plots (one per metric) for multi-subject runs.
plot_qc_psd
    Per-subject before/after PSD with harmonic zoom insets.

Authors
-------
Sina Esmaeili ??? sina.esmaeili@umontreal.ca
Hamza Abdelhedi ??? hamza.abdelhedi@umontreal.ca
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._theme import COLORS, FONTS, pub_figure, pub_legend, style_axes

# ?????? Default method palette (colorblind-safe, matches notebook) ?????????????????????
DEFAULT_METHOD_COLORS = {
    "M0": COLORS["gray"],
    "M1": COLORS["blue"],
    "M2": COLORS["orange"],
    "M3": COLORS["purple"],
}

DEFAULT_METHOD_LABELS = {
    "M0": "Baseline (no cleaning)",
    "M1": "ZapLine (auto)",
    "M2": "ZapLine+ (adaptive)",
    "M3": "Notch filter",
}

DEFAULT_METHOD_ORDER = ["M0", "M1", "M2", "M3"]


# =====================================================================
# Helpers
# =====================================================================


def _resolve_save(fig, save_path, dpi=300):
    """Optionally save *fig* and return it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def _method_color(method, method_colors=None):
    if method_colors and method in method_colors:
        return method_colors[method]
    return DEFAULT_METHOD_COLORS.get(method, "#333333")


def _method_label(method, method_labels=None):
    if method_labels and method in method_labels:
        return method_labels[method]
    return DEFAULT_METHOD_LABELS.get(method, method)


# =====================================================================
# plot_qc_psd ??? per-subject before/after PSD + harmonic zoom
# =====================================================================


def plot_qc_psd(
    freqs_before,
    gm_before,
    freqs_after,
    gm_after,
    *,
    method_tag="",
    subject="",
    harmonics_hz=None,
    metrics_dict=None,
    fmax=125.0,
    method_colors=None,
    method_labels=None,
    save_path=None,
    show=True,
):
    """Per-subject before/after geometric-mean PSD plus harmonic zoom.

    Parameters
    ----------
    freqs_before, gm_before : ndarray
        Frequency vector & geometric-mean PSD **before** cleaning.
    freqs_after, gm_after : ndarray
        Frequency vector & geometric-mean PSD **after** cleaning.
    method_tag : str
        Short method identifier (e.g. ``"M1"``).
    subject : str
        Subject label (e.g. ``"sub-01"``).
    harmonics_hz : list of float | None
        Harmonic frequencies to zoom on.  Defaults to the first 3 from
        *metrics_dict* or ``[50, 100, 150]``.
    metrics_dict : dict | None
        Output of :func:`compute_all_qa_metrics`.  If provided, per-
        harmonic attenuation & R values are annotated.
    fmax : float
        Maximum frequency for the full-spectrum panel.
    method_colors, method_labels : dict | None
        Override palettes.
    save_path : str | Path | None
        Save figure to this path.
    show : bool
        Call ``plt.show()`` if *True*.

    Returns
    -------
    fig : Figure
    """
    if harmonics_hz is None:
        if metrics_dict and "harmonics_hz" in metrics_dict:
            harmonics_hz = metrics_dict["harmonics_hz"][:3]
        else:
            harmonics_hz = [50.0, 100.0, 150.0]

    n_harm = min(len(harmonics_hz), 3)
    n_cols = 1 + n_harm

    fig, axes = pub_figure(1, n_cols, figsize=(4 * n_cols, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    color = _method_color(method_tag, method_colors)
    label_str = _method_label(method_tag, method_labels)

    # Panel 0: full PSD
    ax = axes[0]
    ax.semilogy(
        freqs_before, gm_before, color=COLORS["before"], alpha=0.5, lw=1, label="Before"
    )
    ax.semilogy(
        freqs_after, gm_after, color=color, lw=1.5, label=f"After ({method_tag})"
    )
    for h in harmonics_hz:
        ax.axvline(h, color=COLORS["line_marker"], ls="--", alpha=0.2)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("PSD (V??/Hz)", fontsize=FONTS["label"])
    ax.set_title(f"{subject} ??? {label_str}", fontsize=FONTS["title"])
    pub_legend(ax)
    ax.set_xlim(0, fmax)
    style_axes(ax)

    # Panels 1+: harmonic zoom
    for i, hf in enumerate(harmonics_hz[:n_harm]):
        ax = axes[1 + i]
        zoom = 8  # Hz
        mb = (freqs_before >= hf - zoom) & (freqs_before <= hf + zoom)
        ma = (freqs_after >= hf - zoom) & (freqs_after <= hf + zoom)
        ax.semilogy(
            freqs_before[mb], gm_before[mb], color=COLORS["before"], alpha=0.5, lw=1
        )
        ax.semilogy(freqs_after[ma], gm_after[ma], color=color, lw=1.5)
        ax.axvline(hf, color=COLORS["line_marker"], ls="--", alpha=0.4)

        title = f"{hf:.0f} Hz"
        if metrics_dict:
            atten = metrics_dict.get("attenuation_per_harmonic_db", [])
            r_vals = metrics_dict.get("R_per_harmonic", [])
            if i < len(atten) and i < len(r_vals):
                title += f"\n??={atten[i]:.1f} dB, R={r_vals[i]:.2f}"
        ax.set_title(title, fontsize=FONTS["tick"])
        ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
        style_axes(ax)

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =====================================================================
# plot_psd_gallery ??? methods ?? harmonics grid
# =====================================================================


def plot_psd_gallery(
    freqs_before,
    gm_before,
    cleaned_psds,
    *,
    harmonics_hz=None,
    fmax=125.0,
    subject="",
    method_order=None,
    method_colors=None,
    method_labels=None,
    save_path=None,
    show=True,
):
    """Grid of PSD panels (methods ?? harmonics) for one or more subjects.

    Parameters
    ----------
    freqs_before : ndarray
        Frequency vector for the baseline PSD.
    gm_before : ndarray
        Geometric-mean PSD before cleaning.
    cleaned_psds : dict
        ``{method_tag: (freqs, gm_psd)}`` for each cleaning method.
    harmonics_hz : list of float | None
        Harmonic frequencies.  Default ``[50, 100, 150]``.
    fmax : float
        Max frequency for full-spectrum column.
    subject : str
        Subject label for the suptitle.
    method_order : list of str | None
        Order of rows. Defaults to sorted keys of *cleaned_psds*.
    method_colors, method_labels : dict | None
        Override palettes.
    save_path : str | Path | None
        Save figure to this path.
    show : bool
        Call ``plt.show()`` if *True*.

    Returns
    -------
    fig : Figure
    """
    if harmonics_hz is None:
        harmonics_hz = [50.0, 100.0, 150.0]
    if method_order is None:
        method_order = sorted(cleaned_psds.keys())

    n_rows = len(method_order)
    n_cols = 1 + len(harmonics_hz)

    fig, axes = pub_figure(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, mtag in enumerate(method_order):
        if mtag not in cleaned_psds:
            for col_j in range(n_cols):
                axes[row_i, col_j].text(
                    0.5,
                    0.5,
                    f"{mtag}\nno data",
                    transform=axes[row_i, col_j].transAxes,
                    ha="center",
                    va="center",
                    fontsize=FONTS["label"],
                )
                axes[row_i, col_j].axis("off")
            continue

        freqs_c, gm_c = cleaned_psds[mtag]
        color = _method_color(mtag, method_colors)
        label_str = _method_label(mtag, method_labels)

        # Column 0: Full PSD
        ax = axes[row_i, 0]
        ax.semilogy(
            freqs_before,
            gm_before,
            color=COLORS["before"],
            alpha=0.4,
            lw=0.8,
            label="Before",
        )
        ax.semilogy(freqs_c, gm_c, color=color, lw=1.2, label=label_str)
        for hf in harmonics_hz:
            ax.axvline(hf, color=COLORS["line_marker"], ls="--", alpha=0.15)
        ax.set_ylabel(mtag, fontsize=FONTS["title"], fontweight="bold")
        if row_i == 0:
            ax.set_title("Full PSD", fontsize=FONTS["title"])
        ax.set_xlim(0, fmax)
        pub_legend(ax, fontsize=6)
        style_axes(ax)

        # Columns 1+: harmonic zoom
        for col_j, hf in enumerate(harmonics_hz):
            ax = axes[row_i, 1 + col_j]
            zoom = 8
            mb = (freqs_before >= hf - zoom) & (freqs_before <= hf + zoom)
            mc = (freqs_c >= hf - zoom) & (freqs_c <= hf + zoom)
            ax.semilogy(
                freqs_before[mb],
                gm_before[mb],
                color=COLORS["before"],
                alpha=0.4,
                lw=0.8,
            )
            ax.semilogy(freqs_c[mc], gm_c[mc], color=color, lw=1.2)
            ax.axvline(hf, color=COLORS["line_marker"], ls="--", alpha=0.3)
            if row_i == 0:
                ax.set_title(f"{hf:.0f} Hz", fontsize=FONTS["title"])
            style_axes(ax)

    fig.suptitle(
        f"PSD Gallery ??? {subject}" if subject else "PSD Gallery",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =====================================================================
# plot_subject_psd_overlay ??? full spectrum + zoom (all methods on 1 plot)
# =====================================================================


def plot_subject_psd_overlay(
    freqs_before,
    gm_before,
    cleaned_psds,
    *,
    line_freq=50.0,
    fmax=125.0,
    n_harmonics=3,
    subject="",
    method_order=None,
    method_colors=None,
    method_labels=None,
    save_path=None,
    show=True,
):
    """Full-spectrum + fundamental-zoom PSD comparison for one subject.

    Parameters
    ----------
    freqs_before : ndarray
        Baseline frequency vector.
    gm_before : ndarray
        Baseline geometric-mean PSD.
    cleaned_psds : dict
        ``{method_tag: (freqs, gm_psd)}``.
    line_freq : float
        Fundamental line frequency (Hz).
    fmax : float
        Max frequency for full-spectrum panel.
    n_harmonics : int
        Number of harmonics to mark.
    subject : str
        Subject label.
    method_order, method_colors, method_labels : dict | None
        Display overrides.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    if method_order is None:
        method_order = sorted(cleaned_psds.keys())

    fig, axes = pub_figure(1, 2, figsize=(16, 5))

    # Left: full spectrum
    ax = axes[0]
    ax.semilogy(
        freqs_before, gm_before, color=COLORS["before"], alpha=0.4, lw=1, label="Before"
    )
    for mtag in method_order:
        if mtag == "M0" or mtag not in cleaned_psds:
            continue
        f, p = cleaned_psds[mtag]
        ax.semilogy(
            f,
            p,
            color=_method_color(mtag, method_colors),
            lw=1.2,
            label=_method_label(mtag, method_labels),
        )
    for h in range(1, n_harmonics + 2):
        hf = line_freq * h
        if hf < fmax:
            ax.axvline(hf, color=COLORS["line_marker"], ls="--", alpha=0.15)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Geometric-Mean PSD (V??/Hz)", fontsize=FONTS["label"])
    ax.set_title(f"{subject} ??? Full Spectrum Comparison", fontsize=FONTS["title"])
    pub_legend(ax)
    ax.set_xlim(0, fmax)
    style_axes(ax)

    # Right: zoom at fundamental
    ax = axes[1]
    zoom = 10
    mb = (freqs_before >= line_freq - zoom) & (freqs_before <= line_freq + zoom)
    ax.semilogy(
        freqs_before[mb],
        gm_before[mb],
        color=COLORS["before"],
        alpha=0.4,
        lw=1,
        label="Before",
    )
    for mtag in method_order:
        if mtag == "M0" or mtag not in cleaned_psds:
            continue
        f, p = cleaned_psds[mtag]
        m = (f >= line_freq - zoom) & (f <= line_freq + zoom)
        ax.semilogy(
            f[m],
            p[m],
            color=_method_color(mtag, method_colors),
            lw=1.5,
            label=_method_label(mtag, method_labels),
        )
    ax.axvline(line_freq, color=COLORS["line_marker"], ls="--", alpha=0.4)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_title(f"{subject} ??? Zoom at {line_freq} Hz", fontsize=FONTS["title"])
    pub_legend(ax)
    style_axes(ax)

    fig.suptitle(
        f"Per-Subject PSD Comparison ??? {subject}",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =====================================================================
# plot_metric_bars ??? group-mean ?? SEM bar chart
# =====================================================================


def plot_metric_bars(
    df, *, method_order=None, method_colors=None, save_path=None, show=True
):
    """Group-mean (?? SEM) bar chart for each QA metric.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``"method"`` plus the metric columns
        (``R_f0``, ``peak_attenuation_db``, etc.).
    method_order : list of str | None
        Order of bars.  Default ``DEFAULT_METHOD_ORDER``.
    method_colors : dict | None
        Override palette.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    if method_order is None:
        method_order = DEFAULT_METHOD_ORDER

    metric_cols = [
        "R_f0",
        "peak_attenuation_db",
        "below_noise_pct",
        "overclean_proportion",
        "underclean_proportion",
    ]
    metric_labels = [
        "R(f???)  ??? 1",
        "Peak Attenuation (dB)  ???",
        "Sub-Peak ??Power (%)  ??? 0",
        "Overclean Frac  ???",
        "Underclean Frac  ???",
    ]
    lower_better = [True, False, True, True, True]

    n_metrics = len(metric_cols)
    fig, axes = pub_figure(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = np.array([axes])

    for i, (col, label, lb) in enumerate(zip(metric_cols, metric_labels, lower_better)):
        ax = axes[i]
        means, sems = [], []
        for mtag in method_order:
            vals = df.loc[df["method"] == mtag, col].dropna()
            means.append(vals.mean() if len(vals) else 0)
            sems.append(vals.sem() if len(vals) > 1 else 0)

        x = np.arange(len(method_order))
        colors = [_method_color(m, method_colors) for m in method_order]
        bars = ax.bar(
            x,
            means,
            yerr=sems,
            color=colors,
            edgecolor="k",
            linewidth=0.5,
            capsize=3,
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, fontsize=FONTS["tick"])
        ax.set_ylabel(label, fontsize=FONTS["label"])
        style_axes(ax, grid=True)

        # Value annotations
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONTS["annotation"],
            )

        # Star best (skip M0)
        means_no_m0 = means[1:]
        if means_no_m0:
            if lb:
                best_idx = 1 + int(np.argmin(means_no_m0))
            else:
                best_idx = 1 + int(np.argmax(means_no_m0))
            ax.annotate(
                "???",
                xy=(best_idx, means[best_idx]),
                fontsize=14,
                ha="center",
                va="bottom",
                color="gold",
            )

    fig.suptitle(
        "Method Comparison (group mean ?? SEM)",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =====================================================================
# plot_tradeoff_scatter
# =====================================================================


def plot_tradeoff_scatter(
    df,
    *,
    method_order=None,
    method_colors=None,
    method_labels=None,
    save_path=None,
    show=True,
):
    """Attenuation-vs-distortion scatter with group-mean stars.

    The ideal method sits in the **upper-left** corner.

    Parameters
    ----------
    df : DataFrame
        Columns ``"method"``, ``"peak_attenuation_db"``,
        ``"below_noise_pct"``.
    method_order, method_colors, method_labels : dict | None
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    if method_order is None:
        method_order = DEFAULT_METHOD_ORDER

    fig, ax = pub_figure(1, 1, figsize=(8, 6))
    if isinstance(ax, np.ndarray):
        ax = ax.flat[0]

    for mtag in method_order:
        sub_df = df[df["method"] == mtag]
        c = _method_color(mtag, method_colors)
        lbl = _method_label(mtag, method_labels)
        ax.scatter(
            sub_df["below_noise_pct"],
            sub_df["peak_attenuation_db"],
            color=c,
            s=80,
            alpha=0.8,
            edgecolors="k",
            linewidth=0.5,
            label=f"{mtag} ({lbl})",
            zorder=3,
        )
        if len(sub_df) > 1:
            ax.scatter(
                sub_df["below_noise_pct"].mean(),
                sub_df["peak_attenuation_db"].mean(),
                color=c,
                s=200,
                marker="*",
                edgecolors="k",
                linewidth=1,
                zorder=4,
            )

    ax.set_xlabel(
        "Sub-Peak ??Power (%)  ???  closer to 0 is better", fontsize=FONTS["label"]
    )
    ax.set_ylabel(
        "Peak Attenuation (dB)  ???  higher is better", fontsize=FONTS["label"]
    )
    ax.set_title(
        "Attenuation vs Sub-Peak Distortion Trade-off",
        fontsize=FONTS["title"],
        fontweight="bold",
    )
    pub_legend(ax)
    style_axes(ax, grid=True)
    ax.axhline(10, color=COLORS["success"], ls=":", alpha=0.4)
    ax.axvline(0, color=COLORS["accent"], ls=":", alpha=0.4, label="No distortion")

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =====================================================================
# plot_r_comparison ??? bar or paired-dot R(f???)
# =====================================================================


def plot_r_comparison(
    df,
    *,
    method_order=None,
    method_colors=None,
    method_labels=None,
    save_path=None,
    show=True,
):
    """R(f???) bar chart (single subject) or paired-dot plot (multi-subject).

    Parameters
    ----------
    df : DataFrame
        Columns ``"subject"``, ``"method"``, ``"R_f0"``.
    method_order, method_colors, method_labels : dict | None
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    if method_order is None:
        method_order = DEFAULT_METHOD_ORDER

    fig, ax = pub_figure(1, 1, figsize=(8, 6))
    if isinstance(ax, np.ndarray):
        ax = ax.flat[0]

    multi = df["subject"].nunique() > 1

    if multi:
        for sub in df["subject"].unique():
            r_vals = []
            for mtag in method_order:
                v = df.loc[(df["subject"] == sub) & (df["method"] == mtag), "R_f0"]
                r_vals.append(v.iloc[0] if len(v) else float("nan"))
            ax.plot(
                range(len(method_order)),
                r_vals,
                "o-",
                color=COLORS["gray"],
                alpha=0.3,
                markersize=4,
            )
        means = [df.loc[df["method"] == m, "R_f0"].mean() for m in method_order]
        ax.plot(
            range(len(method_order)),
            means,
            "s-",
            color="k",
            markersize=8,
            lw=2,
            label="Group mean",
            zorder=5,
        )
    else:
        r_vals = []
        for m in method_order:
            v = df.loc[df["method"] == m, "R_f0"]
            r_vals.append(v.iloc[0] if len(v) else 0)
        x = np.arange(len(method_order))
        colors = [_method_color(m, method_colors) for m in method_order]
        ax.bar(x, r_vals, color=colors, edgecolor="k", linewidth=0.5, alpha=0.85)
        for xi, rv in zip(x, r_vals):
            ax.text(
                xi,
                rv,
                f"{rv:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONTS["annotation"],
            )

    ax.axhline(1.0, color=COLORS["success"], ls="--", alpha=0.5, label="Ideal (R=1)")
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, fontsize=FONTS["tick"])
    ax.set_ylabel("R(f???) ??? Noise-Surround Ratio", fontsize=FONTS["label"])
    ax.set_title(
        "Residual Line Noise ??? R(f???)", fontsize=FONTS["title"], fontweight="bold"
    )
    pub_legend(ax)
    style_axes(ax, grid=True)

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =====================================================================
# plot_harmonic_attenuation ??? grouped bars per harmonic
# =====================================================================


def plot_harmonic_attenuation(
    freqs_before,
    gm_before,
    cleaned_psds,
    harmonics_hz,
    *,
    subject="",
    method_order=None,
    method_colors=None,
    method_labels=None,
    save_path=None,
    show=True,
):
    """Per-harmonic peak-attenuation grouped-bar chart.

    Parameters
    ----------
    freqs_before : ndarray
        Baseline frequency vector.
    gm_before : ndarray
        Baseline geometric-mean PSD.
    cleaned_psds : dict
        ``{method_tag: (freqs, gm_psd)}``.
    harmonics_hz : list of float
        Harmonic frequencies to evaluate.
    subject : str
        Subject label.
    method_order, method_colors, method_labels : dict | None
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    from ..qa.metrics import peak_attenuation_db as _atten

    if method_order is None:
        method_order = [
            m for m in DEFAULT_METHOD_ORDER if m != "M0" and m in cleaned_psds
        ]

    fig, ax = pub_figure(1, 1, figsize=(10, 5))
    if isinstance(ax, np.ndarray):
        ax = ax.flat[0]

    bar_width = 0.8 / max(len(method_order), 1)
    x = np.arange(len(harmonics_hz))

    for j, mtag in enumerate(method_order):
        if mtag not in cleaned_psds:
            continue
        freqs_c, gm_c = cleaned_psds[mtag]
        atten_per_h = [_atten(freqs_before, gm_before, gm_c, hf) for hf in harmonics_hz]
        ax.bar(
            x + j * bar_width,
            atten_per_h,
            bar_width,
            color=_method_color(mtag, method_colors),
            edgecolor="k",
            linewidth=0.3,
            label=_method_label(mtag, method_labels),
            alpha=0.85,
        )

    ax.set_xticks(x + bar_width * (len(method_order) - 1) / 2)
    ax.set_xticklabels([f"{h:.0f} Hz" for h in harmonics_hz], fontsize=FONTS["tick"])
    ax.set_ylabel("Peak Attenuation (dB)", fontsize=FONTS["label"])
    title = (
        f"Per-Harmonic Attenuation ??? {subject}"
        if subject
        else "Per-Harmonic Attenuation"
    )
    ax.set_title(title, fontsize=FONTS["title"], fontweight="bold")
    pub_legend(ax)
    style_axes(ax, grid=True)

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =====================================================================
# plot_paired_metrics ??? multi-subject paired-dot plots
# =====================================================================


def plot_paired_metrics(
    df, *, method_order=None, method_colors=None, save_path=None, show=True
):
    """Paired dot plots for key metrics across subjects.

    Requires at least 2 subjects.

    Parameters
    ----------
    df : DataFrame
        Columns ``"subject"``, ``"method"``, and metric columns.
    method_order, method_colors : dict | None
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure | None
        *None* if fewer than 2 subjects.
    """
    if df["subject"].nunique() < 2:
        return None

    if method_order is None:
        method_order = DEFAULT_METHOD_ORDER

    pair_metrics = [
        ("peak_attenuation_db", "Peak Attenuation (dB)"),
        ("below_noise_pct", "Sub-Peak ??Power (%)"),
        ("R_f0", "R(f???)"),
    ]

    fig, axes = pub_figure(1, len(pair_metrics), figsize=(6 * len(pair_metrics), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (col, label) in zip(axes.flat, pair_metrics):
        for sub in df["subject"].unique():
            sub_vals = []
            for mtag in method_order:
                v = df.loc[(df["subject"] == sub) & (df["method"] == mtag), col]
                sub_vals.append(v.iloc[0] if len(v) else float("nan"))
            ax.plot(
                range(len(method_order)),
                sub_vals,
                "o-",
                color=COLORS["gray"],
                alpha=0.3,
                markersize=4,
            )

        means = [df.loc[df["method"] == m, col].mean() for m in method_order]
        ax.plot(
            range(len(method_order)),
            means,
            "s-",
            color="k",
            markersize=8,
            lw=2,
            label="Group mean",
            zorder=5,
        )

        ax.set_xticks(range(len(method_order)))
        ax.set_xticklabels(method_order, fontsize=FONTS["tick"])
        ax.set_ylabel(label, fontsize=FONTS["label"])
        pub_legend(ax)
        style_axes(ax, grid=True)

    fig.suptitle(
        "Paired Subject-Level Comparison", fontsize=FONTS["suptitle"], fontweight="bold"
    )
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =====================================================================
# plot_tradeoff_and_r ??? combined 2-panel figure (scatter + R)
# =====================================================================


def plot_tradeoff_and_r(
    df,
    *,
    method_order=None,
    method_colors=None,
    method_labels=None,
    save_path=None,
    show=True,
):
    """Two-panel figure: attenuation-vs-distortion + R(f???).

    Convenience wrapper that combines :func:`plot_tradeoff_scatter` and
    :func:`plot_r_comparison` into a single figure.

    Parameters
    ----------
    df : DataFrame
    method_order, method_colors, method_labels : dict | None
    save_path : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    if method_order is None:
        method_order = DEFAULT_METHOD_ORDER

    fig, axes = pub_figure(1, 2, figsize=(16, 6))

    # ?????? Left: scatter ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    ax = axes[0]
    for mtag in method_order:
        sub_df = df[df["method"] == mtag]
        c = _method_color(mtag, method_colors)
        lbl = _method_label(mtag, method_labels)
        ax.scatter(
            sub_df["below_noise_pct"],
            sub_df["peak_attenuation_db"],
            color=c,
            s=80,
            alpha=0.8,
            edgecolors="k",
            linewidth=0.5,
            label=f"{mtag} ({lbl})",
            zorder=3,
        )
        if len(sub_df) > 1:
            ax.scatter(
                sub_df["below_noise_pct"].mean(),
                sub_df["peak_attenuation_db"].mean(),
                color=c,
                s=200,
                marker="*",
                edgecolors="k",
                linewidth=1,
                zorder=4,
            )
    ax.set_xlabel(
        "Sub-Peak ??Power (%)  ???  closer to 0 is better", fontsize=FONTS["label"]
    )
    ax.set_ylabel(
        "Peak Attenuation (dB)  ???  higher is better", fontsize=FONTS["label"]
    )
    ax.set_title(
        "Attenuation vs Sub-Peak Distortion Trade-off",
        fontsize=FONTS["title"],
        fontweight="bold",
    )
    pub_legend(ax)
    style_axes(ax, grid=True)
    ax.axhline(10, color=COLORS["success"], ls=":", alpha=0.4)
    ax.axvline(0, color=COLORS["accent"], ls=":", alpha=0.4, label="No distortion")

    # ?????? Right: R(f???) ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    ax = axes[1]
    multi = df["subject"].nunique() > 1
    if multi:
        for sub in df["subject"].unique():
            r_vals = []
            for mtag in method_order:
                v = df.loc[(df["subject"] == sub) & (df["method"] == mtag), "R_f0"]
                r_vals.append(v.iloc[0] if len(v) else float("nan"))
            ax.plot(
                range(len(method_order)),
                r_vals,
                "o-",
                color=COLORS["gray"],
                alpha=0.3,
                markersize=4,
            )
        means = [df.loc[df["method"] == m, "R_f0"].mean() for m in method_order]
        ax.plot(
            range(len(method_order)),
            means,
            "s-",
            color="k",
            markersize=8,
            lw=2,
            label="Group mean",
            zorder=5,
        )
    else:
        r_vals = []
        for m in method_order:
            v = df.loc[df["method"] == m, "R_f0"]
            r_vals.append(v.iloc[0] if len(v) else 0)
        x = np.arange(len(method_order))
        colors = [_method_color(m, method_colors) for m in method_order]
        ax.bar(x, r_vals, color=colors, edgecolor="k", linewidth=0.5, alpha=0.85)
        for xi, rv in zip(x, r_vals):
            ax.text(
                xi,
                rv,
                f"{rv:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONTS["annotation"],
            )

    ax.axhline(1.0, color=COLORS["success"], ls="--", alpha=0.5, label="Ideal (R=1)")
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, fontsize=FONTS["tick"])
    ax.set_ylabel("R(f???) ??? Noise-Surround Ratio", fontsize=FONTS["label"])
    ax.set_title(
        "Residual Line Noise ??? R(f???)", fontsize=FONTS["title"], fontweight="bold"
    )
    pub_legend(ax)
    style_axes(ax, grid=True)

    fig.suptitle("Trade-off Analysis", fontsize=FONTS["suptitle"], fontweight="bold")
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
