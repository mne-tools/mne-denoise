"""Visualization helpers for grouped metrics and summary statistics.

This module provides reusable, study-agnostic metric plots for grouped
comparisons, paired subject trajectories, and distribution summaries.

Input model
-----------
Grouped-stat functions in this module assume column-oriented input:

1. Mapping-like object with ``.items()`` (for example: ``dict``).
2. Columns should be 1D and aligned by row.
3. Metric columns should be numeric when used in computations.

Public plots
------------
1. :func:`plot_metric_bars`
2. :func:`plot_tradeoff_scatter`
3. :func:`plot_metric_comparison`
4. :func:`plot_metric_slopes`
5. :func:`plot_metric_violins`
6. :func:`plot_null_distribution`
7. :func:`plot_forest`
8. :func:`plot_harmonic_attenuation` (line-noise-specific helper)

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from ..qa import peak_attenuation_db
from ._seaborn import _suppress_seaborn_plot_warnings, _try_import_seaborn
from .theme import (
    COLORS,
    FONTS,
    _finalize_fig,
    get_series_color,
    style_axes,
    themed_figure,
    themed_legend,
    use_theme,
)

_STATS_STYLE = MappingProxyType(
    {
        "bar_alpha": 0.85,
        "bar_linewidth": 0.5,
        "bar_capsize": 3,
        "scatter_size": 80,
        "scatter_alpha": 0.8,
        "scatter_edge_linewidth": 0.5,
        "mean_scatter_size": 200,
        "mean_marker_size": 8,
        "mean_linewidth": 2.0,
        "subject_trace_alpha": 0.3,
        "subject_trace_marker_size": 4,
        "paired_line_alpha": 0.1,
        "paired_linewidth": 0.4,
        "reference_linewidth": 0.8,
        "reference_alpha": 0.5,
        "annotation_star_size": 14,
        "strip_size": 3,
        "strip_alpha": 0.7,
        "strip_jitter": 0.12,
        "forest_marker_size": 5,
        "forest_baseline_mean_marker_size": 9,
        "forest_pooled_marker_size": 10,
        "hist_alpha": 0.5,
        "hist_linewidth": 0.5,
        "legend_fontsize_small": 7,
    }
)


def _plot_subject_trajectories(
    ax,
    subject_values,
    groups,
    metric_values,
    group_order,
    *,
    style="o-",
    alpha=None,
    linewidth=None,
    markersize=None,
    zorder=None,
):
    """Plot paired subject trajectories and return finite group means."""
    subjects = list(dict.fromkeys(np.asarray(subject_values, dtype=object).tolist()))
    traces = np.full((len(subjects), len(group_order)), np.nan, dtype=float)
    for subject_idx, subject in enumerate(subjects):
        subject_mask = subject_values == subject
        for group_idx, group in enumerate(group_order):
            vals = metric_values[subject_mask & (groups == group)]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                traces[subject_idx, group_idx] = vals[0]
        kws = {
            "color": COLORS["stat_subject"],
            "alpha": _STATS_STYLE["subject_trace_alpha"] if alpha is None else alpha,
        }
        if markersize is not None:
            kws["markersize"] = markersize
        if linewidth is not None:
            kws["lw"] = linewidth
        if zorder is not None:
            kws["zorder"] = zorder
        ax.plot(range(len(group_order)), traces[subject_idx], style, **kws)
    return np.nanmean(traces, axis=0)


def plot_metric_bars(
    data,
    metric_cols,
    metric_labels=None,
    lower_better=None,
    group_col="group",
    group_order=None,
    group_colors=None,
    group_labels=None,
    title="Metric Comparison (group mean ± SEM)",
    fname=None,
    show=True,
):
    """Plot grouped bar charts for one or more scalar metrics.

    Parameters
    ----------
    data : mapping of str to array-like
        Columnar data mapping. All columns must be 1D and have equal length.
    metric_cols : list of str
        Metric columns to visualize.
    metric_labels : list of str | None
        Axis labels corresponding to ``metric_cols``. If None, labels are
        derived from metric names.
    lower_better : list of bool | None
        Whether smaller values indicate better performance for each metric.
        If None, no best-marker star is added.
    group_col : str
        Column name identifying comparison groups.
    group_order : list of str | None
        Explicit order for group bars. If None, first-seen order is used.
    group_colors, group_labels : dict | None
        Optional color/label overrides keyed by group name.
    title : str
        Figure-level title.
    fname : path-like | None
        Optional output path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_metric_bars
    >>> data = {
    ...     "group": np.array(["A", "A", "B", "B"]),
    ...     "score": np.array([0.9, 1.0, 0.7, 0.8]),
    ... }
    >>> fig = plot_metric_bars(
    ...     data, metric_cols=["score"], group_col="group", show=False
    ... )
    """
    columns = {name: np.asarray(values) for name, values in data.items()}
    groups = np.asarray(columns[group_col], dtype=object)

    metric_cols = list(metric_cols)

    if metric_labels is None:
        metric_labels = [str(col).replace("_", " ").strip() for col in metric_cols]
    else:
        metric_labels = list(metric_labels)
    if lower_better is None:
        lower_better = [None] * len(metric_cols)
    else:
        lower_better = list(lower_better)

    if group_order is None:
        group_order = list(dict.fromkeys(groups.tolist()))
    else:
        group_order = list(group_order)

    with use_theme():
        n_metrics = len(metric_cols)
        fig, axes = themed_figure(1, n_metrics, figsize=(4 * n_metrics, 5))
        if n_metrics == 1:
            axes = np.array([axes])

        for axis_index, (col, label, is_lower_better) in enumerate(
            zip(metric_cols, metric_labels, lower_better)
        ):
            ax = axes[axis_index]
            metric_values = np.asarray(columns[col], dtype=float)
            means, sems = [], []
            for group in group_order:
                vals = metric_values[groups == group]
                vals = vals[np.isfinite(vals)]
                means.append(vals.mean() if vals.size else np.nan)
                if vals.size > 1:
                    sems.append(float(vals.std(ddof=1) / np.sqrt(vals.size)))
                else:
                    sems.append(0.0)

            x = np.arange(len(group_order))
            colors = [
                group_colors[group]
                if group_colors and group in group_colors
                else get_series_color(idx)
                for idx, group in enumerate(group_order)
            ]
            bars = ax.bar(
                x,
                means,
                yerr=sems,
                color=colors,
                edgecolor=COLORS["edge"],
                linewidth=_STATS_STYLE["bar_linewidth"],
                capsize=_STATS_STYLE["bar_capsize"],
                alpha=_STATS_STYLE["bar_alpha"],
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    group_labels[group]
                    if group_labels and group in group_labels
                    else group
                    for group in group_order
                ],
                fontsize=FONTS["tick"],
            )
            ax.set_ylabel(label, fontsize=FONTS["label"])
            style_axes(ax, grid=True)

            for bar, mean_value in zip(bars, means):
                if np.isnan(mean_value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_value,
                    f"{mean_value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONTS["annotation"],
                )

            finite_means = np.asarray(means, dtype=float)
            if is_lower_better is not None and np.isfinite(finite_means).any():
                best_index = (
                    int(np.nanargmin(finite_means))
                    if is_lower_better
                    else int(np.nanargmax(finite_means))
                )
                ax.annotate(
                    "★",
                    xy=(best_index, finite_means[best_index]),
                    fontsize=_STATS_STYLE["annotation_star_size"],
                    ha="center",
                    va="bottom",
                    color=COLORS["stat_highlight"],
                )

        fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
        return _finalize_fig(fig, show=show, fname=fname)


def plot_tradeoff_scatter(
    data,
    x_col,
    y_col,
    group_col="group",
    group_order=None,
    group_colors=None,
    group_labels=None,
    x_label=None,
    y_label=None,
    title="Metric Trade-off",
    reference_x=None,
    reference_y=None,
    ax=None,
    fname=None,
    show=True,
):
    """Plot a grouped x/y trade-off scatter with optional group means.

    Parameters
    ----------
    data : mapping of str to array-like
        Columnar mapping with group and metric columns.
    x_col, y_col : str
        Metric columns for x and y axes.
    group_col : str
        Grouping column name.
    group_order : list of str | None
        Optional group order. If None, first-seen order is used.
    group_colors, group_labels : dict | None
        Optional style overrides keyed by group name.
    x_label, y_label : str | None
        Axis labels. If None, derived from metric names.
    title : str
        Axes title.
    reference_x, reference_y : float | None
        Optional vertical/horizontal reference lines.
    ax : matplotlib.axes.Axes | None
        Existing axes. If None, create a new figure.
    fname : path-like | None
        Optional output path when creating a new figure.
    show : bool
        Whether to display the figure when creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_tradeoff_scatter
    >>> data = {
    ...     "group": np.array(["A", "A", "B", "B"]),
    ...     "distortion": np.array([0.1, 0.2, 0.4, 0.3]),
    ...     "attenuation": np.array([8.0, 9.0, 5.0, 6.0]),
    ... }
    >>> fig = plot_tradeoff_scatter(
    ...     data, group_col="group", x_col="distortion", y_col="attenuation", show=False
    ... )
    """
    columns = {name: np.asarray(values) for name, values in data.items()}
    groups = np.asarray(columns[group_col], dtype=object)

    x = np.asarray(columns[x_col], dtype=float)
    y = np.asarray(columns[y_col], dtype=float)

    if x_label is None:
        x_label = str(x_col).replace("_", " ").strip()
    if y_label is None:
        y_label = str(y_col).replace("_", " ").strip()

    if group_order is None:
        group_order = list(dict.fromkeys(groups.tolist()))
    else:
        group_order = list(group_order)

    with use_theme():
        finalize = ax is None
        if ax is None:
            fig, ax = themed_figure(1, 1, figsize=(8, 6))
            if isinstance(ax, np.ndarray):
                ax = ax.flat[0]
        else:
            fig = ax.figure

        for idx, group in enumerate(group_order):
            mask = groups == group
            x_group = x[mask]
            y_group = y[mask]
            finite_mask = np.isfinite(x_group) & np.isfinite(y_group)
            x_group = x_group[finite_mask]
            y_group = y_group[finite_mask]
            color = (
                group_colors[group]
                if group_colors and group in group_colors
                else get_series_color(idx)
            )
            label = (
                group_labels[group] if group_labels and group in group_labels else group
            )

            ax.scatter(
                x_group,
                y_group,
                color=color,
                s=_STATS_STYLE["scatter_size"],
                alpha=_STATS_STYLE["scatter_alpha"],
                edgecolors=COLORS["edge"],
                linewidth=_STATS_STYLE["scatter_edge_linewidth"],
                label=label,
                zorder=3,
            )
            if x_group.size > 1:
                ax.scatter(
                    x_group.mean(),
                    y_group.mean(),
                    color=color,
                    s=_STATS_STYLE["mean_scatter_size"],
                    marker="*",
                    edgecolors=COLORS["edge"],
                    linewidth=_STATS_STYLE["mean_linewidth"] / 2,
                    zorder=4,
                )

        ax.set_xlabel(x_label, fontsize=FONTS["label"])
        ax.set_ylabel(y_label, fontsize=FONTS["label"])
        ax.set_title(title, fontsize=FONTS["title"], fontweight="bold")
        if reference_y is not None:
            ax.axhline(
                reference_y,
                color=COLORS["stat_reference"],
                ls=":",
                lw=_STATS_STYLE["reference_linewidth"],
                alpha=_STATS_STYLE["reference_alpha"],
            )
        if reference_x is not None:
            ax.axvline(
                reference_x,
                color=COLORS["stat_reference"],
                ls=":",
                lw=_STATS_STYLE["reference_linewidth"],
                alpha=_STATS_STYLE["reference_alpha"],
            )
        themed_legend(ax)
        style_axes(ax, grid=True)

        if finalize:
            return _finalize_fig(fig, show=show, fname=fname)
        return fig


def plot_metric_comparison(
    data,
    metric_col,
    metric_label=None,
    group_col="group",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    title="Metric Comparison",
    reference_value=None,
    reference_label="Reference",
    ax=None,
    fname=None,
    show=True,
):
    """Plot one metric as grouped bars or paired subject trajectories.

    Parameters
    ----------
    data : mapping of str to array-like
        Columnar mapping with at least ``group_col`` and one numeric metric.
    metric_col : str
        Metric column to visualize.
    metric_label : str | None
        Y-axis label. If None, derived from ``metric_col``.
    group_col : str
        Grouping column name.
    subject_col : str
        Subject identifier column for paired overlays.
    group_order : list of str | None
        Optional explicit group order. If None, first-seen order is used.
    group_colors, group_labels : dict | None
        Optional style overrides keyed by group.
    title : str
        Axes title.
    reference_value : float | None
        Optional horizontal reference line.
    reference_label : str
        Legend label for ``reference_value``.
    ax : matplotlib.axes.Axes | None
        Existing axes. If None, create a new figure.
    fname : path-like | None
        Optional output path when creating a new figure.
    show : bool
        Whether to display the figure when creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_metric_comparison
    >>> data = {
    ...     "subject": np.array(["s1", "s1", "s2", "s2"]),
    ...     "group": np.array(["A", "B", "A", "B"]),
    ...     "score": np.array([1.1, 0.8, 1.0, 0.7]),
    ... }
    >>> fig = plot_metric_comparison(
    ...     data,
    ...     group_col="group",
    ...     subject_col="subject",
    ...     metric_col="score",
    ...     show=False,
    ... )
    """
    columns = {name: np.asarray(values) for name, values in data.items()}
    groups = np.asarray(columns[group_col], dtype=object)
    subjects = np.asarray(columns[subject_col], dtype=object)
    metric_values = np.asarray(columns[metric_col], dtype=float)

    if metric_label is None:
        metric_label = str(metric_col).replace("_", " ").strip()

    if group_order is None:
        group_order = list(dict.fromkeys(groups.tolist()))
    else:
        group_order = list(group_order)

    with use_theme():
        finalize = ax is None
        if ax is None:
            fig, ax = themed_figure(1, 1, figsize=(8, 6))
            if isinstance(ax, np.ndarray):
                ax = ax.flat[0]
        else:
            fig = ax.figure

        multi_subject = len(list(dict.fromkeys(subjects.tolist()))) > 1

        if multi_subject:
            means = _plot_subject_trajectories(
                ax,
                subjects,
                groups,
                metric_values,
                group_order,
                style="o-",
                markersize=_STATS_STYLE["subject_trace_marker_size"],
            )
            ax.plot(
                range(len(group_order)),
                means,
                "s-",
                color=COLORS["stat_mean"],
                markersize=_STATS_STYLE["mean_marker_size"],
                lw=_STATS_STYLE["mean_linewidth"],
                label="Group mean",
                zorder=5,
            )
        else:
            metric_vals = []
            for group in group_order:
                values = metric_values[groups == group]
                values = values[np.isfinite(values)]
                metric_vals.append(values[0] if values.size else np.nan)
            x = np.arange(len(group_order))
            colors = [
                group_colors[group]
                if group_colors and group in group_colors
                else get_series_color(idx)
                for idx, group in enumerate(group_order)
            ]
            ax.bar(
                x,
                metric_vals,
                color=colors,
                edgecolor=COLORS["edge"],
                linewidth=_STATS_STYLE["bar_linewidth"],
                alpha=_STATS_STYLE["bar_alpha"],
            )
            for xi, metric_value in zip(x, metric_vals):
                if np.isnan(metric_value):
                    continue
                ax.text(
                    xi,
                    metric_value,
                    f"{metric_value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONTS["annotation"],
                )

        if reference_value is not None:
            ax.axhline(
                reference_value,
                color=COLORS["stat_reference"],
                ls="--",
                lw=_STATS_STYLE["reference_linewidth"],
                alpha=_STATS_STYLE["reference_alpha"],
                label=reference_label,
            )
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(
            [
                group_labels[group] if group_labels and group in group_labels else group
                for group in group_order
            ],
            fontsize=FONTS["tick"],
        )
        ax.set_ylabel(metric_label, fontsize=FONTS["label"])
        ax.set_title(title, fontsize=FONTS["title"], fontweight="bold")
        themed_legend(ax)
        style_axes(ax, grid=True)

        if finalize:
            return _finalize_fig(fig, show=show, fname=fname)
        return fig


def plot_harmonic_attenuation(
    freqs_before,
    gm_before,
    cleaned_psds,
    harmonics_hz,
    subject="",
    series_order=None,
    series_colors=None,
    series_labels=None,
    title=None,
    fname=None,
    show=True,
):
    """Plot grouped per-harmonic attenuation bars for line-noise studies.

    Parameters
    ----------
    freqs_before : array-like
        Frequency axis of the reference PSD.
    gm_before : array-like
        Reference geometric-mean PSD.
    cleaned_psds : dict[str, tuple[array-like, array-like]]
        Mapping from series name to ``(freqs, psd)`` after denoising.
    harmonics_hz : array-like of float
        Harmonic frequencies to evaluate.
    subject : str
        Optional subject label included in default title.
    series_order : list[str] | None
        Plotting order for series. If None, keys from ``cleaned_psds`` are used.
    series_colors, series_labels : dict | None
        Optional color/label overrides keyed by series name.
    title : str | None
        Custom axes title.
    fname : path-like | None
        Optional output path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Notes
    -----
    This helper is intentionally domain-specific (line-frequency harmonics)
    and complements the otherwise study-agnostic grouped-stat plots.
    """
    if series_order is None:
        series_order = list(cleaned_psds.keys())

    with use_theme():
        fig, ax = themed_figure(1, 1, figsize=(10, 5))
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]

        bar_width = 0.8 / max(len(series_order), 1)
        x = np.arange(len(harmonics_hz))

        for idx, series_name in enumerate(series_order):
            if series_name not in cleaned_psds:
                continue
            _, gm_clean = cleaned_psds[series_name]
            attenuation = [
                peak_attenuation_db(freqs_before, gm_before, gm_clean, harmonic)
                for harmonic in harmonics_hz
            ]
            ax.bar(
                x + idx * bar_width,
                attenuation,
                bar_width,
                color=series_colors[series_name]
                if series_colors and series_name in series_colors
                else get_series_color(idx),
                edgecolor=COLORS["edge"],
                linewidth=_STATS_STYLE["bar_linewidth"],
                label=series_labels[series_name]
                if series_labels and series_name in series_labels
                else series_name,
                alpha=_STATS_STYLE["bar_alpha"],
            )

        ax.set_xticks(x + bar_width * (len(series_order) - 1) / 2)
        ax.set_xticklabels(
            [f"{harmonic:.0f} Hz" for harmonic in harmonics_hz],
            fontsize=FONTS["tick"],
        )
        ax.set_ylabel("Peak Attenuation (dB)", fontsize=FONTS["label"])
        if title is None:
            title = (
                f"Per-Harmonic Attenuation — {subject}"
                if subject
                else "Per-Harmonic Attenuation"
            )
        ax.set_title(title, fontsize=FONTS["title"], fontweight="bold")
        themed_legend(ax)
        style_axes(ax, grid=True)

        return _finalize_fig(fig, show=show, fname=fname)


def plot_metric_slopes(
    data,
    metric_cols=None,
    metric_labels=None,
    metric_specs=None,
    group_col="group",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    reference_lines=None,
    suptitle=None,
    title="Paired Subject-Level Comparison",
    fname=None,
    show=True,
):
    """Plot subject-level paired trajectories for one or more metrics.

    Parameters
    ----------
    data : mapping of str to array-like
        Columnar mapping with subject/group identifiers and metric columns.
    metric_cols : list of str | None
        Metric columns to plot. Used only when ``metric_specs`` is None.
    metric_labels : list of str | None
        Display labels aligned with ``metric_cols``.
    metric_specs : list[tuple[str, str]] | None
        Explicit list of ``(metric_col, metric_label)`` pairs.
    group_col : str
        Grouping column name.
    subject_col : str
        Subject identifier column name.
    group_order : list of str | None
        Optional group order. If None, first-seen order is used.
    group_colors, group_labels : dict | None
        Optional style overrides keyed by group.
    reference_lines : dict | None
        Optional horizontal reference lines per metric:
        ``{metric_col: [(y_value, style_dict), ...]}``.
    suptitle, title : str | None
        Figure title. ``suptitle`` overrides ``title`` when provided.
    fname : path-like | None
        Optional output path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_metric_slopes
    >>> data = {
    ...     "subject": np.array(["s1", "s1", "s2", "s2"]),
    ...     "group": np.array(["A", "B", "A", "B"]),
    ...     "metric": np.array([1.0, 0.8, 1.1, 0.7]),
    ... }
    >>> fig = plot_metric_slopes(
    ...     data, metric_cols=["metric"], group_col="group", show=False
    ... )
    """
    columns = {name: np.asarray(values) for name, values in data.items()}
    subject_values = np.asarray(columns[subject_col], dtype=object)
    groups = np.asarray(columns[group_col], dtype=object)
    if metric_specs is None:
        metric_cols = list(metric_cols)
        if metric_labels is None:
            metric_labels = [
                str(metric_col).replace("_", " ").strip() for metric_col in metric_cols
            ]
        else:
            metric_labels = list(metric_labels)
        metric_specs = list(zip(metric_cols, metric_labels))
    else:
        metric_specs = list(metric_specs)

    if group_order is None:
        group_order = list(dict.fromkeys(groups.tolist()))
    else:
        group_order = list(group_order)

    with use_theme():
        fig, axes = themed_figure(
            1, len(metric_specs), figsize=(6 * len(metric_specs), 5)
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        tick_labels = [
            group_labels[group] if group_labels and group in group_labels else group
            for group in group_order
        ]

        for ax, (metric_col, metric_label) in zip(axes.flat, metric_specs):
            metric = np.asarray(columns[metric_col], dtype=float)
            means = _plot_subject_trajectories(
                ax,
                subject_values,
                groups,
                metric,
                group_order,
                style="o-",
                markersize=_STATS_STYLE["subject_trace_marker_size"],
            )

            ax.plot(
                range(len(group_order)),
                means,
                "s-",
                color=COLORS["stat_mean"],
                markersize=_STATS_STYLE["mean_marker_size"],
                lw=_STATS_STYLE["mean_linewidth"],
                label="Group mean",
                zorder=5,
            )

            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(tick_labels, fontsize=FONTS["tick"])
            ax.set_ylabel(metric_label, fontsize=FONTS["label"])
            if reference_lines and metric_col in reference_lines:
                for y_val, style in reference_lines[metric_col]:
                    ax.axhline(y_val, **style)
            themed_legend(ax)
            style_axes(ax, grid=True)

        fig.suptitle(
            suptitle or title,
            fontsize=FONTS["suptitle"],
            fontweight="bold",
        )
        return _finalize_fig(fig, show=show, fname=fname)


def plot_metric_violins(
    data,
    metric_cols,
    metric_labels=None,
    group_col="group",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    baseline_group=None,
    reference_lines=None,
    show_paired=True,
    suptitle=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot violin + strip distributions with optional paired subject lines.

    Parameters
    ----------
    data : mapping of str to array-like
        Columnar mapping with group/subject columns and one or more metrics.
    metric_cols : list of str
        Metric columns to render.
    metric_labels : list of str | None
        Labels corresponding to ``metric_cols``.
    group_col : str
        Grouping column name.
    subject_col : str
        Subject identifier column name.
    group_order : list of str | None
        Optional group order. If None, first-seen order is used.
    group_colors, group_labels : dict | None
        Optional style overrides keyed by group.
    baseline_group : str | None
        Optional group used to draw a baseline mean line.
    reference_lines : dict | None
        Optional horizontal reference lines per metric.
    show_paired : bool
        Whether to draw subject-level paired lines.
    suptitle : str | None
        Figure-level title.
    figsize : tuple | None
        Figure size in inches. Defaults to ``(4 * n_metrics, 5.5)``.
    fname : path-like | None
        Optional output path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ImportError
        If seaborn is unavailable.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_metric_violins
    >>> data = {
    ...     "subject": np.array(["s1", "s1", "s2", "s2"]),
    ...     "group": np.array(["A", "B", "A", "B"]),
    ...     "metric": np.array([0.2, 0.6, 0.1, 0.5]),
    ... }
    >>> fig = plot_metric_violins(
    ...     data, ["metric"], group_col="group", subject_col="subject", show=False
    ... )
    """
    columns = {name: np.asarray(values) for name, values in data.items()}
    sns = _try_import_seaborn()

    metric_cols = list(metric_cols)
    if metric_labels is None:
        metric_labels = [
            str(metric_col).replace("_", " ").strip() for metric_col in metric_cols
        ]
    else:
        metric_labels = list(metric_labels)
    groups = np.asarray(columns[group_col], dtype=object)
    subject_values = np.asarray(columns[subject_col], dtype=object)
    if group_order is None:
        group_order = list(dict.fromkeys(groups.tolist()))
    else:
        group_order = list(group_order)

    n_metrics = len(metric_cols)
    n_subjects = len(list(dict.fromkeys(subject_values.tolist())))
    if figsize is None:
        figsize = (4 * n_metrics, 5.5)

    with use_theme():
        fig, axes = themed_figure(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])

        pretty_order = [
            group_labels[group] if group_labels and group in group_labels else group
            for group in group_order
        ]
        palette = {
            (
                group_labels[group] if group_labels and group in group_labels else group
            ): (
                group_colors[group]
                if group_colors and group in group_colors
                else get_series_color(idx)
            )
            for idx, group in enumerate(group_order)
        }

        for ax, metric_col, metric_label in zip(axes.flat, metric_cols, metric_labels):
            metric = np.asarray(columns[metric_col], dtype=float)
            x_vals = []
            y_vals = []
            for group in group_order:
                group_vals = metric[groups == group]
                group_vals = group_vals[np.isfinite(group_vals)]
                if group_vals.size == 0:
                    continue
                group_name = (
                    group_labels[group]
                    if group_labels and group in group_labels
                    else group
                )
                x_vals.extend([group_name] * group_vals.size)
                y_vals.extend(group_vals.tolist())

            if not y_vals:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=FONTS["label"],
                )
                style_axes(ax)
                continue

            x_arr = np.asarray(x_vals, dtype=object)
            y_arr = np.asarray(y_vals, dtype=float)

            with _suppress_seaborn_plot_warnings():
                sns.violinplot(
                    x=x_arr,
                    y=y_arr,
                    hue=x_arr,
                    order=pretty_order,
                    hue_order=pretty_order,
                    palette=palette,
                    inner=None,
                    linewidth=_STATS_STYLE["reference_linewidth"],
                    alpha=_STATS_STYLE["subject_trace_alpha"],
                    ax=ax,
                    cut=0,
                    density_norm="width",
                    legend=False,
                )
                sns.stripplot(
                    x=x_arr,
                    y=y_arr,
                    hue=x_arr,
                    order=pretty_order,
                    hue_order=pretty_order,
                    palette=palette,
                    size=_STATS_STYLE["strip_size"],
                    alpha=_STATS_STYLE["strip_alpha"],
                    jitter=_STATS_STYLE["strip_jitter"],
                    ax=ax,
                    zorder=5,
                    legend=False,
                )

            if show_paired and n_subjects > 1:
                _plot_subject_trajectories(
                    ax,
                    subject_values,
                    groups,
                    metric,
                    group_order,
                    style="-",
                    alpha=_STATS_STYLE["paired_line_alpha"],
                    linewidth=_STATS_STYLE["paired_linewidth"],
                    zorder=1,
                )

            if baseline_group is not None:
                base_values = metric[groups == baseline_group]
                base_values = base_values[np.isfinite(base_values)]
            else:
                base_values = np.array([])
            if base_values.size:
                ax.axhline(
                    base_values.mean(),
                    color=COLORS["stat_reference"],
                    ls="--",
                    lw=_STATS_STYLE["reference_linewidth"],
                    alpha=_STATS_STYLE["reference_alpha"],
                )

            if reference_lines and metric_col in reference_lines:
                for y_val, style in reference_lines[metric_col]:
                    ax.axhline(y_val, **style)

            ax.set_xlabel("")
            ax.set_ylabel(metric_label, fontsize=FONTS["label"])
            ax.tick_params(axis="x", labelsize=FONTS["tick"], rotation=30)
            style_axes(ax, grid=True)
            ax.xaxis.grid(False)

        fig.suptitle(
            suptitle or f"Metric Distributions (N = {n_subjects})",
            fontsize=FONTS["suptitle"],
            fontweight="bold",
        )
        return _finalize_fig(fig, show=show, fname=fname)


def plot_null_distribution(
    null_values,
    observed,
    metric_label="Statistic",
    ci=95,
    n_bins=60,
    suptitle=None,
    series_color=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot a null-distribution histogram with observed statistic and CI.

    Parameters
    ----------
    null_values : array-like
        Samples from the null distribution.
    observed : float
        Observed statistic to compare against the null.
    metric_label : str
        Label for the x-axis.
    ci : float
        Central interval width in percent.
    n_bins : int
        Number of histogram bins.
    suptitle : str | None
        Figure title override.
    series_color : str | None
        Color for the observed-statistic marker/annotation.
    figsize : tuple | None
        Figure size in inches.
    fname : path-like | None
        Optional output path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    p_value : float
        Two-sided empirical p-value under ``null_values``.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_null_distribution
    >>> rng = np.random.default_rng(0)
    >>> null = rng.normal(0.0, 0.1, 1000)
    >>> fig, p = plot_null_distribution(null, observed=0.25, show=False)
    """
    null_values = np.asarray(null_values)
    if figsize is None:
        figsize = (8, 5)
    if series_color is None:
        series_color = COLORS["stat_mean"]

    with use_theme():
        fig, ax = themed_figure(1, 1, figsize=figsize)

        ax.hist(
            null_values,
            bins=n_bins,
            color=COLORS["stat_subject"],
            alpha=_STATS_STYLE["hist_alpha"],
            edgecolor=COLORS["separator"],
            linewidth=_STATS_STYLE["hist_linewidth"],
            density=True,
            zorder=2,
            label=f"Null (N = {len(null_values):,})",
        )

        alpha_tail = (100 - ci) / 2
        lo, hi = np.percentile(null_values, [alpha_tail, 100 - alpha_tail])
        ax.axvspan(
            lo,
            hi,
            color=COLORS["stat_ci"],
            alpha=0.12,
            zorder=1,
            label=f"{ci}% CI [{lo:+.3f}, {hi:+.3f}]",
        )
        ax.axvline(
            lo,
            color=COLORS["stat_reference"],
            ls=":",
            lw=_STATS_STYLE["reference_linewidth"],
            alpha=0.6,
        )
        ax.axvline(
            hi,
            color=COLORS["stat_reference"],
            ls=":",
            lw=_STATS_STYLE["reference_linewidth"],
            alpha=0.6,
        )

        ax.axvline(
            observed,
            color=series_color,
            lw=2.5,
            ls="--",
            zorder=5,
            label=f"Observed = {observed:.3f}",
        )

        p_value = float(np.mean(np.abs(null_values) >= np.abs(observed)))

        ax.annotate(
            f"p = {p_value:.4f}",
            xy=(observed, ax.get_ylim()[1] * 0.92),
            fontsize=FONTS["annotation"],
            fontweight="bold",
            ha="left" if observed > np.median(null_values) else "right",
            va="top",
            color=series_color,
            xytext=(8, 0),
            textcoords="offset points",
        )

        ax.set_xlabel(metric_label, fontsize=FONTS["label"])
        ax.set_ylabel("Density", fontsize=FONTS["label"])
        themed_legend(ax, fontsize=_STATS_STYLE["legend_fontsize_small"])
        style_axes(ax)

        fig.suptitle(
            suptitle or f"Null Distribution - {metric_label}",
            fontsize=FONTS["suptitle"],
            fontweight="bold",
        )
        return _finalize_fig(fig, show=show, fname=fname), p_value


def plot_forest(
    data,
    metric_col,
    ci_col=None,
    se_col=None,
    group_col="group",
    subject_col="subject",
    target_group=None,
    baseline_group=None,
    group_colors=None,
    group_labels=None,
    metric_label=None,
    reference_line=0.0,
    suptitle=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot per-subject point estimates with confidence intervals.

    Parameters
    ----------
    data : mapping of str to array-like
        Columnar mapping with group, subject, and metric columns.
    metric_col : str
        Metric column to display on the x-axis.
    ci_col : str | None
        Optional half-width CI column for each subject estimate.
    se_col : str | None
        Optional SE column. If provided and ``ci_col`` is absent, CI is
        approximated as ``1.96 * SE``.
    group_col : str
        Grouping column name.
    subject_col : str
        Subject identifier column name.
    target_group : str | None
        Group to plot as primary forest series. Defaults to the last
        first-seen group.
    baseline_group : str | None
        Optional baseline group to overlay with faint points and mean marker.
    group_colors, group_labels : dict | None
        Optional style overrides keyed by group.
    metric_label : str | None
        X-axis label. If None, derived from ``metric_col``.
    reference_line : float | None
        Optional vertical reference line value.
    suptitle : str | None
        Figure title override.
    figsize : tuple | None
        Figure size in inches.
    fname : path-like | None
        Optional output path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.viz import plot_forest
    >>> data = {
    ...     "subject": np.array(["s1", "s2", "s1", "s2"]),
    ...     "group": np.array(["A", "A", "B", "B"]),
    ...     "effect": np.array([0.2, 0.4, 0.8, 0.9]),
    ... }
    >>> fig = plot_forest(data, metric_col="effect", group_col="group", show=False)
    """
    columns = {name: np.asarray(values) for name, values in data.items()}
    groups_col = np.asarray(columns[group_col], dtype=object)
    subject_col_values = np.asarray(columns[subject_col], dtype=object)
    metric = np.asarray(columns[metric_col], dtype=float)

    groups = list(dict.fromkeys(groups_col.tolist()))

    if target_group is None:
        target_group = groups[-1]
    if metric_label is None:
        metric_label = str(metric_col).replace("_", " ").strip().title()

    target_mask = groups_col == target_group
    subjects = np.asarray(subject_col_values[target_mask], dtype=object)
    values = metric[target_mask]

    order_idx = np.argsort(np.where(np.isfinite(values), values, np.inf))
    subjects = subjects[order_idx]
    values = values[order_idx]
    n_subjects = len(subjects)

    if ci_col is not None:
        ci_values = np.asarray(columns[ci_col], dtype=float)[target_mask]
        half_width = ci_values[order_idx]
    elif se_col is not None:
        se_values = np.asarray(columns[se_col], dtype=float)[target_mask]
        half_width = (1.96 * se_values)[order_idx]
    else:
        sd = values[np.isfinite(values)].std(ddof=1) if n_subjects > 1 else 1.0
        half_width = np.full(n_subjects, 1.96 * sd / np.sqrt(max(n_subjects, 1)))

    if figsize is None:
        figsize = (8, max(4, n_subjects * 0.35 + 2))

    with use_theme():
        fig, ax = themed_figure(1, 1, figsize=figsize)

        y_pos = np.arange(n_subjects)
        target_color = (
            group_colors[target_group]
            if group_colors and target_group in group_colors
            else get_series_color(groups.index(target_group))
        )
        target_label = (
            group_labels[target_group]
            if group_labels and target_group in group_labels
            else target_group
        )

        if baseline_group is not None:
            base_color = (
                group_colors[baseline_group]
                if group_colors and baseline_group in group_colors
                else get_series_color(groups.index(baseline_group))
            )
            base_label = (
                group_labels[baseline_group]
                if group_labels and baseline_group in group_labels
                else baseline_group
            )
            base_metric = metric[groups_col == baseline_group]
            base_subjects = np.asarray(
                subject_col_values[groups_col == baseline_group], dtype=object
            )
            for i, subject in enumerate(subjects):
                mask = base_subjects == subject
                base_values = base_metric[mask]
                base_values = base_values[np.isfinite(base_values)]
                if base_values.size:
                    ax.plot(
                        base_values[0],
                        y_pos[i],
                        "o",
                        color=base_color,
                        markersize=_STATS_STYLE["forest_marker_size"],
                        alpha=0.35,
                        zorder=2,
                    )
            finite_base = base_metric[np.isfinite(base_metric)]
            if finite_base.size:
                base_mean = finite_base.mean()
                ax.plot(
                    base_mean,
                    -1.2,
                    "D",
                    color=base_color,
                    markersize=_STATS_STYLE["forest_baseline_mean_marker_size"],
                    zorder=6,
                    alpha=0.5,
                    label=f"{base_label} mean = {base_mean:.3f}",
                )

        ax.errorbar(
            values,
            y_pos,
            xerr=half_width,
            fmt="o",
            color=target_color,
            ecolor=target_color,
            elinewidth=1.2,
            capsize=_STATS_STYLE["bar_capsize"],
            markersize=_STATS_STYLE["forest_marker_size"],
            alpha=_STATS_STYLE["bar_alpha"],
            zorder=4,
            label=target_label,
        )

        finite_values = values[np.isfinite(values)]
        target_mean = float(finite_values.mean()) if finite_values.size else np.nan
        target_se = (
            float(finite_values.std(ddof=1) / np.sqrt(finite_values.size))
            if finite_values.size > 1
            else 0.0
        )
        ax.errorbar(
            target_mean,
            -1.2,
            xerr=1.96 * target_se,
            fmt="D",
            color=target_color,
            ecolor=target_color,
            elinewidth=_STATS_STYLE["mean_linewidth"],
            capsize=_STATS_STYLE["bar_capsize"] + 1,
            markersize=_STATS_STYLE["forest_pooled_marker_size"],
            zorder=6,
            label=f"Pooled mean = {target_mean:.3f}",
        )

        if reference_line is not None:
            ax.axvline(
                reference_line,
                color=COLORS["stat_reference"],
                ls="--",
                lw=_STATS_STYLE["reference_linewidth"],
                alpha=_STATS_STYLE["reference_alpha"],
            )

        ax.set_yticks(list(y_pos) + [-1.2])
        ax.set_yticklabels(list(subjects) + ["Pooled"], fontsize=FONTS["tick"])
        ax.set_xlabel(metric_label, fontsize=FONTS["label"])
        ax.set_ylabel("")
        ax.invert_yaxis()
        style_axes(ax, grid=True)
        ax.yaxis.grid(False)
        themed_legend(
            ax, fontsize=_STATS_STYLE["legend_fontsize_small"], loc="lower right"
        )

        fig.suptitle(
            suptitle
            or f"Forest Plot - {group_labels[target_group] if group_labels and target_group in group_labels else target_group}",
            fontsize=FONTS["suptitle"],
            fontweight="bold",
        )
        return _finalize_fig(fig, show=show, fname=fname)
