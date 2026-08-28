"""Public and quantitative contracts for grouped statistic plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_forest,
    plot_harmonic_attenuation,
    plot_metric_bars,
    plot_metric_comparison,
    plot_metric_slopes,
    plot_metric_violins,
    plot_null_distribution,
    plot_tradeoff_scatter,
    plot_window_count_series,
)


@pytest.fixture
def metric_data():
    """Return deterministic paired values for two groups."""
    return {
        "subject": np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object),
        "group": np.array(["A", "B", "A", "B", "A", "B"], dtype=object),
        "score": np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0]),
        "distortion": np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.4]),
        "attenuation": np.array([8.0, 6.0, 9.0, 7.0, 10.0, 8.0]),
    }


def test_plot_window_count_series_preserves_counts_and_mean():
    """Window counts use the sample index and a correctly labeled mean."""
    counts = np.array([1.0, 3.0, 5.0])
    fig = plot_window_count_series(counts, show=False)

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_title() == "Window Count Series")
    assert [patch.get_height() for patch in ax.patches] == counts.tolist()
    assert ax.get_xlabel() == "Window"
    assert ax.get_ylabel() == "Count"
    assert "Mean (3)" in {line.get_label() for line in ax.lines}


def test_plot_metric_bars_aggregates_groups_and_labels_metric(metric_data):
    """Metric bars show group means, labels, and the lower-is-better marker."""
    fig = plot_metric_bars(
        metric_data,
        metric_cols=["score"],
        group_col="group",
        group_order=["A", "B"],
        group_labels={"A": "Baseline", "B": "Cleaned"},
        lower_better=[True],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "score")
    assert [patch.get_height() for patch in ax.patches] == [2.0, 4.0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "Baseline",
        "Cleaned",
    ]
    assert ax.get_ylabel() == "score"
    assert any(text.get_text() == "★" for text in ax.texts)


def test_plot_tradeoff_scatter_maps_xy_groups_and_reference_lines():
    """Trade-off points preserve supplied x/y columns and reference values."""
    data = {
        "group": np.array(["A", "B"], dtype=object),
        "x": np.array([0.1, 0.2]),
        "y": np.array([2.0, 4.0]),
    }
    fig = plot_tradeoff_scatter(
        data,
        x_col="x",
        y_col="y",
        group_col="group",
        reference_x=0.15,
        reference_y=3.0,
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_title() == "Metric Trade-off")
    offsets = np.vstack([collection.get_offsets() for collection in ax.collections])
    assert any(np.allclose(point, [0.1, 2.0]) for point in offsets)
    assert any(np.allclose(point, [0.2, 4.0]) for point in offsets)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert any(np.allclose(line.get_xdata(), [0.15, 0.15]) for line in ax.lines)
    assert any(np.allclose(line.get_ydata(), [3.0, 3.0]) for line in ax.lines)


def test_plot_metric_comparison_supports_paired_and_single_subject(metric_data):
    """Metric comparisons aggregate paired subjects and retain the bar fallback."""
    paired = plot_metric_comparison(
        metric_data,
        metric_col="score",
        group_col="group",
        subject_col="subject",
        group_order=["A", "B"],
        reference_value=0.0,
        show=False,
    )
    assert isinstance(paired, plt.Figure)
    paired_axis = next(ax for ax in paired.axes if ax.get_ylabel() == "score")
    mean_line = next(
        line for line in paired_axis.lines if line.get_label() == "Group mean"
    )
    np.testing.assert_allclose(mean_line.get_ydata(), [2.0, 4.0])
    assert paired_axis.get_ylabel() == "score"

    single = {
        "subject": ["s1", "s1"],
        "group": ["A", "B"],
        "score": [1.0, 2.0],
    }
    single_fig = plot_metric_comparison(
        single, metric_col="score", group_col="group", subject_col="subject", show=False
    )
    assert isinstance(single_fig, plt.Figure)


def test_plot_harmonic_attenuation_reports_db_values():
    """Per-harmonic bars use the QA attenuation definition in dB."""
    freqs = np.arange(0.0, 121.0, 0.5)
    before = np.ones_like(freqs)
    after = before.copy()
    for harmonic in (50.0, 100.0):
        after[np.abs(freqs - harmonic) <= 2.0] = 0.1

    fig = plot_harmonic_attenuation(
        freqs,
        before,
        {"Cleaned": (freqs, after)},
        harmonics_hz=[50.0, 100.0],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Peak Attenuation (dB)")
    assert [patch.get_height() for patch in ax.patches] == pytest.approx([10.0, 10.0])
    assert ax.get_ylabel() == "Peak Attenuation (dB)"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["50 Hz", "100 Hz"]


def test_plot_metric_slopes_reports_subject_aggregate(metric_data):
    """Paired slopes include the group-mean trajectory for each metric."""
    fig = plot_metric_slopes(
        metric_data,
        metric_cols=["score"],
        group_col="group",
        subject_col="subject",
        group_order=["A", "B"],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    axis = next(ax for ax in fig.axes if ax.get_ylabel() == "score")
    mean_line = next(line for line in axis.lines if line.get_label() == "Group mean")
    np.testing.assert_allclose(mean_line.get_ydata(), [2.0, 4.0])
    assert axis.get_ylabel() == "score"


def test_plot_metric_violins_supports_pairs_baselines_and_references(metric_data):
    """Distribution plots retain paired data and optional reference lines."""
    fig = plot_metric_violins(
        metric_data,
        metric_cols=["score"],
        group_col="group",
        subject_col="subject",
        baseline_group="A",
        reference_lines={"score": [(3.5, {"color": "black"})]},
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "score")
    assert ax.get_ylabel() == "score"
    assert any(np.allclose(line.get_ydata(), [3.5, 3.5]) for line in ax.lines)


def test_plot_null_distribution_returns_empirical_p_value_and_ci():
    """Null plots return the two-sided empirical p-value and show the CI."""
    null = np.array([-1.0, -0.5, 0.5, 1.0])
    fig, p_value = plot_null_distribution(
        null, observed=1.0, metric_label="Suppression (dB)", ci=50, show=False
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_xlabel() == "Suppression (dB)")
    assert p_value == pytest.approx(0.5)
    assert ax.get_xlabel() == "Suppression (dB)"
    assert any("50% CI" in label for label in ax.get_legend_handles_labels()[1])


def test_plot_forest_preserves_confidence_error_and_group_labels(metric_data):
    """Forest plots expose per-subject intervals and the pooled estimate."""
    data = dict(metric_data)
    data["ci"] = np.full(len(data["score"]), 0.25)
    fig = plot_forest(
        data,
        metric_col="score",
        ci_col="ci",
        group_col="group",
        subject_col="subject",
        target_group="B",
        baseline_group="A",
        group_labels={"A": "Baseline", "B": "Cleaned"},
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_xlabel() == "Score")
    assert ax.get_xlabel() == "Score"
    legend_labels = ax.get_legend_handles_labels()[1]
    assert "Cleaned" in legend_labels
    assert any(label.startswith("Pooled mean =") for label in legend_labels)
    assert any(label.startswith("Baseline mean =") for label in legend_labels)
