"""Tests for general summary-level denoising plots."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.viz import _summary_panels as summary_panels
from mne_denoise.viz import summary


def test_comparison_viz_show(fitted_dss, synthetic_data):
    """Test show=True code paths for the denoising summary plot."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
        epochs_clean = fitted_dss.transform(synthetic_data)
        summary.plot_denoising_summary(
            synthetic_data,
            epochs_clean,
            info=synthetic_data.info,
            times=synthetic_data.times,
            show=True,
        )


def test_denoising_summary(fitted_dss, synthetic_data):
    """Test denoising summary plots."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    raw_orig = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    raw_clean = mne.io.RawArray(epochs_clean.get_data()[0], synthetic_data.info)

    fig = summary.plot_denoising_summary(
        raw_orig,
        raw_clean,
        info=raw_orig.info,
        times=raw_orig.times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig = summary.plot_denoising_summary(
        synthetic_data,
        epochs_clean,
        info=synthetic_data.info,
        times=synthetic_data.times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_fname_parameter_comparison(fitted_dss, synthetic_data, tmp_path):
    """Test fname parameter on the denoising summary plot."""
    epochs_clean = fitted_dss.transform(synthetic_data)

    summary_path = tmp_path / "summary.png"
    fig = summary.plot_denoising_summary(
        synthetic_data,
        epochs_clean,
        info=synthetic_data.info,
        times=synthetic_data.times,
        show=False,
        fname=str(summary_path),
    )
    assert isinstance(fig, plt.Figure)
    assert summary_path.exists()
    plt.close(fig)


def test_denoising_summary_invalid_times(fitted_dss, synthetic_data):
    """Test plot_denoising_summary raises ValueError for mismatched times."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    with pytest.raises(ValueError, match="times must be 1D and match"):
        summary.plot_denoising_summary(
            synthetic_data,
            epochs_clean,
            info=synthetic_data.info,
            times=synthetic_data.times[:-1],  # Mismatched length
            show=False,
        )


def test_plot_metric_tradeoff_summary():
    """Test plot_metric_tradeoff_summary composer."""
    data = {
        "subject": ["s1", "s1", "s2", "s2"],
        "method": ["A", "B", "A", "B"],
        "below_noise_distortion_db": [1.0, 2.0, 1.5, 2.5],
        "peak_attenuation_db": [10, 15, 12, 18],
        "R_f0": [1.1, 1.2, 1.05, 1.3],
    }

    fig = summary.plot_metric_tradeoff_summary(
        data,
        group_col="method",
        subject_col="subject",
        x_col="below_noise_distortion_db",
        y_col="peak_attenuation_db",
        metric_col="R_f0",
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_component_cleaning_summary_standard():
    """Component cleaning summary works with standard inputs."""
    rng = np.random.default_rng(0)
    freqs = np.linspace(0.0, 80.0, 161)
    psd_before = rng.random((4, freqs.size))
    psd_after = psd_before * 0.7

    fig = summary.plot_component_cleaning_summary(
        scores=np.array([2.1, 1.6, 1.2, 0.9]),
        selected_count=2,
        patterns=rng.standard_normal((5, 4)),
        removed=rng.standard_normal((5, 200)),
        sources=rng.standard_normal((4, 200)),
        sfreq=200.0,
        freqs=freqs,
        psd_before=psd_before,
        psd_after=psd_after,
        line_freq=50.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test without scores and removed data for summary table fallback
    fig = summary.plot_component_cleaning_summary(
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with provided summary_rows
    fig = summary.plot_component_cleaning_summary(
        summary_rows=[("Custom", "Row")],
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_component_cleaning_summary_segmented():
    """Component cleaning summary works with segmented metadata."""
    rng = np.random.default_rng(1)
    segment_info = [
        {"start": 0, "end": 100, "count": 2, "metric": 49.8},
        {"start": 100, "end": 200, "count": 1, "metric": 50.1},
        {"start": 200, "end": 300, "count": 3, "metric": 49.9},
    ]

    fig = summary.plot_component_cleaning_summary(
        patterns=rng.standard_normal((5, 3)),
        sfreq=100.0,
        segment_info=segment_info,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_signal_diagnostics_summary():
    """Signal diagnostics summary works for array and MNE inputs."""
    rng = np.random.default_rng(2)
    n_times = 250
    times = np.arange(n_times) / 250.0
    signals = {
        "before": rng.standard_normal((4, n_times)),
        "after": rng.standard_normal((4, n_times)) * 0.8,
    }
    windows = [(0.08, 0.14, "early"), (0.18, 0.24, "late"), (10.0, 11.0, "missing")]
    group_order = ["before", "after"]
    group_colors = {"before": "#4C72B0", "after": "#55A868"}
    group_labels = {"before": "Before", "after": "After"}

    # Test with array input and a missing window
    fig = summary.plot_signal_diagnostics_summary(
        signals,
        channel=1,
        channel_label="C2",
        times=times,
        group_order=group_order,
        reference_group="before",
        group_colors=group_colors,
        group_labels=group_labels,
        windows=windows,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with MNE Evoked input
    info = mne.create_info(["C1", "C2", "C3", "C4"], 250.0, "eeg")
    evoked_before = mne.EvokedArray(signals["before"], info, tmin=0)
    evoked_after = mne.EvokedArray(signals["after"], info, tmin=0)
    mne_signals = {"before": evoked_before, "after": evoked_after}

    fig = summary.plot_signal_diagnostics_summary(
        mne_signals,
        channel="C2",
        channel_names=["C1", "C2", "C3", "C4"],
        times=times,
        group_order=group_order,
        reference_group="before",
        group_colors=group_colors,
        group_labels=group_labels,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with 3D array input
    signals_3d = {
        "before": rng.standard_normal((2, 4, n_times)),
        "after": rng.standard_normal((2, 4, n_times)),
    }
    fig = summary.plot_signal_diagnostics_summary(
        signals_3d,
        channel=1,
        channel_label="C2",
        times=times,
        group_order=group_order,
        reference_group="before",
        group_colors=group_colors,
        group_labels=group_labels,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_signal_diagnostics_summary_single_axis():
    """Test plot_signal_diagnostics_summary with single axis fallback."""
    rng = np.random.default_rng(2)
    n_times = 100
    times = np.linspace(0, 1, n_times)
    signals = {
        "g1": rng.standard_normal((2, n_times)),
        "g2": rng.standard_normal((2, n_times)),
    }

    with patch("mne_denoise.viz.summary.themed_figure") as mock_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mock_fig.return_value = (fig, ax)

        summary.plot_signal_diagnostics_summary(
            signals,
            channel=0,
            channel_label="CH",
            times=times,
            group_order=["g1", "g2"],
            reference_group="g1",
            group_colors={"g1": "r", "g2": "b"},
            group_labels={"g1": "G1", "g2": "G2"},
            show=False,
        )
        plt.close(fig)


def test_plot_signal_diagnostics_summary_validation():
    """Signal diagnostics summary input validation."""
    rng = np.random.default_rng(22)
    n_times = 200
    times = np.arange(n_times) / 200.0
    signals = {
        "before": rng.standard_normal((3, n_times)),
        "after": rng.standard_normal((3, n_times)),
    }

    # Empty signals
    with pytest.raises(ValueError, match="signals must be a non-empty mapping"):
        summary.plot_signal_diagnostics_summary(
            {},
            channel=0,
            times=times,
            group_order=[],
            reference_group="",
            group_colors={},
            group_labels={},
        )

    # Empty group_order
    with pytest.raises(ValueError, match="group_order must contain at least one"):
        summary.plot_signal_diagnostics_summary(
            signals,
            channel=0,
            times=times,
            group_order=[],
            reference_group="before",
            group_colors={},
            group_labels={},
        )

    # reference_group not in group_order
    with pytest.raises(ValueError, match="reference_group must be present"):
        summary.plot_signal_diagnostics_summary(
            signals,
            channel=0,
            times=times,
            group_order=["before"],
            reference_group="after",
            group_colors={},
            group_labels={},
        )

    # Invalid signal shape
    bad_signals = {"before": rng.standard_normal(10)}
    with pytest.raises(ValueError, match="Each signal must be 2D"):
        summary.plot_signal_diagnostics_summary(
            bad_signals,
            channel=0,
            times=times,
            group_order=["before"],
            reference_group="before",
            group_colors={},
            group_labels={},
        )

    # Mismatched dimensions
    mismatched_signals = {
        "before": rng.standard_normal((3, 200)),
        "after": rng.standard_normal((3, 201)),
    }
    with pytest.raises(ValueError, match="All signals must have matching"):
        summary.plot_signal_diagnostics_summary(
            mismatched_signals,
            channel=0,
            times=times,
            group_order=["before", "after"],
            reference_group="before",
            group_colors={"before": "r", "after": "b"},
            group_labels={"before": "B", "after": "A"},
        )

    # Mismatched channel names
    with pytest.raises(ValueError, match="channel_names length must match"):
        summary.plot_signal_diagnostics_summary(
            signals,
            channel=0,
            times=times,
            channel_names=["Fz", "Cz"],
            group_order=["before"],
            reference_group="before",
            group_colors={"before": "r"},
            group_labels={"before": "B"},
        )

    # String selector without names
    with pytest.raises(
        ValueError, match="String channel selectors require channel names"
    ):
        summary.plot_signal_diagnostics_summary(
            signals,
            channel="Cz",
            times=times,
            group_order=["before"],
            reference_group="before",
            group_colors={},
            group_labels={},
        )

    # Missing channel label when no names
    with pytest.raises(ValueError, match="channel_label must be provided"):
        summary.plot_signal_diagnostics_summary(
            signals,
            channel=0,
            times=times,
            group_order=["before"],
            reference_group="before",
            group_colors={"before": "r"},
            group_labels={"before": "B"},
        )

    # Invalid times
    with pytest.raises(ValueError, match="times must be 1D and match"):
        summary.plot_signal_diagnostics_summary(
            signals,
            channel=0,
            channel_label="C",
            times=np.zeros(10),
            group_order=["before"],
            reference_group="before",
            group_colors={"before": "r"},
            group_labels={"before": "B"},
        )


def test_plot_condition_interaction_summary():
    """Condition interaction summary works with nested mappings."""
    times = np.linspace(-0.2, 0.5, 300)
    traces = {
        "cond_a": {
            "before": np.sin(2 * np.pi * 4 * times),
            "after": np.sin(2 * np.pi * 4 * times) * 0.6,
        },
        "cond_b": {
            "before": np.cos(2 * np.pi * 3 * times),
            "after": np.cos(2 * np.pi * 3 * times) * 0.7,
        },
    }
    errors = {
        "cond_a": {
            "before": np.full(times.size, 0.1),
            "after": np.full(times.size, 0.08),
        },
        "cond_b": {
            "before": np.full(times.size, 0.12),
            "after": np.full(times.size, 0.09),
        },
    }
    # Test with automatic order and colors
    fig = summary.plot_condition_interaction_summary(
        traces,
        times=times,
        errors=errors,
        windows=[(0.08, 0.14, "w1")],
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with explicit orders and labels
    fig = summary.plot_condition_interaction_summary(
        traces,
        times=times,
        condition_order=["cond_b", "cond_a"],
        group_order=["after", "before"],
        group_colors={"before": "k", "after": "r"},
        group_labels={"before": "B", "after": "A"},
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with missing group in one condition
    traces_partial = {
        "cond_a": {"before": np.zeros(300)},
        "cond_b": {"after": np.zeros(300)},
    }
    fig = summary.plot_condition_interaction_summary(
        traces_partial, times=times, show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test errors: empty traces
    with pytest.raises(ValueError, match="traces must be a non-empty mapping"):
        summary.plot_condition_interaction_summary({}, times=times)

    # Test error: invalid times
    with pytest.raises(ValueError, match="times must be 1D"):
        summary.plot_condition_interaction_summary(traces, times=np.zeros((2, 2)))

    # Test error: mismatched trace size
    with pytest.raises(
        ValueError, match="All traces must be 1D and match times length"
    ):
        summary.plot_condition_interaction_summary(
            {"c": {"g": np.zeros(10)}}, times=times
        )

    # Test error: mismatched trace shape (2D)
    with pytest.raises(ValueError, match="All traces must be 1D"):
        summary.plot_condition_interaction_summary(
            {"c": {"g": np.zeros((2, 300))}}, times=times
        )

    # Test error: mismatched trace/error shape
    bad_errors = {"cond_a": {"before": np.zeros(10)}}
    with pytest.raises(ValueError, match="errors traces must match"):
        summary.plot_condition_interaction_summary(
            traces, times=times, errors=bad_errors
        )


def test_plot_group_condition_interaction_summary():
    """Group condition interaction summary works with nested mappings."""
    times = np.linspace(-0.2, 0.5, 300)
    traces = {
        "before": {
            "cond_a": np.sin(2 * np.pi * 4 * times),
            "cond_b": np.sin(2 * np.pi * 5 * times),
        },
        "after": {
            "cond_a": np.sin(2 * np.pi * 4 * times) * 0.7,
            "cond_b": np.sin(2 * np.pi * 5 * times) * 0.6,
        },
    }
    fig = summary.plot_group_condition_interaction_summary(
        traces,
        times=times,
        windows=[(0.08, 0.14, "w1")],
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with explicit orders and labels
    fig = summary.plot_group_condition_interaction_summary(
        traces,
        times=times,
        group_order=["after", "before"],
        condition_order=["cond_b", "cond_a"],
        condition_labels={"cond_a": "A", "cond_b": "B"},
        errors=traces,  # Just to test error plotting path
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with missing condition in one group
    traces_partial = {"before": {"c1": np.zeros(300)}, "after": {"c2": np.zeros(300)}}
    summary.plot_group_condition_interaction_summary(
        traces_partial, times=times, show=False
    )

    # Test error: empty traces
    with pytest.raises(ValueError, match="traces must be a non-empty mapping"):
        summary.plot_group_condition_interaction_summary({}, times=times)

    # Test error: mismatched times/shape
    with pytest.raises(
        ValueError, match="All traces must be 1D and match times length"
    ):
        summary.plot_group_condition_interaction_summary(traces, times=np.zeros(10))

    with pytest.raises(ValueError, match="All traces must be 1D"):
        summary.plot_group_condition_interaction_summary(
            {"g": {"c": np.zeros((2, 300))}}, times=times
        )

    # Test error: mismatched error shape
    with pytest.raises(ValueError, match="errors traces must match"):
        summary.plot_group_condition_interaction_summary(
            traces, times=times, errors={"before": {"cond_a": np.zeros(10)}}
        )

    # Test error: times not 1D
    with pytest.raises(ValueError, match="times must be 1D"):
        summary.plot_group_condition_interaction_summary(traces, times=np.zeros((2, 2)))


def test_plot_endpoint_metrics_summary():
    """Endpoint metrics summary works for mapping input."""
    data = {
        "subject": np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object),
        "group": np.array(["A", "B", "A", "B", "A", "B"], dtype=object),
        "score": np.array([1.2, 0.9, 1.1, 0.8, 1.3, 0.85]),
    }
    null = np.random.default_rng(4).normal(0.0, 1.0, size=1000)

    # Full data and null distribution
    fig = summary.plot_endpoint_metrics_summary(
        data,
        metric_col="score",
        group_col="group",
        subject_col="subject",
        group_order=["A", "B"],
        group_colors={"A": "r", "B": "b"},
        group_labels={"A": "Alpha", "B": "Beta"},
        reference_value=1.0,
        null_distribution=null,
        observed_value=1.25,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test without null distribution (table fallback) and single values
    fig = summary.plot_endpoint_metrics_summary(
        data,
        metric_col="score",
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with single group and missing values
    data_limited = {
        "subject": ["s1", "s2"],
        "group": ["ptr", "ptr"],
        "score": [1.0, np.nan],
    }
    fig = summary.plot_endpoint_metrics_summary(
        data_limited,
        metric_col="score",
        group_order=["A", "B"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with null distribution but no observed value
    fig = summary.plot_endpoint_metrics_summary(
        data,
        metric_col="score",
        null_distribution=null,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with empty null distribution
    fig = summary.plot_endpoint_metrics_summary(
        data,
        metric_col="score",
        null_distribution=[],
        observed_value=1.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with single axis fallback
    with patch("mne_denoise.viz.summary.themed_figure") as mock_fig:
        # Use a real subplot to avoid MagicMock missing attribute errors like 'spines'
        fig_real = plt.figure()
        ax_real = fig_real.add_subplot(111)
        mock_fig.return_value = (fig_real, ax_real)
        summary.plot_endpoint_metrics_summary(data, metric_col="score", show=False)
        plt.close(fig_real)


def test_panels_placeholders():
    """Test placeholder paths for all summary panels."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # _plot_selection_scores_panel
    summary_panels._plot_selection_scores_panel(ax, scores=None, selected_count=0)
    assert "No scores available" in [t.get_text() for t in ax.texts]

    # _plot_patterns_panel
    ax.clear()
    summary_panels._plot_patterns_panel(fig, ax.get_subplotspec(), patterns=None)
    assert "No patterns available" in [t.get_text() for t in fig.axes[-1].texts]

    # _plot_removed_power_panel
    ax.clear()
    summary_panels._plot_removed_power_panel(fig, ax, removed=None)
    assert any("No removal" in t.get_text() for t in ax.texts)

    # _plot_source_trace_panel
    ax.clear()
    summary_panels._plot_source_trace_panel(ax, sources=None)
    assert "No source data" in [t.get_text() for t in ax.texts]

    # _plot_source_trace_panel missing sfreq
    ax.clear()
    summary_panels._plot_source_trace_panel(ax, sources=np.zeros((4, 100)), sfreq=None)
    assert any("sfreq required" in t.get_text() for t in ax.texts)

    # _plot_before_after_psd_panel
    ax.clear()
    summary_panels._plot_before_after_psd_panel(ax, freqs=None)
    assert "No PSD data" in [t.get_text() for t in ax.texts]

    # _plot_segment_metric_panel empty
    ax.clear()
    summary_panels._plot_segment_metric_panel(ax, segment_info=[{}])
    assert any("No metric" in t.get_text() for t in ax.texts)

    # _plot_segment_metric_panel with frequency/fine_freq
    ax.clear()
    summary_panels._plot_segment_metric_panel(ax, [{"frequency": 10.5}])
    summary_panels._plot_segment_metric_panel(ax, [{"fine_freq": 10.5}])
    assert len(ax.get_lines()) > 0

    # _plot_segment_boundaries_panel missing sfreq
    ax.clear()
    summary_panels._plot_segment_boundaries_panel(ax, segment_info=[], sfreq=0)
    assert "sfreq required" in [t.get_text() for t in ax.texts]
    plt.close(fig)


def test_panels_errors():
    """Test error handling in summary panels."""
    fig = plt.figure()
    spec = fig.add_gridspec(1, 1)[0]
    ax = fig.add_subplot(spec)

    with pytest.raises(ValueError, match="patterns must be 2D"):
        summary_panels._plot_patterns_panel(fig, spec, patterns=np.zeros(10))

    with pytest.raises(ValueError, match="removed must be 2D or 3D"):
        summary_panels._plot_removed_power_panel(fig, ax, removed=np.zeros(10))

    with pytest.raises(ValueError, match="sources must be 2D or 3D"):
        summary_panels._plot_source_trace_panel(ax, sources=np.zeros(10), sfreq=100)
    plt.close(fig)


def test_panels_topomaps(fitted_dss):
    """Test topomap paths for patterns and removal panels."""
    fig = plt.figure()
    spec = fig.add_gridspec(1, 1)[0]
    ax = fig.add_subplot(spec)
    info = fitted_dss.info_

    # Patterns topomap
    patterns = np.random.randn(info["nchan"], 4)
    summary_panels._plot_patterns_panel(fig, spec, patterns=patterns, info=info)
    # Check that multiple axes were created for topomaps
    assert len(fig.axes) > 1

    # Removal topomap
    fig.clear()
    ax = fig.add_subplot(111)
    removed = np.random.randn(info["nchan"], 100)
    summary_panels._plot_removed_power_panel(fig, ax, removed=removed, info=info)
    # Check for colorbar
    assert any("colorbar" in str(a.get_label()).lower() for a in fig.axes)
    plt.close(fig)


def test_panels_fallbacks():
    """Test specific fallbacks and options in panels."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Patterns with channel names
    patterns = np.zeros((4, 2))
    summary_panels._plot_patterns_panel(
        fig, ax.get_subplotspec(), patterns=patterns, channel_names=["A", "B", "C", "D"]
    )

    # Removal with channel names
    ax.clear()
    summary_panels._plot_removed_power_panel(
        fig, ax, removed=np.zeros((3, 10)), channel_names=["A", "B", "C"]
    )

    # PSD with/without line frequency
    ax.clear()
    summary_panels._plot_before_after_psd_panel(
        ax, freqs=[1, 2], psd_before=[1, 1], psd_after=[1, 1], line_freq=50
    )
    summary_panels._plot_before_after_psd_panel(
        ax, freqs=[1, 2], psd_before=[1, 1], psd_after=[1, 1], line_freq=None
    )

    # Segment counts from n_selected
    ax.clear()
    summary_panels._plot_segment_counts_panel(ax, [{"n_selected": 5}])

    # Segment metric from fine_freq
    ax.clear()
    summary_panels._plot_segment_metric_panel(ax, [{"fine_freq": 10.5}])
    plt.close(fig)


def test_panels_3d_averaging():
    """Test 3D input handling in removal and source panels."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 3D Removed
    removed_3d = np.random.randn(2, 4, 100)
    summary_panels._plot_removed_power_panel(fig, ax, removed=removed_3d)

    # 3D Sources
    ax.clear()
    sources_3d = np.random.randn(2, 4, 100)
    summary_panels._plot_source_trace_panel(ax, sources=sources_3d, sfreq=100)
    plt.close(fig)
