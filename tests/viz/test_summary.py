"""Public contracts for summary-level denoising plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_component_cleaning_summary,
    plot_condition_interaction_summary,
    plot_denoising_summary,
    plot_endpoint_metrics_summary,
    plot_group_condition_interaction_summary,
    plot_metric_tradeoff_summary,
    plot_signal_diagnostics_summary,
)


def _titles(fig):
    """Return non-empty axes titles without depending on panel placement."""
    return [
        ax.get_title(loc=location)
        for ax in fig.axes
        for location in ("left", "center", "right")
        if ax.get_title(loc=location)
    ]


def test_denoising_summary_composes_public_panels_and_saves(
    fitted_dss, synthetic_data, tmp_path
):
    """The dashboard exposes its documented map, PSD, and GFP panels."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    summary_path = tmp_path / "summary.png"

    fig = plot_denoising_summary(
        synthetic_data,
        epochs_clean,
        info=synthetic_data.info,
        times=synthetic_data.times,
        show=False,
        fname=summary_path,
    )

    assert isinstance(fig, plt.Figure)
    assert summary_path.exists()
    titles = _titles(fig)
    assert "Power Ratio Map" in titles
    assert "PSD Comparison" in titles
    assert "Temporal Signal Comparison (GFP)" in titles
    gfp_axis = next(ax for ax in fig.axes if "Temporal Signal" in ax.get_title())
    assert {line.get_label() for line in gfp_axis.lines} >= {"Before", "After"}
    assert gfp_axis.get_xlabel() == "Time"
    assert gfp_axis.get_ylabel() == "Global Field Power"


def test_denoising_summary_is_headless_and_validates_times(
    fitted_dss, synthetic_data, monkeypatch
):
    """The public summary can be generated headlessly and rejects bad axes."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    monkeypatch.setattr(plt, "show", lambda: None)

    fig = plot_denoising_summary(
        synthetic_data,
        epochs_clean,
        info=synthetic_data.info,
        times=synthetic_data.times,
        show=True,
    )
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError, match="times must be 1D and match"):
        plot_denoising_summary(
            synthetic_data,
            epochs_clean,
            info=synthetic_data.info,
            times=synthetic_data.times[:-1],
            show=False,
        )


def test_metric_tradeoff_summary_composes_metric_panels():
    """The metric composer preserves the two public primitive summaries."""
    data = {
        "subject": np.array(["s1", "s1", "s2", "s2"]),
        "method": np.array(["A", "B", "A", "B"]),
        "distortion": np.array([1.0, 2.0, 1.5, 2.5]),
        "attenuation": np.array([10.0, 15.0, 12.0, 18.0]),
        "ratio": np.array([1.1, 1.2, 1.05, 1.3]),
    }

    fig = plot_metric_tradeoff_summary(
        data,
        group_col="method",
        subject_col="subject",
        x_col="distortion",
        y_col="attenuation",
        metric_col="ratio",
        group_order=["A", "B"],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    titles = _titles(fig)
    assert "Metric Trade-off" in titles
    assert "Metric Comparison" in titles


def test_component_cleaning_summary_supports_core_and_segmented_inputs():
    """Component summaries expose core panels and the segmented alternative."""
    rng = np.random.default_rng(0)
    freqs = np.linspace(0.0, 80.0, 161)
    psd_before = rng.random((5, freqs.size))
    psd_after = psd_before * 0.7

    fig = plot_component_cleaning_summary(
        scores=np.array([2.1, 1.6, 1.2]),
        selected_count=1,
        patterns=rng.standard_normal((5, 3)),
        removed=rng.standard_normal((5, 200)),
        sources=rng.standard_normal((3, 200)),
        sfreq=200.0,
        freqs=freqs,
        psd_before=psd_before,
        psd_after=psd_after,
        line_freq=50.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    titles = _titles(fig)
    assert any("Component scores" in title for title in titles)
    assert any("Component patterns" in title for title in titles)
    assert any("Power spectral density" in title for title in titles)

    segment_info = [
        {"start": 0, "end": 100, "count": 2, "metric": 49.8},
        {"start": 100, "end": 200, "count": 1, "metric": 50.1},
    ]
    segmented = plot_component_cleaning_summary(
        patterns=rng.standard_normal((5, 3)),
        sfreq=100.0,
        segment_info=segment_info,
        show=False,
    )
    assert isinstance(segmented, plt.Figure)
    assert any("Segment selection counts" in title for title in _titles(segmented))
    assert any("Component patterns" in title for title in _titles(segmented))


def test_signal_diagnostics_summary_maps_channel_groups_and_windows():
    """Signal summaries preserve channel mapping, groups, differences, and windows."""
    times = np.linspace(-0.2, 0.5, 120)
    base = np.vstack(
        [np.sin(2 * np.pi * 4 * times), np.cos(2 * np.pi * 3 * times), times]
    )
    signals = {
        "before": np.stack([base, base * 1.1]),
        "after": np.stack([base * 0.5, base * 0.6]),
    }

    fig = plot_signal_diagnostics_summary(
        signals,
        channel="C2",
        channel_names=["C1", "C2", "C3"],
        times=times,
        group_order=["before", "after"],
        reference_group="before",
        group_colors={"before": "#4C72B0", "after": "#55A868"},
        group_labels={"before": "Before", "after": "After"},
        windows=[(0.08, 0.14, "early")],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    titles = _titles(fig)
    assert any("Channel overlay (C2)" in title for title in titles)
    assert any("GFP comparison" in title for title in titles)
    assert any("Difference vs before" in title for title in titles)
    overlay = next(ax for ax in fig.axes if "Channel overlay" in ax.get_title())
    assert {line.get_label() for line in overlay.lines} >= {"Before", "After"}
    assert overlay.get_xlabel() == "Time"
    assert overlay.get_ylabel() == "Amplitude"
    assert overlay.patches


def test_signal_diagnostics_summary_rejects_ambiguous_channel_metadata():
    """Channel selectors fail clearly when the documented metadata is absent."""
    signals = {"before": np.zeros((2, 20)), "after": np.ones((2, 20))}
    with pytest.raises(ValueError, match="String channel selectors require"):
        plot_signal_diagnostics_summary(
            signals,
            channel="Cz",
            times=np.arange(20),
            group_order=["before", "after"],
            reference_group="before",
            group_colors={},
            group_labels={},
        )


def test_condition_interaction_summary_preserves_traces_errors_and_order():
    """Condition panels retain explicit ordering, labels, traces, and uncertainty."""
    times = np.linspace(-0.2, 0.5, 100)
    traces = {
        "cond_a": {
            "before": np.sin(2 * np.pi * 4 * times),
            "after": 0.6 * np.sin(2 * np.pi * 4 * times),
        },
        "cond_b": {
            "before": np.cos(2 * np.pi * 3 * times),
            "after": 0.7 * np.cos(2 * np.pi * 3 * times),
        },
    }
    errors = {
        condition: {group: np.full(times.size, 0.1) for group in groups}
        for condition, groups in traces.items()
    }

    fig = plot_condition_interaction_summary(
        traces,
        times=times,
        errors=errors,
        condition_order=["cond_b", "cond_a"],
        group_order=["after", "before"],
        group_labels={"before": "Before", "after": "After"},
        windows=[(0.08, 0.14, "early")],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    axes = [ax for ax in fig.axes if ax.get_title() in {"cond_a", "cond_b"}]
    assert [ax.get_title() for ax in axes] == ["cond_b", "cond_a"]
    assert {line.get_label() for line in axes[0].lines} >= {"Before", "After"}
    assert axes[0].collections
    assert axes[0].patches
    assert axes[0].get_xlabel() == "Time"


def test_group_condition_interaction_summary_preserves_conditions_and_errors():
    """Group panels retain condition labels and optional error bands."""
    times = np.linspace(-0.2, 0.5, 100)
    traces = {
        "before": {
            "cond_a": np.sin(2 * np.pi * 4 * times),
            "cond_b": np.sin(2 * np.pi * 5 * times),
        },
        "after": {
            "cond_a": 0.7 * np.sin(2 * np.pi * 4 * times),
            "cond_b": 0.6 * np.sin(2 * np.pi * 5 * times),
        },
    }
    errors = {
        group: {condition: np.full(times.size, 0.05) for condition in values}
        for group, values in traces.items()
    }

    fig = plot_group_condition_interaction_summary(
        traces,
        times=times,
        errors=errors,
        group_order=["after", "before"],
        condition_order=["cond_b", "cond_a"],
        condition_labels={"cond_a": "A", "cond_b": "B"},
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    axes = [ax for ax in fig.axes if ax.get_title() in {"after", "before"}]
    assert [ax.get_title() for ax in axes] == ["after", "before"]
    assert {line.get_label() for line in axes[0].lines} >= {"A", "B"}
    assert axes[0].collections
    assert axes[0].get_ylabel() == "Amplitude"


def test_endpoint_metrics_summary_supports_null_and_optional_summary_panel():
    """Endpoint summaries expose grouped metrics and the optional null panel."""
    data = {
        "subject": np.array(["s1", "s1", "s2", "s2", "s3", "s3"]),
        "group": np.array(["A", "B", "A", "B", "A", "B"]),
        "score": np.array([1.2, 0.9, 1.1, 0.8, 1.3, 0.85]),
    }
    null = np.linspace(-2.0, 2.0, 101)

    fig = plot_endpoint_metrics_summary(
        data,
        metric_col="score",
        group_order=["A", "B"],
        group_labels={"A": "Alpha", "B": "Beta"},
        reference_value=1.0,
        null_distribution=null,
        observed_value=1.25,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    null_axis = next(ax for ax in fig.axes if "Null distribution" in ax.get_title())
    assert {line.get_label() for line in null_axis.lines} >= {"Observed"}
    assert any(text.get_text().startswith("p=") for text in null_axis.texts)

    no_null = plot_endpoint_metrics_summary(data, metric_col="score", show=False)
    assert isinstance(no_null, plt.Figure)
    assert any("Summary" in title for title in _titles(no_null))
