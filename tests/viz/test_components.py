"""Public contracts for component-level visualization functions."""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_component_epochs_image,
    plot_component_patterns,
    plot_component_score_curve,
    plot_component_spectrogram,
    plot_component_summary,
    plot_component_time_series,
    plot_window_score_traces,
)


class ArrayOnlyEst:
    """Small fitted-estimator stand-in without MNE metadata."""

    patterns_ = np.ones((5, 3))
    sources_ = np.arange(600.0).reshape(3, 200)
    eigenvalues_ = np.array([1.0, 0.5, 0.1])


class NoScoreEst:
    """Estimator stand-in with no component score attribute."""


def test_plot_component_score_curve_supports_modes_and_reports_scores(fitted_dss):
    """Score curves expose the documented modes and a useful y-axis label."""
    labels = {
        "raw": "Score / Eigenvalue",
        "cumulative": "Cumulative Score (Normalized)",
        "ratio": "Power Ratio",
    }
    for mode, label in labels.items():
        fig = plot_component_score_curve(fitted_dss, mode=mode, show=False)
        assert isinstance(fig, plt.Figure)
        score_axis = next(ax for ax in fig.axes if ax.get_title() == "Component Scores")
        assert score_axis.get_ylabel() == label

    with pytest.raises(ValueError, match="mode must be one of"):
        plot_component_score_curve(fitted_dss, mode="unsupported", show=False)
    with pytest.raises(ValueError, match="does not expose component scores"):
        plot_component_score_curve(NoScoreEst(), show=False)


def test_plot_component_patterns_supports_topomap_and_line_fallback(
    fitted_dss, synthetic_data
):
    """Patterns honor component selection and the explicit topomap inputs."""
    picks = np.arange(len(synthetic_data.ch_names))
    topomap = plot_component_patterns(
        fitted_dss,
        info=synthetic_data.info,
        picks=picks,
        n_components=[1],
        show=False,
    )
    assert isinstance(topomap, plt.Figure)
    assert any("Comp 1" in ax.get_title() for ax in topomap.axes)

    line_plot = plot_component_patterns(fitted_dss, n_components=2, show=False)
    assert isinstance(line_plot, plt.Figure)
    pattern_axis = next(
        ax for ax in line_plot.axes if ax.get_title() == "Component Patterns"
    )
    assert pattern_axis.get_xlabel() == "Channel"
    assert pattern_axis.get_ylabel() == "Pattern Weight"

    with pytest.raises(ValueError, match="info is required"):
        plot_component_patterns(fitted_dss, picks=[0], show=False)
    with pytest.raises(ValueError, match="No components selected"):
        plot_component_patterns(fitted_dss, n_components=[], show=False)


def test_component_primitives_support_zapline(fitted_zapline):
    """The generic component primitives accept fitted standard ZapLine."""
    score_fig = plot_component_score_curve(fitted_zapline, show=False)
    pattern_fig = plot_component_patterns(fitted_zapline, n_components=2, show=False)
    assert isinstance(score_fig, plt.Figure)
    assert isinstance(pattern_fig, plt.Figure)


def test_plot_component_summary_contains_requested_public_sections(
    fitted_dss, synthetic_data
):
    """A component summary includes selected patterns, traces, PSDs, and CI."""
    fig = plot_component_summary(
        fitted_dss,
        data=synthetic_data,
        info=synthetic_data.info,
        picks=np.arange(len(synthetic_data.ch_names)),
        times=synthetic_data.times,
        n_components=[0, 2],
        psd_fmax=40.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    titles = [ax.get_title() for ax in fig.axes]
    assert any("Comp 0 Pattern" in title for title in titles)
    assert any("Comp 2 Time Course" in title for title in titles)
    psd_axes = [ax for ax in fig.axes if ax.get_title() == "PSD"]
    assert psd_axes
    assert all(ax.get_xlim()[1] == pytest.approx(40.0) for ax in psd_axes)
    assert any(ax.collections for ax in fig.axes if "Time Course" in ax.get_title())

    no_ci = plot_component_summary(
        fitted_dss,
        data=synthetic_data,
        times=synthetic_data.times,
        n_components=[0],
        plot_ci=False,
        show=False,
    )
    time_axes = [ax for ax in no_ci.axes if "Time Course" in ax.get_title()]
    assert time_axes
    assert all(not ax.collections for ax in time_axes)


def test_plot_component_summary_validates_info_sfreq_and_selection(fitted_dss):
    """Missing metadata and empty public selections fail clearly."""
    with pytest.raises(ValueError, match="Data must be provided"):
        plot_component_summary(fitted_dss, show=False)

    with pytest.raises(ValueError, match="sfreq is required"):
        plot_component_summary(ArrayOnlyEst(), show=False)

    with pytest.raises(ValueError, match="No components selected"):
        plot_component_summary(ArrayOnlyEst(), sfreq=100.0, n_components=[], show=False)


def test_plot_component_epochs_image_preserves_component_indices_and_axes(
    fitted_dss, synthetic_data
):
    """Epoch images expose the requested component labels and sample axis."""
    fig = plot_component_epochs_image(
        fitted_dss,
        data=synthetic_data,
        n_components=[0, 2],
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    assert {ax.get_title() for ax in fig.axes} >= {"Comp 0", "Comp 2"}
    assert any(ax.get_ylabel() == "Epochs" for ax in fig.axes)
    assert any(ax.get_xlabel() == "Time (samples)" for ax in fig.axes)

    bad = type("BadSources", (), {"sources_": np.zeros(3)})()
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        plot_component_epochs_image(bad, show=False)


def test_plot_component_time_series_normalizes_traces_and_uses_time_axis(
    fitted_dss, synthetic_data
):
    """Time-series plots use supplied times and z-score each component."""
    times = np.linspace(-0.2, 0.5, synthetic_data.get_data().shape[-1])
    fig = plot_component_time_series(
        fitted_dss,
        data=synthetic_data,
        n_components=[0],
        times=times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    time_axis = next(ax for ax in fig.axes if ax.get_title() == "Component Time Series")
    line = next(iter(time_axis.lines))
    np.testing.assert_allclose(line.get_xdata(), times)
    assert np.std(line.get_ydata()) == pytest.approx(1.0)
    assert time_axis.get_xlabel() == "Time"

    with pytest.raises(ValueError, match="times must have length"):
        plot_component_time_series(
            fitted_dss,
            data=synthetic_data,
            times=times[:-1],
            show=False,
        )


def test_plot_component_spectrogram_exposes_frequency_and_time_units():
    """Spectrograms use the requested frequency grid and documented units."""
    data = np.sin(2.0 * np.pi * 10.0 * np.arange(200) / 100.0)
    fig = plot_component_spectrogram(
        data,
        sfreq=100.0,
        freqs=np.array([5.0, 10.0, 15.0]),
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_title() == "Component Spectrogram")
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Frequency (Hz)"
    assert any(ax.get_ylabel() == "Power" for ax in fig.axes)

    with pytest.raises(ValueError, match="fmax must be strictly positive"):
        plot_component_spectrogram(data, sfreq=100.0, fmax=0.0, show=False)


def test_component_plot_fname_and_window_score_threshold(tmp_path):
    """Public file output and threshold annotations work without inspecting layout."""
    path = tmp_path / "scores.png"
    fig = plot_component_score_curve(ArrayOnlyEst(), show=False, fname=str(path))
    assert isinstance(fig, plt.Figure)
    assert path.exists()

    scores = np.arange(6.0).reshape(2, 3)
    fig = plot_window_score_traces(scores, threshold=0.5, show=False)
    assert isinstance(fig, plt.Figure)
    score_axis = next(ax for ax in fig.axes if ax.get_title() == "Window Score Traces")
    assert any(np.allclose(line.get_ydata(), 0.5) for line in score_axis.lines)
    with pytest.raises(ValueError, match="2D array"):
        plot_window_score_traces(np.ones(3), show=False)


def test_plot_component_summary_handles_fitted_subset_patterns():
    """Short fitted pattern arrays do not fail when the selected channels match."""
    info = mne.create_info(
        ch_names=["Fp1", "Fp2", "EOG"], sfreq=100.0, ch_types=["eeg", "eeg", "eog"]
    )
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
    )

    class SubsetEstimator:
        patterns_ = np.ones((2, 1))
        sources_ = np.ones((1, 100))

    fig = plot_component_summary(
        SubsetEstimator(),
        info=info,
        picks=[0, 1],
        sfreq=100.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
