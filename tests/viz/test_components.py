"""Tests for mne_denoise.viz.components functions."""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.dss import DSS
from mne_denoise.viz import (
    plot_component_epochs_image,
    plot_component_patterns,
    plot_component_score_curve,
    plot_component_spectrogram,
    plot_component_summary,
    plot_component_time_series,
)
from mne_denoise.viz.components import _resolve_component_indices


class ArrayOnlyEst:
    """Mock estimator with patterns and sources but no MNE info."""

    patterns_ = np.random.randn(5, 3)
    sources_ = np.random.randn(3, 200)
    eigenvalues_ = np.array([1.0, 0.5, 0.1])

    def get_params(self, deep=True):
        return {}

    def transform(self, data):
        return self.sources_


class NoScoreEst:
    """Mock estimator with no scores."""

    pass


def test_plot_component_score_curve(fitted_dss):
    """Test score curve plotting."""
    fig = plot_component_score_curve(fitted_dss, mode="raw", show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_score_curve(fitted_dss, mode="cumulative", show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_score_curve(fitted_dss, mode="ratio", show=False)
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    plot_component_score_curve(fitted_dss, ax=ax, show=False)

    # Test n_selected_ branch specifically
    SelectedEst = type(
        "SelectedEst", (), {"scores_": np.array([2.0, 1.0]), "n_selected_": 1}
    )
    plot_component_score_curve(SelectedEst(), show=False)

    with pytest.raises(ValueError, match="mode must be one of"):
        plot_component_score_curve(fitted_dss, mode="bad-mode", show=False)

    with pytest.raises(ValueError, match="does not expose component scores"):
        plot_component_score_curve(NoScoreEst(), show=False)

    MockBadScores = type("MockBadScores", (), {"scores_": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="must be a non-empty 1D array"):
        plot_component_score_curve(MockBadScores(), show=False)

    # Test n_removed_ branch
    CutoffEst = type(
        "CutoffEst", (), {"scores_": np.array([2.0, 1.0, 0.5]), "n_removed_": 1}
    )
    fig = plot_component_score_curve(CutoffEst(), show=False)
    assert isinstance(fig, plt.Figure)

    assert _resolve_component_indices(None, 10, 5) == [0, 1, 2, 3, 4]
    assert _resolve_component_indices(2, 10, 5) == [0, 1]
    assert _resolve_component_indices([1, 3], 10, 5) == [1, 3]
    with pytest.raises(ValueError, match="indices out of range"):
        _resolve_component_indices([11], 10, 5)


def test_plot_component_patterns(fitted_dss, synthetic_data):
    """Test topomap plotting."""
    picks = np.arange(len(synthetic_data.ch_names))
    fig = plot_component_patterns(
        fitted_dss, info=synthetic_data.info, picks=picks, show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_component_patterns(
        fitted_dss,
        info=synthetic_data.info,
        picks=picks,
        n_components=2,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    # Test grid padding (ax.axis('off') for unused axes)
    class ManyCompEst:
        patterns_ = np.zeros((5, 5))
        info_ = synthetic_data.info

        def get_params(self, deep=True):
            return {}

    fig = plot_component_patterns(
        ManyCompEst(),
        n_components=5,
        info=synthetic_data.info,
        picks=[0, 1, 2],
        show=False,
    )
    assert len(fig.axes) > 5

    class MockEst:
        patterns_ = np.zeros((5, 3))

    fig = plot_component_patterns(MockEst(), show=False)
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    fig_ret = plot_component_patterns(MockEst(), ax=ax, show=False)
    assert fig_ret is fig

    with pytest.raises(ValueError, match="info is required"):
        plot_component_patterns(fitted_dss, picks=[0, 1], show=False)

    fig, ax = plt.subplots()
    plot_component_patterns(
        fitted_dss,
        info=synthetic_data.info,
        picks=[0, 1],
        n_components=[0],
        ax=ax,
        show=False,
    )
    assert ax.get_title() == "Comp 0"

    with pytest.raises(ValueError, match="No components selected"):
        plot_component_patterns(fitted_dss, n_components=[], show=False)

    BadDimEst = type("BadDimEst", (), {"patterns_": np.zeros((5,))})
    with pytest.raises(ValueError, match="must be a 2D array"):
        plot_component_patterns(BadDimEst(), show=False)

    with pytest.raises(
        ValueError, match="ax can only be used when plotting a single topomap"
    ):
        plot_component_patterns(
            fitted_dss,
            info=synthetic_data.info,
            picks=[0, 1],
            n_components=2,
            ax=plt.subplots()[1],
            show=False,
        )


def test_component_primitives_support_zapline(fitted_zapline):
    """Component primitives should also work with fitted ZapLine estimators."""
    fig = plot_component_score_curve(fitted_zapline, show=False)
    assert isinstance(fig, plt.Figure)

    fig_ext, custom_ax = plt.subplots()
    ret_fig = plot_component_score_curve(fitted_zapline, ax=custom_ax, show=False)
    assert ret_fig is fig_ext

    fig = plot_component_patterns(fitted_zapline, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_patterns(fitted_zapline, n_components=2, show=False)
    assert isinstance(fig, plt.Figure)

    fig_ext, custom_ax = plt.subplots()
    ret_fig = plot_component_patterns(fitted_zapline, ax=custom_ax, show=False)
    assert ret_fig is fig_ext


def test_plot_component_summary(fitted_dss, synthetic_data):
    """Test component summary dashboard."""
    fig = plot_component_summary(
        fitted_dss,
        data=synthetic_data,
        info=synthetic_data.info,
        picks=np.arange(len(synthetic_data.ch_names)),
        times=synthetic_data.times,
        n_components=2,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_component_summary(
        fitted_dss,
        data=synthetic_data,
        times=synthetic_data.times,
        n_components=2,
        psd_fmax=40.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    psd_axes = [ax for ax in fig.axes if ax.get_title() == "PSD"]
    assert psd_axes
    for ax in psd_axes:
        assert ax.get_xlim()[1] == pytest.approx(40.0)

    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    fig = plot_component_summary(
        fitted_dss, data=raw, times=raw.times, n_components=1, show=False
    )
    assert isinstance(fig, plt.Figure)

    plot_component_summary(
        fitted_dss,
        data=synthetic_data,
        times=synthetic_data.times,
        n_components=[0, 2],
        show=False,
    )

    def dummy_bias(d):
        return d

    DSS(n_components=3, bias=dummy_bias)
    with pytest.raises(ValueError, match="Data must be provided"):
        plot_component_summary(fitted_dss, data=None, show=False)

    # Missing info/picks/data-type combination
    with pytest.raises(ValueError, match="info is required when picks is provided"):
        plot_component_summary(
            ArrayOnlyEst(), data=np.random.randn(1, 200), picks=[0], show=False
        )

    with pytest.raises(
        ValueError, match="sfreq is required when info is not available"
    ):
        plot_component_summary(ArrayOnlyEst(), data=np.random.randn(1, 200), show=False)

    with pytest.raises(ValueError, match="psd_fmax must be strictly positive"):
        plot_component_summary(
            fitted_dss,
            data=synthetic_data,
            times=synthetic_data.times,
            psd_fmax=0,
            show=False,
        )

    with pytest.raises(ValueError, match="times must have length"):
        plot_component_summary(
            fitted_dss,
            data=synthetic_data,
            times=np.arange(synthetic_data.times.size - 1),
            show=False,
        )

    with pytest.raises(ValueError, match="sfreq must be strictly positive"):
        plot_component_summary(ArrayOnlyEst(), sfreq=0, show=False)

    with pytest.raises(ValueError, match="No components selected"):
        plot_component_summary(
            fitted_dss, data=synthetic_data, n_components=[], show=False
        )

    # Test plot_ci branch
    plot_component_summary(
        fitted_dss, data=synthetic_data, n_components=1, plot_ci=False, show=False
    )

    fig = plot_component_summary(
        ArrayOnlyEst(),
        times=np.arange(200) / 250.0,
        sfreq=250.0,
        n_components=2,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    # Cover no Topomap info text branch
    plot_component_summary(
        ArrayOnlyEst(),
        times=np.arange(200) / 250.0,
        sfreq=250.0,
        picks=None,
        show=False,
    )

    # Cover n_cycles and default fmax in spectrogram (via summary)
    plot_component_summary(fitted_dss, data=synthetic_data, show=False)


def test_plot_component_epochs_image(fitted_dss, synthetic_data):
    """Test component image plotting."""
    fig = plot_component_epochs_image(
        fitted_dss, data=synthetic_data, n_components=2, show=False
    )
    assert isinstance(fig, plt.Figure)

    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    fig = plot_component_epochs_image(fitted_dss, data=raw, n_components=1, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_epochs_image(
        fitted_dss,
        data=synthetic_data,
        n_components=[0, 2],
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    BadDimEst = type("BadDimEst", (), {"sources_": np.zeros((3,))})
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        plot_component_epochs_image(BadDimEst(), show=False)

    with pytest.raises(ValueError, match="No components selected"):
        plot_component_epochs_image(
            fitted_dss, data=synthetic_data, n_components=[], show=False
        )


def test_plot_component_time_series(fitted_dss, synthetic_data):
    """Test stacked time series plotting."""
    fig = plot_component_time_series(
        fitted_dss, data=synthetic_data, times=synthetic_data.times, show=False
    )
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError):
        plot_component_time_series(fitted_dss, data=None, show=False)

    class MockEst:
        sources_ = np.random.randn(3, 100)
        eigenvalues_ = np.array([1.0, 0.5, 0.1])

        def get_params(self, deep=True):
            return {}

    fig = plot_component_time_series(MockEst(), times=np.arange(100), show=False)
    assert isinstance(fig, plt.Figure)

    # Test low variance branch
    DeadEst = type("DeadEst", (), {"sources_": np.zeros((1, 100))})
    fig = plot_component_time_series(DeadEst(), show=False)
    assert isinstance(fig, plt.Figure)

    # Test existing ax branch
    fig, ax = plt.subplots()
    plot_component_time_series(fitted_dss, data=synthetic_data, ax=ax, show=False)

    with pytest.raises(ValueError, match="times must have length"):
        plot_component_time_series(
            fitted_dss,
            data=synthetic_data,
            times=np.arange(synthetic_data.times.size - 1),
            show=False,
        )

    with pytest.raises(ValueError, match="No components selected"):
        plot_component_time_series(
            fitted_dss, data=synthetic_data, n_components=[], show=False
        )


def test_plot_component_spectrogram():
    """Test component TFR plotting."""
    sfreq = 100.0
    n_times = 200

    comp_1d = np.random.randn(n_times)
    fig = plot_component_spectrogram(comp_1d, sfreq=sfreq, show=False)
    assert isinstance(fig, plt.Figure)

    comp_2d = np.random.randn(1, n_times)
    fig = plot_component_spectrogram(
        comp_2d, sfreq=sfreq, freqs=np.arange(1, 10), show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_component_spectrogram(comp_1d, sfreq=sfreq, fmax=30, show=False)
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_ylim()[1] == pytest.approx(30.0, abs=1.0)

    fig, ax = plt.subplots()
    ret_fig = plot_component_spectrogram(comp_1d, sfreq=sfreq, ax=ax, show=False)
    assert ret_fig is fig

    with pytest.raises(ValueError, match="fmax must be strictly positive"):
        plot_component_spectrogram(comp_1d, sfreq=sfreq, fmax=0, show=False)

    BadDimComp = np.zeros((1, 1, 1, 1))
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        plot_component_spectrogram(BadDimComp, sfreq=sfreq, show=False)

    # Cover fmax=None and n_cycles=None branches
    # Use long signal to avoid "wavelet longer than signal" error
    comp_long = np.random.randn(1000)
    plot_component_spectrogram(comp_long, sfreq=100.0, fmax=None, show=False)
    plot_component_spectrogram(comp_long, sfreq=100.0, n_cycles=None, show=False)
    # Explicit n_cycles to cover the 'else' (skip) branch
    plot_component_spectrogram(comp_long, sfreq=100.0, n_cycles=5, show=False)


def test_fname_parameter_components(fitted_dss, tmp_path):
    """Test fname parameter on a component plot function."""
    fpath = tmp_path / "scores.png"
    fig = plot_component_score_curve(
        fitted_dss,
        mode="raw",
        show=False,
        fname=str(fpath),
    )
    assert isinstance(fig, plt.Figure)
    assert fpath.exists()

    fpath = tmp_path / "patterns.png"
    fig = plot_component_patterns(
        fitted_dss,
        info=fitted_dss.info_,
        picks=np.arange(fitted_dss.info_["nchan"]),
        show=False,
        fname=str(fpath),
    )
    assert isinstance(fig, plt.Figure)
    assert fpath.exists()


def test_plot_component_summary_zapline_mock():
    """Test is_zapline branch in summary."""
    ZapMock = type(
        "ZapMock",
        (),
        {
            "scores_": np.array([1.0]),
            "patterns_": np.zeros((5, 1)),
            "sources_": np.zeros((1, 100)),
            "line_freq": 60.0,
            "get_params": lambda self, deep=True: {},
            "transform": lambda self, d: self.sources_,
        },
    )
    plot_component_summary(ZapMock(), sfreq=100.0, data=np.zeros((5, 100)), show=False)
