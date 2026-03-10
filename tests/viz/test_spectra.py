"""Tests for mne_denoise.viz.spectra functions."""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_component_psd_comparison,
    plot_narrowband_score_scan,
    plot_psd_comparison,
    plot_psd_gallery,
    plot_psd_overlay,
    plot_psd_zoom_comparison,
    plot_spectrogram_comparison,
    plot_time_frequency_mask,
)


def test_plot_narrowband_score_scan():
    """Test narrowband scan visualization."""
    frequencies = np.arange(5, 30, 0.5)
    eigenvalues = np.random.randn(len(frequencies)) * 0.1 + 0.5

    peak_idx = np.argmin(np.abs(frequencies - 10))
    eigenvalues[peak_idx] = 2.0
    eigenvalues[peak_idx - 1 : peak_idx + 2] += np.array([0.5, 0, 0.5])

    fig = plot_narrowband_score_scan(frequencies, eigenvalues, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_narrowband_score_scan(
        frequencies,
        eigenvalues,
        peak_freq=frequencies[peak_idx],
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_narrowband_score_scan(
        frequencies,
        eigenvalues,
        true_freqs=[10, 22],
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_narrowband_score_scan(
        frequencies,
        eigenvalues,
        peak_freq=10.0,
        true_freqs=[10, 22],
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    fig_ret = plot_narrowband_score_scan(frequencies, eigenvalues, ax=ax, show=False)
    assert fig_ret is fig

    with pytest.raises(ValueError, match="matching first dimensions"):
        plot_narrowband_score_scan(frequencies, eigenvalues[:-1], show=False)


def test_plot_time_frequency_mask():
    """Test TF mask visualization."""
    n_freqs, n_times = 10, 20
    mask = np.random.rand(n_freqs, n_times)
    times = np.arange(n_times)
    freqs = np.arange(n_freqs)

    fig = plot_time_frequency_mask(mask, times, freqs, show=False)
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    fig_ret = plot_time_frequency_mask(mask, times, freqs, ax=ax, show=False)
    assert fig_ret is fig

    with pytest.raises(ValueError, match="mask shape must match"):
        plot_time_frequency_mask(mask[:, :-1], times, freqs, show=False)

    with pytest.raises(ValueError, match="mask must be a 2D array"):
        plot_time_frequency_mask(mask[0], times, freqs, show=False)


def test_plot_psd_comparison(fitted_dss, synthetic_data):
    """Test PSD comparison visualization."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    raw_before = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    raw_after = mne.io.RawArray(epochs_clean.get_data()[0], synthetic_data.info)

    fig, ax = plt.subplots()
    fig_ret = plot_psd_comparison(synthetic_data, epochs_clean, ax=ax, show=False)
    assert fig_ret is fig

    fig = plot_psd_comparison(raw_before, raw_after, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_comparison(synthetic_data, epochs_clean, average=False, show=False)
    assert isinstance(fig, plt.Figure)

    array_before = synthetic_data.get_data()[0]
    array_after = epochs_clean.get_data()[0]
    fig = plot_psd_comparison(
        array_before,
        array_after,
        sfreq=synthetic_data.info["sfreq"],
        line_freq=10.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError, match="sfreq must be provided"):
        plot_psd_comparison(array_before, array_after, show=False)


def test_plot_psd_comparison_with_zapline_arrays(zapline_data, fitted_zapline):
    """Test PSD comparison with ZapLine-style array inputs."""
    data, sfreq = zapline_data
    cleaned = fitted_zapline.transform(data)

    fig = plot_psd_comparison(data, cleaned, sfreq=sfreq, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_comparison(
        data,
        cleaned,
        sfreq=sfreq,
        line_freq=50.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig_ext, custom_ax = plt.subplots()
    ret_fig = plot_psd_comparison(
        data,
        cleaned,
        sfreq=sfreq,
        ax=custom_ax,
        show=False,
    )
    assert ret_fig is fig_ext

    fig = plot_psd_comparison(data, cleaned, sfreq=sfreq, fmax=150.0, show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_component_psd_comparison(fitted_dss, synthetic_data):
    """Test component PSD comparison visualization."""
    sources = fitted_dss.transform(synthetic_data)

    fig = plot_component_psd_comparison(
        synthetic_data,
        sources,
        component_indices=[0, 1, 2],
        sfreq=100,
        peak_freq=10,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_component_psd_comparison(
        synthetic_data,
        sources,
        component_indices=[0, 1],
        sfreq=100,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    sources_raw = fitted_dss.transform(raw)
    fig = plot_component_psd_comparison(
        raw,
        sources_raw,
        component_indices=[0, 1],
        sfreq=100,
        peak_freq=10,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_component_psd_comparison(
        raw,
        sources_raw,
        component_indices=[0, 1],
        sfreq=100,
        fmin=5,
        fmax=20,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError, match="component_indices cannot be empty"):
        plot_component_psd_comparison(
            raw,
            sources_raw,
            component_indices=[],
            sfreq=100,
            show=False,
        )


def test_plot_psd_zoom_comparison():
    """Test PSD comparison with zoom panels."""
    freqs = np.arange(0, 200, 0.5)
    psd_before = np.ones_like(freqs) * 1e-6
    psd_after = np.ones_like(freqs) * 1e-5

    fig = plot_psd_zoom_comparison(
        freqs,
        psd_before,
        freqs,
        psd_after,
        series_name="M1",
        title="sub-01",
        zoom_freqs=[50.0, 100.0],
        zoom_annotations=["atten=10 dB", "atten=8 dB"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_psd_gallery():
    """Test PSD gallery visualization."""
    freqs = np.arange(0, 200, 0.5)
    psd_before = np.ones_like(freqs) * 1e-6
    series_psds = {
        "M1": (freqs, np.ones_like(freqs) * 1e-5),
        "M2": (freqs, np.ones_like(freqs) * 1.1e-5),
    }

    fig = plot_psd_gallery(
        freqs, psd_before, series_psds, zoom_freqs=[50.0], show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_gallery(
        freqs,
        psd_before,
        series_psds,
        zoom_freqs=[50.0, 100.0],
        series_order=["M2", "M1"],
        title="sub-01",
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_gallery(
        freqs,
        psd_before,
        series_psds,
        zoom_freqs=[50.0, 100.0],
        series_order=["M1", "MISSING"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_psd_overlay():
    """Test PSD overlay visualization."""
    freqs = np.arange(0, 200, 0.5)
    psd_before = np.ones_like(freqs) * 1e-5
    series_psds = {
        "M0": (freqs, np.ones_like(freqs) * 1e-6),
        "M1": (freqs, np.ones_like(freqs) * 1.2e-6),
    }

    fig = plot_psd_overlay(freqs, psd_before, series_psds, focus_freq=50.0, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_overlay(
        freqs,
        psd_before,
        series_psds,
        focus_freq=50.0,
        n_harmonics=2,
        title="sub-01",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_spectrogram_comparison(fitted_dss, synthetic_data):
    """Test spectrogram comparison visualization."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    fig = plot_spectrogram_comparison(
        synthetic_data,
        epochs_clean,
        picks=[0, 1],
        times=synthetic_data.times,
        fmin=1,
        fmax=20,
        n_freqs=5,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    array_before = synthetic_data.get_data().mean(axis=0)
    array_after = epochs_clean.get_data().mean(axis=0)
    fig = plot_spectrogram_comparison(
        array_before,
        array_after,
        picks=[0, 1],
        times=synthetic_data.times,
        sfreq=synthetic_data.info["sfreq"],
        fmin=1,
        fmax=20,
        n_freqs=5,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError, match="n_freqs must be at least 2"):
        plot_spectrogram_comparison(
            synthetic_data,
            epochs_clean,
            picks=[0],
            times=synthetic_data.times,
            fmin=1,
            fmax=20,
            n_freqs=1,
            show=False,
        )

    with pytest.raises(ValueError, match="picks must be provided explicitly"):
        plot_spectrogram_comparison(
            synthetic_data,
            epochs_clean,
            picks=None,  # type: ignore[arg-type]
            times=synthetic_data.times,
            show=False,
        )

    with pytest.raises(ValueError, match="sfreq must be provided"):
        plot_spectrogram_comparison(
            array_before,
            array_after,
            picks=[0, 1],
            times=synthetic_data.times,
            show=False,
        )


def test_spectra_internal_helpers():
    """Test internal helper functions in spectra.py."""
    from mne_denoise.viz.spectra import (
        _as_component_data,
        _compute_array_psd,
        _compute_array_spectrogram,
    )

    # _compute_array_psd
    freqs, psd = _compute_array_psd(np.random.randn(100), sfreq=100, fmin=1, fmax=40)
    assert psd.ndim == 2
    freqs, psd = _compute_array_psd(
        np.random.randn(2, 2, 100), sfreq=100, fmin=1, fmax=40
    )
    assert psd.ndim == 2

    # _as_component_data
    data = _as_component_data(np.random.randn(100))
    assert data.ndim == 2
    data = _as_component_data(np.random.randn(2, 100))
    assert data.ndim == 2
    data = _as_component_data(np.random.randn(2, 3, 100))
    assert data.ndim == 2
    with pytest.raises(ValueError, match="components must be 1D, 2D, or 3D"):
        _as_component_data(np.random.randn(2, 2, 2, 100))

    # _compute_array_spectrogram
    with pytest.raises(ValueError, match="Array spectrogram inputs must be 2D"):
        _compute_array_spectrogram(np.random.randn(2, 2, 2, 100), [0], 100, 1, 40, 5)

    # 3D array in spectrogram
    freqs, spec = _compute_array_spectrogram(
        np.random.randn(5, 2, 100), [0], 100, 1, 40, 5
    )
    assert spec.ndim == 2

    # Invalid frequency grid
    with pytest.raises(ValueError, match="Could not compute a valid frequency grid"):
        _compute_array_spectrogram(np.random.randn(2, 10), [0], 10, 50, 60, 5)


def test_plot_narrowband_score_scan_edges():
    """Test edge cases for narrowband score scan."""
    freqs = np.linspace(5, 30, 10)
    scores = np.random.randn(10, 2)

    # frequencies.ndim != 1
    with pytest.raises(ValueError, match="frequencies must be a 1D array"):
        plot_narrowband_score_scan(freqs[:, np.newaxis], scores, show=False)

    # eigenvalues.ndim not in (1, 2)
    with pytest.raises(ValueError, match="eigenvalues must be a 1D or 2D array"):
        plot_narrowband_score_scan(freqs, scores[:, :, np.newaxis], show=False)

    # 2D eigenvalues (multiple components)
    fig = plot_narrowband_score_scan(freqs, scores, show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_psd_comparison_edges(synthetic_data):
    """Test edge cases for PSD comparison."""
    data = synthetic_data.get_data()

    # average=False branch
    fig = plot_psd_comparison(
        data, data, sfreq=synthetic_data.info["sfreq"], average=False, show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_psd_zoom_comparison_edges():
    """Test edge cases for PSD zoom comparison."""
    freqs = np.linspace(0, 100, 50)
    psd = np.ones(50)

    # zoom_freqs invalid
    with pytest.raises(
        ValueError, match="zoom_freqs must be a non-empty 1D array-like"
    ):
        plot_psd_zoom_comparison(freqs, psd, freqs, psd, zoom_freqs=[], show=False)

    # zoom_half_width_hz <= 0
    with pytest.raises(ValueError, match="zoom_half_width_hz must be positive"):
        plot_psd_zoom_comparison(
            freqs, psd, freqs, psd, zoom_freqs=[10], zoom_half_width_hz=0, show=False
        )

    # series overrides and annotations
    fig = plot_psd_zoom_comparison(
        freqs,
        psd,
        freqs,
        psd,
        series_name="A",
        zoom_freqs=[10, 20],
        zoom_annotations=["X", "Y"],
        series_colors={"A": "red"},
        series_labels={"A": "Label A"},
        title="Title",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_psd_gallery_edges():
    """Test edge cases for PSD gallery."""
    freqs = np.linspace(0, 100, 50)
    psd = np.ones(50)
    series_psds = {"A": (freqs, psd)}

    # zoom_freqs invalid
    with pytest.raises(
        ValueError, match="zoom_freqs must be a non-empty 1D array-like"
    ):
        plot_psd_gallery(freqs, psd, series_psds, zoom_freqs=[], show=False)

    # zoom_half_width_hz <= 0
    with pytest.raises(ValueError, match="zoom_half_width_hz must be positive"):
        plot_psd_gallery(
            freqs, psd, series_psds, zoom_freqs=[10], zoom_half_width_hz=0, show=False
        )

    # Single row layout
    fig = plot_psd_gallery(freqs, psd, series_psds, zoom_freqs=[10], show=False)
    assert isinstance(fig, plt.Figure)

    # Custom colors and labels
    fig = plot_psd_gallery(
        freqs,
        psd,
        series_psds,
        zoom_freqs=[10],
        series_colors={"A": "blue"},
        series_labels={"A": "A_Label"},
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_psd_overlay_edges():
    """Test edge cases for PSD overlay."""
    freqs = np.linspace(0, 100, 50)
    psd = np.ones(50)
    series_psds = {"A": (freqs, psd)}

    # focus_half_width_hz <= 0
    with pytest.raises(ValueError, match="focus_half_width_hz must be positive"):
        plot_psd_overlay(
            freqs, psd, series_psds, focus_freq=10, focus_half_width_hz=0, show=False
        )

    # Custom order, colors, labels
    fig = plot_psd_overlay(
        freqs,
        psd,
        series_psds,
        focus_freq=10,
        series_order=["A", "B"],
        series_colors={"A": "green"},
        series_labels={"A": "A_Label"},
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_component_psd_comparison_edges(synthetic_data):
    """Test edge cases for component PSD comparison."""
    data = synthetic_data.get_data()

    # index out of range
    with pytest.raises(ValueError, match="out of range"):
        plot_component_psd_comparison(
            data, data, component_indices=[100], sfreq=100, show=False
        )

    with pytest.raises(
        ValueError, match="sfreq must be provided when components are arrays"
    ):
        plot_component_psd_comparison(
            synthetic_data, data, component_indices=[0], sfreq=None, show=False
        )


def test_plot_psd_zoom_comparison_no_series():
    """Test PSD zoom comparison without series name."""
    freqs = np.linspace(0, 100, 50)
    psd = np.ones(50)
    fig = plot_psd_zoom_comparison(freqs, psd, freqs, psd, zoom_freqs=[10], show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_component_psd_comparison_mne_sfreq(synthetic_data):
    """Test component PSD comparison with MNE sfreq inference."""
    epochs = synthetic_data
    # Use MNE object for components to trigger line 959
    fig = plot_component_psd_comparison(
        epochs, epochs, component_indices=[0], show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_spectrogram_comparison_more_edges(synthetic_data):
    """Test more edge cases for spectrogram comparison."""
    # Ensure raw data is used to avoid any MNE confusion
    data = synthetic_data.get_data()
    times = synthetic_data.times
    sfreq = synthetic_data.info["sfreq"]
    n_times = data.shape[-1]
    assert len(times) == n_times

    evoked = synthetic_data.average()
    fig = plot_spectrogram_comparison(
        evoked, evoked, picks=[0], times=evoked.times, show=False
    )
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError, match="Array spectrogram inputs must be 2D or 3D"):
        very_bad_data = np.zeros((1, 1, 1, n_times))
        plot_spectrogram_comparison(
            very_bad_data,
            very_bad_data,
            picks=[0],
            times=times,
            sfreq=sfreq,
            show=False,
        )


def test_plot_time_frequency_mask_more_edges():
    """Test more edge cases for TF mask."""
    mask = np.zeros((10, 20))
    times = np.zeros((20,))
    freqs = np.zeros((10,))

    # Line 1248: times.ndim != 1
    with pytest.raises(ValueError, match="times and freqs must be 1D arrays"):
        plot_time_frequency_mask(mask, times[:, np.newaxis], freqs, show=False)

    # Line 1248: freqs.ndim != 1
    with pytest.raises(ValueError, match="times and freqs must be 1D arrays"):
        plot_time_frequency_mask(mask, times, freqs[:, np.newaxis], show=False)


def test_plot_component_psd_comparison_raw(synthetic_data):
    """Test component PSD with Raw input for coverage."""
    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    # This should trigger line 959 since raw has .info['sfreq']
    fig = plot_component_psd_comparison(raw, raw, component_indices=[0], show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_spectrogram_comparison_edges(synthetic_data):
    """Test edge cases for spectrogram comparison."""
    epochs = synthetic_data
    half_epochs = epochs.copy()
    half_epochs.crop(tmin=0, tmax=0.5)
    data = epochs.get_data()

    # fmax <= fmin
    with pytest.raises(ValueError, match="fmax must be greater than fmin"):
        plot_spectrogram_comparison(
            epochs, epochs, picks=[0], times=epochs.times, fmin=40, fmax=10, show=False
        )

    # empty picks
    with pytest.raises(ValueError, match="picks cannot be empty"):
        plot_spectrogram_comparison(
            epochs, epochs, picks=[], times=epochs.times, show=False
        )

    # n_times mismatch (MNE)
    with pytest.raises(ValueError, match="share the same n_times"):
        plot_spectrogram_comparison(
            epochs, half_epochs, picks=[0], times=epochs.times, show=False
        )

    # times.size mismatch (MNE)
    with pytest.raises(ValueError, match="times must match the signal n_times"):
        plot_spectrogram_comparison(
            epochs, epochs, picks=[0], times=epochs.times[:-1], show=False
        )

    # Mixed types
    with pytest.raises(ValueError, match="must be both MNE or both arrays"):
        plot_spectrogram_comparison(
            epochs, data, picks=[0], times=epochs.times, show=False
        )

    # times.ndim != 1
    with pytest.raises(ValueError, match="times must be a 1D array"):
        plot_spectrogram_comparison(
            data[0],
            data[0],
            picks=[0],
            times=epochs.times[:, np.newaxis],
            sfreq=100,
            show=False,
        )

    # n_times mismatch (Array)
    with pytest.raises(ValueError, match="share the same n_times"):
        plot_spectrogram_comparison(
            data[0],
            data[0, :, :-1],
            picks=[0],
            times=epochs.times,
            sfreq=100,
            show=False,
        )

    # times.size mismatch (Array)
    with pytest.raises(ValueError, match="times must match the signal n_times"):
        plot_spectrogram_comparison(
            data[0], data[0], picks=[0], times=epochs.times[:-1], sfreq=100, show=False
        )

    # pick index out of range
    with pytest.raises(ValueError, match="Channel picks out of range"):
        plot_spectrogram_comparison(
            data[0], data[0], picks=[100], times=epochs.times, sfreq=100, show=False
        )
