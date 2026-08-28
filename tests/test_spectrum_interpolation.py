"""Scientific and method-specific contracts for spectrum interpolation."""

from __future__ import annotations

import logging

import mne
import numpy as np
import pytest

from mne_denoise.spectrum_interpolation import (
    SpectrumInterpolation,
    interpolate_spectrum,
)


def _band_power(data, f0, sfreq, half_bw=0.5):
    spec = np.abs(np.fft.rfft(data, axis=1))
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
    mask = (freqs >= f0 - half_bw) & (freqs <= f0 + half_bw)
    return float((spec[:, mask] ** 2).sum())


def _synth(sfreq=1000.0, dur=8.0, n_ch=4, line=60.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sfreq * dur)
    times = np.arange(n) / sfreq
    neural = np.sin(2 * np.pi * 10 * times)
    line_noise = 3.0 * np.sin(2 * np.pi * line * times)
    data = rng.standard_normal((n_ch, n)) * 0.5
    data += neural[None, :] + line_noise[None, :]
    return data, sfreq


def test_interpolate_spectrum_attenuates_line_without_suppressing_neural_band():
    """Line power is removed while a non-target neural band is preserved."""
    data, sfreq = _synth()
    clean = interpolate_spectrum(data, sfreq, [60.0])

    line_before = _band_power(data, 60, sfreq)
    line_after = _band_power(clean, 60, sfreq)
    neural_before = _band_power(data, 10, sfreq)
    neural_after = _band_power(clean, 10, sfreq)

    assert clean.shape == data.shape
    assert np.isrealobj(clean)
    assert line_after / line_before < 0.01
    assert np.isclose(neural_after, neural_before, rtol=1e-6)


def test_interpolate_spectrum_interpolates_target_power_and_preserves_spectrum():
    """Target power follows neighboring amplitudes and real spectra reconstruct."""
    rng = np.random.default_rng(4)
    data = rng.standard_normal((2, 1000))
    clean = interpolate_spectrum(
        data,
        1000.0,
        [60.0],
        bandwidth=1.0,
        neighbour_width=2.0,
    )

    frequencies = np.fft.rfftfreq(data.shape[1], d=1 / 1000.0)
    before = np.fft.rfft(data, axis=1)
    after = np.fft.rfft(clean, axis=1)
    target = (frequencies >= 59.0) & (frequencies <= 61.0)
    neighbours = ((frequencies >= 57.0) & (frequencies < 59.0)) | (
        (frequencies > 61.0) & (frequencies <= 63.0)
    )
    outside = ~target

    expected_amplitude = np.sqrt(np.abs(before[:, neighbours]) ** 2).mean(
        axis=1, keepdims=True
    )
    np.testing.assert_allclose(
        np.sqrt(np.abs(after[:, target]) ** 2),
        np.broadcast_to(expected_amplitude, after[:, target].shape),
    )
    np.testing.assert_allclose(
        np.abs(after[:, outside]) ** 2, np.abs(before[:, outside]) ** 2
    )
    np.testing.assert_allclose(
        np.angle(after[:, outside]), np.angle(before[:, outside]), atol=1e-9
    )
    assert np.isrealobj(clean)

    full_spectrum = np.fft.fft(clean, axis=1)
    np.testing.assert_allclose(full_spectrum[:, 1:], np.conj(full_spectrum[:, :0:-1]))


def test_spectrum_interpolation_removes_harmonics_and_accepts_multiple_targets():
    """Harmonic expansion and explicit multi-target lists remove only their lines."""
    data, sfreq = _synth(n_ch=2, dur=4.0)
    times = np.arange(data.shape[1]) / sfreq
    data = data + 1.5 * np.sin(2 * np.pi * 120.0 * times)[None, :]
    data = data + 0.75 * np.sin(2 * np.pi * 180.0 * times)[None, :]

    harmonic = SpectrumInterpolation(
        sfreq=sfreq,
        line_freq=60.0,
        n_harmonics=3,
    )
    harmonic_clean = harmonic.fit_transform(data)

    np.testing.assert_array_equal(harmonic.freqs_, [60.0, 120.0, 180.0])
    for frequency in harmonic.freqs_:
        assert (
            _band_power(harmonic_clean, frequency, sfreq)
            / _band_power(data, frequency, sfreq)
            < 0.01
        )

    explicit = SpectrumInterpolation(
        sfreq=sfreq,
        line_freq=[60.0, 120.0],
        n_harmonics=1,
    )
    explicit_clean = explicit.fit_transform(data)
    np.testing.assert_array_equal(explicit.freqs_, [60.0, 120.0])
    for frequency in explicit.freqs_:
        assert (
            _band_power(explicit_clean, frequency, sfreq)
            / _band_power(data, frequency, sfreq)
            < 0.01
        )
    assert np.isclose(
        _band_power(explicit_clean, 180.0, sfreq),
        _band_power(data, 180.0, sfreq),
        rtol=1e-6,
    )


def test_spectrum_interpolation_default_harmonics_stop_below_nyquist():
    """The default target list contains every harmonic strictly below Nyquist."""
    sfreq = 1000.0
    estimator = SpectrumInterpolation(sfreq=sfreq, line_freq=60.0).fit(
        np.zeros((1, 1000))
    )

    np.testing.assert_array_equal(estimator.freqs_, np.arange(60.0, 500.0, 60.0))
    assert np.all(estimator.freqs_ < sfreq / 2)


def test_interpolate_spectrum_handles_boundary_and_out_of_range_targets():
    """Single-sided neighbors work at the boundaries; out-of-range lines pass through."""
    sfreq = 1000.0
    times = np.arange(1000) / sfreq
    data = (
        2.0
        + 3.0 * np.cos(2 * np.pi * 499.0 * times)
        + 0.2 * np.sin(2 * np.pi * 10.0 * times)
    )[None, :]

    clean = interpolate_spectrum(data, sfreq, [0.5, 499.5])
    for frequency in (0.5, 499.5):
        assert (
            _band_power(clean, frequency, sfreq) / _band_power(data, frequency, sfreq)
            < 0.01
        )
    assert clean.shape == data.shape
    assert np.isrealobj(clean)

    out_of_range = interpolate_spectrum(data, sfreq, [600.0])
    np.testing.assert_allclose(out_of_range, data)


def test_spectrum_interpolation_validates_array_dimensions():
    """The low-level and estimator array paths reject unsupported dimensions."""
    data, sfreq = _synth()
    with pytest.raises(ValueError, match="2D"):
        interpolate_spectrum(data[:, :, None], sfreq, [60.0])

    with pytest.raises(ValueError, match="2D or 3D"):
        SpectrumInterpolation(sfreq=sfreq, line_freq=60.0).fit_transform(
            np.zeros(data.shape[1])
        )


def test_spectrum_interpolation_rejects_distinct_invalid_parameters():
    """Positive widths, frequencies, harmonics, and Nyquist bounds are enforced."""
    data, sfreq = _synth()
    with pytest.raises(ValueError, match="sfreq"):
        interpolate_spectrum(data, 0.0, [60.0])
    with pytest.raises(ValueError, match="freqs"):
        interpolate_spectrum(data, sfreq, [0.0])
    with pytest.raises(ValueError, match="bandwidth"):
        interpolate_spectrum(data, sfreq, [60.0], bandwidth=0.0)
    with pytest.raises(ValueError, match="neighbour_width"):
        interpolate_spectrum(data, sfreq, [60.0], neighbour_width=0.0)

    for n_harmonics in (0, 1.5):
        with pytest.raises(ValueError, match="n_harmonics"):
            SpectrumInterpolation(
                sfreq=sfreq,
                line_freq=60.0,
                n_harmonics=n_harmonics,
            ).fit(data)
    with pytest.raises(ValueError, match="line_freq"):
        SpectrumInterpolation(sfreq=sfreq, line_freq=0.0).fit(data)
    with pytest.raises(ValueError, match="Nyquist"):
        SpectrumInterpolation(sfreq=sfreq, line_freq=[60.0, sfreq / 2]).fit(data)


def test_spectrum_interpolation_sfreq_semantics_for_mne_and_arrays():
    """MNE metadata supplies the sampling rate needed to interpret line frequency."""
    data, sfreq = _synth(n_ch=2)
    with pytest.raises(ValueError, match="sfreq"):
        SpectrumInterpolation(line_freq=60.0).fit_transform(data)

    info = mne.create_info(["EEG0", "EEG1"], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    estimator = SpectrumInterpolation(line_freq=60.0, n_harmonics=1)
    cleaned = estimator.fit_transform(raw)

    assert estimator.sfreq_ == sfreq
    np.testing.assert_array_equal(estimator.freqs_, [60.0])
    assert (
        _band_power(cleaned.get_data(), 60.0, sfreq) / _band_power(data, 60.0, sfreq)
        < 0.01
    )

    with pytest.raises(ValueError, match="disagrees"):
        SpectrumInterpolation(
            sfreq=sfreq / 2,
            line_freq=60.0,
            n_harmonics=1,
        ).fit(raw)


def test_spectrum_interpolation_3d_matches_epochwise_processing():
    """The NumPy epoch path applies the same spectral operation per epoch."""
    data, sfreq = _synth(n_ch=2, dur=2.0)
    epochs = np.stack([data, data * 0.5])
    estimator = SpectrumInterpolation(sfreq=sfreq, line_freq=60.0, n_harmonics=1)
    clean = estimator.fit_transform(epochs)
    expected = np.stack(
        [interpolate_spectrum(epoch, sfreq, [60.0]) for epoch in epochs]
    )

    np.testing.assert_allclose(clean, expected)


def test_spectrum_interpolation_summary_reports_resolved_targets(caplog):
    """Opt-in logging exposes one useful target-frequency summary."""
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        SpectrumInterpolation(
            sfreq=1000.0,
            line_freq=60.0,
            n_harmonics=3,
            bandwidth=1.5,
        ).fit(np.zeros((2, 1000)), verbose=True)

    summaries = [
        record
        for record in caplog.records
        if record.message.startswith("Spectrum interpolation:")
    ]
    assert len(summaries) == 1
    for token in ("target frequencies=[ 60.", "targets=3", "bandwidth=1.5"):
        assert token in summaries[0].message
