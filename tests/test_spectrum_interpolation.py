"""Unit tests for spectrum-interpolation line-noise removal."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

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
    t = np.arange(n) / sfreq
    neural = np.sin(2 * np.pi * 10 * t)
    line_noise = 3.0 * np.sin(2 * np.pi * line * t)
    data = rng.standard_normal((n_ch, n)) * 0.5 + neural[None, :] + line_noise[None, :]
    return data, sfreq


def test_interpolate_spectrum_attenuates_line():
    """The line frequency should be strongly attenuated."""
    data, sfreq = _synth()
    clean = interpolate_spectrum(data, sfreq, [60.0])

    before = _band_power(data, 60, sfreq)
    after = _band_power(clean, 60, sfreq)
    assert after / before < 0.01  # > 20 dB attenuation


def test_interpolate_spectrum_preserves_broadband():
    """Activity away from the line frequency is left intact."""
    data, sfreq = _synth()
    clean = interpolate_spectrum(data, sfreq, [60.0])

    before = _band_power(data, 10, sfreq)
    after = _band_power(clean, 10, sfreq)
    assert np.isclose(after, before, rtol=1e-6)


def test_interpolate_spectrum_preserves_out_of_band_phase():
    """Only the amplitude inside the band changes; phase is preserved."""
    data, sfreq = _synth()
    clean = interpolate_spectrum(
        data, sfreq, [60.0], bandwidth=1.0, neighbour_width=2.0
    )

    spec_b = np.fft.rfft(data, axis=1)
    spec_a = np.fft.rfft(clean, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
    far = np.abs(freqs - 60) > 3.0
    assert np.allclose(np.angle(spec_b[:, far]), np.angle(spec_a[:, far]), atol=1e-9)


def test_interpolate_spectrum_uses_mean_neighbour_amplitude():
    """Each replaced bin should receive the mean neighbouring amplitude."""
    rng = np.random.default_rng(4)
    data = rng.standard_normal((2, 1000))
    clean = interpolate_spectrum(
        data, 1000.0, [60.0], bandwidth=1.0, neighbour_width=2.0
    )

    freqs = np.fft.rfftfreq(data.shape[1], d=1 / 1000.0)
    before = np.abs(np.fft.rfft(data, axis=1))
    after = np.abs(np.fft.rfft(clean, axis=1))
    band = (freqs >= 59.0) & (freqs <= 61.0)
    neighbours = ((freqs >= 57.0) & (freqs < 59.0)) | ((freqs > 61.0) & (freqs <= 63.0))
    expected = before[:, neighbours].mean(axis=1, keepdims=True)
    assert np.allclose(after[:, band], expected)


def test_interpolate_spectrum_shape_and_dtype():
    """Output keeps the input shape and is real-valued."""
    data, sfreq = _synth()
    clean = interpolate_spectrum(data, sfreq, [60.0])
    assert clean.shape == data.shape
    assert np.isrealobj(clean)


def test_interpolate_spectrum_requires_2d():
    """The core function only accepts 2D arrays."""
    data = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="2D"):
        interpolate_spectrum(data, 1000.0, [60.0])


def test_spectrum_interpolation_removes_harmonics():
    """Requested harmonics of the line frequency are all removed."""
    sfreq, dur, n_ch = 1000.0, 8.0, 3
    n = int(sfreq * dur)
    t = np.arange(n) / sfreq
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n_ch, n)) * 0.3
    for h in (60.0, 120.0, 180.0):
        data += np.sin(2 * np.pi * h * t)[None, :]

    si = SpectrumInterpolation(sfreq=sfreq, line_freq=60.0, n_harmonics=3)
    clean = si.fit_transform(data)

    assert np.array_equal(si.freqs_, np.array([60.0, 120.0, 180.0]))
    for h in (60.0, 120.0, 180.0):
        assert _band_power(clean, h, sfreq) / _band_power(data, h, sfreq) < 0.01


def test_spectrum_interpolation_default_harmonics_reach_nyquist():
    """Without n_harmonics, every harmonic below Nyquist is targeted."""
    si = SpectrumInterpolation(sfreq=1000.0, line_freq=60.0).fit(np.zeros((1, 1000)))
    assert np.array_equal(si.freqs_, np.arange(60.0, 500.0, 60.0))


def test_spectrum_interpolation_explicit_frequency_list():
    """A sequence of frequencies is used directly without harmonics."""
    data, sfreq = _synth()
    si = SpectrumInterpolation(sfreq=sfreq, line_freq=[60.0], n_harmonics=0).fit(data)
    assert np.array_equal(si.freqs_, np.array([60.0]))


def test_spectrum_interpolation_raw_roundtrip():
    """A Raw object is cleaned and returned as Raw."""
    data, sfreq = _synth(n_ch=4)
    info = mne.create_info([f"EEG{i:03d}" for i in range(4)], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    out = SpectrumInterpolation(line_freq=60.0).fit_transform(raw)

    assert isinstance(out, mne.io.BaseRaw)
    before = _band_power(data, 60, sfreq)
    after = _band_power(out.get_data(), 60, sfreq)
    assert after / before < 0.01


def test_spectrum_interpolation_epochs_roundtrip():
    """An Epochs object is cleaned per epoch and returned as Epochs."""
    data, sfreq = _synth(n_ch=4, dur=4.0)
    epoch_data = np.stack([data, data * 0.9], axis=0)
    info = mne.create_info([f"EEG{i:03d}" for i in range(4)], sfreq, "eeg")
    epochs = mne.EpochsArray(epoch_data, info, verbose=False)

    out = SpectrumInterpolation(line_freq=60.0).fit_transform(epochs)

    assert isinstance(out, mne.BaseEpochs)
    assert out.get_data().shape == epoch_data.shape
    after = _band_power(out.get_data()[0], 60, sfreq)
    before = _band_power(epoch_data[0], 60, sfreq)
    assert after / before < 0.01


def test_spectrum_interpolation_evoked_roundtrip():
    """An Evoked object is cleaned and returned as Evoked."""
    data, sfreq = _synth(n_ch=4)
    info = mne.create_info([f"EEG{i:03d}" for i in range(4)], sfreq, "eeg")
    evoked = mne.EvokedArray(data, info, verbose=False)

    out = SpectrumInterpolation(line_freq=60.0).fit_transform(evoked)

    assert isinstance(out, mne.Evoked)
    after = _band_power(out.data, 60, sfreq)
    before = _band_power(data, 60, sfreq)
    assert after / before < 0.01


def test_spectrum_interpolation_passes_through_non_data_channels():
    """Non-data channels (e.g. stim) are returned unchanged."""
    data, sfreq = _synth(n_ch=3)
    info = mne.create_info(
        ["EEG000", "EEG001", "STI 014"], sfreq, ["eeg", "eeg", "stim"]
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    out = SpectrumInterpolation(line_freq=60.0).fit_transform(raw)

    stim_in = raw.get_data(picks="stim")
    stim_out = out.get_data(picks="stim")
    assert np.array_equal(stim_in, stim_out)
    # the eeg channels were actually modified
    eeg_in = raw.get_data(picks="eeg")
    eeg_out = out.get_data(picks="eeg")
    assert not np.allclose(eeg_in, eeg_out)


def test_spectrum_interpolation_array_requires_sfreq():
    """A NumPy array input without sfreq raises."""
    data, _ = _synth()
    with pytest.raises(ValueError, match="sfreq"):
        SpectrumInterpolation(line_freq=60.0).fit_transform(data)


def test_spectrum_interpolation_transform_before_fit_raises():
    """Transforming before fitting raises."""
    data, sfreq = _synth()
    si = SpectrumInterpolation(sfreq=sfreq, line_freq=60.0)
    with pytest.raises(NotFittedError):
        si.transform(data)


def test_interpolate_spectrum_skips_out_of_range_frequency():
    """Frequencies at or beyond Nyquist are ignored."""
    data, sfreq = _synth()
    clean = interpolate_spectrum(data, sfreq, [600.0])  # > Nyquist (500)
    assert np.allclose(clean, data, atol=1e-9)


def test_interpolate_spectrum_snaps_to_nearest_bin_when_band_too_narrow():
    """When the band is narrower than the resolution, the nearest bin is used."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((2, 100))  # 100 samples -> coarse 10 Hz bins
    # 65 Hz lies between bins (60/70); band of 1 Hz contains no bin -> snap.
    clean = interpolate_spectrum(data, 1000.0, [65.0], bandwidth=1.0)
    assert clean.shape == data.shape


def test_interpolate_spectrum_handles_single_sided_neighbours():
    """Frequencies near DC / Nyquist use whichever neighbour band exists."""
    data, sfreq = _synth(n_ch=2)  # 1 Hz resolution, bins 0..500
    near_nyquist = interpolate_spectrum(data, sfreq, [499.5])  # right band empty
    near_dc = interpolate_spectrum(data, sfreq, [0.5])  # left band empty
    assert near_nyquist.shape == data.shape
    assert near_dc.shape == data.shape


def test_interpolate_spectrum_rejects_zero_neighbour_width():
    """The reference bands must have positive width."""
    data, sfreq = _synth()
    with pytest.raises(ValueError, match="neighbour_width"):
        interpolate_spectrum(data, sfreq, [60.0], neighbour_width=0.0)


def test_spectrum_interpolation_rejects_bad_ndim():
    """A 1D array reaches the core dispatcher and is rejected."""
    si = SpectrumInterpolation(sfreq=1000.0, line_freq=60.0)
    with pytest.raises(ValueError, match="2D or 3D"):
        si.fit_transform(np.zeros(100))


def test_spectrum_interpolation_no_data_channels_pass_through():
    """An MNE object containing only non-data channels is unchanged."""
    data, sfreq = _synth(n_ch=2)
    info = mne.create_info(["MISC0", "MISC1"], sfreq, ["misc", "misc"])
    raw = mne.io.RawArray(data, info, verbose=False)

    out = SpectrumInterpolation(line_freq=60.0).fit_transform(raw)

    assert isinstance(out, mne.io.BaseRaw)
    assert np.array_equal(out.get_data(), raw.get_data())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sfreq": 0.0}, "sfreq"),
        ({"freqs": [0.0]}, "freqs"),
        ({"bandwidth": 0.0}, "bandwidth"),
        ({"neighbour_width": 0.0}, "neighbour_width"),
    ],
)
def test_interpolate_spectrum_validates_parameters(kwargs, match):
    """The low-level API should reject invalid numerical parameters."""
    data, sfreq = _synth()
    params = {"sfreq": sfreq, "freqs": [60.0]}
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        interpolate_spectrum(data, **params)


@pytest.mark.parametrize("n_harmonics", [0, -1, 1.5, True])
def test_spectrum_interpolation_validates_n_harmonics(n_harmonics):
    """n_harmonics must be a positive integer or None."""
    data, sfreq = _synth()
    si = SpectrumInterpolation(sfreq=sfreq, line_freq=60.0, n_harmonics=n_harmonics)
    with pytest.raises(ValueError, match="n_harmonics"):
        si.fit(data)


def test_spectrum_interpolation_rejects_frequency_at_nyquist():
    """Estimator target frequencies must lie below Nyquist."""
    with pytest.raises(ValueError, match="Nyquist"):
        SpectrumInterpolation(sfreq=1000.0, line_freq=[60.0, 500.0]).fit(
            np.zeros((1, 1000))
        )


def test_spectrum_interpolation_rejects_mismatched_mne_sfreq():
    """Transform must not silently use a sampling rate fitted elsewhere."""
    data, sfreq = _synth(n_ch=2)
    si = SpectrumInterpolation(line_freq=60.0).fit(
        mne.io.RawArray(data, mne.create_info(2, sfreq, "eeg"), verbose=False)
    )
    other = mne.io.RawArray(
        data[:, ::2], mne.create_info(2, sfreq / 2, "eeg"), verbose=False
    )
    with pytest.raises(ValueError, match="sampling frequency"):
        si.transform(other)


def test_spectrum_interpolation_3d_matches_epochwise_processing():
    """The vectorized 3D path should match independent epoch processing."""
    data, sfreq = _synth(n_ch=2, dur=2.0)
    epochs = np.stack([data, data * 0.5])
    si = SpectrumInterpolation(sfreq=sfreq, line_freq=60.0, n_harmonics=1)
    clean = si.fit_transform(epochs)
    expected = np.stack(
        [interpolate_spectrum(epoch, sfreq, [60.0]) for epoch in epochs]
    )
    assert np.allclose(clean, expected)


def test_spectrum_interpolation_rejects_declared_sfreq_conflict():
    """A declared sfreq that contradicts the container is an error, not ignored."""
    data, sfreq = _synth(n_ch=2)
    raw = mne.io.RawArray(data, mne.create_info(2, sfreq, "eeg"), verbose=False)
    with pytest.raises(ValueError, match="disagrees with MNE info sfreq"):
        SpectrumInterpolation(sfreq=sfreq / 2, line_freq=60.0).fit(raw)


def test_spectrum_interpolation_accepts_agreeing_declared_sfreq():
    """A redundant but consistent sfreq is accepted."""
    data, sfreq = _synth(n_ch=2)
    raw = mne.io.RawArray(data, mne.create_info(2, sfreq, "eeg"), verbose=False)
    si = SpectrumInterpolation(sfreq=sfreq, line_freq=60.0).fit(raw)
    assert si.sfreq_ == pytest.approx(sfreq)
