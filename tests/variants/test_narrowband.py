from unittest.mock import patch

import numpy as np
import pytest

from mne_denoise.dss.denoisers.spectral import BandpassBias
from mne_denoise.dss.linear import DSS
from mne_denoise.dss.variants.narrowband import narrowband_dss, narrowband_scan


@pytest.fixture
def osc_data_generator():
    rng = np.random.default_rng(42)
    sfreq = 100
    n_times = 500
    times = np.arange(n_times) / sfreq
    freq = 10

    # Signal: 10 Hz alpha
    signal = np.sin(2 * np.pi * freq * times) * 3.0

    def get_data(shape):
        noise = rng.normal(0, 0.5, shape)
        data = noise.copy()
        # Add signal to first channel
        if len(shape) == 2:  # (n_ch, n_times)
            data[0] += signal
        elif len(shape) == 3:  # (n_epochs, n_ch, n_times)
            data[:, 0, :] += signal
        return data, sfreq, freq, signal

    return get_data


def test_narrowband_dss_array(osc_data_generator):
    data, sfreq, freq, signal = osc_data_generator((3, 500))

    dss = narrowband_dss(sfreq=sfreq, freq=freq, bandwidth=2.0, n_components=1)
    dss.fit(data)

    sources = dss.transform(data)
    corr = np.abs(np.corrcoef(sources[0], signal)[0, 1])
    assert corr > 0.90


def test_narrowband_scan_functional():
    """narrowband_scan should find the correct peak frequency."""
    rng = np.random.default_rng(42)
    sfreq = 100
    n_times = 1000
    times = np.arange(n_times) / sfreq
    n_ch = 3

    # Signal: 13 Hz
    target_freq = 13.0
    signal = np.sin(2 * np.pi * target_freq * times)

    noise = rng.standard_normal((n_ch, n_times)) * 0.1
    data = noise.copy()
    data[0] += signal

    best_dss, freqs, eigs = narrowband_scan(
        data,
        sfreq=sfreq,
        freq_range=(5, 20),
        freq_step=1.0,
        bandwidth=2.0,
        n_components=1,
    )

    best_freq = freqs[np.argmax(eigs)]
    assert abs(best_freq - target_freq) <= 1.0


def test_narrowband_scan_errors():
    """Test error handling in scanning loop."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))

    # Patch narrowband_dss to raise exception for a specific frequency
    with patch("mne_denoise.dss.variants.narrowband.narrowband_dss") as mock_dss:

        def side_effect(sfreq, freq, **kwargs):
            if freq == 10.0:
                raise ValueError("Simulated failure")
            # Return real DSS for others (need to import real function inside or keep ref)
            bias = BandpassBias(freq_band=(freq - 1, freq + 1), sfreq=sfreq)
            dss = DSS(bias=bias)
            return dss

        mock_dss.side_effect = side_effect

        # This will call our mock. 10.0 should fail and satisfy line 150-152
        events = []
        best_dss, freqs, eigs = narrowband_scan(
            data,
            sfreq=100,
            freq_range=(9, 11),
            freq_step=1.0,
            callback=events.append,
        )
        assert len(freqs) == 3  # 9, 10, 11
        # The scan should complete despite 10.0 failing
        assert len(events) == 3
        assert events[1].metric is None
        np.testing.assert_allclose([events[0].metric, events[2].metric], eigs[[0, 2]])


def test_narrowband_scan_rejects_adaptive_before_progress():
    """adaptive=True is rejected before the scan starts."""
    data = np.zeros((3, 100))

    with pytest.raises(ValueError, match="adaptive"):
        narrowband_scan(
            data,
            sfreq=100,
            freq_range=(9, 11),
            adaptive=True,
        )


def test_narrowband_scan_all_fail():
    """Test raising RuntimeError when all frequencies fail."""
    data = np.zeros((3, 100))
    events = []
    with patch("mne_denoise.dss.variants.narrowband.narrowband_dss") as mock_dss:
        mock_dss.side_effect = ValueError("Fail everything")

        with pytest.raises(RuntimeError, match="Failed to fit DSS at any frequency"):
            narrowband_scan(
                data,
                sfreq=100,
                freq_range=(9, 11),
                callback=events.append,
            )

    assert len(events) == 3
    assert [event.metric for event in events] == [None, None, None]
