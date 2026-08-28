import logging
from unittest.mock import patch

import mne
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


def test_narrowband_dss_fit_summary_describes_target_band(osc_data_generator, caplog):
    """The parent DSS summary identifies the effective narrowband target."""
    data, sfreq, freq, _signal = osc_data_generator((3, 500))
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        narrowband_dss(
            sfreq=sfreq,
            freq=freq,
            bandwidth=2.0,
            n_components=1,
        ).fit(data, verbose=True)
    summaries = [r for r in caplog.records if r.message.startswith("DSS:")]
    assert len(summaries) == 1
    assert "BandpassBias(9-11 Hz)" in summaries[0].message


def test_narrowband_dss_raw(osc_data_generator):
    data, sfreq, freq, signal = osc_data_generator((3, 500))
    info = mne.create_info(3, sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = narrowband_dss(sfreq=sfreq, freq=freq, bandwidth=2.0)
    dss.fit(raw)

    sources = dss.transform(raw)
    assert sources.shape == (3, 500)
    corr = np.abs(np.corrcoef(sources[0], signal)[0, 1])
    assert corr > 0.90


def test_narrowband_dss_epochs(osc_data_generator):
    data, sfreq, freq, signal = osc_data_generator((5, 3, 500))
    info = mne.create_info(3, sfreq, "eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)

    dss = narrowband_dss(sfreq=sfreq, freq=freq, bandwidth=2.0, n_components=2)
    dss.fit(epochs)

    sources = dss.transform(epochs)
    assert sources.shape == (5, 2, 500)  # (n_epochs, n_comp, n_times)

    # Signal in first channel
    corr = np.abs(np.corrcoef(sources[0, 0], signal)[0, 1])
    assert corr > 0.90


def test_narrowband_dss_evoked(osc_data_generator):
    data, sfreq, freq, signal = osc_data_generator((5, 3, 500))
    info = mne.create_info(3, sfreq, "eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    evoked = epochs.average()

    dss = narrowband_dss(sfreq=sfreq, freq=freq, bandwidth=2.0, n_components=1)
    dss.fit(evoked)

    source_evoked = dss.transform(evoked)
    assert isinstance(source_evoked, np.ndarray)
    assert source_evoked.shape == (1, 500)
    corr = np.abs(np.corrcoef(source_evoked[0], signal)[0, 1])
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


def test_narrowband_scan_progress_success(osc_data_generator):
    """narrowband_scan emits one event per successful frequency attempt."""
    data, sfreq, _freq, _signal = osc_data_generator((3, 500))
    events = []

    _best_dss, frequencies, eigenvalues = narrowband_scan(
        data,
        sfreq=sfreq,
        freq_range=(9, 11),
        freq_step=1.0,
        bandwidth=2.0,
        n_components=1,
        callback=events.append,
    )

    assert len(events) == len(frequencies) == 3
    assert [event.method for event in events] == ["narrowband_scan"] * 3
    assert [event.stage for event in events] == ["frequency"] * 3
    assert [event.current for event in events] == [1, 2, 3]
    assert [event.total for event in events] == [3, 3, 3]
    assert all(event.component is None for event in events)
    assert all(event.metric is not None for event in events)
    np.testing.assert_allclose([event.metric for event in events], eigenvalues)


def test_narrowband_scan_callback_is_numerically_transparent(osc_data_generator):
    """A callback does not change the fitted DSS or scan results."""
    data, sfreq, _freq, _signal = osc_data_generator((3, 500))
    scan_kws = {
        "sfreq": sfreq,
        "freq_range": (9, 11),
        "freq_step": 1.0,
        "bandwidth": 2.0,
        "n_components": 1,
    }

    best_without, frequencies_without, eigenvalues_without = narrowband_scan(
        data, **scan_kws
    )
    events = []
    best_with, frequencies_with, eigenvalues_with = narrowband_scan(
        data, callback=events.append, **scan_kws
    )

    np.testing.assert_array_equal(frequencies_with, frequencies_without)
    np.testing.assert_allclose(eigenvalues_with, eigenvalues_without)
    for attribute in ("eigenvalues_", "filters_", "patterns_"):
        np.testing.assert_allclose(
            getattr(best_with, attribute), getattr(best_without, attribute)
        )


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


def test_narrowband_scan_logs_returned_successful_winner(caplog):
    """The reported scan winner is the same successful candidate returned."""

    class FakeDSS:
        def __init__(self, freq, eigenvalue):
            self.freq = freq
            self.eigenvalue = eigenvalue

        def fit(self, data, *, verbose=None):
            del data, verbose
            if self.freq == 10.0:
                raise ValueError("simulated candidate failure")
            self.eigenvalues_ = np.array([self.eigenvalue])
            return self

    def make_candidate(sfreq, freq, **kwargs):
        del sfreq, kwargs
        return FakeDSS(freq, {-1.0: -0.2, 1.0: -0.4}.get(freq, -0.3))

    data = np.zeros((3, 100))
    with patch(
        "mne_denoise.dss.variants.narrowband.narrowband_dss",
        side_effect=make_candidate,
    ):
        with caplog.at_level(logging.INFO, logger="mne_denoise"):
            events = []
            best_dss, frequencies, eigenvalues = narrowband_scan(
                data,
                sfreq=100,
                freq_range=(9, 11),
                freq_step=1.0,
                verbose=True,
                callback=events.append,
            )

    assert best_dss.freq == frequencies[0]
    summary = [
        record for record in caplog.records if "Narrowband DSS scan:" in record.message
    ]
    assert len(summary) == 1
    assert "best=9 Hz" in summary[0].message
    assert len(events) == 3
    assert events[1].metric is None
    np.testing.assert_allclose(
        [events[0].metric, events[2].metric], eigenvalues[[0, 2]]
    )


def test_narrowband_scan_callback_exception_propagates(osc_data_generator):
    """A callback exception stops the scan instead of being treated as failure."""
    data, sfreq, _freq, _signal = osc_data_generator((3, 500))
    events = []

    class CallbackSentinel(RuntimeError):
        pass

    def callback(event):
        events.append(event)
        raise CallbackSentinel("stop")

    with patch(
        "mne_denoise.dss.variants.narrowband.narrowband_dss",
        wraps=narrowband_dss,
    ) as mock_dss:
        with pytest.raises(CallbackSentinel, match="stop") as caught:
            narrowband_scan(
                data,
                sfreq=sfreq,
                freq_range=(9, 11),
                freq_step=1.0,
                callback=callback,
            )

    assert type(caught.value) is CallbackSentinel
    assert len(events) == 1
    assert mock_dss.call_count == 1


def test_narrowband_scan_rejects_invalid_callback():
    """narrowband_scan validates callbacks before starting the scan."""
    data = np.zeros((3, 100))

    with pytest.raises(TypeError, match="callback must be callable or None"):
        narrowband_scan(data, sfreq=100, freq_range=(9, 11), callback=1)


def test_narrowband_scan_rejects_adaptive_before_progress():
    """adaptive=True is rejected before any frequency event is emitted."""
    data = np.zeros((3, 100))
    events = []

    with pytest.raises(ValueError, match="adaptive"):
        narrowband_scan(
            data,
            sfreq=100,
            freq_range=(9, 11),
            adaptive=True,
            callback=events.append,
        )

    assert events == []


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
