import numpy as np
import pytest

from mne_denoise.dss import ssvep_dss


@pytest.fixture
def ssvep_data_generator():
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 500
    times = np.arange(n_times) / sfreq
    f0 = 12

    # Fundamental plus the second stimulus harmonic.
    signal = np.sin(2 * np.pi * f0 * times) + 0.5 * np.sin(2 * np.pi * 2 * f0 * times)

    def get_data(shape):
        noise = rng.normal(0, 0.5, shape)
        data = noise.copy()
        # Add signal to first channel (broadcasting)
        if len(shape) == 2:  # (n_ch, n_times)
            data[0] += signal
        elif len(shape) == 3:  # (n_epochs, n_ch, n_times)
            data[:, 0, :] += signal
        return data, sfreq, f0, signal

    return get_data


def test_ssvep_dss_array(ssvep_data_generator):
    data, sfreq, f0, signal = ssvep_data_generator((3, 500))

    dss = ssvep_dss(sfreq=sfreq, stim_freq=f0, n_harmonics=2, n_components=2)
    dss.fit(data)

    assert dss.filters_.shape == (2, 3)
    assert dss.bias.harmonic_frequencies == [f0, 2 * f0]
    np.testing.assert_allclose(dss.bias.weights, [1.0, 0.5])
    assert dss.eigenvalues_[0] >= dss.eigenvalues_[1]

    source = dss.filters_[0] @ data
    corr = np.abs(np.corrcoef(source, signal)[0, 1])
    assert corr > 0.8
