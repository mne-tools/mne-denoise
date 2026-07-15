"""Contract tests for experimental ContinuousDSS."""

import numpy as np
import pytest

from mne_denoise.experimental import ContinuousDSS


def _stream(seed=0, n_channels=8, n_times=4096, sfreq=256.0):
    rng = np.random.default_rng(seed)
    time = np.arange(n_times) / sfreq
    target = np.sin(2 * np.pi * 10 * time)
    pattern = rng.normal(size=n_channels)
    pattern /= np.linalg.norm(pattern)
    return 0.2 * rng.normal(size=(n_channels, n_times)) + np.outer(pattern, target)


def _estimator(**kwargs):
    return ContinuousDSS(
        8,
        256.0,
        n_components=1,
        warmup_blocks=4,
        solve_interval=2,
        block_size=64,
        experimental=True,
        **kwargs,
    )


def test_replay_shape_finite_and_state():
    data = _stream()
    estimator = _estimator().fit()
    output = estimator.transform(data)
    assert output.shape == data.shape
    assert np.all(np.isfinite(output))
    assert estimator.filters_.shape == (1, 8)
    assert estimator.n_solves_ > 0
    assert estimator.get_diagnostics()["information_access"].startswith("causal")


def test_warmup_is_passthrough():
    data = _stream(n_times=64)
    estimator = _estimator().fit()
    np.testing.assert_array_equal(estimator.process_block(data), data)


def test_missing_block_is_explicit_and_does_not_update():
    estimator = _estimator().fit()
    block = _stream(n_times=64)
    block[0, 2] = np.nan
    output = estimator.process_block(block)
    assert np.isnan(output[0, 2])
    assert estimator.n_blocks_seen_ == 0
    assert estimator.failure_counts_["nonfinite_block"] == 1


def test_reordered_channels_are_rejected():
    names = [f"EEG{index:02d}" for index in range(8)]
    estimator = _estimator(channel_names=names).fit()
    with pytest.raises(ValueError, match="channel order changed"):
        estimator.process_block(_stream(n_times=64), names[::-1])


def test_reset_is_deterministic():
    data = _stream()
    estimator = _estimator()
    first = estimator.fit_transform(data)
    second = estimator.fit_transform(data)
    np.testing.assert_allclose(first, second, atol=1e-12, rtol=0)


def test_experimental_opt_in_required():
    with pytest.raises(ValueError, match="experimental"):
        ContinuousDSS(8, 256.0).fit()


def test_mne_raw_metadata_and_channel_order_preserved():
    mne = pytest.importorskip("mne")
    names = [f"EEG{index:02d}" for index in range(8)]
    info = mne.create_info(names, 256.0, "eeg")
    raw = mne.io.RawArray(_stream() * 1e-6, info, verbose=False)
    output = _estimator(channel_names=names).fit_transform(raw)
    assert output.ch_names == raw.ch_names
    assert output.get_data().shape == raw.get_data().shape
