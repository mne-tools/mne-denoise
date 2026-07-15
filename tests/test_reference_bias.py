"""Tests for the explicit reference-aware DSS bias."""

import numpy as np
import pytest

from mne_denoise.dss import ReferenceBias, iterative_dss


def test_reference_bias_recovers_reference_predictable_signal():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(2, 2000))
    mixing = rng.normal(size=(8, 2))
    predictable = mixing @ reference
    data = predictable + 0.05 * rng.normal(size=predictable.shape)
    recovered = ReferenceBias(reference).apply(data)
    correlation = np.corrcoef(predictable.ravel(), recovered.ravel())[0, 1]
    assert correlation > 0.99


def test_reference_bias_is_reference_orthogonal_for_independent_signal():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(2, 10000))
    independent = rng.normal(size=(4, 10000))
    recovered = ReferenceBias(reference).apply(independent)
    assert np.var(recovered) < 0.01 * np.var(independent)


def test_reference_bias_supports_epoched_layout():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(2, 500))
    data = rng.normal(size=(4, 500, 3))
    assert ReferenceBias(reference).apply(data).shape == data.shape


def test_reference_bias_rejects_misaligned_data():
    with pytest.raises(ValueError, match="same number"):
        ReferenceBias(np.ones((2, 100))).apply(np.ones((4, 99)))


def test_symmetric_iterative_dss_calls_nonlinearity_per_component():
    rng = np.random.default_rng(10)
    data = rng.normal(size=(4, 500))
    calls = []

    def one_dimensional_only(source):
        calls.append(source.shape)
        assert source.ndim == 1
        return np.tanh(source)

    filters, sources, _, _ = iterative_dss(
        data,
        one_dimensional_only,
        4,
        method="symmetric",
        max_iter=2,
        random_state=3,
    )
    assert filters.shape == (4, 4)
    assert sources.shape == (4, 500)
    assert calls and all(shape == (500,) for shape in calls)
