"""Tests for Teixeira local SSA and its estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mne_denoise.ssa import (
    LocalSingularSpectrumAnalysis,
    compute_local_ssa,
    local_ssa_clean_channel,
)
from mne_denoise.ssa.local import _mdl_order

from ._utils import band_power

# ---------------------------------------------------------------------------
# Teixeira local SSA
# ---------------------------------------------------------------------------


def test_mdl_matches_paper_equations():
    """The implementation agrees with a direct evaluation of Eqs. (4)-(6)."""
    eigenvalues = np.array([9.0, 4.0, 1.0, 0.8, 0.7])
    n_observations = 80
    selected, scores = _mdl_order(eigenvalues, n_observations)
    expected = []
    dimensions = eigenvalues.size
    for order in range(1, dimensions):
        tail = eigenvalues[order:]
        log_ratio = np.log(np.prod(tail) ** (1 / tail.size) / np.mean(tail))
        likelihood = n_observations * (dimensions - order) * log_ratio
        degrees = order * dimensions - order * (order - 1) / 2 + 1
        expected.append(-likelihood + 0.5 * degrees * np.log(n_observations))
    np.testing.assert_allclose(scores, expected)
    assert selected == np.argmin(expected) + 1


def test_local_ssa_reconstructs_artifact_plus_residual(rng):
    """Reversing clustering and embedding preserves the additive identity."""
    x = rng.standard_normal(160)
    cleaned, info = compute_local_ssa(
        x[np.newaxis], window_length=20, n_clusters=2, random_state=0
    )
    np.testing.assert_allclose(cleaned + info["artifacts"], x[np.newaxis])
    assert info["n_clusters"].tolist() == [2]
    assert all(size >= 20 for size in info["cluster_sizes"][0])
    assert len(info["subspace_dimensions"][0]) == 2


def test_local_ssa_attenuates_artifact_and_preserves_broadband_eeg(rng):
    """The paper interpretation removes coherent drift and retains broadband EEG."""
    sfreq = 128.0
    time = np.arange(1000) / sfreq
    artifact = 6.0 * np.sin(2 * np.pi * 0.5 * time)
    neural = 0.5 * rng.standard_normal(time.size)
    observed = artifact + neural + 0.05 * rng.standard_normal(time.size)
    cleaned = local_ssa_clean_channel(
        observed,
        window_length=41,
        n_clusters=6,
        random_state=0,
    )
    low_before = band_power(observed, sfreq, 0.0, 3.0)
    low_after = band_power(cleaned, sfreq, 0.0, 3.0)
    assert low_after < 0.1 * low_before
    assert np.corrcoef(cleaned, neural)[0, 1] > 0.9
    assert np.dot(cleaned, neural) / np.dot(neural, neural) > 0.85


def test_local_ssa_auto_clustering_and_constant_input():
    """Automatic q selection is deterministic and rank deficiency stays finite."""
    constant = np.full((2, 120), 3.0)
    cleaned, info = compute_local_ssa(
        constant, window_length=20, n_clusters="auto", max_clusters=4
    )
    np.testing.assert_allclose(cleaned, 0.0, atol=1e-12)
    assert np.isfinite(cleaned).all()
    assert np.all(info["n_clusters"] >= 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_clusters": 0},
        {"n_clusters": "invalid"},
        {"max_clusters": 0},
        {"random_state": True},
    ],
)
def test_local_ssa_rejects_invalid_parameters(kwargs):
    """Local clustering contracts reject ambiguous or impossible values."""
    with pytest.raises((TypeError, ValueError)):
        compute_local_ssa(np.ones((1, 100)), window_length=20, **kwargs)


def test_local_estimator_contract_and_mne_roundtrip(drift_data):
    """Local SSA clones, records fit state, and preserves an MNE Raw container."""
    mne = pytest.importorskip("mne")
    X, sfreq = drift_data
    info = mne.create_info([f"EEG{i}" for i in range(2)], sfreq, "eeg")
    raw = mne.io.RawArray(X[:2, :500], info, first_samp=11, verbose=False)
    estimator = LocalSingularSpectrumAnalysis(
        window_length=41, n_clusters=2, random_state=0
    )
    with pytest.raises(NotFittedError):
        estimator.transform(raw)
    cleaned = estimator.fit_transform(raw)
    assert clone(estimator).get_params() == estimator.get_params()
    assert cleaned is not raw
    assert cleaned.first_samp == raw.first_samp
    assert estimator.n_channels_in_ == 2
    assert estimator.n_clusters_.shape == (2,)
