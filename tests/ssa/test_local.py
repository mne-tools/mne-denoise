"""Scientific and method-specific contracts for local SSA."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mne_denoise.ssa import (
    LocalSingularSpectrumAnalysis,
    compute_local_ssa,
    local_ssa_clean_channel,
)
from mne_denoise.ssa.local import _mdl_order

from ._utils import band_power


def test_mdl_matches_paper_equations():
    """The local order selection agrees with Teixeira et al. Eqs. (4)-(6)."""
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


def test_local_ssa_reconstructs_artifact_and_exposes_cluster_diagnostics(rng):
    """Reversing local clustering preserves the additive artifact identity."""
    signal = rng.standard_normal(160)
    cleaned, info = compute_local_ssa(
        signal[np.newaxis],
        window_length=20,
        n_clusters=2,
        random_state=0,
    )

    np.testing.assert_allclose(cleaned + info["artifacts"], signal[np.newaxis])
    assert info["n_clusters"].tolist() == [2]
    assert info["cluster_sizes"][0].sum() == signal.size - 20 + 1
    assert len(info["subspace_dimensions"][0]) == 2
    assert len(info["eigenvalues"][0]) == 2
    assert len(info["mdl_scores"][0]) == 2
    assert np.all(info["subspace_dimensions"][0] >= 1)


def test_compute_local_ssa_progress_reports_cluster_count():
    """The functional callback reports the selected cluster per channel."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 120))
    events = []

    cleaned, info = compute_local_ssa(
        data,
        window_length=10,
        n_clusters=2,
        random_state=0,
        callback=events.append,
    )

    assert cleaned.shape == data.shape
    assert len(events) == data.shape[0]
    assert all(event.method == "local_ssa" for event in events)
    assert all(event.stage == "channel" for event in events)
    assert [event.current for event in events] == [1, 2, 3]
    assert all(event.total == data.shape[0] for event in events)
    assert all(event.component is None for event in events)
    np.testing.assert_array_equal(
        [event.metric for event in events], info["n_clusters"].astype(float)
    )


def test_local_ssa_attenuates_artifact_and_preserves_broadband_eeg(rng):
    """Local SSA removes coherent drift while preserving broadband neural data."""
    sfreq = 128.0
    times = np.arange(1000) / sfreq
    artifact = 6.0 * np.sin(2 * np.pi * 0.5 * times)
    neural = 0.5 * rng.standard_normal(times.size)
    observed = artifact + neural + 0.05 * rng.standard_normal(times.size)

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


def test_local_ssa_auto_clustering_is_deterministic_and_handles_rank_deficiency(
    rng,
):
    """Automatic clustering is repeatable and constant input stays finite."""
    constant = np.full((2, 120), 3.0)
    constant_cleaned, constant_info = compute_local_ssa(
        constant,
        window_length=20,
        n_clusters="auto",
        max_clusters=4,
    )
    np.testing.assert_allclose(constant_cleaned, 0.0, atol=1e-12)
    assert np.isfinite(constant_cleaned).all()
    np.testing.assert_array_equal(constant_info["n_clusters"], [1, 1])

    data = rng.standard_normal((2, 160))
    first, first_info = compute_local_ssa(
        data,
        window_length=20,
        n_clusters="auto",
        max_clusters=4,
        random_state=0,
    )
    second, second_info = compute_local_ssa(
        data,
        window_length=20,
        n_clusters="auto",
        max_clusters=4,
        random_state=0,
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_array_equal(first_info["n_clusters"], second_info["n_clusters"])
    for first_sizes, second_sizes in zip(
        first_info["cluster_sizes"], second_info["cluster_sizes"]
    ):
        np.testing.assert_array_equal(first_sizes, second_sizes)
    for first_dimensions, second_dimensions in zip(
        first_info["subspace_dimensions"], second_info["subspace_dimensions"]
    ):
        np.testing.assert_array_equal(first_dimensions, second_dimensions)


def test_local_ssa_rejects_invalid_cluster_configuration():
    """Impossible and ambiguous cluster configurations fail explicitly."""
    data = np.ones((1, 100))
    for kwargs in (
        {"n_clusters": 0},
        {"n_clusters": "invalid"},
        {"max_clusters": 0},
    ):
        with pytest.raises((TypeError, ValueError)):
            compute_local_ssa(data, window_length=20, **kwargs)


def test_local_estimator_uses_mne_sfreq_for_window_and_reports_summary(
    drift_data, caplog
):
    """MNE sampling metadata resolves the fitted window and local diagnostics."""
    mne = pytest.importorskip("mne")
    data, sfreq = drift_data
    info = mne.create_info(["EEG0", "EEG1"], sfreq, "eeg")
    raw = mne.io.RawArray(data[:2, :500], info, verbose=False)
    estimator = LocalSingularSpectrumAnalysis(
        window_seconds=0.16,
        n_clusters=2,
        random_state=0,
        verbose=True,
    )

    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        cleaned = estimator.fit_transform(raw)

    assert cleaned.get_data().shape == (2, 500)
    assert estimator.sfreq_ == sfreq
    assert estimator.diagnostics_["window_length"] == 40
    assert estimator.n_clusters_.shape == (2,)
    assert len(estimator.subspace_dimensions_) == 2
    summaries = [
        record for record in caplog.records if record.message.startswith("Local SSA:")
    ]
    assert len(summaries) == 1
    for token in ("window=40", "channels=2", "mean clusters="):
        assert token in summaries[0].message


def test_local_estimator_distinguishes_continuous_and_epoch_progress_topology(rng):
    """Continuous records report clusters per channel; epochs report per epoch."""
    data = rng.standard_normal((2, 120))
    continuous_estimator = LocalSingularSpectrumAnalysis(
        window_length=10,
        n_clusters=2,
        random_state=0,
    ).fit(data)
    continuous_events = []
    cleaned = continuous_estimator.transform(data, callback=continuous_events.append)

    assert cleaned.shape == data.shape
    assert len(continuous_events) == data.shape[0]
    assert all(event.method == "local_ssa" for event in continuous_events)
    assert [event.stage for event in continuous_events] == ["channel", "channel"]
    assert [event.current for event in continuous_events] == [1, 2]
    assert all(event.total == 2 for event in continuous_events)
    assert all(event.component is None for event in continuous_events)
    np.testing.assert_array_equal(
        [event.metric for event in continuous_events],
        continuous_estimator.n_clusters_.astype(float),
    )

    epochs = np.stack((data, 0.5 * data))
    epoch_estimator = LocalSingularSpectrumAnalysis(
        window_length=10,
        n_clusters=2,
        random_state=0,
    ).fit(epochs)
    epoch_events = []
    epoched = epoch_estimator.transform(epochs, callback=epoch_events.append)

    assert epoched.shape == epochs.shape
    assert len(epoch_events) == epochs.shape[0]
    assert all(event.method == "local_ssa" for event in epoch_events)
    assert [event.stage for event in epoch_events] == ["epoch", "epoch"]
    assert [event.current for event in epoch_events] == [1, 2]
    assert all(event.total == 2 for event in epoch_events)
    assert all(event.component is None for event in epoch_events)
    assert all(event.metric is None for event in epoch_events)
    assert epoch_estimator.n_clusters_.shape == (2, 2)
