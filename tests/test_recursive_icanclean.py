"""Tests for the experimental stateful recursive iCanClean API."""

from __future__ import annotations

import copy
import json

import mne
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import mne_denoise.icanclean.recursive as recursive_module
from mne_denoise.icanclean import ICanClean, RecursiveICanClean, compute_icanclean


@pytest.fixture(scope="module")
def recursive_data():
    """Create primary/reference data with a known shared artifact."""
    rng = np.random.default_rng(710)
    sfreq = 200.0
    n_primary, n_reference, n_samples = 6, 3, 1600
    times = np.arange(n_samples) / sfreq
    neural_sources = np.vstack(
        (
            np.sin(2 * np.pi * 7.0 * times),
            np.sin(2 * np.pi * 11.0 * times + 0.3),
            np.sin(2 * np.pi * 17.0 * times + 0.7),
        )
    )
    brain = rng.standard_normal((n_primary, 3)) @ neural_sources
    brain += 0.03 * rng.standard_normal(brain.shape)
    artifact = 2.5 * np.sin(2 * np.pi * 2.0 * times)
    artifact += 0.8 * rng.standard_normal(n_samples)
    artifact_topography = rng.standard_normal(n_primary)
    artifact_topography /= np.linalg.norm(artifact_topography)
    primary = brain + 5.0 * np.outer(artifact_topography, artifact)
    reference = rng.standard_normal((n_reference, 1)) @ artifact[None, :]
    reference += 0.05 * rng.standard_normal(reference.shape)
    return primary, reference, brain, artifact_topography, sfreq


@pytest.fixture(scope="module")
def fitted_state(recursive_data):
    """Return one valid numerical checkpoint for corruption tests."""
    primary, reference, _, _, _ = recursive_data
    return _estimator().fit(primary, reference=reference).state_dict()


def _estimator(**kwargs):
    """Return a compact deterministic operating point for tests."""
    defaults = {
        "sfreq": 200.0,
        "threshold": 0.2,
        "max_reject_fraction": 0.5,
        "warmup_samples": 200,
        "update_interval_samples": 50,
        "regularization": 1e-6,
    }
    defaults.update(kwargs)
    return RecursiveICanClean(**defaults)


def test_recursive_api_is_distinct_and_cloneable():
    """The recursive prototype is not an alias for ordinary iCanClean."""
    assert RecursiveICanClean is not ICanClean
    assert "experimental" in RecursiveICanClean.__doc__.lower()
    estimator = _estimator(clean_with="Y", forgetting_factor=0.999)
    assert clone(estimator).get_params() == estimator.get_params()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sfreq": True}, "sfreq"),
        ({"clean_with": "Z"}, "clean_with"),
        ({"threshold": 1.1}, "threshold"),
        ({"max_reject_fraction": -0.1}, "max_reject_fraction"),
        ({"max_reject_fraction": 1.1}, "max_reject_fraction"),
        ({"reref_primary": "median"}, "reref"),
        ({"regularization": 0.0}, "regularization"),
        ({"rank_tolerance": 1.0}, "rank_tolerance"),
        ({"forgetting_factor": 0.0}, "forgetting_factor"),
        ({"forgetting_factor": 1.1}, "forgetting_factor"),
        (
            {"forgetting_factor": 0.9, "memory_duration_s": 1.0},
            "mutually exclusive",
        ),
        (
            {"warmup_samples": 10, "warmup_duration_s": 1.0},
            "mutually exclusive",
        ),
        (
            {"warmup_samples": None, "warmup_duration_s": None},
            "exactly one",
        ),
        ({"warmup_samples": 1}, "warmup_samples"),
        ({"stable_updates": True}, "stable_updates"),
        ({"update_order": "sometimes"}, "update_order"),
        ({"adaptation_mode": "maybe"}, "adaptation_mode"),
    ],
)
def test_configuration_validation(kwargs, match):
    """Numerical, unit, and state-mode contracts fail early."""
    with pytest.raises((TypeError, ValueError), match=match):
        RecursiveICanClean(**kwargs)


def test_time_domain_parameters_resolve_to_samples(recursive_data):
    """Second-valued parameters have explicit sample conversions."""
    primary, reference, _, _, sfreq = recursive_data
    estimator = RecursiveICanClean(
        sfreq=sfreq,
        warmup_samples=None,
        warmup_duration_s=1.0,
        update_interval_samples=None,
        update_interval_s=0.25,
        memory_duration_s=0.5,
    ).partial_fit(primary[:, :400], reference=reference[:, :400])
    assert estimator._warmup_samples_ == 200
    assert estimator._update_interval_samples_ == 50
    assert estimator._forgetting_factor_ == pytest.approx(np.exp(-1 / 100))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "warmup_samples": None,
                "warmup_duration_s": 0.001,
            },
            "fewer than 2 samples",
        ),
        (
            {
                "update_interval_samples": None,
                "update_interval_s": 0.001,
            },
            "fewer than 1 sample",
        ),
        (
            {"memory_duration_s": np.finfo(float).tiny},
            "too short",
        ),
    ],
)
def test_time_domain_parameters_reject_unresolvable_durations(
    recursive_data, kwargs, match
):
    """Physical durations may not be silently clamped to arbitrary samples."""
    primary, reference, _, _, _ = recursive_data
    with pytest.raises(ValueError, match=match):
        _estimator(**kwargs).partial_fit(primary[:, :10], reference=reference[:, :10])


def test_batch_moments_equal_recursive_no_forgetting(recursive_data):
    """No-forgetting sufficient statistics reproduce batch population moments."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator(update_interval_samples=2000).partial_fit(
        primary, reference=reference
    )
    covariance_p, covariance_r, covariance_pr = estimator._covariances()
    np.testing.assert_allclose(estimator._mean_primary_, primary.mean(axis=1))
    np.testing.assert_allclose(estimator._mean_reference_, reference.mean(axis=1))
    np.testing.assert_allclose(covariance_p, np.cov(primary, bias=True))
    np.testing.assert_allclose(covariance_r, np.cov(reference, bias=True))
    centered_p = primary - primary.mean(axis=1, keepdims=True)
    centered_r = reference - reference.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(
        covariance_pr, centered_p @ centered_r.T / primary.shape[1]
    )


@pytest.mark.parametrize("forgetting_factor", [None, 0.995])
def test_partial_fit_is_transport_boundary_invariant(recursive_data, forgetting_factor):
    """Sample-indexed recursive updates do not depend on call boundaries."""
    primary, reference, _, _, _ = recursive_data
    whole = _estimator(forgetting_factor=forgetting_factor).partial_fit(
        primary, reference=reference
    )
    chunked = _estimator(forgetting_factor=forgetting_factor)
    boundaries = [0, 73, 319, 511, 1003, primary.shape[1]]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        chunked.partial_fit(primary[:, start:stop], reference=reference[:, start:stop])
    for attribute in (
        "_mean_primary_",
        "_mean_reference_",
        "_scatter_primary_",
        "_scatter_reference_",
        "_scatter_cross_",
        "artifact_operator_",
        "correlations_",
    ):
        np.testing.assert_allclose(
            getattr(chunked, attribute), getattr(whole, attribute), atol=1e-12
        )
    assert chunked._model_version_ == whole._model_version_
    assert chunked._model_state_hash() == whole._model_state_hash()


def test_causal_process_is_transport_boundary_invariant(recursive_data):
    """Causal output is invariant to transport blocks for a fixed sample stream."""
    primary, reference, _, _, _ = recursive_data
    whole = _estimator()
    expected = whole.process(primary, reference=reference)
    chunked = _estimator()
    outputs = []
    boundaries = [0, 17, 201, 486, 777, 1201, primary.shape[1]]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        outputs.append(
            chunked.process(primary[:, start:stop], reference=reference[:, start:stop])
        )
    observed = np.concatenate(outputs, axis=1)
    np.testing.assert_allclose(observed, expected, atol=1e-12)
    np.testing.assert_allclose(chunked.artifact_operator_, whole.artifact_operator_)
    assert chunked._model_state_hash() == whole._model_state_hash()


def test_frozen_transform_matches_arbitrary_eval_chunks(recursive_data):
    """Offline and chunked evaluation agree when adaptation is frozen."""
    primary, reference, _, _, _ = recursive_data
    calibration = slice(0, 800)
    evaluation = slice(800, None)
    estimator = _estimator().fit(
        primary[:, calibration], reference=reference[:, calibration]
    )
    expected = estimator.transform(
        primary[:, evaluation], reference=reference[:, evaluation]
    )
    chunks = []
    for start, stop in ((800, 901), (901, 1200), (1200, 1600)):
        chunks.append(
            estimator.transform(
                primary[:, start:stop], reference=reference[:, start:stop]
            )
        )
    np.testing.assert_allclose(np.concatenate(chunks, axis=1), expected)


def test_transform_never_adapts_on_evaluation_data(recursive_data):
    """Independent evaluation cannot leak into fitted recursive moments."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator().fit(primary[:, :800], reference=reference[:, :800])
    state_hash = estimator._model_state_hash()
    adapted = estimator._samples_adapted_
    estimator.transform(primary[:, 800:], reference=reference[:, 800:])
    assert estimator._samples_adapted_ == adapted
    assert estimator._model_state_hash() == state_hash


def test_fit_transform_attenuates_shared_artifact(recursive_data):
    """The experimental method removes a known shared reference subspace."""
    primary, reference, brain, topography, _ = recursive_data
    estimator = _estimator().fit(primary, reference=reference)
    cleaned = estimator.transform(primary, reference=reference)
    before = np.var(topography @ (primary - brain))
    after = np.var(topography @ (cleaned - brain))
    assert after < 0.1 * before
    assert estimator.n_removed_ >= 1
    assert estimator.last_update_diagnostics_["rank_ceiling"] <= 3
    assert estimator.last_update_diagnostics_["status"] == "APPLIED"


def test_physical_unit_scale_does_not_change_rank_or_decisions(recursive_data):
    """Relative rank/loading rules work for volt-scale MNE-like amplitudes."""
    primary, reference, _, _, _ = recursive_data
    baseline = _estimator().fit(primary, reference=reference)
    scale = 1e-6
    scaled = _estimator().fit(primary * scale, reference=reference * scale)
    assert (
        scaled.last_update_diagnostics_["rank_primary"]
        == baseline.last_update_diagnostics_["rank_primary"]
    )
    assert (
        scaled.last_update_diagnostics_["rank_reference"]
        == baseline.last_update_diagnostics_["rank_reference"]
    )
    np.testing.assert_array_equal(scaled.removed_idx_, baseline.removed_idx_)
    expected = baseline.transform(primary, reference=reference) * scale
    observed = scaled.transform(primary * scale, reference=reference * scale)
    np.testing.assert_allclose(observed, expected, rtol=5e-8, atol=2e-17)


@pytest.mark.parametrize(
    ("clean_with", "reref_primary", "reref_ref"),
    [
        ("X", False, False),
        ("Y", "fullrank", False),
        ("both", False, "fullrank"),
        ("X", "loserank", "loserank"),
    ],
)
def test_frozen_global_bridge_to_batch_icanclean(
    recursive_data, clean_with, reref_primary, reref_ref
):
    """No-forgetting recursion preserves each batch subtraction contract."""
    primary, reference, _, _, sfreq = recursive_data
    expected, expected_qc = compute_icanclean(
        primary,
        reference,
        sfreq,
        mode="global",
        clean_with=clean_with,
        threshold=0.2,
        max_reject_fraction=0.5,
        reref_primary=reref_primary,
        reref_ref=reref_ref,
        verbose=False,
    )
    estimator = _estimator(
        regularization=1e-12,
        clean_with=clean_with,
        reref_primary=reref_primary,
        reref_ref=reref_ref,
    )
    observed = estimator.fit_transform(primary, reference=reference)
    np.testing.assert_array_equal(
        estimator.removed_idx_, expected_qc["removed_idx_"][0]
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("clean_with", ["X", "Y", "both"])
def test_clean_with_variants_are_finite(recursive_data, clean_with):
    """All published iCanClean basis choices have explicit recursive forms."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator(clean_with=clean_with).fit(primary, reference=reference)
    cleaned = estimator.transform(primary, reference=reference)
    assert cleaned.shape == primary.shape
    assert np.all(np.isfinite(cleaned))


def test_rank_deficient_reference_truncates_null_cca_components(recursive_data):
    """Null covariance directions cannot be reported as removed components."""
    primary, reference, _, _, _ = recursive_data
    rank_one_reference = np.vstack((reference[0], 2 * reference[0], -reference[0]))
    estimator = _estimator(threshold=0.0).fit(primary, reference=rank_one_reference)
    assert estimator.rank_reference_ == 1
    assert estimator.rank_ceiling_ == 1
    assert estimator.correlations_.shape == (1,)
    assert np.all(estimator.removed_idx_ < estimator.rank_ceiling_)


def test_state_checkpoint_replays_future_decisions(recursive_data):
    """A checkpoint reproduces future outputs, updates, and model checksum."""
    primary, reference, _, _, _ = recursive_data
    source = _estimator().partial_fit(primary[:, :700], reference=reference[:, :700])
    checkpoint = source.state_dict()
    replay = _estimator().load_state_dict(checkpoint)
    source_output = source.process(primary[:, 700:], reference=reference[:, 700:])
    replay_output = replay.process(primary[:, 700:], reference=reference[:, 700:])
    np.testing.assert_allclose(replay_output, source_output, atol=0.0, rtol=0.0)
    assert (
        replay.state_dict()["model_state_sha256"]
        == source.state_dict()["model_state_sha256"]
    )
    assert replay.last_process_diagnostics_ == source.last_process_diagnostics_


def test_json_checkpoint_is_lossless_and_replays_warmup(recursive_data):
    """Canonical JSON preserves exact state before a model exists."""
    primary, reference, _, _, _ = recursive_data
    source = _estimator(warmup_samples=400).partial_fit(
        primary[:, :173], reference=reference[:, :173]
    )
    payload = source.state_json()
    assert payload == source.state_json()
    assert "NaN" not in payload and "Infinity" not in payload
    replay = _estimator(warmup_samples=400).load_state_json(payload.encode("utf-8"))
    expected = source.process(primary[:, 173:600], reference=reference[:, 173:600])
    observed = replay.process(primary[:, 173:600], reference=reference[:, 173:600])
    np.testing.assert_array_equal(observed, expected)
    assert replay.state_json() == source.state_json()


def test_state_rejects_configuration_and_payload_changes(recursive_data):
    """Replay cannot silently cross operating points or corrupted arrays."""
    primary, reference, _, _, _ = recursive_data
    state = _estimator().fit(primary, reference=reference).state_dict()
    with pytest.raises(ValueError, match="configuration"):
        _estimator(threshold=0.9).load_state_dict(state)
    corrupt = copy.deepcopy(state)
    corrupt["mean_primary"][0] += 1.0
    with pytest.raises(ValueError, match="checksum"):
        _estimator().load_state_dict(corrupt)


def test_failed_state_load_is_transactional(recursive_data):
    """A rejected checkpoint cannot destroy an existing fitted model."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator().fit(primary[:, :800], reference=reference[:, :800])
    before = estimator.state_json()
    corrupt = estimator.state_dict()
    corrupt["next_update_sample"] += 1
    with pytest.raises(ValueError, match="checksum"):
        estimator.load_state_dict(corrupt)
    assert estimator.state_json() == before


def test_reset_clears_model_and_logs(recursive_data):
    """Reset returns the estimator to a genuinely uninitialized state."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator().fit(primary, reference=reference)
    assert estimator.update_history_
    estimator.reset()
    assert not hasattr(estimator, "artifact_operator_")
    assert not hasattr(estimator, "update_history_")
    with pytest.raises(NotFittedError):
        estimator.transform(primary, reference=reference)
    warmed = estimator.process(primary[:, :20], reference=reference[:, :20])
    np.testing.assert_array_equal(warmed, primary[:, :20])


def test_contamination_gate_freezes_adaptation_state(recursive_data):
    """Rejected adaptation samples leave recursive moments and model unchanged."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator().fit(primary[:, :800], reference=reference[:, :800])
    model_hash = estimator._model_state_hash()
    adapted = estimator._samples_adapted_
    covariance = estimator._scatter_cross_.copy()
    output = estimator.process(
        primary[:, 800:],
        reference=reference[:, 800:],
        adaptation_mask=np.zeros(primary.shape[1] - 800, dtype=bool),
    )
    assert np.all(np.isfinite(output))
    assert estimator._samples_adapted_ == adapted
    np.testing.assert_array_equal(estimator._scatter_cross_, covariance)
    assert estimator._model_state_hash() == model_hash
    assert estimator.last_process_diagnostics_["gated_samples"] == 800


def test_frozen_mode_is_explicit_reference_freeze_control(recursive_data):
    """Frozen process mode cleans without changing the calibrated model."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator(adaptation_mode="frozen").fit(
        primary[:, :800], reference=reference[:, :800]
    )
    model_hash = estimator._model_state_hash()
    adapted = estimator._samples_adapted_
    cleaned = estimator.process(primary[:, 800:], reference=reference[:, 800:])
    assert cleaned.shape == primary[:, 800:].shape
    assert estimator._samples_adapted_ == adapted
    assert estimator._model_state_hash() == model_hash
    assert estimator.last_process_diagnostics_["adaptation_mode"] == "frozen"


def test_process_logs_transport_latency_and_convergence(recursive_data):
    """Every transport block records latency and stable-update diagnostics."""
    primary, reference, _, _, sfreq = recursive_data
    estimator = _estimator(convergence_tol=1e9, stable_updates=2)
    estimator.process(primary, reference=reference)
    process = estimator.last_process_diagnostics_
    assert process["transport_block_samples"] == primary.shape[1]
    assert process["transport_block_duration_s"] == pytest.approx(
        primary.shape[1] / sfreq
    )
    assert process["algorithmic_lookahead_samples"] == 0
    assert not process["current_sample_used_for_own_model"]
    assert process["adaptation_delay_samples"] == 1
    assert process["warmup_passthrough_samples"] == estimator._warmup_samples_
    assert process["cleaned_with_model_samples"] == (
        primary.shape[1] - estimator._warmup_samples_
    )
    assert estimator.converged_
    assert len(estimator.update_history_) > 2
    assert len(process["model_state_sha256"]) == 64


def test_failed_online_update_retains_last_valid_model(recursive_data, monkeypatch):
    """Numerical update failures are diagnosed without discarding a valid model."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator().fit(primary[:, :800], reference=reference[:, :800])
    operator = estimator.artifact_operator_.copy()
    version = estimator._model_version_

    def fail_svd(*args, **kwargs):
        raise recursive_module.la.LinAlgError("locked failure")

    monkeypatch.setattr(recursive_module.la, "svd", fail_svd)
    estimator.partial_fit(primary[:, 800:850], reference=reference[:, 800:850])
    np.testing.assert_array_equal(estimator.artifact_operator_, operator)
    assert estimator._model_version_ == version
    assert estimator.last_update_diagnostics_["status"] == "INADMISSIBLE"
    assert estimator.last_update_diagnostics_["retained_previous_model"]
    assert "locked failure" in estimator.last_update_diagnostics_["reason"]


def test_combined_numpy_preserves_reference_channels(recursive_data):
    """Combined arrays clean primary rows and preserve reference rows exactly."""
    primary, reference, _, _, _ = recursive_data
    combined = np.vstack((primary, reference))
    ref_idx = list(range(primary.shape[0], combined.shape[0]))
    estimator = _estimator(ref_channels=ref_idx)
    cleaned = estimator.fit_transform(combined)
    np.testing.assert_array_equal(cleaned[ref_idx], reference)
    assert not np.allclose(cleaned[: primary.shape[0]], primary)


def _make_raw(recursive_data):
    """Return combined Raw with primary, reference, and stimulus channels."""
    primary, reference, _, _, sfreq = recursive_data
    names = (
        [f"EEG{index}" for index in range(primary.shape[0])]
        + [f"REF{index}" for index in range(reference.shape[0])]
        + ["STI 014"]
    )
    types = ["eeg"] * (primary.shape[0] + reference.shape[0]) + ["stim"]
    stim = np.zeros((1, primary.shape[1]))
    stim[0, 100] = 1
    raw = mne.io.RawArray(
        np.vstack((primary, reference, stim)),
        mne.create_info(names, sfreq, types),
        first_samp=37,
        verbose=False,
    )
    raw.info["bads"] = ["EEG1"]
    raw.set_annotations(mne.Annotations([1.0], [0.1], ["marker"]), emit_warning=False)
    return raw


def test_combined_raw_preserves_metadata_and_aligns_names(recursive_data):
    """Raw subtype, metadata, untouched channels, and fit-name order survive."""
    raw = _make_raw(recursive_data)
    primary_names = [name for name in raw.ch_names if name.startswith("EEG")]
    reference_names = [name for name in raw.ch_names if name.startswith("REF")]
    estimator = _estimator(
        ref_channels=reference_names,
        primary_channels=primary_names,
    ).fit(raw)
    reordered = raw.copy().reorder_channels(list(reversed(raw.ch_names)))
    cleaned = estimator.transform(reordered)
    assert type(cleaned) is type(reordered)
    assert cleaned.ch_names == reordered.ch_names
    assert cleaned.first_samp == reordered.first_samp
    assert cleaned.info["bads"] == reordered.info["bads"]
    assert cleaned.annotations == reordered.annotations
    np.testing.assert_array_equal(
        cleaned.get_data(picks=["STI 014"]),
        reordered.get_data(picks=["STI 014"]),
    )
    assert not np.allclose(
        cleaned.get_data(picks=primary_names),
        reordered.get_data(picks=primary_names),
    )


def test_separate_raw_reference_alignment(recursive_data):
    """Separate reference Raw objects align to calibration channel names."""
    primary, reference, _, _, sfreq = recursive_data
    primary_names = [f"EEG{index}" for index in range(primary.shape[0])]
    reference_names = [f"REF{index}" for index in range(reference.shape[0])]
    primary_raw = mne.io.RawArray(
        primary,
        mne.create_info(primary_names, sfreq, "eeg"),
        verbose=False,
    )
    reference_raw = mne.io.RawArray(
        reference,
        mne.create_info(reference_names, sfreq, "eog"),
        verbose=False,
    )
    estimator = _estimator().fit(primary_raw, reference=reference_raw)
    expected = estimator.transform(primary_raw, reference=reference_raw)
    reordered = reference_raw.copy().reorder_channels(list(reversed(reference_names)))
    observed = estimator.transform(primary_raw, reference=reordered)
    np.testing.assert_allclose(observed.get_data(), expected.get_data())
    assert type(observed) is type(primary_raw)


def test_separate_raw_requires_sample_time_alignment(recursive_data):
    """Equal lengths are insufficient when Raw sample identities differ."""
    primary, reference, _, _, sfreq = recursive_data
    primary_raw = mne.io.RawArray(
        primary,
        mne.create_info([f"EEG{i}" for i in range(primary.shape[0])], sfreq, "eeg"),
        first_samp=10,
        verbose=False,
    )
    reference_raw = mne.io.RawArray(
        reference,
        mne.create_info([f"REF{i}" for i in range(reference.shape[0])], sfreq, "eog"),
        first_samp=11,
        verbose=False,
    )
    with pytest.raises(ValueError, match="first_samp"):
        _estimator().fit(primary_raw, reference=reference_raw)


def test_raw_channel_type_identity_is_locked(recursive_data):
    """Matching names cannot hide a changed MNE channel type or physical unit."""
    primary, reference, _, _, sfreq = recursive_data
    primary_names = [f"EEG{i}" for i in range(primary.shape[0])]
    reference_names = [f"REF{i}" for i in range(reference.shape[0])]
    primary_raw = mne.io.RawArray(
        primary,
        mne.create_info(primary_names, sfreq, "eeg"),
        verbose=False,
    )
    reference_raw = mne.io.RawArray(
        reference,
        mne.create_info(reference_names, sfreq, "eog"),
        verbose=False,
    )
    estimator = _estimator().fit(primary_raw, reference=reference_raw)
    changed_type = mne.io.RawArray(
        primary,
        mne.create_info(primary_names, sfreq, "eog"),
        verbose=False,
    )
    with pytest.raises(ValueError, match="types, or physical units"):
        estimator.transform(changed_type, reference=reference_raw)


def test_raw_streaming_requires_contiguous_blocks_and_replays_timeline(
    recursive_data,
):
    """Raw state records and enforces the next expected transport sample."""
    primary, reference, _, _, sfreq = recursive_data
    primary_names = [f"EEG{i}" for i in range(primary.shape[0])]
    reference_names = [f"REF{i}" for i in range(reference.shape[0])]

    def make_pair(start, stop, *, first_samp):
        primary_raw = mne.io.RawArray(
            primary[:, start:stop],
            mne.create_info(primary_names, sfreq, "eeg"),
            first_samp=first_samp,
            verbose=False,
        )
        reference_raw = mne.io.RawArray(
            reference[:, start:stop],
            mne.create_info(reference_names, sfreq, "eog"),
            first_samp=first_samp,
            verbose=False,
        )
        return primary_raw, reference_raw

    first = make_pair(0, 250, first_samp=1000)
    contiguous = make_pair(250, 500, first_samp=1250)
    gapped = make_pair(250, 500, first_samp=1251)
    estimator = _estimator().partial_fit(first[0], reference=first[1])
    replay = _estimator().load_state_json(estimator.state_json())
    with pytest.raises(ValueError, match="contiguous"):
        estimator.partial_fit(gapped[0], reference=gapped[1])
    estimator.partial_fit(contiguous[0], reference=contiguous[1])
    replay.partial_fit(contiguous[0], reference=contiguous[1])
    assert estimator.state_json() == replay.state_json()


def test_preconditions_and_state_errors(recursive_data):
    """Finite data, aligned assets, rank, and fitted-state checks are hard gates."""
    primary, reference, _, _, _ = recursive_data
    with pytest.raises(NotFittedError):
        _estimator().transform(primary, reference=reference)
    with pytest.raises(ValueError, match="same number of samples"):
        _estimator().partial_fit(primary, reference=reference[:, :-1])
    nonfinite = primary.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _estimator().partial_fit(nonfinite, reference=reference)
    with pytest.raises(ValueError, match="adaptation_mask"):
        _estimator().partial_fit(
            primary, reference=reference, adaptation_mask=np.ones(10)
        )
    with pytest.raises(ValueError, match="calibration failed"):
        _estimator().fit(np.ones_like(primary), reference=np.ones_like(reference))
    with pytest.raises(ValueError, match="frozen"):
        _estimator(adaptation_mode="frozen").process(
            primary[:, :20], reference=reference[:, :20]
        )


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([["not numeric"]], "numeric"),
        (np.ones(3), "shape"),
        (np.empty((0, 3)), "at least one"),
    ],
)
def test_array_input_shape_and_dtype_contracts(data, match):
    """Low-level array coercion gives specific failures for invalid inputs."""
    with pytest.raises((TypeError, ValueError), match=match):
        recursive_module._as_channel_time(data, name="X")


def test_numerical_helpers_cover_valid_and_degenerate_boundaries():
    """Scalar, covariance, and optional-copy helpers enforce their contracts."""
    assert recursive_module._finite_real(1, name="value") == 1.0
    with pytest.raises(ValueError, match="zero numerical energy"):
        recursive_module._inverse_sqrt(
            np.diag([1.0, -1.0]),
            regularization=1e-6,
            rank_tolerance=1e-10,
        )
    with pytest.raises(ValueError, match="positive definite"):
        recursive_module._inverse_sqrt(
            np.eye(2),
            regularization=-2.0,
            rank_tolerance=1e-10,
        )
    assert recursive_module._copy_array_or_none(None) is None
    original = np.arange(3.0)
    copied = recursive_module._copy_array_or_none(original)
    np.testing.assert_array_equal(copied, original)
    assert not np.shares_memory(copied, original)


@pytest.mark.parametrize(
    ("ref_channels", "primary_channels", "match"),
    [
        ([], None, "at least one"),
        ([3], None, "outside"),
        ([0, 0], None, "duplicates"),
        ([0, "REF"], None, "only channel names or only integers"),
        (["REF"], None, "String"),
        ([0, 1, 2], None, "leave at least one primary"),
        ([1], [0, 1], "disjoint"),
    ],
)
def test_combined_array_selector_contracts(ref_channels, primary_channels, match):
    """Combined-input selectors reject ambiguous or invalid channel sets."""
    data = np.arange(60.0).reshape(3, 20)
    estimator = _estimator(
        ref_channels=ref_channels,
        primary_channels=primary_channels,
    )
    with pytest.raises((TypeError, ValueError), match=match):
        estimator.partial_fit(data)


def test_combined_input_requires_reference_assets_and_present_names():
    """A combined stream needs explicit reference assets that exist."""
    data = np.arange(60.0).reshape(3, 20)
    with pytest.raises(ValueError, match="ref_channels are required"):
        _estimator().partial_fit(data)

    raw = mne.io.RawArray(
        data,
        mne.create_info(["P0", "P1", "R0"], 200.0, "eeg"),
        verbose=False,
    )
    with pytest.raises(ValueError, match="missing ref_channels"):
        _estimator(ref_channels=["missing"]).partial_fit(raw)


def _make_separate_raw_pair(primary, reference, sfreq, *, first_samp=0):
    """Create a small aligned primary/reference Raw pair."""
    primary_raw = mne.io.RawArray(
        primary,
        mne.create_info(
            [f"EEG{index}" for index in range(primary.shape[0])], sfreq, "eeg"
        ),
        first_samp=first_samp,
        verbose=False,
    )
    reference_raw = mne.io.RawArray(
        reference,
        mne.create_info(
            [f"REF{index}" for index in range(reference.shape[0])], sfreq, "eog"
        ),
        first_samp=first_samp,
        verbose=False,
    )
    return primary_raw, reference_raw


def test_continuous_representation_and_sampling_rate_contracts(recursive_data):
    """Streaming rejects epochs, mixed representations, and rate ambiguity."""
    primary, reference, _, _, sfreq = recursive_data
    info = mne.create_info(["EEG0", "EEG1"], sfreq, "eeg")
    epochs = mne.EpochsArray(np.ones((1, 2, 10)), info, verbose=False)
    with pytest.raises(TypeError, match="continuous Raw"):
        _estimator().partial_fit(epochs, reference=np.ones((1, 10)))

    raw_pair = _make_separate_raw_pair(primary[:, :20], reference[:, :20], sfreq)
    with pytest.raises(TypeError, match="both be Raw"):
        _estimator().partial_fit(primary[:, :20], reference=raw_pair[1])
    with pytest.raises(ValueError, match="cannot be combined"):
        _estimator(ref_channels=[0]).partial_fit(
            primary[:, :20], reference=reference[:, :20]
        )

    mismatched_primary, _ = _make_separate_raw_pair(
        primary[:, :20], reference[:, :20], sfreq
    )
    _, mismatched_reference = _make_separate_raw_pair(
        primary[:, :20], reference[:, :20], sfreq + 1.0
    )
    with pytest.raises(ValueError, match="sampling frequencies must match"):
        _estimator(sfreq=None).partial_fit(
            mismatched_primary, reference=mismatched_reference
        )

    metadata_pair = _make_separate_raw_pair(
        primary[:, :20], reference[:, :20], sfreq + 1.0
    )
    with pytest.raises(ValueError, match="does not match MNE metadata"):
        _estimator().partial_fit(metadata_pair[0], reference=metadata_pair[1])

    duration_estimator = RecursiveICanClean(
        sfreq=None,
        warmup_samples=None,
        warmup_duration_s=1.0,
        update_interval_samples=10,
    )
    with pytest.raises(ValueError, match="sfreq is required"):
        duration_estimator.partial_fit(primary[:, :20], reference=reference[:, :20])


def test_initialized_geometry_rejects_mode_shape_and_representation_changes(
    recursive_data,
):
    """Initialized streams lock mode, representation, and channel counts."""
    primary, reference, _, _, sfreq = recursive_data
    primary_block = primary[:, :20]
    reference_block = reference[:, :20]

    separate = _estimator().partial_fit(primary_block, reference=reference_block)
    separate.ref_channels = [primary.shape[0]]
    with pytest.raises(ValueError, match="switch between combined and separate"):
        separate.partial_fit(np.vstack((primary_block, reference_block)))

    combined = _estimator(
        ref_channels=list(
            range(primary.shape[0], primary.shape[0] + reference.shape[0])
        )
    ).partial_fit(np.vstack((primary_block, reference_block)))
    combined.ref_channels = None
    with pytest.raises(ValueError, match="switch between combined and separate"):
        combined.partial_fit(primary_block, reference=reference_block)

    geometry = _estimator().partial_fit(primary_block, reference=reference_block)
    with pytest.raises(ValueError, match="Primary channel count changed"):
        geometry.partial_fit(primary_block[:-1], reference=reference_block)
    with pytest.raises(ValueError, match="Reference channel count changed"):
        geometry.partial_fit(primary_block, reference=reference_block[:-1])

    raw_pair = _make_separate_raw_pair(primary_block, reference_block, sfreq)
    with pytest.raises(TypeError, match="switch between Raw and NumPy"):
        geometry.partial_fit(raw_pair[0], reference=raw_pair[1])


def test_initialized_raw_geometry_locks_reference_identity_and_rate(recursive_data):
    """Raw reference metadata and sampling rate remain part of state geometry."""
    primary, reference, _, _, sfreq = recursive_data
    primary_block = primary[:, :20]
    reference_block = reference[:, :20]
    raw_pair = _make_separate_raw_pair(primary_block, reference_block, sfreq)
    estimator = _estimator(sfreq=None).partial_fit(raw_pair[0], reference=raw_pair[1])

    changed_reference = mne.io.RawArray(
        reference_block,
        mne.create_info(raw_pair[1].ch_names, sfreq, "eeg"),
        verbose=False,
    )
    with pytest.raises(ValueError, match="Reference MNE channel"):
        estimator.partial_fit(raw_pair[0], reference=changed_reference)

    changed_rate = _make_separate_raw_pair(primary_block, reference_block, sfreq + 1.0)
    with pytest.raises(ValueError, match="Sampling frequency differs"):
        estimator.partial_fit(changed_rate[0], reference=changed_rate[1])


def test_adaptation_selection_and_fit_parameter_boundaries(recursive_data):
    """Gates, selection caps, calibration size, and fit kwargs are explicit."""
    primary, reference, _, _, _ = recursive_data
    with pytest.raises(ValueError, match="boolean/0/1"):
        _estimator().partial_fit(
            primary[:, :20],
            reference=reference[:, :20],
            adaptation_mask=np.full(20, 2),
        )

    mixed_gate = np.arange(primary.shape[1]) % 2 == 0
    mixed = _estimator().fit(
        primary,
        reference=reference,
        adaptation_mask=mixed_gate,
    )
    assert mixed._samples_adapted_ == int(mixed_gate.sum())

    no_op = _estimator(max_reject_fraction=0.0, verbose=True).fit(
        primary[:, :400], reference=reference[:, :400]
    )
    assert no_op.n_removed_ == 0
    assert no_op.last_update_diagnostics_["status"] == "NO_OP"
    np.testing.assert_array_equal(no_op.artifact_operator_, 0.0)

    capped = _estimator(threshold=0.0, max_reject_fraction=0.1).fit(
        primary[:, :400], reference=reference[:, :400]
    )
    assert capped.n_removed_ == 1

    insufficient_gate = np.zeros(400, dtype=bool)
    insufficient_gate[:100] = True
    with pytest.raises(ValueError, match="accepted samples"):
        _estimator().fit(
            primary[:, :400],
            reference=reference[:, :400],
            adaptation_mask=insufficient_gate,
        )

    with pytest.raises(TypeError, match="Unexpected fit parameters"):
        _estimator().fit_transform(
            primary[:, :20], reference=reference[:, :20], unknown=True
        )
    with pytest.raises(ValueError, match="not been initialized"):
        _estimator().state_dict()


def test_update_before_mode_is_explicitly_current_sample_adaptive(recursive_data):
    """Update-before mode reports and exhibits its current-sample dependency."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator(
        warmup_samples=20,
        update_interval_samples=5,
        update_order="before",
    )
    cleaned = estimator.process(primary[:, :30], reference=reference[:, :30])
    assert cleaned.shape == primary[:, :30].shape
    assert estimator.last_process_diagnostics_["current_sample_used_for_own_model"]
    assert estimator.last_process_diagnostics_["adaptation_delay_samples"] == 0
    assert estimator.last_process_diagnostics_["warmup_passthrough_samples"] == 19


def test_model_update_preconditions_and_numerical_failures(recursive_data, monkeypatch):
    """Early and failed model updates abstain without installing invalid state."""
    primary, reference, _, _, _ = recursive_data
    warming = _estimator().partial_fit(primary[:, :20], reference=reference[:, :20])
    assert not warming._update_model()

    def zero_rank(covariance, *, regularization, rank_tolerance):
        del regularization, rank_tolerance
        return np.eye(covariance.shape[0]), 0, 0.0

    monkeypatch.setattr(recursive_module, "_inverse_sqrt", zero_rank)
    rank_failure = _estimator().partial_fit(
        primary[:, :200], reference=reference[:, :200]
    )
    assert "zero rank" in rank_failure.last_update_diagnostics_["reason"]


@pytest.mark.parametrize("failure", ["solve", "nonfinite"])
def test_regression_failure_paths_retain_inadmissible_state(
    recursive_data, monkeypatch, failure
):
    """Regression failure and non-finite solutions are diagnosed as inadmissible."""
    primary, reference, _, _, _ = recursive_data

    if failure == "solve":

        def fail_solve(*args, **kwargs):
            raise recursive_module.la.LinAlgError("regression locked")

        monkeypatch.setattr(recursive_module.la, "solve", fail_solve)
        match = "artifact regression failed"
    else:

        def nonfinite_solve(matrix, target, **kwargs):
            del matrix, kwargs
            return np.full_like(target, np.nan)

        monkeypatch.setattr(recursive_module.la, "solve", nonfinite_solve)
        match = "non-finite operator"

    estimator = _estimator(threshold=0.0).partial_fit(
        primary[:, :200], reference=reference[:, :200]
    )
    assert estimator.last_update_diagnostics_["status"] == "INADMISSIBLE"
    assert match in estimator.last_update_diagnostics_["reason"]


def test_nonfinite_frozen_output_is_rejected(recursive_data):
    """A corrupted installed operator cannot emit non-finite cleaned samples."""
    primary, reference, _, _, _ = recursive_data
    estimator = _estimator().fit(primary[:, :400], reference=reference[:, :400])
    estimator.artifact_operator_[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite output"):
        estimator.transform(primary[:, :20], reference=reference[:, :20])


def test_checkpoint_codec_round_trips_special_values_and_numpy_scalars():
    """Canonical state encoding handles every supported scalar representation."""
    source = {
        "numpy_scalar": np.int64(7),
        "tuple": (1, 2),
        "nan": np.nan,
        "positive_infinity": np.inf,
        "negative_infinity": -np.inf,
    }
    payload = recursive_module._canonical_state_json(source)
    decoded = recursive_module._decode_state_value(json.loads(payload))
    assert decoded["numpy_scalar"] == 7
    assert decoded["tuple"] == (1, 2)
    assert np.isnan(decoded["nan"])
    assert decoded["positive_infinity"] == np.inf
    assert decoded["negative_infinity"] == -np.inf


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"array": np.array([object()], dtype=object)}, "Object arrays"),
        ({1: "non-string key"}, "string keys"),
        ({"unsupported": {1, 2}}, "Unsupported state value"),
    ],
)
def test_checkpoint_encoder_rejects_nonportable_values(value, match):
    """Checkpoint JSON refuses values without a stable portable identity."""
    with pytest.raises(TypeError, match=match):
        recursive_module._canonical_state_json(value)


@pytest.mark.parametrize(
    ("payload", "error", "match"),
    [
        (b"\xff", ValueError, "valid UTF-8"),
        (123, TypeError, "JSON string"),
        ("{", ValueError, "valid JSON"),
        ("[]", ValueError, "dictionary"),
        (json.dumps({"__tuple__": "bad"}), ValueError, "Malformed tuple"),
        (
            json.dumps({"__float__": "sideways"}),
            ValueError,
            "Malformed non-finite float",
        ),
        (
            json.dumps({"__ndarray__": "", "dtype": "|O", "shape": [0]}),
            ValueError,
            "Malformed array",
        ),
        (
            json.dumps({"__ndarray__": "*", "dtype": "<f8", "shape": [1]}),
            ValueError,
            "Malformed array",
        ),
        (
            json.dumps({"__ndarray__": "", "dtype": "<f8", "shape": [1]}),
            ValueError,
            "Malformed array",
        ),
        (
            json.dumps({"__reserved__": 1}),
            ValueError,
            "Unknown reserved marker",
        ),
    ],
)
def test_checkpoint_decoder_rejects_malformed_payloads(payload, error, match):
    """Public JSON loading rejects malformed encodings before state mutation."""
    with pytest.raises(error, match=match):
        RecursiveICanClean().load_state_json(payload)


def _resign_state(state):
    """Refresh the outer checksum so deeper checkpoint validators execute."""
    state["state_sha256"] = recursive_module._state_payload_hash(state)
    return state


def _corrupt_state(state, corruption):
    """Apply one schema-valid numerical corruption for validation tests."""
    if corruption == "input_mode":
        state["input_mode"] = "windowed"
    elif corruption == "mne_type":
        state["mne_type"] = "epochs"
    elif corruption == "channel_count":
        state["n_primary"] = 0
    elif corruption == "sample_counters":
        state["samples_processed"] = -1
    elif corruption == "effective_weight":
        state["effective_weight"] = np.nan
    elif corruption == "warmup":
        state["warmup_samples_resolved"] = 1
    elif corruption == "forgetting_factor":
        state["forgetting_factor_resolved"] = 0.0
    elif corruption == "next_update":
        state["next_update_sample"] = state["samples_adapted"]
    elif corruption == "model_counter":
        state["model_version"] = -1
    elif corruption == "convergence":
        state["converged"] = not state["converged"]
    elif corruption == "sfreq":
        state["sfreq_resolved"] = np.nan
    elif corruption == "array_timeline":
        state["next_raw_first_samp"] = 1
    elif corruption == "array_signature":
        state["primary_channel_signature"] = (("P0", 1, 1, 0, 1),)
    elif corruption == "raw_signature":
        state["mne_type"] = "raw"
    elif corruption == "primary_names":
        state["primary_ch_names"] = ("P0",)
    elif corruption == "reference_names":
        state["reference_ch_names"] = ("R0",)
    elif corruption == "histories":
        state["update_history"] = ()
    elif corruption == "moment_shape":
        state["mean_primary"] = state["mean_primary"][:-1]
    elif corruption == "primary_symmetry":
        state["scatter_primary"][0, 1] += 1.0
    elif corruption == "reference_symmetry":
        state["scatter_reference"][0, 1] += 1.0
    elif corruption == "artifact_shape":
        state["model"]["artifact_operator_"] = state["model"]["artifact_operator_"][
            :, :-1
        ]
    elif corruption == "model_nonfinite":
        state["model"]["x_weights_"][0, 0] = np.nan
    elif corruption == "ranks":
        state["model"]["rank_primary_"] = 0
    elif corruption == "x_weights_shape":
        state["model"]["x_weights_"] = state["model"]["x_weights_"][:, :-1]
    elif corruption == "y_weights_shape":
        state["model"]["y_weights_"] = state["model"]["y_weights_"][:, :-1]
    elif corruption == "correlations_shape":
        state["model"]["correlations_"] = state["model"]["correlations_"][:-1]
    elif corruption == "correlations_range":
        state["model"]["correlations_"][0] = 2.0
    elif corruption == "removed_indices":
        state["model"]["removed_idx_"] = np.array(
            [state["model"]["rank_ceiling_"]], dtype=int
        )
    elif corruption == "regression_shape":
        state["model"]["regression_"] = state["model"]["regression_"][:, :-1]
    elif corruption == "cleaning_shape":
        state["model"]["cleaning_operator_"] = state["model"]["cleaning_operator_"][
            :, :-1
        ]
    elif corruption == "primary_model_mean":
        state["model"]["model_primary_mean_"] = state["model"]["model_primary_mean_"][
            :-1
        ]
    elif corruption == "reference_model_mean":
        state["model"]["model_reference_mean_"] = state["model"][
            "model_reference_mean_"
        ][:-1]
    elif corruption == "model_samples":
        state["model"]["model_samples_"] = state["samples_adapted"] + 1
    elif corruption == "model_missing":
        state["model"] = {}
    else:
        raise AssertionError(f"Unknown corruption: {corruption}")


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("input_mode", "input_mode"),
        ("mne_type", "mne_type"),
        ("channel_count", "channel counts"),
        ("sample_counters", "sample counters"),
        ("effective_weight", "effective weight"),
        ("warmup", "update intervals"),
        ("forgetting_factor", "forgetting factor"),
        ("next_update", "update boundary"),
        ("model_counter", "model counters"),
        ("convergence", "convergence flag"),
        ("sfreq", "sampling frequency"),
        ("array_timeline", "Only Raw state"),
        ("array_signature", "NumPy state"),
        ("raw_signature", "Raw state"),
        ("primary_names", "primary channel names"),
        ("reference_names", "reference channel names"),
        ("histories", "histories"),
        ("moment_shape", "mean_primary"),
        ("primary_symmetry", "primary scatter"),
        ("reference_symmetry", "reference scatter"),
        ("artifact_shape", "artifact_operator"),
        ("model_nonfinite", "x_weights"),
        ("ranks", "covariance ranks"),
        ("x_weights_shape", "x_weights"),
        ("y_weights_shape", "y_weights"),
        ("correlations_shape", "correlations_ has invalid shape"),
        ("correlations_range", "lie in"),
        ("removed_indices", "removed component"),
        ("regression_shape", "regression"),
        ("cleaning_shape", "cleaning_operator"),
        ("primary_model_mean", "model_primary_mean"),
        ("reference_model_mean", "model_reference_mean"),
        ("model_samples", "model sample count"),
        ("model_missing", "model version"),
    ],
)
def test_checkpoint_numerical_validation(fitted_state, corruption, match):
    """Checksum-valid but inconsistent numerical state is rejected precisely."""
    state = copy.deepcopy(fitted_state)
    _corrupt_state(state, corruption)
    _resign_state(state)
    with pytest.raises(ValueError, match=match):
        _estimator().load_state_dict(state)


def test_checkpoint_schema_version_and_layered_checksums(fitted_state):
    """Schema, version, model fields, and both checksum layers are enforced."""
    with pytest.raises(TypeError, match="dictionary"):
        _estimator().load_state_dict([])

    missing_field = copy.deepcopy(fitted_state)
    missing_field.pop("process_history")
    with pytest.raises(ValueError, match="schema"):
        _estimator().load_state_dict(missing_field)

    wrong_version = copy.deepcopy(fitted_state)
    wrong_version["version"] += 1
    with pytest.raises(ValueError, match="Unsupported"):
        _estimator().load_state_dict(wrong_version)

    incomplete_model = copy.deepcopy(fitted_state)
    incomplete_model["model"].pop("x_weights_")
    with pytest.raises(ValueError, match="model fields"):
        _estimator().load_state_dict(incomplete_model)

    invalid_checksum = copy.deepcopy(fitted_state)
    invalid_checksum["state_sha256"] = "short"
    with pytest.raises(ValueError, match="valid state_sha256"):
        _estimator().load_state_dict(invalid_checksum)

    model_checksum = copy.deepcopy(fitted_state)
    model_checksum["model_state_sha256"] = "0" * 64
    _resign_state(model_checksum)
    with pytest.raises(ValueError, match="numerical payload"):
        _estimator().load_state_dict(model_checksum)


def test_checkpoint_restores_last_process_diagnostics(recursive_data):
    """A processed checkpoint restores optional process diagnostics exactly."""
    primary, reference, _, _, _ = recursive_data
    source = _estimator()
    source.process(primary[:, :250], reference=reference[:, :250])
    replay = _estimator().load_state_dict(source.state_dict())
    assert replay.last_process_diagnostics_ == source.last_process_diagnostics_
