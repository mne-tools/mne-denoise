"""Tests for GEVD multi-channel Wiener filtering."""

from __future__ import annotations

import logging

import mne
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mne_denoise.mwf import (
    MWF,
    MultichannelWienerFilter,
    compute_mwf,
    hf_power_mask,
    mwf_filter,
)
from mne_denoise.mwf.core import _apply_spatial_filter, _compute_operator


@pytest.fixture(scope="module")
def contaminated_data():
    """Known clean signal plus a marked spatial high-frequency artifact."""
    rng = np.random.default_rng(42)
    sfreq = 250.0
    n_channels, n_samples = 8, 4000
    times = np.arange(n_samples) / sfreq
    sources = np.vstack(
        [
            np.sin(2 * np.pi * 7.0 * times),
            np.sin(2 * np.pi * 10.0 * times + 0.4),
            np.sin(2 * np.pi * 13.0 * times + 0.8),
        ]
    )
    clean = rng.standard_normal((n_channels, 3)) @ sources
    clean += 0.05 * rng.standard_normal(clean.shape)
    artifact_mask = np.zeros(n_samples, dtype=bool)
    for start in (400, 1400, 2400, 3300):
        artifact_mask[start : start + 250] = True
    artifact_topography = rng.standard_normal(n_channels)
    artifact_topography /= np.linalg.norm(artifact_topography)
    artifact_waveform = rng.standard_normal(n_samples)
    artifact_waveform[~artifact_mask] = 0.0
    contaminated = clean + 8.0 * np.outer(artifact_topography, artifact_waveform)
    return contaminated, clean, artifact_mask, artifact_topography, sfreq


def test_canonical_name_and_alias():
    """MWF is the documented alias of the canonical class."""
    assert MWF is MultichannelWienerFilter
    assert isinstance(MWF(), MultichannelWienerFilter)


def test_estimator_is_cloneable():
    """Constructor parameters follow sklearn cloning semantics."""
    estimator = MultichannelWienerFilter(rank=3, reg=1e-4, treat_nan="clean")
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()


def test_explicit_mask_is_required(contaminated_data):
    """MWF does not silently invent an artifact definition."""
    data = contaminated_data[0]
    with pytest.raises(ValueError, match="explicit artifact_mask"):
        compute_mwf(data)
    with pytest.raises(ValueError, match="explicit artifact_mask"):
        MultichannelWienerFilter().fit(data)


def test_hf_power_mask_flags_bursts(contaminated_data):
    """The opt-in HF heuristic enriches the known burst intervals."""
    data, _, true_mask, _, sfreq = contaminated_data
    mask = hf_power_mask(data, sfreq, hf_hz=20.0, quantile=0.7)
    assert mask.shape == (data.shape[1],)
    assert mask.dtype == bool
    assert (mask & true_mask).sum() / mask.sum() > 0.8


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sfreq": 0.0}, "sfreq"),
        ({"sfreq": 250.0, "hf_hz": 125.0}, "Nyquist"),
        ({"sfreq": 250.0, "quantile": 1.0}, "quantile"),
        ({"sfreq": 250.0, "smooth_s": 0.0}, "smooth_s"),
    ],
)
def test_hf_power_mask_validates_operating_point(kwargs, match):
    """Physical mask parameters have explicit admissible ranges."""
    with pytest.raises(ValueError, match=match):
        hf_power_mask(np.ones((3, 100)), **kwargs)


def test_hf_power_mask_rejects_too_short_data():
    """Short input fails clearly instead of leaking scipy's pad error."""
    with pytest.raises(ValueError, match="too short"):
        hf_power_mask(np.ones((3, 8)), 250.0)


def test_compute_mwf_diagnostics_and_attenuation(contaminated_data):
    """Known artifact energy falls and fit diagnostics remain explicit."""
    data, _, mask, topography, _ = contaminated_data
    cleaned, diagnostics = compute_mwf(data, mask)
    before = np.var(topography @ data[:, mask])
    after = np.var(topography @ cleaned[:, mask])
    assert after < 0.2 * before
    assert diagnostics["artifact_samples"] == int(mask.sum())
    assert diagnostics["clean_samples"] == int((~mask).sum())
    assert diagnostics["rank_requested"] == "positive"
    assert diagnostics["rank_used"] >= 1
    assert diagnostics["spatial_filter"].shape == (data.shape[0], data.shape[0])


def test_mwf_filter_matches_compute_mwf(contaminated_data):
    """The compact functional API delegates to the same implementation."""
    data, _, mask, _, _ = contaminated_data
    expected, _ = compute_mwf(data, mask, rank=2)
    observed = mwf_filter(data, mask, rank=2)
    np.testing.assert_allclose(observed, expected)


def test_full_rank_matches_covariance_ratio():
    """Full-rank GEVD equals the regularized covariance-ratio MWF."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 3000))
    mask = np.zeros(data.shape[1], dtype=bool)
    mask[:1200] = True
    data[:, mask] += np.outer(rng.standard_normal(5), rng.standard_normal(mask.sum()))
    _, diagnostics = compute_mwf(data, mask, rank="full", reg=1e-4)
    artifact_cov = np.cov(data[:, mask])
    clean_cov = np.cov(data[:, ~mask])
    scale = max(np.trace(artifact_cov) / 5, np.trace(clean_cov) / 5)
    artifact_cov += 1e-4 * scale * np.eye(5)
    clean_cov += 1e-4 * scale * np.eye(5)
    expected = clean_cov @ np.linalg.inv(artifact_cov)
    np.testing.assert_allclose(
        diagnostics["spatial_filter"], expected, rtol=1e-8, atol=1e-10
    )


def test_channel_means_are_preserved(contaminated_data):
    """MWF follows the reference mean-subtract/apply/restore convention."""
    data, _, mask, _, _ = contaminated_data
    shifted = data + np.arange(data.shape[0])[:, None] * 10.0
    cleaned, _ = compute_mwf(shifted, mask)
    np.testing.assert_allclose(cleaned.mean(axis=1), shifted.mean(axis=1), atol=1e-12)


def test_global_unit_rescaling_preserves_operator(contaminated_data):
    """Relative regularization is invariant to a shared physical-unit scale."""
    data, _, mask, _, _ = contaminated_data
    cleaned, diagnostics = compute_mwf(data, mask)
    scaled, scaled_diagnostics = compute_mwf(data * 1e6, mask)
    np.testing.assert_allclose(
        diagnostics["spatial_filter"],
        scaled_diagnostics["spatial_filter"],
        rtol=1e-8,
        atol=1e-10,
    )
    # Generalized eigensolvers vary by a few ulps across BLAS implementations;
    # this still bounds the unit-rescaling error below one part in 10 million.
    np.testing.assert_allclose(scaled, cleaned * 1e6, rtol=1e-7, atol=2e-6)


def test_rank_deficiency_is_regularized():
    """Default loading supports rank-deficient data while zero loading is explicit."""
    rng = np.random.default_rng(2)
    latent = rng.standard_normal((2, 1000))
    mixing = rng.standard_normal((6, 2))
    data = mixing @ latent
    mask = np.zeros(1000, dtype=bool)
    mask[:400] = True
    cleaned, diagnostics = compute_mwf(data, mask, reg=1e-6)
    assert np.all(np.isfinite(cleaned))
    assert diagnostics["clean_covariance_rank"] < data.shape[0]
    with pytest.raises(ValueError, match="positive definite"):
        compute_mwf(data, mask, reg=0.0)


@pytest.mark.parametrize("bad_mask", [np.zeros(10), np.full(4000, 2), ["x"] * 4000])
def test_artifact_mask_validation(contaminated_data, bad_mask):
    """Mask shape and values cannot silently change the training regime."""
    data = contaminated_data[0]
    with pytest.raises((TypeError, ValueError), match="artifact_mask"):
        compute_mwf(data, bad_mask)


def test_nan_mask_policies(contaminated_data):
    """Ignored mask samples stay out of both covariance estimates."""
    data, _, mask, _, _ = contaminated_data
    ternary = mask.astype(float)
    ternary[-200:] = np.nan
    _, ignored = compute_mwf(data, ternary, treat_nan="ignore")
    _, clean = compute_mwf(data, ternary, treat_nan="clean")
    assert ignored["ignored_samples"] == 200
    assert clean["ignored_samples"] == 0
    assert clean["clean_samples"] == ignored["clean_samples"] + 200


def test_clean_reference_contract(contaminated_data):
    """A clean reference can supply the clean covariance explicitly."""
    data, clean, _, _, _ = contaminated_data
    cleaned, diagnostics = compute_mwf(data, clean_reference=clean)
    assert cleaned.shape == data.shape
    assert diagnostics["used_clean_reference"] is True
    assert diagnostics["artifact_samples"] == data.shape[1]


def test_inadmissible_inputs_fail_clearly(contaminated_data):
    """Non-finite, single-channel, and insufficient masks are rejected."""
    data, _, mask, _, _ = contaminated_data
    with pytest.raises(ValueError, match="at least 2 channels"):
        compute_mwf(data[:1], mask)
    nonfinite = data.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_mwf(nonfinite, mask)
    with pytest.raises(ValueError, match="two artifact"):
        compute_mwf(data, np.zeros(data.shape[1], dtype=bool))


def test_estimator_fit_transform_and_frozen_operator(contaminated_data):
    """Transform applies the fitted operator without evaluating a new mask."""
    data, _, mask, _, _ = contaminated_data
    estimator = MultichannelWienerFilter(rank=2).fit(data, artifact_mask=mask)
    evaluation = data[:, :500] * 0.5
    observed = estimator.transform(evaluation)
    means = evaluation.mean(axis=1, keepdims=True)
    expected = estimator.spatial_filter_ @ (evaluation - means) + means
    np.testing.assert_allclose(observed, expected)
    assert estimator.artifact_mask_.shape == mask.shape
    assert estimator.selected_components_.size == 2


def test_fit_transform_accepts_mask(contaminated_data):
    """fit_transform forwards explicit fit assets without leakage ambiguity."""
    data, _, mask, _, _ = contaminated_data
    cleaned = MultichannelWienerFilter().fit_transform(data, artifact_mask=mask)
    assert cleaned.shape == data.shape


def test_transform_before_fit_and_channel_mismatch(contaminated_data):
    """Estimator state and sensor dimensionality are enforced."""
    data, _, mask, _, _ = contaminated_data
    with pytest.raises(NotFittedError):
        MultichannelWienerFilter().transform(data)
    estimator = MultichannelWienerFilter().fit(data, artifact_mask=mask)
    with pytest.raises(ValueError, match="different number of channels"):
        estimator.transform(data[:-1])


def test_explicit_hf_strategy_requires_consistent_sfreq(contaminated_data):
    """Automatic mask creation is opt-in and its sampling rate is checked."""
    data, _, _, _, sfreq = contaminated_data
    estimator = MultichannelWienerFilter(
        mask_strategy="hf_power", sfreq=sfreq, quantile=0.7
    ).fit(data)
    assert estimator.fit_diagnostics_["mask_strategy"] == "hf_power"
    with pytest.raises(ValueError, match="sfreq is required"):
        MultichannelWienerFilter(mask_strategy="hf_power").fit(data)


def _make_raw(contaminated_data):
    """Create Raw with EEG plus an untouched stimulus channel and metadata."""
    data, _, mask, _, sfreq = contaminated_data
    eeg = data[:4]
    stim = np.zeros((1, data.shape[1]))
    stim[0, 123] = 1
    names = [f"EEG{index:02d}" for index in range(4)] + ["STI 014"]
    info = mne.create_info(names, sfreq, ["eeg"] * 4 + ["stim"])
    info["bads"] = ["EEG01"]
    raw = mne.io.RawArray(np.vstack([eeg, stim]), info, first_samp=100, verbose=False)
    raw.set_annotations(mne.Annotations([1.0], [0.2], ["test"]), emit_warning=False)
    return raw, mask


def test_raw_roundtrip_preserves_metadata_and_unpicked_channels(contaminated_data):
    """Raw subtype, first sample, annotations, bads, and stim data survive."""
    raw, mask = _make_raw(contaminated_data)
    estimator = MultichannelWienerFilter().fit(raw, artifact_mask=mask)
    cleaned = estimator.transform(raw)
    assert type(cleaned) is type(raw)
    assert cleaned.first_samp == raw.first_samp
    assert cleaned.info["bads"] == raw.info["bads"]
    assert cleaned.annotations == raw.annotations
    np.testing.assert_array_equal(
        cleaned.get_data(picks=["STI 014"]), raw.get_data(picks=["STI 014"])
    )
    assert not np.allclose(cleaned.get_data(picks="eeg"), raw.get_data(picks="eeg"))


def test_mne_channel_alignment_and_sfreq_mismatch(contaminated_data):
    """Named channels are aligned and conflicting physical time units fail."""
    raw, mask = _make_raw(contaminated_data)
    estimator = MultichannelWienerFilter().fit(raw, artifact_mask=mask)
    reordered = raw.copy().reorder_channels(list(reversed(raw.ch_names)))
    cleaned = estimator.transform(reordered)
    assert cleaned.ch_names == reordered.ch_names
    with pytest.raises(ValueError, match="does not match MNE metadata"):
        MultichannelWienerFilter(sfreq=500.0).fit(raw, artifact_mask=mask)


def test_epochs_roundtrip_and_epoch_mask(contaminated_data):
    """Epoch masks flatten consistently while events and metadata survive."""
    pandas = pytest.importorskip("pandas")
    data, _, mask, _, sfreq = contaminated_data
    epochs_data = data[:4, :2000].reshape(4, 5, 400).transpose(1, 0, 2)
    epoch_mask = mask[:2000].reshape(5, 400)
    info = mne.create_info([f"EEG{i}" for i in range(4)], sfreq, "eeg")
    events = np.column_stack([np.arange(5) * 500, np.zeros(5, int), np.ones(5, int)])
    metadata = pandas.DataFrame({"trial": np.arange(5)})
    epochs = mne.EpochsArray(
        epochs_data,
        info,
        events=events,
        event_id={"event": 1},
        metadata=metadata,
        verbose=False,
    )
    cleaned = MultichannelWienerFilter().fit_transform(epochs, artifact_mask=epoch_mask)
    assert type(cleaned) is type(epochs)
    np.testing.assert_array_equal(cleaned.events, epochs.events)
    assert cleaned.event_id == epochs.event_id
    assert cleaned.metadata.equals(epochs.metadata)
    assert cleaned.get_data().shape == epochs.get_data().shape


def test_evoked_roundtrip(contaminated_data):
    """Evoked comment, nave, time origin, and data shape survive."""
    data, _, mask, _, sfreq = contaminated_data
    info = mne.create_info([f"EEG{i}" for i in range(4)], sfreq, "eeg")
    evoked = mne.EvokedArray(
        data[:4], info, tmin=-0.2, nave=12, comment="condition", verbose=False
    )
    cleaned = MultichannelWienerFilter().fit_transform(evoked, artifact_mask=mask)
    assert type(cleaned) is type(evoked)
    assert cleaned.comment == evoked.comment
    assert cleaned.nave == evoked.nave
    assert cleaned.first == evoked.first
    assert cleaned.data.shape == evoked.data.shape


def test_mne_clean_reference_alignment_and_sfreq(contaminated_data):
    """Reference channels align by name and reference sfreq must match."""
    raw, _ = _make_raw(contaminated_data)
    reference = raw.copy().reorder_channels(list(reversed(raw.ch_names)))
    estimator = MultichannelWienerFilter().fit(raw, clean_reference=reference)
    assert estimator.fit_diagnostics_["used_clean_reference"] is True
    mismatch = reference.copy().resample(125.0, verbose=False)
    with pytest.raises(ValueError, match="sampling frequency must match"):
        MultichannelWienerFilter().fit(raw, clean_reference=mismatch)


def test_input_and_operating_point_validation_branches(contaminated_data):
    """Invalid arrays, masks, ranks, and explicit strategies fail at the boundary."""
    data, _, mask, _, sfreq = contaminated_data

    with pytest.raises(TypeError, match="numeric array"):
        compute_mwf("not numeric")
    with pytest.raises(ValueError, match="shape"):
        compute_mwf(np.ones(10), np.ones(10))
    with pytest.raises(ValueError, match="at least 2 samples"):
        compute_mwf(np.ones((2, 1)), np.ones(1))
    with pytest.raises(ValueError, match="finite real"):
        hf_power_mask(data, True)
    with pytest.raises(ValueError, match="treat_nan"):
        compute_mwf(data, mask, treat_nan="invalid")

    infinite_mask = mask.astype(float)
    infinite_mask[0] = np.inf
    with pytest.raises(ValueError, match="0, 1, or NaN"):
        compute_mwf(data, infinite_mask)

    for bad_rank, error in (
        ("invalid", ValueError),
        (True, TypeError),
        (100, ValueError),
    ):
        with pytest.raises(error, match="rank"):
            compute_mwf(data, mask, rank=bad_rank)
    with pytest.raises(ValueError, match="reg must be >="):
        compute_mwf(data, mask, reg=-1.0)
    with pytest.raises(ValueError, match="Pass artifact_mask or mask_strategy"):
        compute_mwf(data, mask, mask_strategy="hf_power", sfreq=sfreq)
    with pytest.raises(ValueError, match="mask_strategy"):
        compute_mwf(data, mask_strategy="invalid")


def test_mask_policy_and_training_preconditions(contaminated_data):
    """Mask conversion and covariance preconditions remain explicit."""
    data, _, mask, _, _ = contaminated_data
    ternary = mask.astype(float)
    ternary[-200:] = np.nan
    _, diagnostics = compute_mwf(data, ternary, treat_nan="artifact")
    assert diagnostics["artifact_samples"] == int(mask.sum()) + 200
    assert diagnostics["ignored_samples"] == 0

    with pytest.raises(ValueError, match="same number of channels"):
        compute_mwf(data, mask, clean_reference=data[:-1])
    almost_all_artifact = np.ones(data.shape[1], dtype=bool)
    almost_all_artifact[0] = False
    with pytest.raises(ValueError, match="two clean samples"):
        compute_mwf(data, almost_all_artifact)

    zero_data = np.zeros((2, 6))
    zero_mask = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    with pytest.raises(ValueError, match="zero numerical energy"):
        compute_mwf(zero_data, zero_mask)


def test_internal_operator_shape_preconditions():
    """The mathematical core rejects mismatched or undersampled training arrays."""
    kwargs = {"rank": "positive", "artifact_weight": 1.0, "reg": 1e-6}
    with pytest.raises(ValueError, match="same channel count"):
        _compute_operator(np.ones((2, 4)), np.ones((3, 4)), **kwargs)
    with pytest.raises(ValueError, match="at least two"):
        _compute_operator(np.ones((2, 1)), np.ones((2, 4)), **kwargs)
    with pytest.raises(ValueError, match="2D or 3D"):
        _apply_spatial_filter(np.ones(4), np.eye(2))


def test_estimator_diagnostic_and_error_paths(contaminated_data, caplog):
    """Logging, frozen evaluation checks, and unknown fit assets are covered."""
    data, _, mask, _, _ = contaminated_data
    with caplog.at_level(logging.INFO, logger="mne_denoise.mwf.core"):
        estimator = MultichannelWienerFilter(verbose=True).fit(data, artifact_mask=mask)
    assert "MWF fit:" in caplog.text

    nonfinite = data[:, :20].copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator.transform(nonfinite)
    with pytest.raises(TypeError, match="Unexpected fit parameters: unknown"):
        estimator.fit_transform(data, artifact_mask=mask, unknown=True)
