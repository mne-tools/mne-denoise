"""Tests for the experimental DSS-guided soft ASR (``GuidedASR``).

Covers the strict-generalization guarantee (``reconstruction="hard"`` + no bias
operators reproduces ``ASR(method="riemannian_windowed")`` exactly), the
discriminative soft-weight math, soft reconstruction behavior, MNE/ndarray
I/O, and the experimental opt-in / validation guards.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.base import clone

from mne_denoise.asr import ASR, GuidedASR, process_guided_asr
from mne_denoise.asr._guidance import (
    _compute_guidance_covariance,
    _guided_component_weights,
)
from mne_denoise.asr._validation import _validate_covariance_matrix
from mne_denoise.dss.denoisers import BandpassBias, LineNoiseBias, PeakFilterBias

SFREQ = 250.0


def test_asr_family_exposes_sklearn_constructor_state():
    assert ASR(sfreq=SFREQ, picks=None).get_params()["picks"] is None
    assert GuidedASR(sfreq=SFREQ).get_params()["filter_kind"] == ASR(
        sfreq=SFREQ
    ).get_params()["filter_kind"]


def _eeg(n_channels=8, n_times=6000, seed=0, bursts=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / SFREQ
    X = np.zeros((n_channels, n_times))
    for c in range(n_channels):
        X[c] = 0.6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28)) + (
            0.05 * rng.standard_normal(n_times)
        )
    for s in np.linspace(800, n_times - 500, bursts).astype(int):
        spatial = rng.standard_normal(n_channels)
        spatial /= np.linalg.norm(spatial)
        X[:, s : s + 150] += 10.0 * np.outer(spatial, rng.standard_normal(150))
    return X


# ---------------------------------------------------------------------------
# Strict-generalization guarantee
# ---------------------------------------------------------------------------


def test_hard_no_bias_equals_riemannian_windowed():
    X = _eeg()
    ref = np.asarray(
        ASR(
            sfreq=SFREQ,
            cutoff=20.0,
            method="riemannian_windowed",
            picks=None,
            verbose=False,
        ).fit_transform(X)
    )
    guided = np.asarray(
        GuidedASR(
            sfreq=SFREQ,
            cutoff=20.0,
            reconstruction="hard",
            picks=None,
            verbose=False,
        ).fit_transform(X)
    )
    # Soft path with the binary mask must be byte-for-byte the standard backend.
    np.testing.assert_allclose(guided, ref, rtol=0, atol=1e-9)


def test_hard_no_bias_needs_no_experimental_optin():
    # reconstruction="hard" reproduces ASR and requires no experimental flag.
    out = GuidedASR(
        sfreq=SFREQ, cutoff=20.0, reconstruction="hard", picks=None, verbose=False
    ).fit_transform(_eeg())
    assert np.all(np.isfinite(np.asarray(out)))


def test_constructor_tracks_asr_parameter_defaults_and_kinds():
    asr_parameters = inspect.signature(ASR).parameters
    guided_parameters = inspect.signature(GuidedASR).parameters
    for name, parameter in asr_parameters.items():
        if name == "method":
            continue
        assert name in guided_parameters
        assert guided_parameters[name].default == parameter.default
        assert guided_parameters[name].kind == parameter.kind


# ---------------------------------------------------------------------------
# Discriminative soft-weight math (the novel core)
# ---------------------------------------------------------------------------


def test_covariance_validation_does_not_change_scale():
    covariance = np.diag([1.0, 2.0, 3.0, 4.0])
    prepared = _validate_covariance_matrix(
        covariance,
        n_channels=4,
        name="covariance",
    )
    np.testing.assert_allclose(prepared, covariance)
    np.testing.assert_allclose(
        _validate_covariance_matrix(
            100.0 * covariance, n_channels=4, name="covariance"
        ),
        100.0 * covariance,
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        _validate_covariance_matrix(
            np.diag([1.0, 1.0, 1.0, -1.0]),
            n_channels=4,
            name="covariance",
        )


def test_guidance_covariance_uses_equal_scale_invariant_bias_votes():
    class IdentityBias:
        def apply(self, data):
            return data

    class ScaledIdentityBias:
        def apply(self, data):
            return 100.0 * data

    data = np.arange(80.0).reshape(4, 20)
    reference = _compute_guidance_covariance(
        data,
        [IdentityBias()],
        name="biases",
    )
    bank = _compute_guidance_covariance(
        data,
        [IdentityBias(), ScaledIdentityBias()],
        name="biases",
    )
    assert bank.shape == (4, 4)
    np.testing.assert_allclose(np.trace(bank), 1.0)
    np.testing.assert_allclose(bank, reference)


def test_guidance_changes_only_asr_flagged_components():
    V = np.eye(4)
    forced = np.zeros(4, dtype=bool)
    D = np.array([100.0, 1.0, 1.0, 100.0])
    theta2 = np.array([5.0, 5.0, 5.0, 5.0])

    c_pre = np.diag([0.0, 0.0, 0.0, 1.0])
    w_pre = _guided_component_weights(
        D,
        V,
        theta2,
        forced_keep=forced,
        artifact_covariance=None,
        preserve_covariance=c_pre,
        strength=1.0,
    )
    assert w_pre[3] == pytest.approx(1.0)

    c_art = np.diag([1.0, 1.0, 0.0, 0.0])
    w_art = _guided_component_weights(
        D,
        V,
        theta2,
        forced_keep=forced,
        artifact_covariance=c_art,
        preserve_covariance=None,
        strength=1.0,
    )
    assert w_art[0] == pytest.approx(0.0)
    assert w_art[1] == pytest.approx(1.0)  # unflagged is never changed

    w_none = _guided_component_weights(
        D,
        V,
        theta2,
        forced_keep=forced,
        artifact_covariance=None,
        preserve_covariance=None,
        strength=1.0,
    )
    assert np.all((w_none >= 0.0) & (w_none <= 1.0))
    assert w_none[0] == pytest.approx(0.05)
    assert w_none[1] == pytest.approx(1.0)

    w_zero_strength = _guided_component_weights(
        D,
        V,
        theta2,
        forced_keep=forced,
        artifact_covariance=None,
        preserve_covariance=c_pre,
        strength=0.0,
    )
    np.testing.assert_allclose(w_zero_strength, w_none)

    w_isotropic = _guided_component_weights(
        D,
        V,
        theta2,
        forced_keep=forced,
        artifact_covariance=None,
        preserve_covariance=np.eye(4),
        strength=1.0,
    )
    np.testing.assert_allclose(w_isotropic, w_none)


def test_soft_weights_in_unit_interval_in_pipeline():
    gs = GuidedASR(
        sfreq=SFREQ,
        cutoff=20.0,
        reconstruction="soft",
        experimental=True,
        preserve_biases=[PeakFilterBias(10.0, SFREQ)],
        artifact_biases=[LineNoiseBias(50.0, SFREQ)],
        picks=None,
        verbose=False,
    )
    gs.fit_transform(_eeg())
    w = gs.get_diagnostics()["soft_weights"]
    assert w.ndim == 2 and w.shape[1] == 8
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0


# ---------------------------------------------------------------------------
# Behavioral: a preserve bias retains more target-band power than hard ASR
# ---------------------------------------------------------------------------


def test_preserve_bias_retains_more_target_band_than_hard():
    rng = np.random.default_rng(3)
    n = 6000
    t = np.arange(n) / SFREQ
    # Quiet background everywhere; a strong 10 Hz directional "neural event"
    # only in the second half (ASR calibrates on the quiet part and tends to
    # over-clean the event).
    X = 0.05 * rng.standard_normal((8, n))
    pattern = rng.standard_normal(8)
    pattern /= np.linalg.norm(pattern)
    event = np.zeros(n)
    event[n // 2 :] = 4.0 * np.sin(2 * np.pi * 10 * t[n // 2 :])
    X += np.outer(pattern, event)

    common = {
        "sfreq": SFREQ,
        "cutoff": 5.0,
        "calibration": "manual",
        "picks": None,
        "verbose": False,
    }
    hard = np.asarray(
        GuidedASR(reconstruction="hard", **common).fit(X[:, : n // 2]).transform(X)
    )
    soft = np.asarray(
        GuidedASR(
            reconstruction="soft",
            experimental=True,
            preserve_biases=[PeakFilterBias(10.0, SFREQ)],
            **common,
        )
        .fit(X[:, : n // 2])
        .transform(X)
    )

    def band_power(arr):
        seg = arr[:, n // 2 :]
        fft = np.fft.rfft(seg, axis=1)
        freqs = np.fft.rfftfreq(seg.shape[1], 1.0 / SFREQ)
        band = (freqs >= 8) & (freqs <= 12)
        return float(np.sum(np.abs(fft[:, band]) ** 2))

    # The preserve bias should retain at least as much 10 Hz power as hard ASR.
    assert band_power(soft) >= band_power(hard)


# ---------------------------------------------------------------------------
# I/O: ndarray, Raw, Epochs
# ---------------------------------------------------------------------------


def test_ndarray_io_shape_preserved():
    X = _eeg()
    out = GuidedASR(
        sfreq=SFREQ,
        cutoff=20.0,
        reconstruction="soft",
        experimental=True,
        preserve_biases=[BandpassBias((8.0, 12.0), SFREQ)],
        picks=None,
        verbose=False,
    ).fit_transform(X)
    assert np.asarray(out).shape == X.shape


def test_raw_and_epochs_io():
    mne = pytest.importorskip("mne")
    ch = [f"EEG{i:02d}" for i in range(8)]
    info = mne.create_info(ch, SFREQ, "eeg")
    raw = mne.io.RawArray(_eeg() * 1e-6, info, verbose=False)
    gs = GuidedASR(
        sfreq=SFREQ,
        cutoff=20.0,
        reconstruction="soft",
        experimental=True,
        preserve_biases=[PeakFilterBias(10.0, SFREQ)],
        verbose=False,
    )
    raw_out = gs.fit_transform(raw)
    assert raw_out.get_data().shape == raw.get_data().shape

    X = _eeg(n_times=6000)
    epo = mne.EpochsArray(
        X.reshape(8, 3, 2000).transpose(1, 0, 2) * 1e-6, info, verbose=False
    )
    epo_out = gs.fit(epo).transform(epo)
    assert epo_out.get_data().shape == epo.get_data().shape
    assert "soft_weights" in gs.get_diagnostics()


def test_evoked_and_non_data_channels_follow_current_asr_workflow():
    mne = pytest.importorskip("mne")
    eeg = _eeg() * 1e-6
    eog = np.sin(2 * np.pi * np.arange(eeg.shape[1]) / SFREQ)[None, :]
    data = np.vstack([eeg, eog])
    names = [f"EEG{i:02d}" for i in range(8)] + ["EOG"]
    info = mne.create_info(names, SFREQ, ["eeg"] * 8 + ["eog"])
    raw = mne.io.RawArray(data, info, verbose=False)
    evoked = mne.EvokedArray(data, info, verbose=False)
    guided = GuidedASR(
        cutoff=20.0,
        reconstruction="soft",
        experimental=True,
        preserve_biases=[PeakFilterBias(10.0, SFREQ)],
        verbose=False,
    )

    with pytest.warns(UserWarning, match="unpublished, unvalidated"):
        raw_out = guided.fit_transform(raw)
    evoked_out = guided.transform(evoked)

    assert isinstance(evoked_out, mne.Evoked)
    np.testing.assert_allclose(raw_out.get_data(picks=["EOG"]), eog)
    np.testing.assert_allclose(evoked_out.get_data(picks=["EOG"]), eog)


def test_current_diagnostics_rejection_and_sklearn_workflows():
    X = _eeg()
    guided = GuidedASR(
        sfreq=SFREQ,
        cutoff=10.0,
        reconstruction="soft",
        experimental=True,
        preserve_biases=[PeakFilterBias(10.0, SFREQ)],
        window_criterion=0.25,
        window_criterion_tolerances=(-np.inf, 7.0),
        picks=None,
        verbose=False,
    )

    cloned = clone(guided)
    assert cloned.get_params()["window_criterion"] == 0.25
    with pytest.warns(UserWarning, match="unpublished, unvalidated"):
        cleaned, diagnostics = guided.fit_transform(X, return_diagnostics=True)

    assert np.asarray(cleaned).shape == X.shape
    assert "rejection_sample_mask" in diagnostics
    np.testing.assert_array_equal(
        guided.get_rejection_mask(), diagnostics["rejection_sample_mask"]
    )
    assert guided.history_["estimator"] == "GuidedASR"


def test_process_guided_asr_validates_public_parameters():
    X = _eeg()
    fitted = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        method="riemannian_windowed",
        picks=None,
        verbose=False,
    ).fit(X)

    with pytest.raises(ValueError, match="artifact_cov must have shape"):
        process_guided_asr(
            X,
            SFREQ,
            fitted.state_,
            artifact_cov=np.eye(X.shape[0] - 1),
        )
    with pytest.raises(ValueError, match="guidance_strength"):
        process_guided_asr(X, SFREQ, fitted.state_, guidance_strength=1.1)


# ---------------------------------------------------------------------------
# Diagnostics / annotations / guards
# ---------------------------------------------------------------------------


def test_diagnostics_and_annotations():
    mne = pytest.importorskip("mne")
    gs = GuidedASR(
        sfreq=SFREQ,
        cutoff=10.0,
        reconstruction="soft",
        experimental=True,
        preserve_biases=[PeakFilterBias(10.0, SFREQ)],
        picks=None,
        verbose=False,
    )
    gs.fit_transform(_eeg(bursts=8))
    diag = gs.get_diagnostics()
    assert "soft_weights" in diag and "mean_soft_weight" in diag
    assert diag["covariance_geometry"] == "guided"
    ann = gs.to_annotations("repair")
    assert isinstance(ann, mne.Annotations)


def test_soft_requires_experimental_optin():
    with pytest.raises(ValueError, match="experimental"):
        GuidedASR(sfreq=SFREQ, reconstruction="soft", picks=None, verbose=False).fit(
            _eeg()
        )


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"reconstruction": "bogus"}, "reconstruction must be"),
        (
            {"guidance_strength": -0.1, "experimental": True},
            "guidance_strength must be",
        ),
    ],
)
def test_param_guards(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        GuidedASR(sfreq=SFREQ, picks=None, verbose=False, **kwargs).fit(_eeg())
