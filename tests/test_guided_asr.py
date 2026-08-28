"""Tests for the experimental DSS-guided soft ASR (``GuidedASR``).

Covers the strict-generalization guarantee (``reconstruction="hard"`` + no bias
operators reproduces ``ASR(method="riemannian_windowed")`` exactly), the
discriminative soft-weight math, soft reconstruction behavior, MNE/ndarray
I/O, and the experimental opt-in / validation guards.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mne_denoise.asr import ASR, GuidedASR, process_guided_asr
from mne_denoise.asr._guidance import (
    _compute_guidance_covariance,
    _guided_component_weights,
)
from mne_denoise.dss.denoisers import LineNoiseBias, PeakFilterBias

SFREQ = 250.0


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


def test_guided_asr_reports_distinct_calibration_and_guidance_results(caplog):
    """GuidedASR has one shared calibration and one distinct configuration report."""
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        GuidedASR(
            sfreq=SFREQ,
            cutoff=20.0,
            reconstruction="hard",
            picks=None,
        ).fit(_eeg(n_times=4000), verbose=True)
    calibration = [
        record for record in caplog.records if "GuidedASR calibrated:" in record.message
    ]
    guidance = [
        record
        for record in caplog.records
        if record.message.startswith("GuidedASR: reconstruction=")
    ]
    assert len(calibration) == 1
    assert len(guidance) == 1
    assert "reconstruction=hard" in guidance[0].message
    assert "guidance strength=" in guidance[0].message


# ---------------------------------------------------------------------------
# Discriminative soft-weight math (the novel core)
# ---------------------------------------------------------------------------


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


def test_current_diagnostics_and_rejection_workflows():
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

    with pytest.warns(UserWarning, match="unpublished, unvalidated"):
        cleaned, diagnostics = guided.fit_transform(X, return_diagnostics=True)

    assert np.asarray(cleaned).shape == X.shape
    assert "rejection_sample_mask" in diagnostics
    np.testing.assert_array_equal(
        guided.get_rejection_mask(), diagnostics["rejection_sample_mask"]
    )
    assert guided.history_["estimator"] == "GuidedASR"
    guided_diagnostics = guided.get_diagnostics()
    assert guided_diagnostics["covariance_geometry"] == "guided"
    assert guided_diagnostics["soft_weights"].shape[1] == X.shape[0]
    assert 0.0 <= guided_diagnostics["mean_soft_weight"] <= 1.0


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


def test_process_guided_asr_progress_reports_ordered_windows():
    """Guided soft reconstruction emits one ordered event per update."""
    X = _eeg()
    fitted = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        calibration="manual",
        filter_kind="none",
        picks=None,
        verbose=False,
    ).fit(X)

    events = []
    _cleaned, diagnostics = process_guided_asr(
        X,
        SFREQ,
        fitted.state_,
        reconstruction="soft",
        max_dims=0.5,
        lookahead=0.0,
        callback=events.append,
    )
    assert len(events) == diagnostics["n_windows"]
    assert all(event.method == "guided_asr" for event in events)
    assert all(event.stage == "window" for event in events)
    assert [event.current for event in events] == list(range(1, len(events) + 1))
    assert [event.total for event in events] == [len(events)] * len(events)
    assert all(event.component is None for event in events)
    np.testing.assert_allclose(
        [event.metric for event in events],
        diagnostics["n_components_reconstructed"],
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Diagnostics / annotations / guards
# ---------------------------------------------------------------------------


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
