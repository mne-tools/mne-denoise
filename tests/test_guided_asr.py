"""Tests for the experimental DSS-guided soft ASR (``GuidedASR``).

Covers the strict-generalization guarantee (``reconstruction="hard"`` + no bias
operators reproduces ``ASR(method="riemannian_windowed")`` exactly), the
discriminative soft-weight math, soft reconstruction behavior, MNE/ndarray
I/O, and the experimental opt-in / validation guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.asr import ASR, GuidedASR
from mne_denoise.asr.guided import _normalize_cov, _soft_component_weights
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


# ---------------------------------------------------------------------------
# Discriminative soft-weight math (the novel core)
# ---------------------------------------------------------------------------


def test_soft_weights_rescue_and_suppress():
    V = np.eye(4)
    forced = np.zeros(4, dtype=bool)
    D = np.array([1.0, 1.0, 1.0, 100.0])  # comp 3 high-variance -> ASR rejects
    theta2 = np.array([5.0, 5.0, 5.0, 5.0])

    # preserve bias aligned to the high-variance comp -> rescued toward 1
    c_pre = _normalize_cov(np.diag([0.0, 0.0, 0.0, 1.0]))
    w_pre = _soft_component_weights(
        D,
        V,
        theta2,
        forced_keep=forced,
        artifact_cov=None,
        preserve_cov=c_pre,
        soft_weight="wiener",
        scale=1.0,
    )
    assert w_pre[3] > 0.9  # high-variance neural direction rescued

    # artifact bias aligned to a low-variance (ASR-kept) comp -> suppressed
    c_art = _normalize_cov(np.diag([1.0, 0.0, 0.0, 0.0]))
    w_art = _soft_component_weights(
        D,
        V,
        theta2,
        forced_keep=forced,
        artifact_cov=c_art,
        preserve_cov=None,
        soft_weight="wiener",
        scale=1.0,
    )
    assert w_art[0] < 0.1  # ASR-kept but artifact-like direction suppressed

    # no bias -> soft ASR weights, all in [0, 1]
    w_none = _soft_component_weights(
        D,
        V,
        theta2,
        forced_keep=forced,
        artifact_cov=None,
        preserve_cov=None,
        soft_weight="wiener",
        scale=1.0,
    )
    assert np.all((w_none >= 0.0) & (w_none <= 1.0))
    assert w_none[0] == pytest.approx(1.0)  # low-variance kept


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
        ({"soft_weight": "bogus", "experimental": True}, "soft_weight must be"),
    ],
)
def test_param_guards(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        GuidedASR(sfreq=SFREQ, picks=None, verbose=False, **kwargs).fit(_eeg())
