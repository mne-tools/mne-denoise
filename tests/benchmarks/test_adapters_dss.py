"""DSS operator adapters: registration, polarity, and convergence accounting.

Polarity is the highest-risk detail in the whole operator-completeness campaign.
``IterativeDSS.transform`` returns *sources*, while ``DSS.transform`` with a
sensor return type returns a reconstruction that *keeps* the selected subspace.
If an adapter picks the wrong direction nothing raises -- every attenuation and
preservation metric simply inverts. These tests therefore assert the *sign* of a
signed synthetic injection in both directions rather than only checking shapes.
"""
from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.benchmarks import adapters_dss as A
from mne_denoise.benchmarks import comparators as C


def _mixture(n_channels=8, n_times=4000, sfreq=250.0, seed=0):
    """Brain alpha plus a spatially structured 50 Hz interferer."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / sfreq
    brain = np.sin(2 * np.pi * 10 * t)
    line = np.sin(2 * np.pi * 50 * t)
    brain_pattern = rng.standard_normal((n_channels, 1))
    line_pattern = np.abs(rng.standard_normal((n_channels, 1))) + 0.5
    data = brain_pattern * brain + 3.0 * line_pattern * line
    data += 0.05 * rng.standard_normal((n_channels, n_times))
    return data, sfreq


def _band_power(data, sfreq, lo, hi):
    from scipy.signal import welch

    f, p = welch(data, sfreq, nperseg=min(1024, data.shape[-1]))
    m = (f >= lo) & (f <= hi)
    return float(np.mean(p[:, m]))


def test_all_new_ids_are_registered():
    for cid in ("dss_cycle_average_subtract", "dss_reference_bias_subtract",
                "dss_line_bias_subtract", "dss_comb_bias_keep",
                "dss_bandpass_bias_keep", "iterative_dss"):
        assert cid in C.available()


def test_registering_twice_is_refused():
    """Guards against a double import silently shadowing a registered id."""
    with pytest.raises(ValueError, match="already registered"):
        C.register("iterative_dss", lambda **p: None)


def test_contrast_grid_matches_the_generic_bss_factorial():
    """The screen must use the same eleven ids as the completed synthetic factorial."""
    assert sorted(A.CONTRASTS) == sorted([
        "tanh", "robust_tanh", "gauss", "skew", "smooth_tanh", "kurtosis",
        "wiener_mask", "variance_mask", "dct", "quasi_periodic", "spectrogram",
    ])


# -- polarity ---------------------------------------------------------------

def test_line_bias_subtract_removes_the_interferer():
    data, sfreq = _mixture()
    comp = C.get("dss_line_bias_subtract", n_components=2)
    ctx = {"sfreq": sfreq, "line_freq": 50.0}
    res = comp.transform(data, comp.fit(data, ctx), ctx)
    assert res.status == "success"
    before = _band_power(data, sfreq, 48, 52)
    after = _band_power(np.asarray(res.cleaned), sfreq, 48, 52)
    assert after < 0.5 * before, "subtract polarity must REMOVE the biased subspace"
    assert res.parameters["polarity"] == "subtract"


def test_bandpass_bias_keep_retains_the_target_band():
    """Same estimator, opposite polarity: the target band must survive, not vanish."""
    data, sfreq = _mixture()
    comp = C.get("dss_bandpass_bias_keep", n_components=2)
    ctx = {"sfreq": sfreq, "band": (8.0, 12.0)}
    res = comp.transform(data, comp.fit(data, ctx), ctx)
    assert res.status == "success"
    cleaned = np.asarray(res.cleaned)
    alpha = _band_power(cleaned, sfreq, 8, 12)
    line = _band_power(cleaned, sfreq, 48, 52)
    assert alpha > line, "keep polarity must RETAIN the biased subspace"
    assert res.parameters["polarity"] == "keep"


def test_polarity_directions_are_complementary():
    """keep + subtract must reconstruct the input, proving the two are one split."""
    data, sfreq = _mixture()
    ctx = {"sfreq": sfreq, "band": (8.0, 12.0)}
    kept = C.get("dss_bandpass_bias_keep", n_components=2)
    sub = A._LinearDSSOperator("tmp_bandpass_subtract", bias="bandpass",
                               polarity="subtract", n_components=2)
    a = np.asarray(kept.transform(data, kept.fit(data, ctx), ctx).cleaned)
    b = np.asarray(sub.transform(data, sub.fit(data, ctx), ctx).cleaned)
    np.testing.assert_allclose(a + b, data, atol=1e-8)


def test_rank_after_follows_polarity():
    data, sfreq = _mixture(n_channels=8)
    ctx = {"sfreq": sfreq, "band": (8.0, 12.0)}
    kept = C.get("dss_bandpass_bias_keep", n_components=3)
    res = kept.transform(data, kept.fit(data, ctx), ctx)
    assert res.rank_after == 3
    sub = A._LinearDSSOperator("tmp_rank", bias="bandpass", polarity="subtract", n_components=3)
    res = sub.transform(data, sub.fit(data, ctx), ctx)
    assert res.rank_after == 8 - 3


@pytest.mark.parametrize("polarity", ["keep", "subtract"])
def test_iterative_dss_polarity_is_recorded_and_shape_preserving(polarity):
    data, sfreq = _mixture()
    comp = C.get("iterative_dss", contrast="tanh", n_components=2, polarity=polarity,
                 random_state=7)
    ctx = {"sfreq": sfreq}
    res = comp.transform(data, comp.fit(data, ctx), ctx)
    assert res.status in ("success", "failed_convergence")
    if res.status == "success":
        assert np.asarray(res.cleaned).shape == data.shape
        assert res.parameters["polarity"] == polarity
        assert res.random_seed == 7


# -- ctx requirements and failure accounting --------------------------------

def test_missing_reference_is_skipped_not_failed():
    """A missing reference channel is a declared skip, never a silent success."""
    data, sfreq = _mixture()
    comp = C.get("dss_reference_bias_subtract", n_components=2)
    ctx = {"sfreq": sfreq}
    res = comp.transform(data, comp.fit(data, ctx), ctx)
    assert res.status == "skipped_missing_channels"


def test_missing_events_is_skipped_not_failed():
    data, sfreq = _mixture()
    comp = C.get("dss_cycle_average_subtract", n_components=2)
    ctx = {"sfreq": sfreq}
    res = comp.transform(data, comp.fit(data, ctx), ctx)
    assert res.status == "skipped_missing_channels"


def _cardiac_fixture(sfreq=250.0, n_channels=8, n_times=6000, seed=3):
    rng = np.random.default_rng(seed)
    events = np.arange(300, n_times - 300, 250)
    beat = np.exp(-0.5 * ((np.arange(-25, 50)) / 6.0) ** 2)
    pattern = np.abs(rng.standard_normal((n_channels, 1))) + 0.5
    data = 0.1 * rng.standard_normal((n_channels, n_times))
    for p in events:
        data[:, p - 25:p + 50] += 5.0 * pattern * beat
    return data, events, sfreq


def test_cycle_average_subtract_removes_a_periodic_injection():
    """Cardiac-shaped test: a stereotyped waveform at known events is removed."""
    data, events, sfreq = _cardiac_fixture()

    def qrs_rms(x):
        seg = np.stack([x[:, p - 25:p + 50] for p in events])
        return float(np.sqrt((seg.mean(0) ** 2).mean()))

    comp = C.get("dss_cycle_average_subtract", n_components=3)
    ctx = {"sfreq": sfreq, "event_samples": events, "event_window_s": (-0.1, 0.2)}
    res = comp.transform(data, comp.fit(data, ctx), ctx)
    assert res.status == "success"
    assert qrs_rms(np.asarray(res.cleaned)) < 0.5 * qrs_rms(data)


def test_cycle_average_window_must_be_in_seconds():
    """Regression pin for the defect that produced a degenerate v1 cardiac operator.

    ``CycleAverageBias`` scales ``window`` by ``sfreq``. The v1 runner passed a
    window already converted to samples together with ``sfreq``, converting it a
    second time: a -100/+200 ms cycle window became -25/+50 SECONDS. Supplying
    samples here must raise rather than silently produce a nonsense operator.
    """
    data, events, sfreq = _cardiac_fixture()
    comp = C.get("dss_cycle_average_subtract", n_components=3)
    ctx = {"sfreq": sfreq, "event_samples": events,
           "event_window_s": (-25, 50)}  # samples mistaken for seconds
    res = comp.transform(data, comp.fit(data, ctx), ctx)
    assert res.status == "failed_numerical"
    assert "SECONDS" in (res.error or "")


def test_an_annihilating_operator_is_not_reported_as_success():
    """The v1 cardiac operator zeroed every recording and scored a perfect attenuation.

    Attenuation alone cannot distinguish perfect cleaning from signal destruction;
    only the paired preservation endpoint can. Fail closed so such an output can
    never reach a results table as a headline number.
    """
    original = np.random.default_rng(0).standard_normal((8, 1000))
    assert A._degenerate_reason(np.zeros_like(original), original) is not None
    assert "identically zero" in A._degenerate_reason(np.zeros_like(original), original)
    assert A._degenerate_reason(np.full_like(original, np.nan), original) is not None
    assert A._degenerate_reason(original * 0.5, original) is None


def test_invalid_polarity_and_contrast_are_rejected():
    with pytest.raises(ValueError, match="polarity"):
        A._LinearDSSOperator("x", bias="bandpass", polarity="invert")
    with pytest.raises(ValueError, match="unknown bias"):
        A._LinearDSSOperator("x", bias="nope", polarity="keep")
    with pytest.raises(ValueError, match="unknown contrast"):
        C.get("iterative_dss", contrast="nope")


def test_iterative_dss_is_declared_stochastic():
    """Selection and reporting depend on this flag being honest."""
    comp = C.get("iterative_dss", contrast="tanh")
    assert comp.meta.deterministic is False
    linear = C.get("dss_bandpass_bias_keep")
    assert linear.meta.deterministic is True


def test_reference_bias_is_declared_reference_aware():
    """Figure 4 encodes information tier by marker shape; the flag drives it."""
    assert C.get("dss_reference_bias_subtract").meta.information_tier == "reference_aware"
    assert C.get("dss_line_bias_subtract").meta.information_tier == "blind"
