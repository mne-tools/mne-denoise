"""Comparator contract + leakage-barrier tests."""

import numpy as np
import pytest

from mne_denoise.benchmarks import comparators as C
import mne_denoise.benchmarks.adapters  # noqa: F401  (registers autocca/asr/wavelet/... adapters)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((8, 500))


def test_registry_has_trivial_comparators():
    assert "none" in C.available()
    assert "pca_reconstruct" in C.available()


def test_none_is_identity_roundtrip():
    x = _data()
    comp = C.get("none")
    state = comp.fit(x)
    res = comp.transform(x, state)
    assert res.status == "success"
    np.testing.assert_allclose(np.asarray(res.cleaned), x)
    assert res.runtime_seconds is not None and res.cpu_seconds is not None


def test_pca_fit_then_transform_reduces_rank():
    train, evl = _data(1), _data(2)
    comp = C.get("pca_reconstruct", n_components=3)
    state = comp.fit(train)
    res = comp.transform(evl, state)
    assert res.status == "success"
    assert res.rank_after == 3
    assert np.asarray(res.cleaned).shape == evl.shape


def test_autocca_separates_by_autocorrelation():
    # a slow (high-autocorrelation, neural-like) source + a fast (low-autocorrelation,
    # muscle-like) source mixed across channels; autoCCA should keep the slow, drop the fast.
    rng = np.random.default_rng(3)
    sf, n_ch, n_t = 250.0, 8, 4000
    t = np.arange(n_t) / sf
    slow = np.sin(2 * np.pi * 8 * t)
    fast = rng.standard_normal(n_t)
    A = rng.standard_normal((n_ch, 2))
    X = A[:, 0:1] * slow + A[:, 1:2] * fast
    comp = C.get("autocca", rho_threshold=0.9)
    res = comp.transform(X, comp.fit(X))
    assert res.status == "success"
    assert np.asarray(res.cleaned).shape == X.shape          # shape preserved
    assert 1 <= res.rank_after < n_ch                        # dropped the low-autocorrelation component(s)
    from scipy.signal import welch

    def _hf(x):
        f, p = welch(x, sf, nperseg=512, axis=-1)
        return float(np.mean(p[..., f > 40]))
    assert _hf(np.asarray(res.cleaned)) < 0.5 * _hf(X)       # high-frequency (muscle) power suppressed


def test_ssa_removes_slow_drift_preserves_oscillation():
    # large slow drift (ocular/instrumental-like, 0.3 Hz) + alpha (10 Hz); SSA drops the slow.
    rng = np.random.default_rng(5)
    sf, n_ch, n_t = 250.0, 4, 5000
    t = np.arange(n_t) / sf
    drift = 5.0 * np.sin(2 * np.pi * 0.3 * t)
    alpha = np.sin(2 * np.pi * 10 * t)
    X = np.stack([drift + alpha + 0.05 * rng.standard_normal(n_t) for _ in range(n_ch)])
    comp = C.get("ssa", drop_freq_max=3.0)
    res = comp.transform(X, comp.fit(X), {"sfreq": sf})
    assert res.status == "success"
    assert np.asarray(res.cleaned).shape == X.shape

    from scipy.signal import welch

    def _band(x, lo, hi):
        f, p = welch(x, sf, nperseg=1024, axis=-1)
        return float(np.mean(p[..., (f >= lo) & (f <= hi)]))
    cl = np.asarray(res.cleaned)
    assert _band(cl, 0, 1) < 0.3 * _band(X, 0, 1)            # slow drift suppressed
    assert _band(cl, 8, 12) > 0.7 * _band(X, 8, 12)          # alpha preserved


def test_wavelet_threshold_suppresses_broadband_noise():
    rng = np.random.default_rng(7)
    sf, n_ch, n_t = 250.0, 4, 4000
    t = np.arange(n_t) / sf
    alpha = np.sin(2 * np.pi * 10 * t)
    X = np.stack([alpha + 0.8 * rng.standard_normal(n_t) for _ in range(n_ch)])
    comp = C.get("wavelet_threshold")
    res = comp.transform(X, comp.fit(X))
    assert res.status == "success"
    assert np.asarray(res.cleaned).shape == X.shape
    from scipy.signal import welch

    def _hfp(x):
        f, p = welch(x, sf, nperseg=1024, axis=-1)
        return float(np.mean(p[..., f > 40]))
    assert _hfp(np.asarray(res.cleaned)) < _hfp(X)           # broadband HF noise reduced


def test_wica_removes_transient_artifact():
    rng = np.random.default_rng(8)
    sf, n_ch, n_t = 250.0, 6, 4000
    t = np.arange(n_t) / sf
    neural = np.sin(2 * np.pi * 10 * t)
    artifact = np.zeros(n_t)
    for s in range(250, n_t, 500):
        artifact[s : s + 25] += 10.0                          # sparse blink-like transients
    A = rng.standard_normal((n_ch, 2))
    X = A[:, 0:1] * neural + A[:, 1:2] * artifact + 0.1 * rng.standard_normal((n_ch, n_t))
    comp = C.get("wica", n_components=4)
    res = comp.transform(X, comp.fit(X))
    assert res.status == "success"
    assert np.asarray(res.cleaned).shape == X.shape
    assert np.abs(np.asarray(res.cleaned)).max() < 0.7 * np.abs(X).max()   # transient peak suppressed


def test_emd_drops_high_frequency_imfs():
    from scipy.signal import butter, filtfilt, welch

    rng = np.random.default_rng(9)
    sf, n_ch, n_t = 250.0, 3, 3000
    t = np.arange(n_t) / sf
    alpha = np.sin(2 * np.pi * 10 * t)
    b, a = butter(4, [35 / 125, 90 / 125], btype="band")
    muscle = filtfilt(b, a, rng.standard_normal(n_t))
    X = np.stack([alpha + 2.0 * muscle for _ in range(n_ch)])
    comp = C.get("emd", freq_cutoff=25.0)
    res = comp.transform(X, comp.fit(X), {"sfreq": sf})
    assert res.status == "success"
    assert np.asarray(res.cleaned).shape == X.shape

    def _band(x, lo, hi):
        f, p = welch(x, sf, nperseg=1024, axis=-1)
        return float(np.mean(p[..., (f >= lo) & (f <= hi)]))
    cl = np.asarray(res.cleaned)
    assert _band(cl, 35, 90) < 0.6 * _band(X, 35, 90)        # high-frequency muscle IMF removed
    assert _band(cl, 8, 12) > 0.5 * _band(X, 8, 12)          # alpha preserved


def test_new_comparators_handle_epochs_3d():
    # the new comparators must preserve the (n_trials, n_channels, n_times) shape on Epochs
    import mne

    rng = np.random.default_rng(11)
    sf, n_tr, n_ch, n_t = 250.0, 8, 6, 500
    data = rng.standard_normal((n_tr, n_ch, n_t)) * 1e-6
    info = mne.create_info([f"E{i}" for i in range(n_ch)], sf, "eeg")
    epo = mne.EpochsArray(data, info, verbose=False)
    ctx = {"sfreq": sf}
    for cid in ("autocca", "ssa", "wavelet_threshold", "wica"):
        comp = C.get(cid)
        res = comp.transform(epo, comp.fit(epo, ctx), ctx)
        assert res.status == "success", cid
        assert res.cleaned.get_data().shape == data.shape, cid     # 3-D epochs shape preserved


def test_xdawn_supervised_evoked_enhancement():
    # xDAWN is supervised: fit on the combined two-condition train, transform one condition.
    import mne

    rng = np.random.default_rng(12)
    sf, nch, ntr, nt = 256.0, 16, 40, 154
    t = np.arange(nt) / sf - 0.1
    erp = -np.exp(-((t - 0.155) ** 2) / (2 * 0.018 ** 2))
    patt = rng.standard_normal(nch); patt /= np.linalg.norm(patt)
    info = mne.create_info([f"E{i}" for i in range(nch)], sf, "eeg")

    def make(amp, cid):
        data = np.stack([patt[:, None] * (amp * erp)[None, :] * 4e-6
                         + rng.standard_normal((nch, nt)) * 3e-6 for _ in range(ntr)])
        ev = np.column_stack([np.arange(ntr) * nt, np.zeros(ntr, int), np.full(ntr, cid)])
        return mne.EpochsArray(data, info, ev, tmin=-0.1, verbose=False)

    a, b = make(1.0, 1), make(0.5, 2)
    tr = mne.concatenate_epochs([a[::2], b[::2]], verbose=False)
    comp = C.get("xdawn", n_components=4)
    res = comp.transform(a[1::2], comp.fit(tr, {"sfreq": sf}), {"sfreq": sf})
    assert res.status == "success"
    assert res.cleaned.get_data().shape == a[1::2].get_data().shape   # shape preserved
    assert res.rank_after == 4


def test_transform_without_fit_state_raises():
    comp = C.get("pca_reconstruct", n_components=2)
    with pytest.raises(RuntimeError):
        comp.transform(_data(), state=None)  # leakage barrier: must fit first


def test_fit_signature_excludes_eval():
    # the contract: fit(train, ctx) — there is no `eval` parameter to peek at.
    import inspect

    params = list(inspect.signature(C.Comparator.fit).parameters)
    assert "evaluation" not in params and "eval" not in params


def test_comparator_result_rejects_bad_status():
    with pytest.raises(ValueError):
        C.ComparatorResult(status="totally_made_up")


def test_comparator_result_summary_is_scalar_only():
    res = C.ComparatorResult(cleaned=np.zeros((2, 2)), rank_after=2)
    s = res.summary()
    assert "cleaned" not in s and s["rank_after"] == 2


def test_meta_reference_aware_sets_tier():
    m = C.ComparatorMeta("x", reference_aware=True)
    assert m.information_tier == "reference_aware"


def test_meta_bad_fit_scope_raises():
    with pytest.raises(ValueError):
        C.ComparatorMeta("x", fit_scope="whenever")
