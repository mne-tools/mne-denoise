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


def test_spectrum_interp_removes_line_preserves_band():
    import mne

    rng = np.random.default_rng(2)
    sf, nch, T = 500.0, 6, 5000
    t = np.arange(T) / sf
    data = np.stack([np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)
                     + 0.1 * rng.standard_normal(T) for _ in range(nch)]) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info([f"E{i}" for i in range(nch)], sf, "eeg"), verbose=False)
    comp = C.get("spectrum_interp")
    ctx = {"sfreq": sf, "line_freq": 60.0}
    res = comp.transform(raw, comp.fit(raw, ctx), ctx)
    assert res.status == "success"
    from scipy.signal import welch

    def bp(x, lo, hi):
        f, p = welch(x, sf, nperseg=1024, axis=-1)
        return float(np.mean(p[..., (f >= lo) & (f <= hi)]))
    cl = res.cleaned.get_data()
    assert bp(cl, 59, 61) < 0.1 * bp(data, 59, 61)   # line removed
    assert bp(cl, 9, 11) > 0.8 * bp(data, 9, 11)     # alpha band preserved


def test_eemd_cca_removes_muscle_preserves_alpha():
    pytest.importorskip("PyEMD")
    from scipy.signal import butter, filtfilt, welch

    rng = np.random.default_rng(3)
    sf, nch, T = 200.0, 2, 1200
    t = np.arange(T) / sf
    alpha = np.sin(2 * np.pi * 10 * t)
    b, a = butter(4, [35 / 100, 90 / 100], btype="band")
    muscle = filtfilt(b, a, rng.standard_normal(T))
    A = rng.standard_normal((nch, 2))
    X = A[:, 0:1] * alpha + A[:, 1:2] * 2.0 * muscle
    comp = C.get("eemd_cca", trials=6, max_imf=6)
    res = comp.transform(X, comp.fit(X), {"sfreq": sf})
    assert res.status == "success"
    assert np.asarray(res.cleaned).shape == X.shape

    def bp(z, lo, hi):
        f, p = welch(z, sf, nperseg=512, axis=-1)
        return float(np.mean(p[..., (f >= lo) & (f <= hi)]))
    assert bp(np.asarray(res.cleaned), 35, 90) < 0.5 * bp(X, 35, 90)   # broadband muscle reduced


def test_mwf_removes_muscle_subspace():
    # MWF (Somers 2018): clean and artifact segments -> Wiener filter removes the artifact subspace.
    from scipy.signal import butter, filtfilt, welch

    rng = np.random.default_rng(5)
    sf, nch, T = 250.0, 8, 6000
    t = np.arange(T) / sf
    alpha = np.sin(2 * np.pi * 10 * t)
    b, a = butter(4, [35 / 125, 90 / 125], btype="band")
    muscle = filtfilt(b, a, rng.standard_normal(T)); muscle[: T // 2] = 0.0   # 2nd half only
    A = rng.standard_normal((nch, 2))
    X = A[:, 0:1] * alpha + A[:, 1:2] * 3.0 * muscle + 0.1 * rng.standard_normal((nch, T))
    comp = C.get("mwf", quantile=0.5)
    res = comp.transform(X, comp.fit(X), {"sfreq": sf})
    assert res.status == "success"
    assert np.asarray(res.cleaned).shape == X.shape

    def bp(x, lo, hi):
        f, p = welch(x, sf, nperseg=1024, axis=-1)
        return float(np.mean(p[..., (f >= lo) & (f <= hi)]))
    cl = np.asarray(res.cleaned)
    assert bp(cl, 35, 90) < 0.3 * bp(X, 35, 90)       # muscle subspace removed
    assert bp(cl, 8, 12) > 0.7 * bp(X, 8, 12)         # alpha preserved


def test_adjust_flags_and_removes_blink_ic():
    # ADJUST (Mognon 2011): flag the frontal high-kurtosis blink IC and remove it.
    import mne

    rng = np.random.default_rng(7)
    ch = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3", "Cz", "C4",
          "P3", "Pz", "P4", "O1", "O2", "T7", "T8", "P7", "P8", "Oz"]
    sf, nt, nep = 200.0, 200, 60
    info = mne.create_info(ch, sf, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    t = np.arange(nt) / sf
    blink_topo = np.array([3, 3, 2, 2, 1.5, 2, 2, .5, .3, .5, .2, .1, .2, .05, .05, .4, .4, .15, .15, .05])
    alpha_topo = rng.standard_normal(len(ch))
    data = np.zeros((nep, len(ch), nt))
    for e in range(nep):
        alpha = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
        blink = np.zeros(nt)
        if e % 2 == 0:
            c = int(rng.integers(40, 160)); blink[c - 10:c + 10] += np.hanning(20) * 8
        data[e] = ((alpha_topo[:, None] * alpha + blink_topo[:, None] * blink) * 1e-6
                   + rng.standard_normal((len(ch), nt)) * 0.3e-6)
    epo = mne.EpochsArray(data, info, verbose=False)
    comp = C.get("adjust", n_components=15)
    res = comp.transform(epo, comp.fit(epo, {"sfreq": sf}), {"sfreq": sf})
    assert res.status == "success"
    assert res.diagnostics["n_excluded"] >= 1                        # flagged the blink IC

    def fp(x):
        return float((x[:, [0, 1], :] ** 2).mean())                  # Fp1/Fp2 power
    assert fp(res.cleaned.get_data()) < 0.5 * fp(data)               # frontal blink removed


def test_mara_flags_and_removes_blink_ic(monkeypatch):
    # MARA (Winkler 2011): flag an ocular IC via the six-feature linear discriminant and remove it.
    # MARA keys on the current-density norm (leadfield-based, needs 10-20 names) and the spectral
    # shape, so the artifact IC is built as a realistic blink: a focal frontal-outlier topography
    # with a slow monophasic time course (high skewness, steep 1/f, ~no alpha-band power).
    pytest.importorskip("scipy")
    monkeypatch.setenv("MNE_DENOISE_ENABLE_GPL_MARA", "1")
    import mne

    rng = np.random.default_rng(7)
    ch = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3", "Cz", "C4",
          "P3", "Pz", "P4", "O1", "O2", "T7", "T8", "P7", "P8", "Oz"]
    sf, nt, nep = 200.0, 400, 80
    info = mne.create_info(ch, sf, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    t = np.arange(nt) / sf
    # focal frontal blink topography (near-zero away from frontal sites -> large pattern range)
    blink_topo = np.array([10, 10, 4, 4, 2, 4, 4, .2, .1, .2, .05, .02, .05, .01, .01, .3, .3, .05, .05, .01])
    alpha_topo = np.abs(rng.standard_normal(len(ch))); alpha_topo[:7] *= 0.2   # alpha posterior-weighted
    data = np.zeros((nep, len(ch), nt))
    for e in range(nep):
        alpha = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
        blink = np.zeros(nt)
        if rng.random() < 0.6:
            c = int(rng.integers(60, nt - 60)); w = 40
            blink[c - w:c + w] += np.hanning(2 * w) ** 0.5 * 12               # broad, sharp-onset deflection
        data[e] = ((alpha_topo[:, None] * alpha + blink_topo[:, None] * blink) * 1e-6
                   + rng.standard_normal((len(ch), nt)) * 0.2e-6)
    epo = mne.EpochsArray(data, info, verbose=False)
    comp = C.get("mara", n_components=15)
    res = comp.transform(epo, comp.fit(epo, {"sfreq": sf}), {"sfreq": sf})
    assert res.status == "success"
    assert res.diagnostics["n_excluded"] >= 1                        # flagged at least one artifact IC

    def fp(x):
        return float((x[:, [0, 1], :] ** 2).mean())                  # Fp1/Fp2 power
    assert fp(res.cleaned.get_data()) < 0.5 * fp(data)               # frontal blink removed


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
