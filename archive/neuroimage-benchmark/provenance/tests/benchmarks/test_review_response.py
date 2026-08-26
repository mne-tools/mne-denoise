"""Tests for the adversarial-review-response additions:
SSS/tSSS comparator (MF4), FBCCA/TRCA SSVEP decoders (MF3), and the muscle
alpha-ERD contrast (MF1). Script-internal helpers are imported lazily inside the
tests so a heavy top-level import never breaks collection."""

import pathlib
import sys

import numpy as np
import pytest

from mne_denoise.benchmarks import comparators as C
import mne_denoise.benchmarks.adapters  # noqa: F401  (registers sss/tsss/... adapters)

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# --- MF4: SSS / tSSS comparator -----------------------------------------
def test_sss_tsss_registered():
    avail = C.available()
    assert "sss" in avail and "tsss" in avail


def test_sss_degrades_gracefully_without_geometry():
    """A plain RawArray has no device geometry, so maxwell_filter must fail --
    the adapter must return a failed_numerical status, never raise."""
    mne = pytest.importorskip("mne")
    n_ch, sf = 20, 300.0
    info = mne.create_info([f"MEG{i:03d}" for i in range(n_ch)], sf, ["mag"] * n_ch)
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((n_ch, int(sf * 6))) * 1e-13, info, verbose=False)
    comp = C.get("sss")
    res = comp.transform(raw, comp.fit(raw))
    assert res.status in ("failed_numerical", "unavailable_dependency", "success")
    if res.status == "success":
        assert np.asarray(res.cleaned.get_data()).shape == raw.get_data().shape


# --- MF3: FBCCA + TRCA decoders -----------------------------------------
def test_fbcca_and_trca_recover_frequency():
    pytest.importorskip("mne")
    from run_ssvep_arm import _fbcca, _ref, _trca_decode

    sf, nch, ntar, nblk = 250.0, 8, 3, 4
    nsamp = int(sf * 4)
    freqs = np.array([9.0, 11.0, 13.0])
    t = np.arange(nsamp) / sf
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nch, ntar))
    seg = np.zeros((nch, nsamp, ntar, nblk))          # (ch, samp, target, block)
    for j in range(ntar):
        s = np.sin(2 * np.pi * freqs[j] * t)
        for b in range(nblk):
            seg[:, :, j, b] = A[:, j:j + 1] * s + rng.standard_normal((nch, nsamp)) * 0.3
    refs = [_ref(f, sf, nsamp, 3) for f in freqs]
    hits = sum(_fbcca(seg[:, :, j, 0], refs, sf) == j for j in range(ntar))
    assert hits >= 2                                   # clean high-SNR trials mostly correct
    acc = _trca_decode(seg, sf)
    assert 0.0 <= acc <= 1.0 and acc > 1.0 / ntar      # beats chance


# --- MF1: muscle alpha-ERD contrast -------------------------------------
def test_alpha_erd_sign_and_edges():
    pytest.importorskip("mne")
    from run_muscle_arm import _alpha_erd

    sf = 250.0
    n = int(sf * 60)
    t = np.arange(n) / sf
    rng = np.random.default_rng(0)
    emg = np.abs(rng.standard_normal((4, n)))
    emg[:, n // 2:] *= 5.0                              # 2nd half = high-motion (active)
    alpha = np.sin(2 * np.pi * 10 * t)
    alpha[n // 2:] *= 0.3                               # alpha desynchronizes when active
    eeg = rng.standard_normal((8, n)) * 0.1 + alpha[None, :]
    val = _alpha_erd(eeg, emg, sf)
    assert np.isfinite(val) and val < 0                # ERD is negative
    assert np.isnan(_alpha_erd(eeg[:, :10], emg[:, :10], sf))  # too short -> NaN, no crash
