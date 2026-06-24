#!/usr/bin/env python
"""Mobile-EEG evoked DSS-enhancement endpoint (ds003620; mobile-ERP arm, held-out).

Completes the mobile-ERP arm beyond the C0 baseline (run_mobile_erp_c0.py). The auditory
evoked response is a SINGLE condition (the released markers do not separate oddball targets),
so we score the enhancement of the vertex N1-P2 evoked SNR rather than an A-B contrast. DSS
(AverageBias) spatial filters are fit on the ODD trials and applied to the held-out EVEN
trials (anti-circularity); the endpoint is the per-subject paired change in evoked SNR
(N1-P2 peak-to-peak / pre-stimulus RMS). Preservation is the odd-vs-even morphology
correlation. An ONSET-SHIFTED null (events displaced 0.8 s into the inter-stimulus interval,
carrying no stimulus-locked response) gives the baseline against which enhancement is tested.

    srun ... python scripts/run_mobile_erp_dss.py --root /project/.../ds003620 --out /tmp/mobile_dss.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import mne_denoise.benchmarks.adapters  # noqa: F401,E402  (registers the dss comparator)
from mne_denoise.benchmarks import comparators  # noqa: E402


def _epochs(root, subject, shift_s=0.0):
    import mne

    vhdr = glob.glob(f"{root}/{subject}/eeg/{subject}_task-oddball_eeg.vhdr")
    if not vhdr:
        return None
    raw = mne.io.read_raw_brainvision(vhdr[0], preload=True, verbose=False)
    raw.pick("eeg").filter(1.0, 20.0, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    if len(events) < 50:
        return None
    if shift_s:
        sf = float(raw.info["sfreq"])
        events = events.copy()
        events[:, 0] += int(shift_s * sf)
        events = events[events[:, 0] < raw.n_times - int(0.5 * sf)]
    epo = mne.Epochs(raw, events, tmin=-0.1, tmax=0.5, baseline=(-0.1, 0.0),
                     preload=True, reject=None, verbose=False)
    return epo


def _roi_idx(epo):
    return [i for i, c in enumerate(epo.ch_names) if c.upper() in ("CZ", "FCZ", "CPZ")]


def _wave(avg_data, roi):
    return avg_data[roi].mean(0) if roi else np.sqrt((avg_data ** 2).mean(0))   # ROI mean or GFP


def _snr(avg_data, times, roi):
    sig = _wave(avg_data, roi)
    n1 = (times >= 0.07) & (times <= 0.16)
    p2 = (times >= 0.15) & (times <= 0.30)
    base = (times >= -0.1) & (times <= 0.0)
    ptp = abs(float(sig[p2].max()) - float(sig[n1].min())) if roi else float(sig[p2].max() - sig[n1].min())
    noise = float(sig[base].std()) or 1e-30
    return ptp / noise


def _subject(root, subject):
    epo = _epochs(root, subject)
    if epo is None or len(epo) < 50:
        return None
    roi = _roi_idx(epo)
    t = epo.times
    ctx = {"sfreq": float(epo.info["sfreq"])}
    odd, even = epo[::2], epo[1::2]
    try:
        cmp = comparators.get("dss")
        st = cmp.fit(odd, ctx)
        ec = cmp.transform(even, st, ctx)
        oc = cmp.transform(odd, st, ctx)
    except Exception as exc:  # noqa: BLE001
        print(f"  {subject}: dss fail ({type(exc).__name__})")
        return None
    if ec.status != "success" or oc.status != "success":
        return None
    snr_c0 = _snr(even.average().data, t, roi)
    snr_dss = _snr(ec.cleaned.average().data, t, roi)
    post = (t >= 0.0) & (t <= 0.4)
    morph = float(np.corrcoef(_wave(oc.cleaned.average().data, roi)[post],
                              _wave(ec.cleaned.average().data, roi)[post])[0, 1])
    # onset-shifted null: same DSS pipeline at a non-stimulus latency
    null_snr = None
    en = _epochs(root, subject, shift_s=0.8)
    if en is not None and len(en) >= 50:
        on, ev = en[::2], en[1::2]
        try:
            sn = cmp.fit(on, ctx)
            rn = cmp.transform(ev, sn, ctx)
            if rn.status == "success":
                null_snr = _snr(rn.cleaned.average().data, en.times, roi)
        except Exception:  # noqa: BLE001
            pass
    return {"snr_c0": snr_c0, "snr_dss": snr_dss, "morph": morph, "null_snr": null_snr}


def _hedges_dz(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return 0.0
    return float(d.mean() / d.std(ddof=1) * (1 - 3 / (4 * n - 9)))


def _boot_ci(fn, *arrs, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(a, float) for a in arrs]
    m = len(arrs[0])
    vals = [fn(*[a[rng.integers(0, m, m)] for a in arrs]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    subs = sorted(pathlib.Path(s).name for s in glob.glob(f"{a.root}/sub-*") if pathlib.Path(s).is_dir())
    c0, dss, morph, null = [], [], [], []
    for sub in subs:
        try:
            r = _subject(a.root, sub)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sub}: skip ({type(exc).__name__})")
            continue
        if r is None:
            continue
        c0.append(r["snr_c0"]); dss.append(r["snr_dss"]); morph.append(r["morph"])
        if r["null_snr"] is not None:
            null.append(r["null_snr"])
        print(f"  {sub}: snr_c0={r['snr_c0']:.2f} snr_dss={r['snr_dss']:.2f} "
              f"d={r['snr_dss']-r['snr_c0']:+.2f} morph={r['morph']:.2f}")
    c0, dss, morph = np.array(c0), np.array(dss), np.array(morph)
    delta = dss - c0
    # paired enhancement effect (Hedges dz of SNR change) + bootstrap CI
    dz = _hedges_dz(delta)
    dz_lo, dz_hi = _boot_ci(lambda d: _hedges_dz(d), delta)
    mr = float(np.median(morph))
    mr_lo, mr_hi = _boot_ci(lambda x: float(np.median(x)), morph)
    # onset-shifted null: one-sample g of the null SNR's deviation, and permutation p
    null = np.array(null) if null else np.array([])
    null_dz = _hedges_dz(null - np.median(c0)) if len(null) > 2 else 0.0
    nlo, nhi = _boot_ci(lambda x: _hedges_dz(x - np.median(c0)), null) if len(null) > 2 else (0.0, 0.0)
    # permutation: is the real paired enhancement larger than sign-flipped (null) enhancement?
    rng = np.random.default_rng(7)
    obs = float(delta.mean())
    perm = [float((delta * rng.choice([-1, 1], len(delta))).mean()) for _ in range(10000)]
    pval = (1 + np.sum(np.abs(perm) >= abs(obs))) / (1 + len(perm))
    out = {
        "mobile_erp_ds003620.dss_delta.dz": round(dz, 2),
        "mobile_erp_ds003620.dss_delta.ci_lo": round(dz_lo, 2),
        "mobile_erp_ds003620.dss_delta.ci_hi": round(dz_hi, 2),
        "mobile_erp_ds003620.morph_r.median": round(mr, 2),
        "mobile_erp_ds003620.morph_r.ci_lo": round(mr_lo, 2),
        "mobile_erp_ds003620.morph_r.ci_hi": round(mr_hi, 2),
        "mobile_erp_ds003620.null_g.value": round(null_dz, 2),
        "mobile_erp_ds003620.null_g.ci_lo": round(nlo, 2),
        "mobile_erp_ds003620.null_g.ci_hi": round(nhi, 2),
        "mobile_erp_ds003620.perm_p.value": round(float(pval), 4),
        "_n": int(len(c0)), "_n_null": int(len(null)),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print("RESULT:", json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
