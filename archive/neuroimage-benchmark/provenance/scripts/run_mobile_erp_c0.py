#!/usr/bin/env python
"""Mobile-EEG auditory evoked-response C0 baseline (ds003620; reframed mobile-ERP arm).

ds003620's released markers do not distinguish oddball targets from standards (all stimuli are
``S 1``), so the original P300 *discrimination* is not computable. We instead quantify the
auditory *evoked response* itself under mobility: every stimulus is epoched, and the vertex
N1-P2 peak-to-peak amplitude indexes the evoked response. The C0 (uncorrected) endpoints are the
across-subjects effect size (one-sample Hedges g of the peak-to-peak amplitude) and the P2 peak
latency. Emits mobile_erp_ds003620.g.c0 and mobile_erp_ds003620.latency_ms.c0.

Run inside a compute allocation (epoching 44 subjects):
    srun ... python scripts/run_mobile_erp_c0.py --root /project/.../ds003620 --out /tmp/mobile_erp.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st

import numpy as np


def _peak(root, subject):
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
    epo = mne.Epochs(raw, events, tmin=-0.1, tmax=0.5, baseline=(-0.1, 0.0),
                     preload=True, reject=None, verbose=False)
    ev = epo.average()
    # vertex ROI (auditory N1-P2 maximum); fall back to GFP if no central electrode
    roi = [c for c in ev.ch_names if c.upper() in ("CZ", "FCZ", "CPZ")]
    t = ev.times
    sig = ev.copy().pick(roi).data.mean(0) if roi else ev.data.std(0)
    n1 = (t >= 0.07) & (t <= 0.16)
    p2 = (t >= 0.15) & (t <= 0.30)
    n1_amp = float(np.min(sig[n1])) if roi else float(np.max(sig[n1]))
    p2_idx = np.where(p2)[0][np.argmax(sig[p2])]
    p2_amp = float(sig[p2_idx])
    ptp = abs(p2_amp - n1_amp)               # N1-P2 peak-to-peak (robust auditory ERP index)
    return ptp, float(t[p2_idx] * 1000.0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    subs = sorted(pathlib.Path(s).name for s in glob.glob(f"{a.root}/sub-*") if pathlib.Path(s).is_dir())
    amps, lats = [], []
    for sub in subs:
        try:
            r = _peak(a.root, sub)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sub}: skip ({type(exc).__name__})")
            continue
        if r is None:
            continue
        amps.append(r[0]); lats.append(r[1])
        print(f"  {sub}: ptp={r[0]:.2e} p2_lat={r[1]:.0f}ms")
    n = len(amps)
    g = (st.mean(amps) / st.pstdev(amps)) * (1 - 3 / (4 * n - 9)) if n > 2 and st.pstdev(amps) > 0 else 0.0
    out = {"mobile_erp_ds003620.g.c0": round(g, 2),
           "mobile_erp_ds003620.latency_ms.c0": int(round(st.mean(lats))) if lats else 0,
           "_n": n}
    json.dump(out, open(a.out, "w"), indent=1)
    print("RESULT:", json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
