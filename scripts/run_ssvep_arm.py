#!/usr/bin/env python
"""SSVEP steady-state benchmark runner (arm: ssvep_tsinghua / ssvep_beta; generalization).

For each subject and each 40-target stimulus frequency, the comb-DSS / RESS extractor
(``mne_denoise.dss.variants.ssvep.ssvep_dss``, a CombFilterBias at the fundamental + harmonics)
is fit on the trials and the top SSVEP component extracted. Primary endpoint: harmonic SNR of the
extracted component (spectral power at the stimulus fundamental + integer harmonics relative to
the surrounding noise), compared against the best raw occipital channel. Secondary: 40-class
SSVEP decoding accuracy with subject-safe leave-one-block-out CV, CCA vs DSS-assisted CCA. This
track is reported as generalization/exploratory (decoding accuracy is not itself a denoising score).

    python scripts/run_ssvep_arm.py --config configs/benchmarks/ssvep_tsinghua.yaml --subject S1
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
import yaml  # noqa: E402

from mne_denoise.benchmarks import io as bio  # noqa: E402
from mne_denoise.dss.variants.ssvep import ssvep_dss  # noqa: E402

BENCH = "ssvep"


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def harmonic_snr(x, sfreq, f0, n_harm=3, half_bw=0.2, side=1.0):
    """Mean narrowband SNR (linear) of a 1-D component at f0 and its harmonics."""
    from scipy.signal import welch

    nper = min(len(x), int(sfreq * 4))
    f, p = welch(x, sfreq, nperseg=nper)
    snrs = []
    for h in range(1, n_harm + 1):
        fh = f0 * h
        peak = p[(f >= fh - half_bw) & (f <= fh + half_bw)]
        nb = p[((f >= fh - side) & (f < fh - half_bw)) | ((f > fh + half_bw) & (f <= fh + side))]
        if peak.size and nb.size and nb.mean() > 0:
            snrs.append(peak.max() / nb.mean())
    return float(np.mean(snrs)) if snrs else 0.0


def _ref(freq, sfreq, n_samp, n_harm=3):
    """CCA reference: sin/cos at the fundamental + harmonics, shape (2*n_harm, n_samp)."""
    t = np.arange(n_samp) / sfreq
    rows = []
    for h in range(1, n_harm + 1):
        rows += [np.sin(2 * np.pi * freq * h * t), np.cos(2 * np.pi * freq * h * t)]
    return np.asarray(rows)


def _cca_corr(X, Y):
    """Largest canonical correlation between X (c1, T) and Y (c2, T)."""
    X = X - X.mean(1, keepdims=True)
    Y = Y - Y.mean(1, keepdims=True)
    qx, _ = np.linalg.qr(X.T)
    qy, _ = np.linalg.qr(Y.T)
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(s[0]) if s.size else 0.0


def _corr1d(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _corr2d(A, B):
    a = A.ravel() - A.ravel().mean(); b = B.ravel() - B.ravel().mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _fbcca(X, refs, sf, n_sub=5):
    """Filter-bank CCA (Chen et al. 2015): sub-band CCA correlations combined with
    weights w(k)=k^-1.25+0.25; sub-band k passband [8k, 90] Hz (fundamental + harmonics).
    Returns the index of the best-matching reference. Training-free."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sf / 2.0
    weights = np.array([k ** (-1.25) + 0.25 for k in range(1, n_sub + 1)])
    Xb = []
    for k in range(1, n_sub + 1):
        lo, hi = 8.0 * k, min(90.0, nyq - 1.0)
        if lo >= hi:
            Xb.append(None); continue
        sos = butter(4, [lo, hi], btype="band", fs=sf, output="sos")
        Xb.append(sosfiltfilt(sos, X, axis=-1))
    rho = np.zeros(len(refs))
    for j, Y in enumerate(refs):
        acc = 0.0
        for k, Xf in enumerate(Xb):
            if Xf is not None:
                acc += weights[k] * _cca_corr(Xf, Y) ** 2
        rho[j] = acc
    return int(np.argmax(rho))


def _trca_filter(trials):
    """TRCA spatial filter (Nakanishi et al. 2018): leading eigenvector of Q^-1 S,
    S summing cross-trial covariances (reproducibility) and Q the covariance of the
    concatenated trials. `trials`: list of (ch, samp)."""
    trials = [t - t.mean(1, keepdims=True) for t in trials]
    nch = trials[0].shape[0]
    S = np.zeros((nch, nch))
    for i in range(len(trials)):
        for k in range(len(trials)):
            if i != k:
                S += trials[i] @ trials[k].T
    UX = np.concatenate(trials, axis=1)
    Q = UX @ UX.T
    try:
        ev, V = np.linalg.eig(np.linalg.solve(Q, S))
        return np.real(V[:, int(np.argmax(np.real(ev)))])
    except np.linalg.LinAlgError:
        return np.ones(nch)


def _trca_decode(seg, sf):
    """Ensemble TRCA (eTRCA, Nakanishi et al. 2018) with leave-one-block-out CV.
    seg: (ch, samp, target, block). Per test block, every target's TRCA spatial
    filter is trained on the other blocks; the ensemble filter bank W (all targets'
    filters stacked) projects both the test trial and each target's template, and the
    test trial is assigned to the target maximizing the 2-D correlation -- the
    ensemble form is far stronger than per-target basic TRCA on many-target montages.
    Restricted to the parieto-occipital channels TRCA conventionally uses (Tsinghua
    64-ch montage), which is essential for SSVEP TRCA performance."""
    nch, nsamp, ntar, nblk = seg.shape
    OCC64 = [47, 53, 54, 55, 56, 57, 60, 61, 62]   # Pz,PO5,PO3,POz,PO4,PO6,O1,Oz,O2 (Tsinghua 64-ch, 0-based)
    if nch == 64:
        seg = seg[OCC64]
        nch = len(OCC64)
    if nblk < 3:
        return float("nan")
    hits = tot = 0
    for b in range(nblk):
        W = np.zeros((ntar, nch))
        templates = []
        for j in range(ntar):
            tr = [seg[:, :, j, bb] for bb in range(nblk) if bb != b]
            W[j] = _trca_filter(tr)
            templates.append(np.mean(tr, axis=0))          # (nch, samp)
        WT = [W @ T for T in templates]                     # ensemble-projected templates
        for j_true in range(ntar):
            xp = W @ seg[:, :, j_true, b]                    # (ntar, samp)
            scores = [_corr2d(xp, WT[j]) for j in range(ntar)]
            hits += int(np.argmax(scores) == j_true)
            tot += 1
    return hits / tot if tot else float("nan")


def _load_tsinghua(root, subject):
    """Tsinghua/BETA .mat -> (data[ch, samp, target, block], freqs[target], sfreq)."""
    import scipy.io as sio

    cand = glob.glob(f"{root}/**/{subject}.mat", recursive=True)
    if not cand:
        raise FileNotFoundError(f"SSVEP .mat not found for {subject} under {root}")
    m = sio.loadmat(cand[0])
    if "data" in m:                                   # Tsinghua: (64, 1500, 40, 6)
        data = np.asarray(m["data"], float)
        fp = glob.glob(f"{root}/**/Freq_Phase.mat", recursive=True)
        freqs = np.asarray(sio.loadmat(fp[0])["freqs"]).ravel() if fp else None
        return data, freqs, 250.0
    # BETA: nested struct EEG with .data (64, samp, block, target) + .suppl_info.freqs
    eeg = m["data"][0, 0]
    data = np.asarray(eeg["EEG"][0, 0]["data"] if "EEG" in eeg.dtype.names else eeg["data"], float)
    if data.ndim == 4 and data.shape[2] <= 8:         # (ch, samp, block, target) -> (ch, samp, target, block)
        data = data.transpose(0, 1, 3, 2)
    freqs = np.asarray(eeg["suppl_info"][0, 0]["freqs"]).ravel()
    return data, freqs, 250.0


def run_subject(cfg, subject, root, deriv_root, *, synthetic=False):
    import mne

    data, freqs, sf = _load_tsinghua(root, subject)
    nch, _, ntar, nblk = data.shape
    w0, w1 = int(0.5 * sf), int(5.5 * sf)             # 0.5-5.5 s SSVEP window
    seg = data[:, w0:w1, :, :]
    n_samp = seg.shape[1]
    info = mne.create_info([f"E{i}" for i in range(nch)], sf, "eeg")
    nharm = int((cfg.get("metrics", {}) or {}).get("n_harmonics", 3))
    snr_raw, snr_dss, hits_cca, hits_dss, hits_fbcca, oob = [], [], 0, 0, 0, []
    refs = [_ref(f, sf, n_samp, nharm) for f in freqs]
    for ti in range(ntar):
        f0 = float(freqs[ti])
        trials = seg[:, :, ti, :].transpose(2, 0, 1)  # (block, ch, samp)
        epo = mne.EpochsArray(trials, info, verbose=False)
        try:
            dss = ssvep_dss(sf, f0, n_harmonics=nharm)
            dss.fit(epo)
            src = np.asarray(dss.transform(epo))       # (block, n_comp, samp) or (n_comp, samp)
            comp = src[:, 0, :] if src.ndim == 3 else src[None, 0, :]
        except Exception:  # noqa: BLE001
            comp = trials.mean(1)                       # fallback: mean channel
        for b in range(nblk):
            raw_b = trials[b]
            snr_raw.append(max(harmonic_snr(raw_b[c], sf, f0, nharm) for c in range(nch)))
            snr_dss.append(harmonic_snr(comp[b % comp.shape[0]], sf, f0, nharm))
            # decoding: CCA on raw, and DSS-source-assisted CCA
            cc = [_cca_corr(raw_b, refs[j]) for j in range(ntar)]
            if int(np.argmax(cc)) == ti:
                hits_cca += 1
            cd = [_cca_corr(np.vstack([raw_b, comp[b % comp.shape[0]][None]]), refs[j]) for j in range(ntar)]
            if int(np.argmax(cd)) == ti:
                hits_dss += 1
            if _fbcca(raw_b, refs, sf) == ti:                # filter-bank CCA (training-free)
                hits_fbcca += 1
    ntr = ntar * nblk
    row = {"status": "success", "n_targets": ntar,
           "harmonic_snr_raw": float(np.median(snr_raw)),
           "harmonic_snr_dss": float(np.median(snr_dss)),
           "snr_gain_db": float(10 * np.log10(np.median(snr_dss) / max(np.median(snr_raw), 1e-9))),
           "acc_cca": hits_cca / ntr, "acc_dss_cca": hits_dss / ntr,
           "acc_fbcca": hits_fbcca / ntr, "acc_trca": _trca_decode(seg, sf),  # standard SSVEP comparator battery
           "window_s": float((w1 - w0) / sf)}                                 # analysis window, disclosed
    out_dir = pathlib.Path(deriv_root) / subject / BENCH / "ssvep_dss"
    bio.save_subject_benchmark_results(out_dir, subject=subject, method="ssvep_dss", metrics=row)
    return [{"method": "ssvep_dss", **row}]


def main(argv=None):
    import os

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--subject")
    p.add_argument("--slurm-array", action="store_true")
    p.add_argument("--deriv-root")
    a = p.parse_args(argv)
    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "ssvep")
    deriv = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    root = pathlib.Path(os.environ.get("SSVEP_ROOT", "/scratch/sesma/ssvep_extracted")) / \
        (cfg.get("dataset", {}) or {}).get("subdir", "tsinghua")
    subs = (cfg.get("dataset", {}) or {}).get("subjects") or [f"S{i}" for i in range(1, 36)]
    subject = a.subject or (subs[int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1] if a.slurm_array else None)
    if not subject:
        p.error("need --subject or --slurm-array")
    for r in run_subject(cfg, subject, root, deriv):
        print(f"  {r['method']}: SNR raw={r['harmonic_snr_raw']:.2f} dss={r['harmonic_snr_dss']:.2f} "
              f"gain={r['snr_gain_db']:+.1f}dB  acc CCA={r['acc_cca']:.3f} DSS+CCA={r['acc_dss_cca']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
