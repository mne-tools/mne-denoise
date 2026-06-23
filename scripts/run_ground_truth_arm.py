#!/usr/bin/env python
"""Ground-truth source-recovery runner (arms: ground_truth_generic, ground_truth_forward).

Pure simulation. Each replicate has known clean sources S and mixing A. Methods unmix
on TRAIN, are matched to the clean brain sources on TRAIN (Hungarian; sign/scale frozen
on train), applied to TEST, and scored against known truth: RRMSE, clean-source
correlation, SIR/SDR/SAR, Amari (square identifiable mixing only). No real data; the
unit of inference is the replicate (nested over A x SNR x seed).

    python scripts/run_ground_truth_arm.py --config configs/benchmarks/ground_truth_generic.yaml --synthetic
    python scripts/run_ground_truth_arm.py --config ... --all
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from mne_denoise.benchmarks import io as bio
from mne_denoise.benchmarks import simulation as sim
from mne_denoise.qa import ground_truth as gt

BENCH = "ground_truth"


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _fit_unmix(method: str, X_train: np.ndarray, n_comp: int):
    """Return ``(transform, W)``: ``transform(X) -> sources (n_comp, T)`` and an
    estimated unmixing ``W (n_comp, n_ch)`` (or None)."""
    if method == "pca":
        from sklearn.decomposition import PCA

        p = PCA(n_components=n_comp, whiten=True, svd_solver="full")
        p.fit(X_train.T)
        return (lambda X: p.transform(X.T).T), p.components_
    if method == "fastica":
        from sklearn.decomposition import FastICA

        ica = FastICA(n_components=n_comp, whiten="unit-variance", max_iter=600, random_state=0)
        ica.fit(X_train.T)
        return (lambda X: ica.transform(X.T).T), ica.components_
    if method == "infomax":
        import mne

        W = mne.preprocessing.infomax(X_train.T, extended=True, random_state=0)  # (n_ch, n_ch)
        return (lambda X: (W @ X)[:n_comp]), W[:n_comp]
    if method == "picard":
        from picard import picard

        _, W, _ = picard(X_train, n_components=n_comp, ortho=False, random_state=0, max_iter=300)
        return (lambda X: W @ X), W
    if method == "iterative_dss":
        from mne_denoise.dss import iterative_dss
        from mne_denoise.dss.denoisers import TanhMaskDenoiser, beta_tanh

        filt, _, _, _ = iterative_dss(X_train, TanhMaskDenoiser(), n_comp, method="symmetric",
                                      beta=beta_tanh, max_iter=300)
        return (lambda X: filt @ X), filt
    raise KeyError(method)


def _align(src_test, m, n_ref):
    """Reorder sign/scale-applied recovered sources to (n_ref, T) aligned to references."""
    applied = sim.apply_source_matching(src_test, m)
    out = np.zeros((n_ref, applied.shape[1]))
    for i, ref_j in enumerate(m.perm):
        if 0 <= ref_j < n_ref:
            out[ref_j] = applied[i]
    return out


def run_replicate(rep, methods, *, min_corr=0.5):
    bi = np.asarray(rep.brain_idx)
    S_tr_b, S_te_b = rep.S_train[bi], rep.S_test[bi]
    n_brain = int(np.size(bi))
    n_comp = rep.A.shape[1]
    rows = []
    for method in methods:
        try:
            W = None
            if method == "oracle":
                est = S_te_b.copy()  # known clean sources = upper bound
            else:
                transform, W = _fit_unmix(method, rep.X_train, n_comp)
                m = sim.fit_source_matching(transform(rep.X_train), S_tr_b, min_corr=min_corr)
                est = _align(transform(rep.X_test), m, n_brain)
            row = {
                "status": "success",
                "rrmse": float(gt.rrmse(est, S_te_b)),
                "corr_clean": float(gt.correlation_with_clean(est, S_te_b)),
                "snr_db": float(rep.snr_db),
                "seed": int(rep.seed),
                "regime": rep.regime,
            }
            if W is not None and rep.is_square:
                try:
                    row["amari"] = float(gt.amari_index(W, rep.A))
                except Exception:  # noqa: BLE001
                    pass
            rows.append({"method": method, **row})
        except Exception as exc:  # noqa: BLE001
            rows.append({"method": method, "status": "failed_numerical",
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows


def _grid(cfg, *, synthetic):
    s = cfg.get("simulation", {}) or {}
    raw = f"{s.get('regime', '')} {cfg.get('arm', '')}"
    regime = "forward" if "forward" in raw else "generic"
    snrs = s.get("snr_levels_db")
    snrs = snrs if isinstance(snrs, list) else [-6.0, 0.0, 6.0]
    nrep = s.get("n_replicates")
    nrep = (3 if synthetic else (nrep if isinstance(nrep, int) else 20))
    return regime, snrs, nrep


def run(cfg, deriv_root, *, synthetic=False):
    regime, snrs, nrep = _grid(cfg, synthetic=synthetic)
    methods = list(cfg.get("methods_under_test", []) or []) + \
        list((cfg.get("comparators", {}) or {}).get("required", []) or [])
    # map config ids to local unmixers; keep order, de-dup
    alias = {"iterativedss": "iterative_dss", "iterative_dss_tanh": "iterative_dss",
             "iterative_dss_symmetric": "iterative_dss", "fastICA": "fastica"}
    methods = list(dict.fromkeys(alias.get(m, m) for m in methods)) or \
        ["pca", "fastica", "iterative_dss", "oracle"]
    allrows = []
    for snr in snrs:
        for seed in range(nrep):
            rep = sim.simulate_replicate(regime=regime, n_channels=8, n_brain=4, n_artifact=4,
                                         n_train=2000, n_test=2000, snr_db=float(snr), seed=seed)
            for r in run_replicate(rep, methods):
                r["replicate"] = f"snr{snr}_seed{seed}"
                allrows.append(r)
                if not synthetic:
                    out = pathlib.Path(deriv_root) / f"snr{snr}_seed{seed}" / BENCH / r["method"]
                    bio.save_subject_benchmark_results(out, subject=f"snr{snr}_seed{seed}",
                                                       method=r["method"], metrics=r)
    return allrows, methods


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--all", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--deriv-root", default=None)
    a = p.parse_args(argv)
    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "ground_truth")
    deriv_root = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    rows, methods = run(cfg, deriv_root, synthetic=a.synthetic)
    # summarise per method (mean over replicates)
    import statistics
    print(f"arm={arm}  replicates={len({r.get('replicate') for r in rows})}  methods={methods}")
    print(f"  {'method':16}{'rrmse':>9}{'corr':>8}{'amari':>9}  n")
    for mth in methods:
        rs = [r for r in rows if r.get("method") == mth and r.get("status") == "success"]
        if not rs:
            print(f"  {mth:16}{'FAILED':>9}"); continue
        me = lambda k: round(statistics.mean([r[k] for r in rs if k in r]), 4) if any(k in r for r in rs) else "-"
        print(f"  {mth:16}{str(me('rrmse')):>9}{str(me('corr_clean')):>8}{str(me('amari')):>9}  {len(rs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
