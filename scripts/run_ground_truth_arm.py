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


def _rjd(A, eps=1e-12, maxiter=200):
    """Real Jacobi joint approximate diagonaliser (Cardoso & Souloumiac 1996).
    A: (k, n, n) stack of symmetric matrices -> rotation V (n, n) s.t. each
    V.T @ A_i @ V is approximately diagonal."""
    k, n, _ = A.shape
    A = A.copy().astype(float)
    V = np.eye(n)
    for _ in range(maxiter):
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                h = np.vstack([A[:, p, p] - A[:, q, q], A[:, p, q] + A[:, q, p]])  # (2, k)
                hh = h @ h.T
                ton, toff = hh[0, 0] - hh[1, 1], hh[0, 1] + hh[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) > eps:
                    changed = True
                    Ap, Aq = A[:, p, :].copy(), A[:, q, :].copy()        # rotate rows
                    A[:, p, :], A[:, q, :] = c * Ap + s * Aq, -s * Ap + c * Aq
                    Ap, Aq = A[:, :, p].copy(), A[:, :, q].copy()        # rotate cols
                    A[:, :, p], A[:, :, q] = c * Ap + s * Aq, -s * Ap + c * Aq
                    Vp, Vq = V[:, p].copy(), V[:, q].copy()
                    V[:, p], V[:, q] = c * Vp + s * Vq, -s * Vp + c * Vq
        if not changed:
            break
    return V


def _whiten(X, n_comp):
    """Centre + PCA-whiten X (n_ch, T) to (n_comp, T); return (Z, Wh)."""
    Xc = X - X.mean(1, keepdims=True)
    d, E = np.linalg.eigh((Xc @ Xc.T) / Xc.shape[1])
    idx = np.argsort(d)[::-1][:n_comp]
    Wh = (E[:, idx] / np.sqrt(np.maximum(d[idx], 1e-18))).T          # (n_comp, n_ch)
    return Wh @ Xc, Wh


def _sobi(X, n_comp, n_lags=20):
    """SOBI (Belouchrani 1997): joint-diagonalise time-lagged covariances."""
    Z, Wh = _whiten(X, n_comp)
    T = Z.shape[1]
    covs = []
    for tau in range(1, n_lags + 1):
        Rt = (Z[:, tau:] @ Z[:, :-tau].T) / (T - tau)
        covs.append(0.5 * (Rt + Rt.T))
    V = _rjd(np.asarray(covs))
    return V.T @ Wh                                                  # (n_comp, n_ch)


def _jade(X, n_comp):
    """JADE (Cardoso & Souloumiac 1993): joint-diagonalise 4th-order cumulant matrices."""
    Z, Wh = _whiten(X, n_comp)
    m, T = Z.shape
    CM = []
    for p in range(m):
        for q in range(p, m):
            M = (Z * (Z[p] * Z[q])) @ Z.T / T                       # E[z_k z_l z_p z_q]
            M = M - (np.eye(m) if p == q else 0.0)
            M[p, q] -= 1.0
            M[q, p] -= 1.0
            CM.append(0.5 * (M + M.T) * (1.0 if p == q else np.sqrt(2.0)))
    V = _rjd(np.asarray(CM))
    return V.T @ Wh


def _fit_unmix(method: str, X_train: np.ndarray, n_comp: int):
    """Return ``(transform, W)``: ``transform(X) -> sources (n_comp, T)`` and an
    estimated unmixing ``W (n_comp, n_ch)`` (or None)."""
    if method in ("sobi", "jade"):
        W = (_sobi if method == "sobi" else _jade)(X_train, n_comp)
        return (lambda X: W @ (X - X.mean(1, keepdims=True))), W
    if method == "amica":
        from amica_python import Amica, AmicaConfig   # GPU-accelerated AMICA (Palmer et al.); JAX uses GPU when available

        cfg = AmicaConfig(pcakeep=int(n_comp), max_iter=2000, do_newton=True)
        model = Amica(cfg, random_state=0).fit(np.asarray(X_train, dtype=float))
        W = np.asarray(model.unmixing_matrix_sensor_)          # (n_comp, n_ch) full sensor-space unmixing
        return (lambda X: np.asarray(model.transform(np.asarray(X, dtype=float)))), W
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

        K, W, _ = picard(X_train, n_components=n_comp, ortho=False, random_state=0, max_iter=300)
        WK = W @ K  # compose whitening (K) with the whitened-space unmixing (W)
        return (lambda X: WK @ X), WK
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
    n_comp = min(rep.A.shape[1], rep.X_train.shape[0])   # clamp to channel count (enables the low-density sweep)
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
