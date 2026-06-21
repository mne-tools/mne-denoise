"""Ground-truth recovery metrics — valid ONLY where the truth is known.

These are the metrics confined to the simulation / phantom arms (see
``docs/benchmarks/BENCHMARK_PLAN.md`` §4).  Each function **gates** its inputs and
raises ``ValueError`` on misuse rather than silently returning a meaningless
number:

* :func:`rrmse`, :func:`correlation_with_clean` — need a known clean target.
* :func:`bss_eval` (SIR/SDR/SAR) — need known matched reference sources.
* :func:`amari_index` — needs a known **square** identifiable mixing (``W @ A``).
* :func:`mixing_recovery_error` — compatible dimensions.
* :func:`contamination_precision_recall` — a known boolean mask.
"""

from __future__ import annotations

import numpy as np


def _arr(x, name):
    a = np.asarray(x, float)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    return a


# ---------------------------------------------------------------------------
# Error / correlation vs a known clean target
# ---------------------------------------------------------------------------
def rrmse(estimate, target) -> float:
    """Relative RMSE ``||est - target|| / ||target||`` (temporal)."""
    est, tgt = _arr(estimate, "estimate"), _arr(target, "target")
    if est.shape != tgt.shape:
        raise ValueError(f"shape mismatch {est.shape} vs {tgt.shape}")
    denom = float(np.linalg.norm(tgt))
    if denom == 0:
        raise ValueError("target has zero norm; RRMSE undefined")
    return float(np.linalg.norm(est - tgt) / denom)


def rrmse_spectral(estimate, target, *, axis: int = -1) -> float:
    """Relative RMSE of magnitude spectra (rFFT along ``axis``)."""
    est, tgt = _arr(estimate, "estimate"), _arr(target, "target")
    if est.shape != tgt.shape:
        raise ValueError(f"shape mismatch {est.shape} vs {tgt.shape}")
    me = np.abs(np.fft.rfft(est, axis=axis))
    mt = np.abs(np.fft.rfft(tgt, axis=axis))
    denom = float(np.linalg.norm(mt))
    if denom == 0:
        raise ValueError("target spectrum has zero norm")
    return float(np.linalg.norm(me - mt) / denom)


def correlation_with_clean(estimate, target) -> float:
    """Mean per-row Pearson correlation with the clean target."""
    est, tgt = _arr(estimate, "estimate"), _arr(target, "target")
    if est.shape != tgt.shape:
        raise ValueError(f"shape mismatch {est.shape} vs {tgt.shape}")
    est = np.atleast_2d(est)
    tgt = np.atleast_2d(tgt)
    cs = []
    for e, t in zip(est, tgt):
        e = e - e.mean(); t = t - t.mean()
        d = np.linalg.norm(e) * np.linalg.norm(t)
        cs.append(float(e @ t / d) if d else 0.0)
    return float(np.mean(cs))


# ---------------------------------------------------------------------------
# BSS evaluation (SIR / SDR / SAR)
# ---------------------------------------------------------------------------
def bss_eval(estimate, references, target_index: int) -> dict:
    """SDR/SIR/SAR of one estimated source against known reference sources.

    Projection-based decomposition (Vincent et al., 2006).  ``references`` is
    ``(n_sources, n_samples)`` — required (raises if ``None``).
    """
    if references is None:
        raise ValueError("bss_eval requires known reference sources")
    s_hat = _arr(estimate, "estimate").ravel()
    S = _arr(references, "references")
    if S.ndim != 2:
        raise ValueError("references must be 2-D (n_sources, n_samples)")
    if s_hat.shape[0] != S.shape[1]:
        raise ValueError("estimate length must match reference n_samples")
    if not (0 <= target_index < S.shape[0]):
        raise ValueError("target_index out of range")

    tgt = S[target_index]
    # s_target: projection onto the target reference
    s_target = (s_hat @ tgt) / (tgt @ tgt + 1e-12) * tgt
    # projection onto the full reference subspace (least squares)
    coef, *_ = np.linalg.lstsq(S.T, s_hat, rcond=None)
    s_proj = S.T @ coef
    e_interf = s_proj - s_target
    e_artif = s_hat - s_proj

    def _db(num, den):
        return 10.0 * np.log10((num + 1e-20) / (den + 1e-20))

    sdr = _db(np.sum(s_target ** 2), np.sum((e_interf + e_artif) ** 2))
    sir = _db(np.sum(s_target ** 2), np.sum(e_interf ** 2))
    sar = _db(np.sum((s_target + e_interf) ** 2), np.sum(e_artif ** 2))
    return {"sdr": float(sdr), "sir": float(sir), "sar": float(sar)}


# ---------------------------------------------------------------------------
# Mixing-matrix metrics
# ---------------------------------------------------------------------------
def amari_index(W, A) -> float:
    """Amari distance of ``P = W @ A`` from a scaled permutation (0 = perfect).

    Requires a **square** product (``W @ A`` of shape ``(n, n)``) — i.e. a known,
    identifiable mixing.  Raises otherwise (per the validity-gating rule).
    """
    W, A = _arr(W, "W"), _arr(A, "A")
    P = W @ A
    n, m = P.shape
    if n != m:
        raise ValueError(
            f"Amari index requires a square W@A; got {P.shape}. "
            "Use it only with a known identifiable (square) mixing."
        )
    absP = np.abs(P)
    row = np.sum(np.sum(absP, axis=1) / (absP.max(axis=1) + 1e-20) - 1.0)
    col = np.sum(np.sum(absP, axis=0) / (absP.max(axis=0) + 1e-20) - 1.0)
    return float((row + col) / (2.0 * n))


def mixing_recovery_error(A_est, A_true) -> float:
    """Relative Frobenius error after per-column sign+scale+permutation alignment."""
    Ae, At = _arr(A_est, "A_est"), _arr(A_true, "A_true")
    if Ae.shape != At.shape:
        raise ValueError(f"shape mismatch {Ae.shape} vs {At.shape}")
    from scipy.optimize import linear_sum_assignment

    # cosine similarity between columns
    en = Ae / (np.linalg.norm(Ae, axis=0, keepdims=True) + 1e-12)
    tn = At / (np.linalg.norm(At, axis=0, keepdims=True) + 1e-12)
    sim = np.abs(en.T @ tn)
    row, col = linear_sum_assignment(-sim)
    aligned = np.zeros_like(At)
    for r, c in zip(row, col):
        scale = (At[:, c] @ Ae[:, r]) / (Ae[:, r] @ Ae[:, r] + 1e-12)
        aligned[:, c] = scale * Ae[:, r]
    return float(np.linalg.norm(aligned - At) / (np.linalg.norm(At) + 1e-12))


def contamination_precision_recall(pred_mask, true_mask) -> dict:
    """Precision/recall/F1 of a predicted contamination mask vs the known mask."""
    p = np.asarray(pred_mask, bool)
    t = np.asarray(true_mask, bool)
    if p.shape != t.shape:
        raise ValueError(f"mask shape mismatch {p.shape} vs {t.shape}")
    tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t)); fn = int(np.sum(~p & t))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
