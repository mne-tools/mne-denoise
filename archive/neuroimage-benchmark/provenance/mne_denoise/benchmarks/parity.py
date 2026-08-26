"""Numerical-equivalence ("parity") helpers — three evidence levels.

See ``docs/benchmarks/RECONCILIATION.md``:

* **L1** exact numerical — same fixture, aligned output within a declared tolerance.
* **L2** subspace/algorithmic — projector / principal-angle / repair-decision
  agreement after sign+permutation alignment (the right level for ICA/DSS-like
  outputs, which are only defined up to sign and order).
* **L3** canonical-behaviour replication — reproduce a published scalar/qualitative
  result within tolerance.

Golden fixtures must be redistributable; **CI must never require MATLAB or other
proprietary software**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class ParityResult:
    level: str
    passed: bool
    metric: float
    tolerance: float
    detail: dict


# ---------------------------------------------------------------------------
# L1 — exact numerical
# ---------------------------------------------------------------------------
def numerical_parity(a, b, *, rtol: float = 1e-6, atol: float = 1e-8) -> ParityResult:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    passed = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    return ParityResult("L1", passed, max_abs, atol,
                        {"rtol": rtol, "max_abs_err": max_abs})


# ---------------------------------------------------------------------------
# L2 — subspace / algorithmic
# ---------------------------------------------------------------------------
def principal_angles(A, B) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spaces of A, B."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(s[::-1], -1.0, 1.0))


def subspace_parity(A, B, *, tol_deg: float = 1.0) -> ParityResult:
    """Pass if the largest principal angle between span(A) and span(B) ≤ tol."""
    ang = np.degrees(principal_angles(A, B))
    worst = float(ang.max()) if ang.size else 0.0
    return ParityResult("L2", worst <= tol_deg, worst, tol_deg,
                        {"angles_deg": ang.tolist()})


def align_components(W_a, W_b):
    """Match rows of ``W_b`` to ``W_a`` up to sign+permutation (max |corr|).

    Returns ``(perm, signs, corr)`` so that ``W_b[perm] * signs[:, None]`` is the
    best sign-and-order-aligned version of ``W_b`` for comparison with ``W_a``.
    """
    W_a, W_b = np.asarray(W_a, float), np.asarray(W_b, float)
    if W_a.shape != W_b.shape:
        raise ValueError("component matrices must share shape")
    # correlation between every pair of components
    a = W_a - W_a.mean(1, keepdims=True)
    b = W_b - W_b.mean(1, keepdims=True)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    corr = an @ bn.T                       # (n_a, n_b)
    row, col = linear_sum_assignment(-np.abs(corr))
    perm = col[np.argsort(row)]
    signs = np.sign([corr[r, c] for r, c in zip(row, col)])
    signs[signs == 0] = 1.0
    matched = np.array([abs(corr[r, c]) for r, c in zip(row, col)])
    return perm, signs, matched


# ---------------------------------------------------------------------------
# L3 — canonical behaviour replication
# ---------------------------------------------------------------------------
def behaviour_parity(value: float, reference: float, *, rtol: float = 0.1) -> ParityResult:
    denom = abs(reference) if reference else 1.0
    rel = abs(float(value) - float(reference)) / denom
    return ParityResult("L3", rel <= rtol, rel, rtol,
                        {"value": float(value), "reference": float(reference)})
