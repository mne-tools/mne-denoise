"""Semi-synthetic multichannel mixtures for the ground-truth track.

Two regimes (see ``docs/benchmarks/BENCHMARK_PLAN.md`` §5):

* **generic** — random, well-conditioned (optionally square) mixing ``A``; valid
  for Amari / source-recovery metrics.
* **forward** — a physiologically-flavoured lead field (spatially-smooth brain
  columns, frontal ocular columns, temporal muscle columns).  Source-recovery
  ground truth is **EEG-only**; MEG evidence is empirical + parity.

Each replicate uses **one** mixing ``A`` for both the train and the held-out test
segment (``X = A·S``); ``A`` is independent across replicates.  Source matching is
solved on the **train** split and applied to test (never rematched on test).
The oracle is ``A_brain · S_brain``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MixtureReplicate:
    A: np.ndarray                 # (n_channels, n_sources)
    S_train: np.ndarray           # (n_sources, n_train)
    X_train: np.ndarray           # (n_channels, n_train)
    S_test: np.ndarray            # (n_sources, n_test)
    X_test: np.ndarray            # (n_channels, n_test)
    labels: np.ndarray            # (n_sources,) of {"brain","artifact"}
    snr_db: float
    seed: int
    regime: str

    @property
    def brain_idx(self) -> np.ndarray:
        return np.flatnonzero(self.labels == "brain")

    @property
    def artifact_idx(self) -> np.ndarray:
        return np.flatnonzero(self.labels == "artifact")

    @property
    def is_square(self) -> bool:
        return self.A.shape[0] == self.A.shape[1]


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------
def _nongaussian_sources(n: int, t: int, rng: np.random.Generator) -> np.ndarray:
    """Mutually-independent, mostly non-Gaussian sources (ICA-identifiable)."""
    src = np.empty((n, t))
    tt = np.arange(t)
    for i in range(n):
        kind = i % 4
        if kind == 0:                      # super-Gaussian (Laplace)
            src[i] = rng.laplace(size=t)
        elif kind == 1:                    # sinusoid (sub-Gaussian)
            f = 0.01 + 0.04 * rng.random()
            src[i] = np.sin(2 * np.pi * f * tt + rng.random() * 6.28)
        elif kind == 2:                    # AR(1) coloured noise
            a = 0.7
            e = rng.standard_normal(t)
            s = np.zeros(t)
            for k in range(1, t):
                s[k] = a * s[k - 1] + e[k]
            src[i] = s
        else:                              # bursty (sparse) source
            s = rng.standard_normal(t)
            mask = rng.random(t) < 0.05
            s[~mask] *= 0.1
            src[i] = s
    src -= src.mean(1, keepdims=True)
    src /= src.std(1, keepdims=True) + 1e-12
    return src


def _well_conditioned_mixing(
    n_channels: int, n_sources: int, rng: np.random.Generator, max_cond: float | None
) -> np.ndarray:
    A = rng.standard_normal((n_channels, n_sources))
    if max_cond is not None:
        for _ in range(100):
            if np.linalg.cond(A) <= max_cond:
                break
            A = rng.standard_normal((n_channels, n_sources))
    return A


def _ring_positions(n: int) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(ang), np.sin(ang)]


def _smooth_topography(center: np.ndarray, pos: np.ndarray, width: float) -> np.ndarray:
    d2 = np.sum((pos - center) ** 2, axis=1)
    col = np.exp(-d2 / (2 * width ** 2))
    return col / (np.linalg.norm(col) + 1e-12)


def _forward_mixing(n_channels, n_brain, n_artifact, rng) -> np.ndarray:
    """Spatially-smooth lead field: brain anywhere, ocular frontal, muscle temporal."""
    pos = _ring_positions(n_channels)
    cols = []
    for _ in range(n_brain):               # brain: random smooth blobs
        c = pos[rng.integers(n_channels)] + 0.2 * rng.standard_normal(2)
        cols.append(_smooth_topography(c, pos, width=0.8))
    for j in range(n_artifact):            # ocular frontal (top), muscle temporal (sides)
        c = np.array([0.0, 1.0]) if j % 2 == 0 else np.array([1.0, 0.0])
        cols.append(_smooth_topography(c, pos, width=0.5))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Replicate construction
# ---------------------------------------------------------------------------
def simulate_replicate(
    *,
    regime: str = "generic",
    n_channels: int = 8,
    n_brain: int = 4,
    n_artifact: int = 4,
    n_train: int = 2000,
    n_test: int = 2000,
    snr_db: float = 0.0,
    seed: int = 0,
    max_cond: float | None = 50.0,
) -> MixtureReplicate:
    """Build one replicate; train and test share the SAME mixing ``A``."""
    rng = np.random.default_rng(seed)
    n_sources = n_brain + n_artifact
    if n_channels < n_sources:
        raise ValueError("n_channels must be >= n_brain + n_artifact")
    if regime == "generic":
        A = _well_conditioned_mixing(n_channels, n_sources, rng, max_cond)
    elif regime == "forward":
        A = _forward_mixing(n_channels, n_brain, n_artifact, rng)
    else:
        raise ValueError(f"unknown regime {regime!r}")

    labels = np.array(["brain"] * n_brain + ["artifact"] * n_artifact)

    def _make(t: int) -> tuple[np.ndarray, np.ndarray]:
        S = _nongaussian_sources(n_sources, t, rng)
        # scale artifacts vs brain to hit the target input SNR (sensor space)
        brain = labels == "brain"
        gain = 10 ** (-snr_db / 20.0)
        S[~brain] *= gain
        X = A @ S
        return S, X

    S_tr, X_tr = _make(n_train)
    S_te, X_te = _make(n_test)
    return MixtureReplicate(A, S_tr, X_tr, S_te, X_te, labels, float(snr_db), int(seed), regime)


def oracle_reconstruction(A: np.ndarray, S: np.ndarray, brain_idx: np.ndarray) -> np.ndarray:
    """Upper-bound clean sensor signal from the KNOWN brain sources: A_brain·S_brain."""
    A, S = np.asarray(A, float), np.asarray(S, float)
    bi = np.asarray(brain_idx, int)
    return A[:, bi] @ S[bi, :]


# ---------------------------------------------------------------------------
# Source matching — solve on TRAIN, apply to test
# ---------------------------------------------------------------------------
@dataclass
class SourceMatching:
    perm: np.ndarray              # reference index for each recovered row
    signs: np.ndarray
    scales: np.ndarray
    matched_corr: np.ndarray
    unmatched: list[int]
    collapsed: list[int]


def fit_source_matching(
    recovered_train: np.ndarray, reference_train: np.ndarray, *, min_corr: float = 0.5
) -> SourceMatching:
    """Hungarian assignment (max |corr|) + per-pair sign/scale, on TRAIN only."""
    R = np.asarray(recovered_train, float)
    G = np.asarray(reference_train, float)
    nr = R.shape[0]
    rc = R - R.mean(1, keepdims=True)
    gc = G - G.mean(1, keepdims=True)
    rn = rc / (np.linalg.norm(rc, axis=1, keepdims=True) + 1e-12)
    gn = gc / (np.linalg.norm(gc, axis=1, keepdims=True) + 1e-12)
    corr = rn @ gn.T                          # (n_recovered, n_reference)
    row, col = linear_sum_assignment(-np.abs(corr))
    perm = np.full(nr, -1, int)
    signs = np.ones(nr)
    scales = np.ones(nr)
    matched = np.zeros(nr)
    collapsed: list[int] = []
    for r, c in zip(row, col):
        perm[r] = c
        matched[r] = abs(corr[r, c])
        # signed least-squares coefficient mapping recovered -> reference (train).
        # Split into sign + magnitude so apply() multiplies by (signs * scales) == coef.
        denom = float(R[r] @ R[r]) + 1e-12
        coef = float((G[c] @ R[r]) / denom)
        signs[r] = 1.0 if coef >= 0 else -1.0
        scales[r] = abs(coef)
        if np.std(R[r]) < 1e-9:
            collapsed.append(int(r))
    unmatched = [int(r) for r in range(nr) if matched[r] < min_corr]
    return SourceMatching(perm, signs, scales, matched, unmatched, collapsed)


def apply_source_matching(recovered_test: np.ndarray, m: SourceMatching) -> np.ndarray:
    """Apply a TRAIN-fitted matching (perm+sign+scale) to held-out recovered sources."""
    R = np.asarray(recovered_test, float)
    return (R * (m.signs * m.scales)[:, None])
