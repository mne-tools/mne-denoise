"""MARA six-feature extraction + trained linear classifier (Winkler et al. 2011).

DOI 10.1186/1744-9081-7-30. GPL-3.0 (see ``LICENSE.md``). Faithful Python reimplementation of the EEGLAB
``MARA.m`` (https://github.com/irenne/MARA, (C) 2013 Winkler & Waldburger).

Per ICA component, MARA computes six features and applies a linear discriminant:

  1. Current Density Norm  ``log ||M100 @ pattern||`` -- M100 is the sLORETA
     current-density operator built from the ICBM152 leadfield (lambda=100, an
     average-reference centering matrix H, and sLORETA depth weights W). The
     scalp pattern is std-normalised across channels first.
  2. Range Within Pattern  ``log(max(pattern) - min(pattern))`` over channels.
  3. Mean Local Skewness   ``log(mean_w |skewness(ic_t in 15 s window w)|)``.
  4. lambda                the ``x2`` parameter of the log-spectrum fit.
  5. 8-13 Hz power         mean dB power over the integer bins 8..13 Hz.
  6. FitError              ``log ||fit - spectrum||^2`` on the 8..15 Hz bins.

Spectral model (dB spectrum, six anchor points, nonlinear least squares):

  ``P_hat(f) = exp(x1) * f**(-exp(x2)) - x3``   with start ``[4, -2, 54]``.

Classifier: LDA decision ``g = w . feats + b``; component is an ARTIFACT when
``g <= 0`` (kept when ``g > 0``). The weight vector and bias below are the
published MARA classifier, derived (closed-form pooled-covariance LDA, the
MATLAB ``classify`` default) from the 1290 hand-labelled training components in
``fv_training_MARA.mat``. Verified: resubstitution accuracy 92.6%.
"""

from __future__ import annotations

import os

import numpy as np

# --- trained linear discriminant (feature order: [CDN, Range, Skew, lambda, 8-13Hz, FitError]) ---
# Closed-form pooled-covariance LDA fitted to fv_training_MARA.mat (706 artifact + 584 brain,
# group-size priors). g = MARA_W . feats + MARA_B ;  g > 0 -> brain (keep), g <= 0 -> artifact.
MARA_W = np.array(
    [-4.46127217, -5.44380629, -1.75665472, -0.22903768, 0.25026760, 0.60238187],
    dtype=float,
)
MARA_B = -1.98138234

_LEADFIELD_FILE = os.path.join(
    os.path.dirname(__file__), "data", "inv_matrix_icbm152.mat"
)

# module-level cache so the 5.7 MB leadfield + M100 are not rebuilt per call
_LEADFIELD_CACHE: dict | None = None
_M100_CACHE: dict[tuple[str, ...], tuple] = {}


# ---------------------------------------------------------------------------
# sLORETA current-density operator (port of get_M100_ADE / sloreta_invweights)
# ---------------------------------------------------------------------------
def _load_leadfield() -> dict:
    """Load and cache the vendored ICBM152 leadfield and channel labels.

    Returns the leadfield ``L`` (n_ch, n_dip, 3) and channel labels ``clab``.
    """
    global _LEADFIELD_CACHE
    if _LEADFIELD_CACHE is None:
        from scipy.io import loadmat

        m = loadmat(_LEADFIELD_FILE)
        L = np.asarray(m["L"], dtype=float)
        clab = [str(c[0]) for c in m["clab"][0]]
        _LEADFIELD_CACHE = {"L": L, "clab": clab}
    return _LEADFIELD_CACHE


def _sloreta_invweights(LL: np.ndarray) -> np.ndarray:
    """Block-diagonal sLORETA depth weights for leadfield tensor ``LL`` (M, N, 3).

    Port of MARA's ``sloreta_invweights``: column-center the (M, 3N) leadfield,
    ``T = L' pinv(L L')``, per-voxel weight ``W_vv = (T_v L_v)^(-1/2)``, then
    reorder columns so orientations are grouped (matching the MATLAB reshape).
    """
    M, N, NDUM = LL.shape
    # MATLAB: L = reshape(permute(LL,[1 3 2]), M, N*NDUM)  (column-major)
    Lmat = np.transpose(LL, (0, 2, 1)).reshape(M, N * NDUM, order="F")
    Lmat = Lmat - Lmat.mean(axis=0, keepdims=True)
    T = Lmat.T @ np.linalg.pinv(Lmat @ Lmat.T)
    Wblocks = np.zeros((N * NDUM, N * NDUM))
    for v in range(N):
        sl = slice(NDUM * v, NDUM * v + NDUM)
        block = T[sl, :] @ Lmat[:, sl]  # (3, 3)
        w, V = np.linalg.eigh((block + block.T) / 2.0)  # symmetric inverse sqrt
        w = np.clip(w, 1e-12, None)
        Wblocks[sl, sl] = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    # MATLAB: ind = [idum:NDUM:N*NDUM] for idum=1..NDUM  -> group by orientation
    ind = np.concatenate([np.arange(idum, N * NDUM, NDUM) for idum in range(NDUM)])
    return Wblocks[np.ix_(ind, ind)]


def _get_m100(clab_desired: list[str], lam: float = 100.0):
    """Build MARA's ``M100`` operator for channels common to data and leadfield.

    Matches ``clab_desired`` against the leadfield channels. Returns
    ``(M100, common_upper)`` where ``common_upper`` is the sorted list of matched
    channel names (upper-case) giving the row order ``M100`` expects for the
    pattern. Cached per channel set; ``M100`` is ``None`` if nothing matches.
    """
    lf = _load_leadfield()
    L, clab = lf["L"], lf["clab"]
    upper_clab = [c.upper() for c in clab]
    desired_upper = {c.upper() for c in clab_desired}
    common = sorted(set(upper_clab) & desired_upper)  # MATLAB intersect: sorted, unique
    key = tuple(common)
    if key in _M100_CACHE:
        return _M100_CACHE[key]
    if not common:
        _M100_CACHE[key] = (None, common)
        return None, common
    ia = [upper_clab.index(c) for c in common]  # indices into leadfield channels
    nch, n_dip = len(ia), L.shape[1]
    F = L[ia, :, :].reshape(nch, 3 * n_dip, order="F")  # forward for matched channels
    H = np.eye(nch) - np.ones((nch, nch)) / nch  # centering (average reference)
    W = _sloreta_invweights(L)  # depth weights from FULL leadfield
    Lop = H @ F @ W
    d, U = np.linalg.eigh(Lop @ Lop.T)  # inv(L L' + lam I) via eig
    di = 1.0 / (d + lam)
    di[d < 1e-10] = 0.0
    M100 = Lop.T @ (U @ np.diag(di) @ U.T) @ H
    _M100_CACHE[key] = (M100, common)
    return M100, common


# ---------------------------------------------------------------------------
# spatial features (Current Density Norm, Range Within Pattern)
# ---------------------------------------------------------------------------
def _spatial_features(patterns: np.ndarray, ch_names: list[str]) -> tuple:
    """Features 1-2 for all components.

    ``patterns`` is (n_ch, n_ic) raw scalp topographies (``ica.get_components()``,
    i.e. EEGLAB ``icawinv``). Returns ``(cdn, rng)`` arrays of length n_ic; ``cdn``
    is ``None`` when no channel label matches the leadfield (CDN skipped).
    """
    # MARA: patterns = patterns ./ std(patterns, 0, 1)  -- std across channels per component
    std = patterns.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    pat = patterns / std

    rng = np.log(pat.max(axis=0) - pat.min(axis=0))  # feature 2: log peak-to-peak

    cdn = None
    M100, common = _get_m100(ch_names)
    if M100 is not None:
        upper = [c.upper() for c in ch_names]
        order = [upper.index(c) for c in common]  # reorder pattern rows to M100's order
        proj = M100 @ pat[order, :]  # (3*n_dip, n_ic)
        cdn = np.log(
            np.sqrt(np.sum(proj**2, axis=0))
        )  # feature 1: log current-density norm
    return cdn, rng


# ---------------------------------------------------------------------------
# spectral fit (lambda, FitError) + 8-13 Hz power
# ---------------------------------------------------------------------------
def _spectral_model(x, f):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(x[0]) / np.power(f, np.exp(x[1])) - x[2]


def _fit_spectrum(pX: np.ndarray, pY: np.ndarray) -> np.ndarray:
    """Fit ``exp(x1) f^(-exp(x2)) - x3`` to the six anchor points (start [4,-2,54])."""
    from scipy.optimize import least_squares

    x0 = np.array([4.0, -2.0, 54.0])
    try:
        res = least_squares(
            lambda x: _spectral_model(x, pX) - pY, x0, method="trf",
            bounds=([-30.0, -5.0, -np.inf], [30.0, 5.0, np.inf]), max_nfev=10000,
        )
        return res.x
    except Exception:  # noqa: BLE001 -- fall back to a derivative-free optimiser
        from scipy.optimize import minimize

        r = minimize(
            lambda x: float(np.sum((_spectral_model(x, pX) - pY) ** 2)),
            x0,
            method="Nelder-Mead",
        )
        return np.asarray(r.x)


def _freq_idx(freq: np.ndarray, target: float) -> int:
    """Index of the bin whose frequency equals ``target`` (MARA uses integer-Hz bins)."""
    hits = np.where(np.isclose(freq, target))[0]
    return int(hits[0]) if hits.size else int(np.argmin(np.abs(freq - target)))


def _band_min(
    freq: np.ndarray, pxx: np.ndarray, lo: float, hi: float
) -> tuple[float, float]:
    """Spectral minimum (value, frequency) over the band ``[lo, hi]`` Hz."""
    sl = slice(_freq_idx(freq, lo), _freq_idx(freq, hi) + 1)
    seg = pxx[sl]
    j = int(np.argmin(seg))
    return float(seg[j]), float(freq[sl][j])


def _time_freq_features(sources: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute time/frequency features 3-6 for all components.

    Returns (n_ic, 4) = [mean_local_skewness, lambda, power_8_13, fit_error].
    ``sources`` is (n_ic, n_times) continuous component activations.
    """
    from scipy.signal import welch
    from scipy.stats import skew

    # MARA: downsample to ~100 Hz by integer striding (no anti-alias filter)
    factor = max(int(np.floor(sfreq / 100.0)), 1)
    data = sources[:, ::factor]
    fs = int(round(sfreq / factor))

    # standardise each component to unit variance over time
    std = data.std(axis=1, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    comps = data / std

    n_ic = comps.shape[0]
    out = np.empty((n_ic, 4), dtype=float)
    nperseg = fs  # 1-second rectangular window
    win = np.ones(nperseg)
    for ic in range(n_ic):
        # --- spectrum (1 Hz bins), to dB exactly as MARA ---
        freq, pxx = welch(
            comps[ic],
            fs=fs,
            window=win,
            noverlap=0,
            nfft=fs,
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        pxx = 10.0 * np.log10(pxx * fs / 2.0)

        # --- feature 5: mean dB power over integer bins 8..13 Hz ---
        p = sum(pxx[_freq_idx(freq, i)] for i in range(8, 14))
        power_8_13 = p / (13 - 8 + 1)

        # --- six anchor points for the spectral fit ---
        p1x, p1y = 2.0, pxx[_freq_idx(freq, 2)]
        p2x, p2y = 3.0, pxx[_freq_idx(freq, 3)]
        p3y, p3x = _band_min(freq, pxx, 5, 13)  # local min in 5-13 Hz
        p4x = p3x - 1.0
        p4y = pxx[_freq_idx(freq, p4x)]
        p5y, p5x = _band_min(freq, pxx, 33, 39)  # local min in 33-39 Hz
        p6x = p5x + 1.0
        p6y = pxx[_freq_idx(freq, p6x)]
        pX = np.array([p1x, p2x, p3x, p4x, p5x, p6x])
        pY = np.array([p1y, p2y, p3y, p4y, p5y, p6y])

        fitted = _fit_spectrum(pX, pY)
        lam = fitted[1]  # feature 4: lambda = x2

        # --- feature 6: FitError = log ||fit - spectrum||^2 on bins 8..15 Hz ---
        sl = slice(_freq_idx(freq, 8), _freq_idx(freq, 15) + 1)
        resid = _spectral_model(fitted, freq[sl]) - pxx[sl]
        fit_error = np.log(np.linalg.norm(resid) ** 2)

        # --- feature 3: mean abs local skewness over 15 s windows, logged ---
        interval = 15
        skews = []
        i = 1
        while i < comps.shape[1] / fs - interval:
            seg = comps[ic, int(i * fs) : int((i + interval) * fs)]
            skews.append(abs(float(skew(seg, bias=True))))
            i += interval
        if not skews:  # short recording: single window
            skews.append(abs(float(skew(comps[ic], bias=True))))
        mean_skew = np.log(np.mean(skews))

        out[ic] = [mean_skew, lam, power_8_13, fit_error]
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def mara_features(ica, inst) -> np.ndarray:
    """Return the (n_ic, 6) MARA feature matrix for ``ica`` fitted on ``inst``.

    Columns: [CurrentDensityNorm, RangeWithinPattern, MeanLocalSkewness, lambda,
    Power8_13Hz, FitError]. If no channel label matches the leadfield, the
    Current Density Norm column is filled with 0.0 (its effect drops out only if
    you also zero its weight; here it is simply unavailable -- see
    :func:`mara_bad_components`, which handles the CDN-missing case by classifying
    on the remaining features).
    """
    info = inst.info
    ch_names = list(info["ch_names"])
    patterns = np.asarray(ica.get_components(), dtype=float)  # (n_ch, n_ic) = icawinv
    sources = np.asarray(ica.get_sources(inst).get_data(), dtype=float)
    if sources.ndim == 3:  # Epochs -> concatenate trials
        n_ep, n_ic, n_t = sources.shape
        sources = sources.transpose(1, 0, 2).reshape(n_ic, n_ep * n_t)

    cdn, rng = _spatial_features(patterns, ch_names)
    tf = _time_freq_features(sources, float(info["sfreq"]))  # (n_ic, 4)
    n_ic = tf.shape[0]
    feats = np.empty((n_ic, 6), dtype=float)
    feats[:, 0] = cdn if cdn is not None else 0.0
    feats[:, 1] = rng
    feats[:, 2:] = tf
    return feats


def mara_bad_components(ica, inst) -> list[int]:
    """Return the sorted list of artifact IC indices per MARA (Winkler et al. 2011).

    Fits nothing -- applies the published LDA (:data:`MARA_W`, :data:`MARA_B`) to
    the six features. A component is an artifact when ``g = w . feats + b <= 0``.
    If no channel matches the leadfield, the Current Density Norm is unavailable
    and the decision is made on the remaining five features (its term is dropped
    from both the weight vector and the bias is left unchanged).
    """
    feats = mara_features(ica, inst)
    info = inst.info
    _, common = _get_m100(list(info["ch_names"]))
    if common:
        g = feats @ MARA_W + MARA_B
    else:  # CDN missing -> drop that term
        g = feats[:, 1:] @ MARA_W[1:] + MARA_B
    return sorted(int(i) for i in np.where(g <= 0.0)[0])


def mara_artifact_prob(ica, inst) -> np.ndarray:
    """Posterior probability of being an artifact for each IC.

    MARA's ``info.posterior_artefactprob``: ``1 / (1 + exp(g))``.
    """
    feats = mara_features(ica, inst)
    _, common = _get_m100(list(inst.info["ch_names"]))
    g = feats @ MARA_W + MARA_B if common else feats[:, 1:] @ MARA_W[1:] + MARA_B
    return 1.0 / (1.0 + np.exp(g))
