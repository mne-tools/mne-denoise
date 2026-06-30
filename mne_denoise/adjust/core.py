"""ADJUST automatic EEG IC artifact classifier (Mognon et al. 2011, DOI 10.1111/j.1469-8986.2010.01061.x).

Flags ICA components as blink / horizontal-eye / generic-discontinuity artifacts from spatial
features (SAD spatial average difference, SED spatial eye difference, GDSF generic discontinuity)
and temporal features (MEV maximum epoch variance, TK temporal kurtosis), each thresholded by an
EM two-Gaussian split. The classic feature-based ocular ICA classifier (predecessor of ICLabel).
Operates on epoched data.
"""

import numpy as np


def _bimodal_threshold(x):
    """EM two-Gaussian split of feature values across components; artifact = above the higher
    mode. Returns +inf when the feature is not separable (so nothing is flagged on it)."""
    from sklearn.mixture import GaussianMixture

    x = np.asarray(x, dtype=float)
    if x.size < 4 or np.ptp(x) < 1e-12:
        return np.inf
    gm = GaussianMixture(2, random_state=0).fit(x.reshape(-1, 1))
    mu = gm.means_.ravel()
    grid = np.linspace(mu.min(), mu.max(), 256).reshape(-1, 1)
    above = grid[gm.predict(grid) == int(np.argmax(mu))]
    return float(above.min()) if above.size else float(mu.mean())


def adjust_bad_components(ica, epochs):
    """Return the sorted list of artifact IC indices per ADJUST (Mognon et al. 2011)."""
    import mne
    from scipy.stats import kurtosis

    info = epochs.info
    picks = mne.pick_types(info, eeg=True)
    pos = np.array([info["chs"][i]["loc"][:3] for i in picks], dtype=float)
    comps = np.abs(ica.get_components())                  # (n_ch, n_ic) |scalp topographies|
    src = ica.get_sources(epochs).get_data()             # (n_epochs, n_ic, n_times)
    n_ic = comps.shape[1]

    x, y = pos[:, 0], pos[:, 1]                           # left-right, anterior-posterior
    front, back = y > np.percentile(y, 70), y < np.percentile(y, 30)
    left, right = x < np.percentile(x, 30), x > np.percentile(x, 70)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    nn = np.argsort(d, axis=1)[:, 1:6]                    # 5 nearest neighbours per channel

    SAD = comps[front].mean(0) - comps[back].mean(0)                       # blink: frontal-dominant
    SED = np.abs(comps[left].mean(0) - comps[right].mean(0))              # horizontal eye: L/R asym
    GDSF = np.max([np.abs(comps[i] - comps[nn[i]].mean(0)) for i in range(len(pos))], axis=0)
    ev = src.var(axis=2)                                                   # (n_epochs, n_ic)
    MEV = ev.max(0) / (ev.mean(0) + 1e-12)                                # transient burst
    TK = np.array([float(np.mean(kurtosis(src[:, k, :], axis=1))) for k in range(n_ic)])

    tSAD, tTK, tSED, tMEV, tGDSF = (_bimodal_threshold(f) for f in (SAD, TK, SED, MEV, GDSF))
    bad = set()
    for k in range(n_ic):
        if SAD[k] > tSAD and TK[k] > tTK:
            bad.add(k)                                   # blink
        if SED[k] > tSED and MEV[k] > tMEV:
            bad.add(k)                                   # horizontal eye movement
        if GDSF[k] > tGDSF and MEV[k] > tMEV:
            bad.add(k)                                   # generic discontinuity
    return sorted(bad)
