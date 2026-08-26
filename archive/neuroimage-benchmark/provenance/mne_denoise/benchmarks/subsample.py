"""Montage-aware channel subsampling for the low-density / wearable robustness arm.

Pick a spatially-spread subset of EEG channels so a high-density recording can be re-scored
at decreasing channel counts. This makes the wearable-review claim empirical: source
separation (ICA/PCA/DSS) loses effectiveness as the channel count falls below ~8-32, while
ASR / wavelet / autoCCA are comparatively robust.

Farthest-point greedy sampling on 3-D electrode positions (spatially uniform coverage);
strided fallback when positions are missing. ``must_include`` force-keeps channels a metric
depends on (e.g. the ocular posterior ROI), which are seeded before the spread fill.
"""
from __future__ import annotations

import numpy as np


def _channel_positions(info, picks):
    pos = np.full((len(picks), 3), np.nan)
    for k, i in enumerate(picks):
        loc = np.asarray(info["chs"][i]["loc"][:3], dtype=float)
        if loc.shape == (3,):
            pos[k] = loc
    return pos


def farthest_point_channels(info, picks, n_keep, *, must_include=()):
    """Return a spatially-spread subset of ``picks`` (indices into ``info['chs']``) of length
    ``min(n_keep, len(picks))``. ``must_include`` channels (a subset of ``picks``) are kept
    first, then farthest-point greedy fills the rest on 3-D positions. Strided fallback if
    positions are unavailable.
    """
    picks = list(picks)
    must = [p for p in must_include if p in picks]
    if n_keep >= len(picks):
        return picks
    if n_keep <= len(must):
        return sorted(must)[:n_keep]
    pos = _channel_positions(info, picks)
    ok = np.isfinite(pos).all(axis=1) & ~np.all(pos == 0, axis=1)
    idx_of = {p: k for k, p in enumerate(picks)}
    if int(ok.sum()) < n_keep:  # not enough real positions -> strided fallback (keep must + stride)
        rest = [p for p in picks if p not in must]
        take = max(0, n_keep - len(must))
        if take and rest:
            sel = np.unique(np.linspace(0, len(rest) - 1, take).round().astype(int))
            rest = [rest[i] for i in sel]
        else:
            rest = []
        return sorted(set(must) | set(rest))
    chosen = list(must)
    if not chosen:  # seed at the channel nearest the centroid
        cand = np.where(ok)[0]
        chosen = [picks[int(cand[np.argmin(np.linalg.norm(pos[cand] - pos[cand].mean(0), axis=1))])]]
    while len(chosen) < n_keep:
        chosen_k = [idx_of[c] for c in chosen]
        d = np.min([np.linalg.norm(pos - pos[k], axis=1) for k in chosen_k], axis=0)
        d[~ok] = -np.inf
        d[chosen_k] = -np.inf
        chosen.append(picks[int(np.argmax(d))])
    return sorted(set(chosen))
