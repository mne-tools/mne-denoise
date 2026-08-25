# iCanClean

`ICanClean` removes latent artifact subspaces that the primary channels share
with a reference block, using canonical correlation analysis. A component is
rejected when its squared canonical correlation exceeds a threshold:

```python
bad_mask = r2 >= threshold
```

That one line is the whole selection rule, and choosing `threshold` well is the
only thing standing between a useful cleaner and a silent no-op.

## The threshold is not a transferable number

`threshold` is an **absolute** squared canonical correlation, but the scale it
lives on is set by the data and by how the reference block is built. Measured on
one recording (ds004505, 120 scalp and 120 outward-facing noise channels):

| reference construction | max r² | median r² | components ≥ 0.25 |
|---|---|---|---|
| dual-layer (physical noise electrodes) | **0.024** | 0.0013 | 0 of 120 |
| pseudo-reference (band-stopped copy of the EEG) | **0.825** | 0.642 | 119 of 120 |

The same threshold means entirely different things in the two cases — the medians
differ by roughly 500×. A grid of `0.25 … 0.95` removes **exactly zero**
components in the dual-layer case, because the lowest value tested is already ten
times the highest achievable correlation. The same grid removes almost everything
in the pseudo-reference case.

The scale also moves *within* a single homogeneous cohort: across five subjects of
one dataset the achievable maximum ranged 0.023 → 0.078, a 3.4× spread.

**Practical consequence:** a threshold taken from a paper, or tuned on one
recording, should not be assumed to transfer. Check what your data can actually
reach before choosing one:

```python
from mne_denoise.icanclean import ICanClean

icc = ICanClean(sfreq=raw.info["sfreq"], ref_channels=noise_ch, threshold=0.7)
icc.fit_transform(raw)
print(icc.max_r2_)  # highest r2 actually observed, per window
print(icc.thresholds_)  # the threshold applied, per window
```

If `max_r2_` sits below `thresholds_`, the estimator was a pass-through and
`n_removed_` will be zero — not because the data was clean, but because the
threshold was unreachable.

## `threshold='null'` — let the data set the scale

Rather than supplying a constant, ask what r² *sampling noise alone* would
produce for the current window length and channel counts, and reject only what
exceeds it:

```python
icc = ICanClean(
    sfreq=raw.info["sfreq"],
    ref_channels=noise_ch,
    threshold="null",  # calibrated per window
    null_random_state=0,  # reproducible surrogates
)
```

The null is built by circularly shifting the reference block and recomputing the
spectrum, which preserves each channel's autocorrelation and power spectrum while
destroying cross-block alignment. (A plain sample permutation would destroy the
autocorrelation too, giving surrogates that cannot reach the correlations real
data reaches — an anticonservative null.)

This matters most where a fixed threshold fails silently. Canonical correlations
are upward-biased when a window is short relative to `n_primary + n_reference`; as
that ratio approaches 1, every correlation approaches 1 whether or not anything is
shared. Components falsely removed on independent data, 40 primary and 40
reference channels:

| n / (p + q) | `'null'` threshold | `'null'` | fixed 0.65 | fixed 0.85 |
|---|---|---|---|---|
| 1.2 | 0.999 | **0.20** | 20.4 | 14.4 |
| 2.0 | 0.992 | **0.12** | 16.8 | 10.6 |
| 10.0 | 0.904 | **0.00** | 11.4 | 2.6 |
| 300.0 | 0.124 | **0.04** | 0.0 | 0.0 |

The threshold self-adapts from 0.999 to 0.124 with no input. Note the second row:
a 2 s window at 250 Hz on a 120 + 120 montage gives a ratio of 2.08, and a fixed
`threshold=0.85` there removes about a quarter of the components from data with
nothing in common.

Safety does not come from timidity — with 0, 1, 3 and 8 genuinely shared
components injected, `'null'` recovers exactly 0, 1, 3 and 8.

**What it does not do.** `'null'` decides whether a component shares *real*
variance with the reference. It does not decide whether that variance is
*artifact*. With a pseudo-reference — a band-stopped copy of the primary block —
almost every component shares real variance, so `'null'` will select broadly
there. It solves the degeneracy problem, not the selectivity problem.

## Window length and conditioning

`segment_len` sets the window that is cleaned. `stats_segment_len`, when larger,
sets a wider window that the CCA is *estimated* on, while only the inner
`segment_len` is written back. This is how you get short, responsive correction
windows backed by statistically adequate estimates:

```python
icc = ICanClean(
    sfreq=250.0,
    ref_channels=noise_ch,
    mode="sliding",
    segment_len=2.0,  # corrected span
    stats_segment_len=32.0,  # estimation span
    threshold="null",
)
```

As a rule of thumb, keep `n_samples` at least ten times `n_primary + n_reference`
in whichever window the CCA is estimated on. `samples_per_variable_` reports the
achieved ratio.

## Operating modes

| mode | CCA decompositions | what it does |
|---|---|---|
| `'global'` | 1 | one decomposition on the whole recording, subtracted once |
| `'sliding'` | one per window | fresh decomposition per window, overlap-added |
| `'calibrated'` | 1 | one global decomposition, reused for window-local scoring |
| `'hybrid'` | 1 + one per window | a global pass, then a sliding pass on its output |

All four are **batch**: each sees the whole recording before returning cleaned
data, and none updates its decomposition sample by sample.

Which is best is **artifact-dependent**, not universal. On a stationary artifact a
long window exploits more data; on a non-stationary one a short window tracks the
change. Measured on real recordings, best operating point at 90% alpha retention:

| artifact | `'global'` | `'sliding'` 4 s |
|---|---|---|
| ocular (blink, n = 40) | **89.8%** removed | 61.3% |
| cardiac (n = 8) | 37.8% | **41.6%** |

`'calibrated'` calibrates on the same data it cleans — there is no separate
calibration recording — and costs roughly one decomposition rather than one per
window. Note its `correlations_` are window-local Pearson correlations squared,
not canonical correlations squared, and are **not** sorted descending, so row *k*
is not comparable with the other modes.

`'hybrid'` is an mne-denoise extension, not part of the published algorithm; the
reference implementation applies iCanClean exactly once. It is motivated by the
authors' open question about whether "incorporating larger windows of data"
helps — pass 1 estimates on the whole recording, pass 2 then tracks what is left.
Current evidence is suggestive but underpowered: across 40 subjects it removed
more artifact than the best single-pass mode at every preservation floor, but
every confidence interval included zero.

```{note}
`'hybrid'` is a **two-pass batch** refinement. It is not an online or recursive
estimator: both passes see the entire recording, and neither updates its
decomposition as samples arrive.

The 2023 paper raises two distinct possibilities in one sentence — using larger
windows, and computing CCA *recursively*. Only the first is addressed here. A
recursive formulation, in the sense of moments updated incrementally for causal
streaming use, is a different design on a different axis, and nothing on this
page should be read as evaluating one.
```

## Reference modes

The reference block can be physical noise electrodes (`ref_channels`), or derived
from the EEG itself with `pseudo_ref=True` plus a `filter_ref` band-stop that
keeps out-of-band drift and EMG while removing the brain band.

**iCanClean performs as well as its reference observes the artifact you are
scoring.** A mechanically-coupled outward-facing noise layer tracks head *motion*;
it is largely blind above ~12 Hz. Scoring it against broadband muscle asks it to
remove something it cannot see. On treadmill walking, where the artifact is
gait-locked motion, dual-layer raises the good-component count by ~23%; on a
whole-body sport scored against neck EMG, the same configuration removes nothing.

## References

- Downey & Ferris (2022), *The iCanClean Algorithm*, arXiv:2201.11798
- Downey & Ferris (2023), *Sensors* **23**(19):8214
- Gonsisko, Ferris & Downey (2023), *Sensors* **23**(2):928

```{note}
A public U.S. patent application has been filed for the iCanClean method
(US20230363718A1). Patent applications, and any resulting patents, may affect
commercial use.
```
