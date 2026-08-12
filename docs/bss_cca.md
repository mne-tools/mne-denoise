(bss_cca)=
# Reference-free BSS-CCA

The `mne_denoise.bss_cca` module implements blind source separation by
canonical correlation analysis (BSS-CCA) [^1], a reference-free method for
attenuating broadband muscle (EMG) artifacts.

CCA is solved between the recording $x(t)$ and a delayed copy
$y(t) = x(t - 1)$ of itself. The resulting components come out ordered by
decreasing lagged correlation. Muscle activity is a summation of asynchronous
motor-unit potentials, so it resembles temporally white noise and concentrates
in the **lowest** components; those are dropped and the remainder is projected
back to the sensors.

Unlike {class}`~mne_denoise.icanclean.ICanClean`, no reference channel is
needed — which is the point, since a good EMG reference is impractical to
record.

## Estimator API

```python
from mne_denoise.bss_cca import BSSCCA

model = BSSCCA(n_remove=3)
model.fit(training_data)
cleaned = model.transform(evaluation_data)
```

`fit()` learns a channel mean and a fixed spatial operator. `transform()`
always uses those, so a sample gets the same result whether it is transformed
alone, in a temporal chunk, or among other epochs. By default the fitted mean
is added back so the output keeps the input's offset; pass
`preserve_mean=False` for the mean-free reconstruction of Equation (7) in
[^1].

For Raw, Epochs, and Evoked inputs the estimator selects one homogeneous data
channel type and returns a copy of the same container, preserving timing,
annotations, events, epoch metadata, averaging information, bad-channel marks,
and unselected channels. Fit and transform must use the same channel names in
the same order.

## Choosing what to remove

Exactly one of `n_remove` or `rho_threshold` is required. There is no default,
deliberately:

```python
BSSCCA(n_remove=3)  # drop the 3 lowest-correlation components
BSSCCA(rho_threshold=0.75)  # keep components with correlation >= 0.75
```

`n_remove` is the operating knob used throughout [^1] — its figures remove 3,
7, 14, and 15 of the lowest-autocorrelated components — and the clinical
protocol in [^2] has a neurophysiologist choose the count per epoch. It is
also stable: it means the same thing when the fitted rank drops because a
channel was interpolated.

`rho_threshold` is a package convenience. [^1] describes autocorrelation
thresholding as unvalidated future work ("Further research will indicate
whether thresholding on the autocorrelation index will be sufficient"), and no
value generalizes. Real recordings produce a compressed correlation spectrum —
a genuine EMG component sits around 0.3–0.5, and the worked example in [^2]
reports 0.49 — so a high fixed threshold can reject essentially everything. If
no component reaches the threshold, the estimator logs a warning and removes
all of them rather than silently substituting a different operating point.

Sweep `n_remove` and look at where the result stops improving; that is the
procedure Figure 4 of [^1] uses.

## Preprocessing matters

Both source papers band-pass filter before decomposing — [^2] uses 0.3–35 Hz
plus a notch — and both use an average-referenced montage. Do the same.

The reason is not cosmetic. Canonical correlations are derived from singular
values and are therefore **non-negative**: a component dominated by energy near
the Nyquist frequency is *anti*-correlated at lag 1, yet it receives a large
positive correlation and ranks among the most "brain-like" components. Since
near-Nyquist energy is exactly what high-frequency EMG and line-noise leakage
look like, an unfiltered recording can rank artifact at the top of the
ordering.

`autocorrelations_` reports the **signed** lag-1 autocorrelation of each
component so you can see this directly:

```python
model = BSSCCA(n_remove=3).fit(raw)
aliased = model.autocorrelations_ < 0
```

Any `True` entry is a component whose correlation ranking is inverted relative
to its actual temporal structure. `filter_asymmetry_` gives a second check: it
is near zero when the two canonical filters agree, which is the condition under
which the canonical correlation can be read as an autocorrelation at all.

## Lag

The lag is one sample by default, which is what [^1] specifies. Declare it in
samples or in physical time:

```python
BSSCCA(n_remove=3)  # lag = 1 sample
BSSCCA(lag_samples=2, n_remove=3)
BSSCCA(lag_seconds=0.004, sfreq=250.0, n_remove=3)  # NumPy input
BSSCCA(lag_seconds=0.004, n_remove=3).fit(raw)  # Raw supplies sfreq
```

Pairs are truncated at the endpoints — never wrapped — and for epoched input
they are formed strictly within each epoch, so no pair spans an epoch boundary.

## Global and block-wise operation

By default one operator is learned for all the data. EMG is non-stationary,
though: the artifact topography changes with which muscle contracts. Every
application in [^1] is a single 10-second epoch, and [^2] repeats the procedure
"for every 10-s epoch of each EEG segment". Pass `segment_len` to reproduce
that:

```python
cleaned = BSSCCA(segment_len=10.0, n_remove=3).fit_transform(raw)
```

Blocks are contiguous and non-overlapping, matching the papers. `overlap` is a
package extension: a positive fraction blends neighbouring blocks with the
package's overlap-add helper, which smooths block boundaries at the cost of
departing from the published scheme.

A block-wise operator is *piecewise in time* — block $k$ applies to the samples
block $k$ was learned on — so `transform()` requires input with the same number
of samples as `fit()` saw. For epoched input, each epoch is already a block, so
`segment_len` is rejected.

## One-shot array API

`compute_bss_cca()` learns and applies in a single call and returns
diagnostics:

```python
from mne_denoise.bss_cca import compute_bss_cca

cleaned, info = compute_bss_cca(data, n_remove=3, sfreq=250.0)
info["correlations"]  # canonical correlations, descending
info["autocorrelations"]  # signed lag-1 autocorrelation
info["kept_mask"]  # retained components
info["input_rank"]  # < n_channels when the data is rank deficient
```

## Assumptions and limitations

BSS-CCA assumes the sources are mutually uncorrelated with differing
autocorrelation structure, that mixing is linear and instantaneous, and that
there are no more sources than sensors [^1]. In practice:

- **The ordering assumption is regime-dependent.** It rests on neural activity
  being more autocorrelated than muscle. A high-frequency neural target, or a
  temporally structured artifact, can invert it. Report artifact attenuation
  and neural preservation together, and freeze the lag and selection rule
  before evaluation.
- **Rank deficiency is handled but reported.** Average referencing, channel
  interpolation, and flat channels all reduce the rank; `input_rank_` tells you
  the number of components actually available.
- **Sample count matters.** Canonical correlations are biased upward when
  samples are scarce. Fewer lagged pairs than channels is rejected outright,
  and a thin margin is warned about.
- **Selection is not automated for you.** [^1] calls the method
  "user-dependent and semi-automatic" as applied in [^2].

## Not implemented

- Multiple time-lag / multi-set CCA.
- Automatic lag selection.
- Interactive per-block component scrolling (the GUI workflow of [^2]).

## References

[^1]: De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., &
    Van Huffel, S. (2006). Canonical correlation analysis applied to remove
    muscle artifacts from the electroencephalogram. *IEEE Transactions on
    Biomedical Engineering*, 53(12), 2583–2587.
    <https://doi.org/10.1109/TBME.2006.879459>

[^2]: Vergult, A., De Clercq, W., Palmini, A., Vanrumste, B., Dupont, P.,
    Van Huffel, S., & Van Paesschen, W. (2007). Improving the interpretation of
    ictal scalp EEG: BSS-CCA algorithm for muscle artifact removal.
    *Epilepsia*, 48(5), 950–958.
    <https://doi.org/10.1111/j.1528-1167.2007.01031.x>
