# ZapLine

ZapLine combines period-locked smoothing with DSS to remove power-line noise
and its harmonics {footcite:p}`decheveigne2020_zapline`. The adaptive path
extends this workflow with frequency detection, segmentation, and
per-segment quality checks in the ZapLine-plus direction
{footcite:p}`klug_kloosterman2022_zapline_plus`.

## Usage

```python
from mne_denoise.zapline import ZapLine

model = ZapLine(sfreq=1000.0, line_freq=50.0, n_select="auto")
clean = model.fit_transform(raw)
```

## Key points

- line_freq is the fundamental in hertz; n_harmonics controls the targets in
  the line-noise bias.
- n_select is a component count or "auto", not a percentage of removed power.
  nfft, rank, nkeep, and reg affect the fitted DSS operation.
- Standard mode uses a fitted operator with fit/transform. Adaptive mode
  requires fit_transform because filters are fitted per segment.
- In adaptive mode, line_freq=None enables configured frequency detection;
  segmentation, local peak refinement, spectral QA, and optional hybrid cleanup
  are controlled by adaptive_params.
- NumPy input is (n_channels, n_times) or
  (n_epochs, n_channels, n_times). MNE Raw, Epochs, and Evoked inputs are
  supported and copied.

A reduction at the line frequency can also attenuate an overlapping neural
rhythm. Inspect the fitted component and segment diagnostics.

## Number of removed components

ZapLine splits the data into a part smoothed over one line period, which keeps
no line, and a residual, and removes DSS components from the residual only. The
residual also holds most of the activity above about half the line frequency,
so each removed component lowers broadband power in its spatial direction as
well as the line.

In standard mode, `n_select="auto"` takes the larger of two counts on the DSS
scores (`eigenvalues_`): the scores that stand out as outliers, and the
components above the largest drop between consecutive scores within the top
half of the scores. When both counts are zero but a
line peak is detected, ZapLine evaluates 1, 2, ... components and keeps the
first whose spectrum passes its spectral check; when none passes, it keeps the
last count the check rated too weak, or every component.

## Line noise from many sources

When each sensor picks up its own line noise, for example through poor
contacts, the line spreads over many DSS components with similar scores:
several large `eigenvalues_` that decay smoothly instead of one or two that
stand out. ZapLine then needs most components to remove the line and lowers
broadband power accordingly. Check `n_removed_` against the number of
channels; to bound the broadband cost, set `n_select` to a small count and
remove what is left channel by channel with
{class}`~mne_denoise.spectrum_interpolation.SpectrumInterpolation` or a notch
filter.

## Subtracting only the line band

The removed components also carry broadband activity, so subtracting them
whole can raise power away from the line on single channels. With
`line_bandwidth`, an mne-denoise extension, only their part within
`line_bandwidth` Hz of each targeted harmonic is subtracted, and the spectrum
outside these bands is unchanged:

```python
model = ZapLine(sfreq=200.0, line_freq=60.0, line_bandwidth=1.0)  # 59-61 Hz
```

ZapLine selects the same components as without it. A line broader than the
band is removed less completely. On short inputs, or epochs that are not
contiguous, part of the line falls outside the band and is not removed; apply
it to the continuous recording before epoching.

## References

```{footbibliography}
```
