# Reference-free BSS-CCA

BSS-CCA separates channel data using canonical correlation with a lagged copy
of the signal. Components with low lagged correlation can be removed as a
muscle-artifact strategy without a reference channel
{footcite:p}`declercq2006_bss_cca,vergult2007_bss_cca,hotelling1936_cca`.

## Usage

```python
from mne_denoise.bss_cca import BSSCCA

model = BSSCCA(
    sfreq=250.0,
    lag_samples=1,
    n_remove=2,
)
clean = model.fit_transform(data)  # data: (n_channels, n_times)
```

## Key points

- Input is channel-first 2-D data or MNE Raw, Epochs, or Evoked where
  supported; fit learns operators and transform reuses them.
- Choose a selection rule: n_remove, rho_threshold, or slope_threshold.
  rho_threshold and slope_threshold can be combined; a component is then
  removed when either rule flags it. reject chooses whether the low- or
  high-correlation end of the canonical-correlation spectrum is treated as
  artifactual.
- slope_threshold removes the components whose log-log power spectrum within
  slope_band (default 7–75 Hz) rises more steeply than the threshold, the
  muscle signature shown on recordings under neuromuscular blockade
  {footcite:p}`fitzgibbon2016_emg_paralysis,janani2018_bss_cca_slope`. The
  slope is fitted as in the reference implementation of Janani et al.: Welch
  spectra with 1 s Hann segments, least squares on log power against log
  frequency. It needs a sampling frequency, and the threshold has no default.
- Remove line noise first, for example with ZapLine, or keep the line
  frequency out of slope_band: it flattens spectra and changes the selection.
- lag_samples or lag_seconds and preprocessing affect the
  canonical-correlation ordering.
- With MNE input, channels in `info["bads"]` are left out of the fit and pass
  through unchanged.
- Segmented operation fits one operator per block. Vergult et al. cleaned
  contiguous 10 s epochs independently
  {footcite:p}`vergult2007_bss_cca`; overlapping blocks and the cross-fade
  between them, in the middle of their overlap, are additions of this package.

## Published operating points

Janani et al. {footcite:p}`janani2018_bss_cca_slope` remove components in two
stages: correlation below 0.19, then slope above a threshold. Their values come
from specific recordings and should be validated before reuse:

| Parameter | Published value | Measured on |
|---|---|---|
| lag_seconds | 0.002 | 1 kHz data; the authors recommend at least 500 Hz |
| rho_threshold | 0.19 | one 64-channel recording of environmental noise (cap in water) |
| slope_threshold | -0.48 (conservative), -0.76, -0.98 | 6 participants before and under neuromuscular blockade, 115 channels |
| slope_band | 7–75 Hz | the default here |

The slope threshold depends on the decomposition: the same work used -0.31 for
Infomax ICA. With fewer channels, components mix brain and muscle more, so a
threshold may remove more brain activity; check that rhythms such as alpha
survive.

```python
model = BSSCCA(
    lag_seconds=0.002,  # Janani et al.; at least one sample
    rho_threshold=0.19,  # stage 1, validate on your data
    slope_threshold=-0.48,  # stage 2, validate on your data
    segment_len=10.0,
    overlap=0.5,
)
clean_raw = model.fit_transform(raw)  # sfreq is read from raw.info
```

The ordering is a signal-model assumption, not a universal artifact label.
Check the result against controls and the signal of interest.

## References

```{footbibliography}
```
