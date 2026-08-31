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

## References

```{footbibliography}
```
