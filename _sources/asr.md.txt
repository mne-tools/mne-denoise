# Artifact Subspace Reconstruction (ASR)

ASR estimates a reference covariance from relatively clean calibration data,
detects windows whose component variance exceeds that reference, and
reconstructs the affected subspace. It is intended for transient,
high-variance artifacts in continuous EEG
{footcite:p}`kothe_jung2016_asr,chang2018_asr`.

## Usage

```python
from mne_denoise.asr import ASR

asr = ASR(sfreq=250.0, cutoff=20.0)
clean = asr.fit_transform(data)  # data: (n_channels, n_times)
```

For supported MNE objects, pass the object directly. `fit` calibrates the
state, `transform` applies it without mutating the input, and
`fit_transform` composes both operations.

## Key points

- Calibration should represent the clean covariance of the processed channel
  type. `calibration="auto"` selects windows from robust RMS statistics; an
  explicit calibration mask or array can supply a trusted period.
- `cutoff` is a multiplier in the calibrated component space. Lower values
  generally reconstruct more components, but its effect depends on calibration
  and processing settings.
- `method="standard"` is the default. `"riemannian_windowed"` uses the
  windowed Riemannian backend; `"riemannian"` requires
  `experimental=True` {footcite:p}`blum2019_riemannian_asr`.
- `AdaptiveASR` updates calibration state between chunks;
  `JugglerASR` changes reference-sample selection
  {footcite:p}`tsai2024_adaptive_asr,kim2025_juggler_asr`.
- `GuidedASR` adds artifact and preserve covariance guidance. Soft guided
  reconstruction is experimental and requires explicit `experimental=True`.
- Inspect the fitted diagnostics and repaired spans. Artifact attenuation alone
  does not establish preservation of the signal of interest.

## References

```{footbibliography}
```
