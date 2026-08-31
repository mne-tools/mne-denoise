# SSP-SIR

SSP-SIR combines an artifact signal-space projection with source-informed
reconstruction through a lead field. It targets high-variance muscle artifact
in TMS-evoked EEG {footcite:p}`mutanen2016_sspsir`.

## Usage

```python
from mne_denoise.sspsir import SSPSIR

model = SSPSIR(n_components=3, art_window=(0.005, 0.050))
clean = model.fit_transform(epochs)
```

## Key points

- n_components is an artifact-component count or a high-frequency variance
  fraction. M sets the source-informed reconstruction rank.
- art_window=(tmin, tmax) estimates the artifact subspace in that interval.
  blend="constant" applies the projected operator throughout; the default can
  use a time-local blend.
- An individual forward solution is preferred. Compatible EEG MNE input with a
  montage can use the spherical fallback when `forward` is omitted. MEG or
  mixed-channel MNE input and NumPy input require an explicit forward.
- NumPy input uses (n_channels, n_times) or
  (n_epochs, n_channels, n_times). Supported MNE containers are copied.
- The fitted lead field, artifact topographies, operators, kernel, and effective
  rank are available as diagnostics.

The projection can remove signal that shares the selected artifact subspace.
Check the lead-field reference and reconstruction rank when interpreting
evoked responses {footcite:p}`mutanen2022_source_artifact,mutanen2024_sspsir_simulation`.

## References

```{footbibliography}
```
