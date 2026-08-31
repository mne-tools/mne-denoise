# SOUND

SOUND estimates channel-specific noise using a lead field and iteratively
constructs a linear sensor-space cleaning operator
{footcite:p}`mutanen2018_sound,mutanen2022_source_artifact`.

## Usage

```python
from mne_denoise.sound import SOUND

model = SOUND(reference="best")
clean = model.fit_transform(raw)
```

Pass an MNE Raw, Epochs, or Evoked object directly. For NumPy input, provide a
lead field with matching channel order and use the corresponding array layout
documented by the API.

## Key points

- reference="best" selects a low-noise single-channel reference for the
  estimation and maps the result back to an average-referenced full-channel
  operator. reference="average" uses the all-channel solver and assumes
  average-referenced input.
- forward supplies an individual lead field. Without one, MNE input can use
  the spherical fallback derived from its montage; arrays require forward.
- lambda_, n_iter, and tol control regularization and convergence.
  sigma_source="evoked" estimates epoched noise from the trial average;
  "trials" uses concatenated trials.
- The fitted operator_, sigmas_, convergence_, and best_channel_ are useful
  diagnostics. The selected reference index is returned by
  compute_sound_ref_best.
- Re-referencing the data or lead field changes the fitted operation. Use the
  same channel order and reference for both.

SOUND reduces estimated sensor noise; evaluate any change to the signal of
interest separately.

## References

```{footbibliography}
```
