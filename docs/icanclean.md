# iCanClean

iCanClean uses canonical correlation between a primary recording and a
physical or pseudo-reference recording to remove shared variance. The
reference can be built from selected channels or from a supplied reference
array {footcite:p}`downey_ferris2022_icanclean,downey_ferris2023_icanclean_phantom`.

## Usage

```python
from mne_denoise.icanclean import ICanClean

model = ICanClean(
    sfreq=250.0,
    ref_channels=[6, 7],
)
clean = model.fit_transform(data)  # data: (n_channels, n_times)
```

## Key points

- The primary input and reference must have matching observations. The
  functional API accepts channel-first continuous arrays; the estimator also
  supports its documented MNE containers.
- Modes control the cleaning workflow: sliding, global, calibrated, and hybrid
  are supported. A physical reference uses supplied channels; pseudo-reference
  construction is enabled separately with pseudo_ref=True and requires a
  suitable filter_ref.
- Thresholds can use an explicit threshold or the circular-shift null
  option. null_r2_threshold returns a quantile of the maximum squared CCA
  correlation across surrogate shifts; a high shared correlation does not by
  itself identify artifact.
- fit is a compatibility no-op for this transductive estimator; cleaning is
  estimated during transform/fit_transform.

## References

```{footbibliography}
```
