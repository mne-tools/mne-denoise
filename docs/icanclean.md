# iCanClean

iCanClean uses canonical correlation between a primary recording and a
physical or pseudo-reference recording to remove shared variance. The
reference can be built from selected channels or from a supplied reference
array {footcite:p}`downey_ferris2022_icanclean,downey_ferris2023_icanclean_phantom`.

## Usage

```python
from mne_denoise.icanclean import ICanClean

model = ICanClean(sfreq=250.0, mode="pseudo")
clean = model.fit_transform(data)  # reference construction is mode-dependent
```

## Key points

- The primary input and reference must have matching observations. The
  functional API accepts channel-first continuous arrays; the estimator also
  supports its documented MNE containers.
- Modes control how the reference is constructed. A physical reference uses
  supplied channels; pseudo-reference modes derive reference channels from the
  primary data and filtering settings.
- Thresholds can use an explicit r2_threshold or the circular-shift null
  option. null_r2_threshold returns a quantile of the maximum squared CCA
  correlation across surrogate shifts; a high shared correlation does not by
  itself identify artifact.
- fit is a compatibility no-op for this transductive estimator; cleaning is
  estimated during transform/fit_transform.

## References

```{footbibliography}
```
