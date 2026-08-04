# Multi-channel Wiener filtering

`mne_denoise.mwf.MultichannelWienerFilter` implements the zero-delay GEVD
multi-channel Wiener filter described by Somers, Francart, and Bertrand (2018).
`MWF` is a short compatibility alias of the same class.

MWF is semi-supervised. It needs examples of both artifact-present and clean EEG
to estimate their covariance matrices. The core algorithm does not discover
artifacts automatically, and the origin of the mask is part of the scientific
operating point.

## Explicit-mask workflow

Mask values have precise meanings:

- `1`: artifact-present training sample;
- `0`: clean training sample;
- `NaN`: ignored, or reassigned through `treat_nan`.

```python
from mne_denoise.mwf import MultichannelWienerFilter

mwf = MultichannelWienerFilter(rank="positive")
mwf.fit(train_raw, artifact_mask=train_mask)
cleaned = mwf.transform(eval_raw)
```

`transform()` uses only the frozen spatial operator. It does not inspect the
evaluation recording or create a new mask.

For epoched input, a mask can have shape `(n_epochs, n_times)` or be a flat
epoch-major vector. MNE channel names are aligned at transform time, and channels
not selected for fitting are preserved unchanged.

## Independent clean reference

A separate clean recording can supply the clean covariance. If no mask is
provided, all samples in `X` train the artifact-present covariance:

```python
mwf.fit(artifact_training_raw, clean_reference=clean_reference_raw)
```

The training and reference data must have the same channels, channel scaling,
physical units, and—when both are MNE objects—the same sampling frequency.

## Optional high-frequency mask authoring

`hf_power_mask()` is a convenience heuristic, not part of the reference MWF
algorithm. It can be used explicitly:

```python
from mne_denoise.mwf import hf_power_mask

mask = hf_power_mask(train_data, sfreq=250.0, hf_hz=20.0, quantile=0.7)
mwf.fit(train_data, artifact_mask=mask)
```

Or requested as an explicit estimator strategy:

```python
mwf = MultichannelWienerFilter(
    mask_strategy="hf_power",
    sfreq=250.0,
    hf_hz=20.0,
    quantile=0.7,
)
mwf.fit(train_data)
```

This detector can label genuine high-frequency neural activity as artifact. Its
cutoff, quantile, and smoothing duration must therefore be validated for the
acquisition regime.

## GEVD rank and diagnostics

The default `rank="positive"` matches the reference MATLAB toolbox's `poseig`
setting: only positive artifact eigenvalues are retained. `rank="full"` applies
the full-rank covariance-ratio filter, and an integer retains that many leading
GEVD directions.

After fitting, the estimator exposes:

- `generalized_eigenvalues_` and `artifact_eigenvalues_`;
- `selected_components_`;
- `artifact_mask_` and `artifact_fraction_`;
- `fit_diagnostics_`, including sample counts, covariance ranks, and the actual
  diagonal loading.

Relative diagonal loading makes the operator invariant to a shared global unit
rescaling. It cannot correct channel-specific unit mismatches.

## Evidence boundary

This implementation is derived from the authors' public MATLAB equations and
locks internal invariants such as full-rank covariance-ratio equivalence. It does
not yet claim external numerical parity with the MATLAB toolbox or validated
performance for a particular acquisition regime. The reference toolbox also
supports temporal delay embedding; this implementation currently covers the
zero-delay method only.

## References

1. Somers, B., Francart, T., & Bertrand, A. (2018). A generic EEG artifact
   removal algorithm based on the multi-channel Wiener filter. *Journal of
   Neural Engineering*, 15(3), 036007. https://doi.org/10.1088/1741-2552/aaac92
2. Authors' MATLAB implementation:
   https://github.com/exporl/mwf-artifact-removal
