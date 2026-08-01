# Lagged CCA (reference-free BSS-CCA)

`mne_denoise.cca` implements reference-free canonical-correlation blind source
separation for broadband muscle-artifact attenuation (De Clercq et al., 2006).
It solves CCA between the multichannel signal and a delayed copy of that signal,
then orders components by lagged correlation.

The lag is part of the operating point. It must be stated explicitly in samples
or seconds; the API never inserts an undocumented one-sample lag.

```python
from mne_denoise.cca import LaggedCCA

# Sample-domain declaration for an array.
estimator = LaggedCCA(lag_samples=1, rho_threshold=0.9)
estimator.fit(train_data)
cleaned = estimator.transform(evaluation_data)

# Equivalent physical-time declaration for MNE data at 250 Hz.
cleaned_raw = LaggedCCA(lag_seconds=0.004).fit_transform(raw)
```

For one-shot array use, `compute_lagged_cca` returns both the cleaned data and
diagnostics:

```python
from mne_denoise.cca import compute_lagged_cca

cleaned, diagnostics = compute_lagged_cca(
    data,
    lag_seconds=0.004,
    sfreq=250.0,
    rho_threshold=0.9,
)
```

For Epochs, lagged pairs are formed within each epoch. No synthetic pair is
created across epoch boundaries. `fit` and `transform` remain separate so a
fixed operator can be evaluated without train/evaluation leakage.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `lag_samples` | Positive lag in samples; mutually exclusive with `lag_seconds`. |
| `lag_seconds` | Positive physical lag; requires MNE sampling metadata or `sfreq`. |
| `sfreq` | Sampling frequency for NumPy data when using `lag_seconds`. |
| `rho_threshold` | Retain components whose lagged correlation meets this threshold. |
| `n_keep` | Retain exactly this many leading components instead of thresholding. |

The method assumes broadband artifacts have lower short-lag correlation than
the neural activity to preserve. That assumption is regime-dependent: a high-
frequency neural target or a temporally structured artifact can reverse the
ordering. Report attenuation and neural-preservation endpoints together, and
freeze the lag and selection rule before evaluation.

## Reference

De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., & Van Huffel, S.
(2006). Canonical correlation analysis applied to remove muscle artifacts from
the electroencephalogram. *IEEE Transactions on Biomedical Engineering*,
53(12), 2583–2587. https://doi.org/10.1109/TBME.2006.879459
