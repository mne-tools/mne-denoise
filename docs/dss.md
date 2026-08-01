# DSS Module Documentation

## Overview

The `mne_denoise.dss` module provides a comprehensive implementation of **Denoising Source Separation (DSS)** algorithms for M/EEG signal processing. DSS is a powerful spatial filtering technique that finds linear projections maximizing a criterion of interest (the "bias").

## Quick Start

```python
import numpy as np
from mne_denoise.dss import DSS, BandpassBias
from mne_denoise.zapline import ZapLine

# Example: Extract alpha rhythm
data = np.random.randn(64, 10000)  # 64 channels, 10000 samples
bias = BandpassBias(freq_band=(8, 12), sfreq=500)

dss = DSS(bias=bias, n_components=5, component_action="extract")
alpha_sources = dss.fit_transform(data)

# Retain the leading alpha component in sensor space
alpha_only = DSS(
    bias=bias,
    n_components=5,
    component_action="retain",
    component_selection=1,
).fit_transform(data)

# Example: Remove line noise
est = ZapLine(line_freq=50, sfreq=500)
est.fit(data)
cleaned_data = est.transform(data)
```

## Core Components

### Linear DSS

The core linear DSS algorithm maximizes the ratio of biased variance to baseline variance.

```python
from mne_denoise.dss import compute_dss, DSS

# Low-level API
filters, patterns, eigenvalues, explained_var = compute_dss(
    data, biased_data, n_components=5
)

# High-level API (sklearn-style source extraction)
dss = DSS(
    bias=my_bias_function,
    n_components=5,
    component_action="extract",
)
dss.fit(data)
sources = dss.transform(data)
reconstructed = dss.inverse_transform(sources[:3])  # Keep top 3

# Explicit artifact subtraction; fit_transform is exactly fit().transform()
cleaner = DSS(
    bias=artifact_bias,
    n_components=5,
    component_action="subtract",
    component_selection="auto",
)
cleaned = cleaner.fit_transform(data)
```

### Iterative (Nonlinear) DSS

For nonlinear source separation using fixed-point iteration:

```python
from mne_denoise.dss import IterativeDSS, KurtosisDenoiser

denoiser = KurtosisDenoiser(nonlinearity="tanh")
it_dss = IterativeDSS(denoiser, n_components=5, max_iter=100)
it_dss.fit(data)
sources = it_dss.transform(data)
```

### DSS-ZapLine (Line Noise Removal)

Remove 50/60 Hz line noise and harmonics:

```python
from mne_denoise.zapline import ZapLine, dss_zapline_plus

# Clean line noise (fixed frequency)
est = ZapLine(line_freq=50, sfreq=500, n_remove="auto")
est.fit(data)
cleaned = est.transform(data)

# Adaptive cleaning (ZapLine-plus)
result = dss_zapline_plus(data, sfreq=500)

# Check metrics
print(f"Power reduction: {est.n_removed_} components")
```

## Bias Functions (Denoisers)

### Linear Biases

| Class              | Use Case             | Description                              |
| ------------------ | -------------------- | ---------------------------------------- |
| `AverageBias`      | Evoked responses     | Epoch averaging for phase-locked signals |
| `BandpassBias`     | Rhythm extraction    | Narrow-band filter for oscillations      |
| `NotchBias`        | Line noise isolation | Isolate specific frequency               |
| `CycleAverageBias` | Artifact removal     | Cycle-locked averaging for ECG/blinks    |
| `LagAveragingBias` | Predictable signals  | Average explicitly selected sample lags  |

Example usage:

```python
import mne

from mne_denoise.dss import (
    AverageBias,
    CycleAverageBias,
    DSS,
    LagAveragingBias,
)

# Evoked response enhancement
epochs_data = np.random.randn(64, 200, 100)  # channels x times x epochs
bias = AverageBias()

# ECG artifact removal from MNE Raw events. MNE event positions use the
# acquisition coordinate system, so declare the origin and offset explicitly.
events, _, _ = mne.preprocessing.find_ecg_events(raw)
bias = CycleAverageBias(
    event_samples=events[:, 0],
    window=(-0.1, 0.3),
    window_unit="seconds",
    sfreq=raw.info["sfreq"],
    event_origin="raw",
    first_samp=raw.first_samp,
)

# Extract components emphasized by a bias-side average of selected lags.
lag_bias = LagAveragingBias(shifts=[1, 2, 5, 10])
lag_dss = DSS(bias=lag_bias, n_components=5, component_action="extract")
```

`LagAveragingBias` and `lag_averaging_dss` are the canonical names for the
released bias-side operation. `TimeShiftBias` and `time_shift_dss` remain as
0.x compatibility names and emit `FutureWarning`. They are not true
time-shift DSS: the `TimeShiftDSS` name is reserved for a future estimator that
augments the data with lagged copies and learns spatiotemporal filters.

### Experimental cardiac DSS recipe

`CardiacDSS` is a narrowly scoped recipe for fitting linear DSS with a
QRS-synchronized `CycleAverageBias`. It requires explicit event coordinates,
coordinate units, event origin, component action, and component count:

```python
from mne.preprocessing import find_ecg_events
from mne_denoise.dss import CardiacDSS

events, _, _ = find_ecg_events(raw)
cardiac = CardiacDSS(
    qrs_events=events[:, 0],
    event_unit="samples",
    event_origin="raw",
    first_samp=raw.first_samp,
    window=(-0.2, 0.4),
    window_unit="seconds",
    component_action="subtract",
    component_selection=1,
)
cleaned = cardiac.fit_transform(raw)
print(cardiac.get_diagnostics().to_dict())
```

For MNE Epochs or channel-first 3-D NumPy data, events are
`(epoch_index, coordinate_within_epoch)` pairs. The epoch index is zero-based;
the coordinate uses `event_unit` and zero means the first sample stored in that
epoch. Windows crossing an epoch boundary are excluded instead of being allowed
to borrow samples from a neighboring epoch. Events are consumed by `fit` only;
`transform` applies the fitted spatial filters without reusing event positions.

If too few complete windows remain, subtraction safely abstains and returns an
exact copy. Retention and extraction are marked inadmissible because their output
cannot be defined without a fit. `min_valid_events` is a user policy, not a
validated scientific threshold. Automatic component selection is intentionally
not exposed by this recipe; `component_selection` is an explicit integer.

> **Experimental and unvalidated:** these contracts and synthetic unit tests do
> not demonstrate artifact attenuation or neural-signal preservation on real
> M/EEG data. Do not treat availability of this estimator as evidence of support
> for a recording regime.

### Nonlinear Biases

| Class                        | Use Case               | Description                     |
| ---------------------------- | ---------------------- | ------------------------------- |
| `VarianceMaskDenoiser`       | Transient detection    | Emphasize high-variance regions |
| `KurtosisDenoiser`           | Super-Gaussian sources | Maximize kurtosis               |
| `TemporalSmoothnessDenoiser` | Slow sources           | Emphasize temporal smoothness   |

## Preprocessing Utilities

### Bad Channel Detection

```python
from mne_denoise.dss import detect_bad_channels, interpolate_bad_channels

bad_mask, details = detect_bad_channels(data, z_threshold=3.5)
print(f"Bad channels: {np.where(bad_mask)[0]}")

data_clean = interpolate_bad_channels(data, bad_mask, method="spline")
```

### Robust DSS

Automatic bad channel/segment handling:

```python
from mne_denoise.dss import RobustDSS

rdss = RobustDSS(
    bias=my_bias,
    n_components=5,
    detect_bad_channels=True,
    detect_bad_segments=True,
)
rdss.fit(data, sfreq=500)

print(f"Excluded {rdss.bad_channels_.sum()} channels")
sources = rdss.transform(data)
```

## MNE-Python Integration

When MNE-Python is installed, additional functions are available:

```python
from mne_denoise.dss import apply_dss_to_epochs, apply_zapline_to_raw

# Enhance evoked responses in epochs
epochs_clean = apply_dss_to_epochs(epochs, bias="evoked", n_components=5)

# Remove line noise from raw
raw_clean = apply_zapline_to_raw(raw, line_freq=50)

# Extract DSS components for visualization
info = get_dss_components(epochs, bias="alpha", n_components=10)
```

## Algorithm Details

### Linear DSS Algorithm

Following NoiseTools `nt_dss0.m`:

1. Compute baseline covariance: `C0 = X @ X.T / n`
2. Compute biased covariance: `C1 = f(X) @ f(X).T / n`
3. Compute the baseline-covariance whitener: `W = diag(1/sqrt(λ)) @ V.T`
4. Apply the same transform to both axes of C1: `C2 = W @ C1 @ W.T`
5. Eigendecomposition of C2: `[V2, Λ2] = eig(C2)`
6. DSS filters: `todss = V2.T @ W`
7. Normalize for unit variance

This baseline-covariance whitening is an intrinsic part of the DSS generalized
eigenvalue solution. It is separate from the optional `whiten=True` sensor
pre-whitening, which balances mixed MNE channel types or applies a supplied
noise covariance before DSS computes its own baseline and biased covariances.

### Iterative DSS Algorithm

Following Särelä & Valpola (2005):

1. Center the data and whiten it from its empirical covariance
2. Initialize weight vector `w`
3. Compute source: `s = w' @ X_white`
4. Apply nonlinear function: `s' = f(s)`
5. Update: `w_new = X_white @ s' / n`
6. Normalize: `w = w_new / ||w_new||`
7. Repeat until convergence

## References

1. Särelä, J., & Valpola, H. (2005). Denoising Source Separation. _Journal of Machine Learning Research_, 6, 233-272.
2. de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering. _Journal of Neuroscience Methods_, 171(2), 331-339.
3. de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove power line artifacts. _NeuroImage_, 207, 116356.

## API Reference

See the docstrings of individual functions for detailed parameter descriptions:

```python
help(compute_dss)
help(DSS)
help(ZapLine)
help(IterativeDSS)
```
