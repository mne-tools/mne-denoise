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

dss = DSS(bias=bias, n_components=5)
dss.fit(data)
alpha_sources = dss.transform(data)

# Example: Remove line noise
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

# High-level API (sklearn-style)
dss = DSS(bias=my_bias_function, n_components=5)
dss.fit(data)
sources = dss.transform(data)
reconstructed = dss.inverse_transform(sources[:3])  # Keep top 3
```

### Iterative (Nonlinear) DSS

For nonlinear source separation using fixed-point iteration:

```python
from mne_denoise.dss import IterativeDSS, KurtosisDenoiser

denoiser = KurtosisDenoiser(nonlinearity='tanh')
it_dss = IterativeDSS(denoiser, n_components=5, max_iter=100)
it_dss.fit(data)
sources = it_dss.transform(data)
```

### DSS-ZapLine (Line Noise Removal)

Remove 50/60 Hz line noise and harmonics:

```python
from mne_denoise.zapline import ZapLine, dss_zapline_plus

# Clean line noise (fixed frequency)
est = ZapLine(line_freq=50, sfreq=500, n_remove='auto')
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
| `TrialAverageBias` | Evoked responses     | Epoch averaging for phase-locked signals |
| `BandpassBias`     | Rhythm extraction    | Narrow-band filter for oscillations      |
| `NotchBias`        | Line noise isolation | Isolate specific frequency               |
| `CycleAverageBias` | Artifact removal     | Cycle-locked averaging for ECG/blinks    |

Example usage:

```python
from mne_denoise.dss import TrialAverageBias, CycleAverageBias

# Evoked response enhancement
epochs_data = np.random.randn(64, 200, 100)  # channels x times x epochs
bias = TrialAverageBias()

# ECG artifact removal
r_peaks = find_r_peaks(ecg_signal)
bias = CycleAverageBias(event_samples=r_peaks, window=(-0.1, 0.3), sfreq=500)
```

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

data_clean = interpolate_bad_channels(data, bad_mask, method='spline')
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
epochs_clean = apply_dss_to_epochs(epochs, bias='evoked', n_components=5)

# Remove line noise from raw
raw_clean = apply_zapline_to_raw(raw, line_freq=50)

# Extract DSS components for visualization
info = get_dss_components(epochs, bias='alpha', n_components=10)
```

## Algorithm Details

### Linear DSS Algorithm

Following NoiseTools `nt_dss0.m`:

1. Compute baseline covariance: `C0 = X @ X.T / n`
2. Compute biased covariance: `C1 = f(X) @ f(X).T / n`
3. PCA whitening from C0: `W = V @ diag(1/sqrt(λ))`
4. Apply whitening to C1: `C2 = W' @ C1 @ W`
5. Eigendecomposition of C2: `[V2, Λ2] = eig(C2)`
6. DSS filters: `todss = V @ W @ V2`
7. Normalize for unit variance

### Iterative DSS Algorithm

Following Särelä & Valpola (2005):

1. Initialize weight vector `w`
2. Compute source: `s = w' @ X`
3. Apply nonlinear function: `s' = f(s)`
4. Update: `w_new = X @ s' / n`
5. Normalize: `w = w_new / ||w_new||`
6. Repeat until convergence

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
