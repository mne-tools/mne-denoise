# Multi-channel Wiener Filter (MWF)

## Overview

The `mne_denoise.mwf` module implements the **multi-channel Wiener filter (MWF)**
(Somers, Francart & Bertrand 2018), a generic, reference-free spatial artifact
cleaner and the spatial-filter core of the **RELAX** pipeline (Bailey et al.
2023).

With the recording split into artifact-present and artifact-free segments, the
clean signal is recovered by

```
X_clean = R_clean · R_artifact⁻¹ · X
```

where `R_artifact` is the covariance over artifact segments (signal + artifact)
and `R_clean` the covariance over artifact-free segments (signal only). During
clean segments the two covariances coincide, so the filter is ~identity (no
over-cleaning); during artifact segments it projects out the artifact subspace.
No reference channel is required — artifact segments are marked from broadband
high-frequency power (or a caller-supplied mask).

> **Note.** MWF is a *general* cleaner, not an artifact-specific method. Because
> its clean/artifact split is driven by broadband HF power, it can attenuate
> genuine neural high-frequency activity when the two covariances are poorly
> separated. Validate preservation of the band of interest on your data. It is
> provided here as the well-cited RELAX-core building block.

## Quick Start

```python
import numpy as np
from mne_denoise.mwf import MWF

data = np.random.randn(32, 20000)  # 32 channels, 20000 samples

# Leakage-safe estimator API: learn the operator on train, apply to eval.
est = MWF(sfreq=250.0)
est.fit(data)
cleaned = est.transform(data)
print(f"fit on {est.artifact_fraction_:.0%} artifact samples")
```

Supplying an explicit artifact mask (no HF detector, no `sfreq` needed):

```python
cleaned = MWF().fit_transform(data, mask=my_artifact_mask)
```

With MNE-Python objects the sampling frequency is read from `info`:

```python
raw_clean = MWF().fit_transform(raw)
```

## One-shot functional API

```python
from mne_denoise.mwf import compute_mwf, hf_power_mask, mwf_filter

cleaned, info = compute_mwf(data, sfreq=250.0)   # (cleaned, {mask, artifact_fraction})
mask = hf_power_mask(data, sfreq=250.0)           # broadband HF artifact detector
cleaned = mwf_filter(data, mask)                  # raw Wiener filter given a mask
```

## Parameters

| Parameter  | Description                                                            |
| ---------- | --------------------------------------------------------------------- |
| `sfreq`    | Sampling frequency (Hz). Optional for MNE input / when a mask is given. |
| `hf_hz`    | High-pass cutoff (Hz) for the HF artifact detector.                    |
| `quantile` | HF-power quantile above which samples are flagged as artifact.         |
| `reg`      | Diagonal-loading factor for covariance invertibility.                 |

## References

1. Somers, B., Francart, T., & Bertrand, A. (2018). A generic EEG artifact
   removal algorithm based on the multi-channel Wiener filter. _Journal of Neural
   Engineering_, 15(3), 036007. https://doi.org/10.1088/1741-2552/aaac92
2. Bailey, N. W., et al. (2023). RELAX: An automated pre-processing pipeline for
   cleaning EEG data — Part 1. _Clinical Neurophysiology_, 149, 178-201.
   https://doi.org/10.1016/j.clinph.2023.01.007
