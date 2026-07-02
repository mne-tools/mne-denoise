# Auto-CCA (Reference-free BSS-CCA)

## Overview

The `mne_denoise.cca` module implements **reference-free canonical-correlation
blind source separation (BSS-CCA)**, also called *auto-CCA*, for muscle / EMG
artifact removal (De Clercq et al. 2006).

BSS-CCA runs canonical correlation analysis between the multichannel signal and
a **one-sample-lagged copy of itself**. This orders the recovered components by
*temporal autocorrelation*: slow, highly autocorrelated components (neural
rhythms) get high canonical correlations, while broadband, weakly autocorrelated
components (EMG / muscle) get low ones. Dropping the low-autocorrelation
components and reconstructing removes muscle activity while preserving neural
structure.

Auto-CCA is the **reference-free counterpart** to
[`ICanClean`](../mne_denoise/icanclean): both are CCA-based spatial cleaners, but
iCanClean cancels artifacts shared with dedicated reference channels, whereas
auto-CCA needs no reference and separates on autocorrelation alone.

## Quick Start

```python
import numpy as np
from mne_denoise.cca import AutoCCA

data = np.random.randn(32, 10000)  # 32 channels, 10000 samples

# Leakage-safe estimator API: learn on train, apply to eval.
est = AutoCCA(rho_threshold=0.9)
est.fit(data)
cleaned = est.transform(data)

print(f"kept {est.n_kept_} / removed {est.n_removed_} components")
```

With MNE-Python objects (a single homogeneous channel type is auto-picked):

```python
raw_clean = AutoCCA(rho_threshold=0.9).fit_transform(raw)
```

## One-shot functional API

For a quick clean of a single array (learns and applies in one call — use the
`AutoCCA` estimator instead when you need a train/evaluation split):

```python
from mne_denoise.cca import compute_autocca

cleaned, info = compute_autocca(data, rho_threshold=0.9)
print(info["correlations"])   # canonical autocorrelations, descending
print(info["n_removed"])      # components dropped as muscle
```

## Parameters

| Parameter       | Description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `rho_threshold` | Keep components whose autocorrelation is ≥ this value; drop the rest as EMG. |
| `n_keep`        | If set, keep exactly the top-`n_keep` components by autocorrelation.         |
| `verbose`       | MNE-style logging verbosity.                                                 |

## Algorithm Details

Given data `X` (channels × samples):

1. Form the one-sample-lagged copy `Y` of `X` (no wrap-around).
2. Solve CCA between `X` and `Y`, giving canonical filters `A` and canonical
   correlations `R` (the per-component autocorrelations), sorted descending.
3. Build a keep-mask (`R ≥ rho_threshold`, or the top-`n_keep`).
4. Form the channel-space cleaning operator
   `C = (A · diag(keep) · pinv(A))ᵀ` and reconstruct
   `X_clean = C · (X − mean) + mean`.

## References

1. De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., & Van Huffel,
   S. (2006). Canonical correlation analysis applied to remove muscle artifacts
   from the electroencephalogram. _IEEE Transactions on Biomedical Engineering_,
   53(12), 2583-2587. https://doi.org/10.1109/TBME.2006.879459
2. Safieddine, D., et al. (2012). Removal of muscle artifact from EEG data:
   comparison between stochastic (ICA and CCA) and deterministic (EMD and
   wavelet-based) approaches. _EURASIP Journal on Advances in Signal
   Processing_, 2012, 127. https://doi.org/10.1186/1687-6180-2012-127
