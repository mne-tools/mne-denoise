# Singular Spectrum Analysis (SSA)

## Overview

The `mne_denoise.ssa` module implements **Singular Spectrum Analysis (SSA)**
(Golyandina & Zhigljavsky 2013) for removing slow-drift, ocular, and
quasi-periodic (e.g. cardiac) artifacts from M/EEG.

SSA embeds each channel into a trajectory (Hankel) matrix, decomposes it by SVD
into *eigentriples*, and reconstructs the series from a chosen subset by diagonal
averaging. Grouping eigentriples by the **dominant frequency** of their
reconstructed component lets SSA drop slow ocular/drift structure (and, with an
explicit band, quasi-periodic cardiac structure) while preserving oscillatory
neural activity.

SSA is **per-recording and unsupervised**: it learns no operator transferred
from `fit` to `transform` (`fit` is a no-op), so it needs no train/evaluation
split.

## Quick Start

```python
import numpy as np
from mne_denoise.ssa import SingularSpectrumAnalysis

data = np.random.randn(32, 10000)  # 32 channels, 10000 samples

# Drop slow drift / ocular structure (dominant frequency <= 3 Hz).
est = SingularSpectrumAnalysis(sfreq=250.0, drop_freq_max=3.0)
cleaned = est.fit_transform(data)
print(f"dropped per channel: {est.dropped_counts_}")
```

Targeting a specific band (e.g. a ~1 Hz cardiac component):

```python
cleaned = SingularSpectrumAnalysis(sfreq=250.0, drop_band=(0.8, 1.6)).fit_transform(
    data
)
```

With MNE-Python objects the sampling frequency is read from `info`:

```python
raw_clean = SingularSpectrumAnalysis(drop_freq_max=3.0).fit_transform(raw)
```

## One-shot functional API

```python
from mne_denoise.ssa import compute_ssa, ssa_clean_channel

# Multichannel array -> (cleaned, diagnostics)
cleaned, info = compute_ssa(data, sfreq=250.0, drop_freq_max=3.0)

# Single channel
cleaned_ch = ssa_clean_channel(data[0], sfreq=250.0, drop_freq_max=3.0)
```

## Parameters

| Parameter       | Description                                                                    |
| --------------- | ------------------------------------------------------------------------------ |
| `sfreq`         | Sampling frequency (Hz). Optional for MNE input (read from `info`).            |
| `window_length` | SSA embedding window length. Defaults to `min(sfreq/2, max_window)`.           |
| `drop_freq_max` | Drop components with dominant frequency ≤ this value (Hz).                     |
| `drop_band`     | If given, drop components whose dominant frequency falls in this band instead. |
| `n_check`       | Number of top eigentriples (by variance) examined per channel.                |
| `max_window`    | Upper bound on the embedding window length.                                    |

## Algorithm Details

For each channel `x` of length `N`:

1. Choose window length `L`; form the `(L, K)` trajectory matrix
   (`K = N − L + 1`).
2. SVD the trajectory matrix into eigentriples `(σᵢ, uᵢ, vᵢ)`.
3. Reconstruct each of the top `n_check` components by diagonal averaging
   (hankelization) and estimate its dominant frequency `f_dom`.
4. Drop components with `f_dom ≤ drop_freq_max` (or `f_dom ∈ drop_band`).
5. Return the signal minus the summed dropped components.

## References

1. Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for Time
   Series. _Springer_. https://doi.org/10.1007/978-3-642-34913-3
2. Teixeira, A. R., et al. (2006). On the use of clustering and local singular
   spectrum analysis to remove ocular artifacts from EEG. _IJCNN_.
   https://doi.org/10.1109/IJCNN.2006.246999
