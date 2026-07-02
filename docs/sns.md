# Sensor Noise Suppression (SNS)

## Overview

The `mne_denoise.sns` module implements **Sensor Noise Suppression (SNS)**
(de Cheveigné & Simon 2008), a purely spatial, reference-free method that
suppresses noise **specific to individual sensors**.

Each channel is regenerated from a least-squares projection onto the subspace
spanned by its most-correlated *neighbour* channels. The rationale: genuine
brain signal is spatially correlated across sensors and is therefore well
predicted by the other channels, whereas sensor-specific noise (amplifier noise,
a flaky electrode, SQUID noise) is uncorrelated across sensors and cannot be
predicted — so it is removed by the projection. Applied as a single spatial
operator `W` (`X_clean = W @ X`) with `W[k, k] = 0`, so a channel is never
regenerated from itself.

> **Scope.** SNS targets *sensor-specific* noise, not physiological artifacts
> (ocular, muscle, cardiac), which are themselves spatially correlated and so are
> largely retained. Use SNS as a first-stage sensor cleanup, complementary to the
> artifact-specific denoisers (DSS, iCanClean, auto-CCA, ...).

## Quick Start

```python
import numpy as np
from mne_denoise.sns import SNS

data = np.random.randn(64, 20000)  # 64 channels, 20000 samples

# Leakage-safe estimator API: learn the operator on train, apply to eval.
est = SNS(n_neighbors=0)   # 0 = use all other channels
est.fit(data)
cleaned = est.transform(data)
```

On dense arrays, restricting to the most-correlated neighbours is faster and more
robust, and `skip` can drop the closest channels (which may share local noise):

```python
cleaned = SNS(n_neighbors=20, skip=1).fit_transform(data)
```

With MNE-Python objects:

```python
raw_clean = SNS(n_neighbors=40).fit_transform(raw)
```

## One-shot functional API

```python
from mne_denoise.sns import compute_sns, compute_sns_weights

cleaned, info = compute_sns(data, n_neighbors=20)   # (cleaned, {weights, n_neighbors})

# Build the operator directly from a covariance matrix:
W, n = compute_sns_weights(cov, n_neighbors=20, skip=0)
cleaned = W @ data
```

## Parameters

| Parameter     | Description                                                                |
| ------------- | ------------------------------------------------------------------------- |
| `n_neighbors` | Neighbour channels used to regenerate each channel (`0` = all others).     |
| `skip`        | Number of closest neighbours to skip (avoid channels sharing local noise). |
| `verbose`     | MNE-style logging verbosity.                                               |

## Algorithm Details

Given demeaned data `X` (channels × samples) with covariance `C = X Xᵀ / n`:

1. Convert `C` to a correlation matrix for scale-invariant neighbour selection.
2. For each channel `k`, sort the other channels by squared correlation with `k`,
   skip the closest `skip`, and take the top `n_neighbors` as its neighbour set
   `N`.
3. Solve the least-squares projection weights `b = pinv(C[N, N]) · C[N, k]` and
   place them in the operator: `W[k, N] = b`, `W[k, k] = 0`.
4. Apply `X_clean = W · X` — each channel replaced by its regeneration from its
   neighbours.

The projection is basis-invariant to the PCA-whitening used by the reference
NoiseTools / meegkit implementations.

## References

1. de Cheveigné, A., & Simon, J. Z. (2008). Sensor noise suppression. _Journal
   of Neuroscience Methods_, 168(1), 195-202.
   https://doi.org/10.1016/j.jneumeth.2007.09.012
2. de Cheveigné, A. NoiseTools (`nt_sns`), and the meegkit Python port
   (https://github.com/nbara/python-meegkit) — reference implementations.
