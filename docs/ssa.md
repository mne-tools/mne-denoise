# Singular Spectrum Analysis

This package provides two channel-wise, univariate SSA workflows. Basic SSA
decomposes each channel and groups components by dominant frequency; Local SSA
clusters delay vectors and reconstructs selected local subspaces
{footcite:p}`golyandina_zhigljavsky2013_ssa,teixeira2006_local_ssa`.

## Basic SSA

```python
from mne_denoise.ssa import SingularSpectrumAnalysis

basic = SingularSpectrumAnalysis(sfreq=250.0, drop_freq_max=3.0)
clean = basic.fit_transform(data)
```

window_length or window_seconds sets the delay-coordinate embedding. Basic SSA
returns an additive component decomposition internally and removes components
selected by the dominant-frequency rule. The estimator is transductive: each
record supplied to transform is decomposed independently.

## Local SSA

```python
from mne_denoise.ssa import LocalSingularSpectrumAnalysis

local = LocalSingularSpectrumAnalysis(window_length=40, n_clusters="auto")
clean = local.fit_transform(data)
```

Local SSA clusters delay vectors, fits a local PCA model in each cluster, uses
the implemented MDL rule to select subspace dimensions, and subtracts the
reconstruction. Clustering and reconstruction are recomputed for each record.

## Key points

- Both workflows operate independently per channel; they are not multivariate
  SSA.
- The embedding window, frequency rule, cluster count, and random seed affect
  the result.
- NumPy input is channel-first. Supported MNE containers retain their layout
  and metadata.
- These are record-dependent operations: fit records parameters and layout,
  while transform analyzes the records supplied to it.

## References

```{footbibliography}
```
