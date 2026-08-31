# Reference-free BSS-CCA

BSS-CCA separates channel data using canonical correlation with a lagged copy
of the signal. Components with low lagged correlation can be removed as a
muscle-artifact strategy without a reference channel
{footcite:p}`declercq2006_bss_cca,vergult2007_bss_cca,hotelling1936_cca`.

## Usage

```python
from mne_denoise.bss_cca import BSSCCA

model = BSSCCA(
    sfreq=250.0,
    lag_samples=1,
    n_remove=2,
)
clean = model.fit_transform(data)  # data: (n_channels, n_times)
```

## Key points

- Input is channel-first 2-D data or MNE Raw, Epochs, or Evoked where
  supported; fit learns operators and transform reuses them.
- Choose n_remove or use rho_threshold to determine the removed components.
  reject chooses whether the low- or high-correlation end of the
  canonical-correlation spectrum is treated as artifactual.
- lag_samples or lag_seconds and preprocessing affect the
  canonical-correlation ordering.
- Segmented operation fits one operator per block; overlap controls block
  overlap and diagnostics report the fitted block operators.

The ordering is a signal-model assumption, not a universal artifact label.
Check the result against controls and the signal of interest.

## References

```{footbibliography}
```
