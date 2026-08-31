(icanclean)=
# iCanClean

## Overview

iCanClean uses canonical correlation analysis (CCA) between a primary recording
and a reference block to identify shared component variance. It reconstructs
the primary block after subtracting selected canonical variates. The reference
can be a physical noise recording or a filtered pseudo-reference derived from
the primary channels. The reference-based strategy is described by Downey and
Ferris {footcite:p}`downey_ferris2022_icanclean`; the pseudo-reference
configuration and phantom evaluation are described separately
{footcite:p}`downey_ferris2023_icanclean_phantom`.

Shared variance is not synonymous with artifact. Neural activity observed by a
reference, or neural activity retained by a pseudo-reference, can also be
selected. Evaluate artifact attenuation and preservation of the signal of
interest independently.

## Minimal API

Use ``ICanClean`` when working with NumPy arrays or supported MNE containers:

```python
from mne_denoise.icanclean import ICanClean

cleaner = ICanClean(
    sfreq=250.0,
    ref_channels=noise_channels,
    mode="sliding",
    segment_len=2.0,
    threshold=0.7,
)
cleaner.fit(raw)
clean_raw = cleaner.transform(raw)
```

``fit`` is a scikit-learn compatibility no-op: the actual CCA and cleaning
occur in ``transform`` because the operator may be estimated globally or per
window. ``fit_transform`` calls those operations in sequence. MNE ``Raw``,
``Epochs``, and ``Evoked`` inputs are copied; channel names, order, bad-channel
metadata, and relevant container metadata are preserved according to the
shared MNE contract. NumPy input is continuous ``(n_channels, n_times)`` data.

The functional ``compute_icanclean`` interface performs one continuous NumPy
pass and returns ``(cleaned, qc)``. Its ``qc`` mapping contains per-window
correlations, removed-component indices, fitted coefficient matrices, and
window counts.

## Reference construction

With physical reference channels, ``ref_channels`` identifies sensors expected
to observe the artifact. The dual-electrode literature gives a concrete
example of this design for mobile EEG; its assumptions concern the mechanical
coupling and spatial relationship of the reference electrodes
{footcite:p}`nordin2018_dual_electrode`. A reference that does not observe the
artifact cannot identify it through CCA.

``pseudo_ref=True`` instead filters a copy of the primary channels using
``filter_ref`` and uses that filtered block as the reference. This is the
pseudo-reference extension described in the iCanClean phantom work
{footcite:p}`downey_ferris2023_icanclean_phantom`. It is not equivalent to a
physical noise recording: the reference is derived from the signal being
cleaned, so the filter passband and the neural spectrum determine what is
shared.

The filter is used to construct the CCA reference. The original primary data
are used for reconstruction. Inspect the fitted channel lists and the
``correlations_``/``removed_idx_`` diagnostics before interpreting the output.

## Thresholds and modes

The threshold is an absolute squared canonical correlation. Its interpretation
depends on the number of samples, the primary/reference ranks, the window
length, and the reference construction. Do not transfer a threshold from one
recording or reference design without checking the resulting correlation
spectrum.

``threshold="auto"`` is a package rule based on the running correlation
distribution. ``threshold="null"`` uses circularly shifted reference
surrogates to estimate a finite-sample threshold. The null threshold is a
package-defined, research-facing extension: it addresses sampling-degeneracy
of CCA, not whether shared variance is artifact. Its calibrated use does not
exactly simulate the fixed global CCA weights and should be checked with
independent or held-out data.

The estimator supports four batch modes:

| Mode | Decomposition | Interpretation |
| --- | --- | --- |
| ``"global"`` | One over the full recording | One selected basis is subtracted once. |
| ``"sliding"`` | One per cleaning window | Window-local bases are overlap-added. |
| ``"calibrated"`` | One global CCA | Fixed global basis is scored in each window. |
| ``"hybrid"`` | Global pass followed by sliding pass | A package-specific two-pass composition. |

The first three modes correspond to distinct operator reuse choices in the
implementation. ``"hybrid"`` is an mne-denoise extension, not a claim made by
the iCanClean publication. None of these modes is an online recursive
algorithm.

``stats_segment_len`` can make the CCA estimation window broader than the
inner cleaning window in sliding workflows. More samples can improve numerical
conditioning, but it does not make the reference more specific to the target
artifact.

## Fitting and diagnostics

The estimator stores the fitted channel lists and, after transformation,
``correlations_``, ``n_removed_``, ``removed_idx_``, ``filters_``,
``patterns_``, and ``n_windows_``. The correlation arrays describe the fitted
CCA or the mode-specific local scores; they are not automatically a measure of
artifact purity.

For a reproducible evaluation:

1. define the primary/reference channels and preprocessing before fitting;
2. fit on training data only when the operator will be evaluated on held-out
   data;
3. report mode, window length, overlap, threshold, reference construction, and
   re-referencing choices; and
4. quantify both attenuation of the target artifact and changes to a neural
   control or other signal-preservation measure.

The Gonsisko et al. study evaluates the interaction of iCanClean and ICA in
mobile brain imaging {footcite:p}`gonsisko2023_icanclean_ica`; that study does
not turn every package mode or threshold into a general guarantee.

## Published method versus package behavior

The CCA-based reference-noise strategy is the published method. The
pseudo-reference configuration follows the separately published phantom
workflow. Running modes, overlap-add orchestration, ``threshold="auto"``, the
circular-shift ``"null"`` threshold, and the hybrid composition are
implementation conventions or mne-denoise extensions. Unit tests establish
software behavior; they do not independently validate scientific performance
for every artifact, montage, or population.

## References

```{footbibliography}
```
