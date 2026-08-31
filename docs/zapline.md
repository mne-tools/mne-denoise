(zapline)=
# ZapLine

## Overview

ZapLine removes narrow-band power-line noise and harmonics by combining a
period-locked smooth/residual decomposition with Denoising Source Separation
(DSS). The original method is described by de Cheveigné
{footcite:p}`decheveigne2020_zapline`.

The package also exposes an adaptive mode related to ZapLine-plus. It can
detect frequencies, segment non-stationary data, refine the target in each
segment, and adapt the component-selection and quality-assurance loop. That
mode is an implementation of the package's ZapLine-plus path, not a claim that
all of its defaults or fallbacks are prescribed by the paper
{footcite:p}`klug_kloosterman2022_zapline_plus`.

Reducing power at a line frequency does not establish preservation of neural
signals. A neural rhythm that overlaps a removed band can be attenuated too.

## Standard ZapLine

```python
from mne_denoise.zapline import ZapLine

zapline = ZapLine(sfreq=1000.0, line_freq=50.0, n_select="auto")
zapline.fit(raw)
clean_raw = zapline.transform(raw)
```

The standard path:

1. separates a moving-average smooth branch from the residual using the line
   period;
2. fits ``LineNoiseBias`` DSS filters to the residual;
3. selects leading line-noise components; and
4. subtracts their reconstructed artifact from the residual before restoring
   the smooth branch.

``line_freq`` is the fundamental frequency in hertz. ``n_harmonics`` controls
the harmonic targets in the spectral bias. With ``n_select="auto"``, the
estimator first uses the package DSS selector and, when a line peak is present
but that selector returns zero, evaluates leading component counts with a
ZapLine-specific spectral-QA fallback. This fallback is a package behavior;
it is not a universal component-selection rule.

Standard NumPy input has shape ``(n_channels, n_times)`` or
``(n_epochs, n_channels, n_times)``. MNE ``Raw``, ``Epochs``, and ``Evoked``
are supported. ``fit`` and ``transform`` return the same broad input type;
MNE results are copied, and the default homogeneous-channel selection leaves
other channels untouched. ``whiten=True`` opts into joint processing of
supported MNE data channel types or per-channel standardization for arrays;
report that choice because it changes the sensor-space fit.

## Adaptive ZapLine-plus path

Use ``fit_transform`` for adaptive processing:

```python
from mne_denoise.zapline import ZapLine

zapline = ZapLine(
    sfreq=500.0,
    line_freq=None,
    adaptive=True,
    adaptive_params={"process_harmonics": True},
)
clean = zapline.fit_transform(data)
```

In the current implementation, adaptive processing can:

* search a configured frequency range when ``line_freq=None``;
* process detected harmonics when requested;
* split continuous data with covariance-stationarity segmentation;
* refine each target frequency to a local spectral peak;
* check artifact presence and spectral QA per segment;
* retry with changed selection settings; and
* optionally use a narrow notch fallback when ``hybrid_fallback=True``.

Each segment has its own fitted DSS operator. ``crossfade`` is a package
option that blends neighboring segment outputs with a raised-cosine window;
the default ``0`` performs hard concatenation. Adaptive diagnostics are kept
in ``adaptive_results_`` and ``segment_results_`` where applicable, including
target frequencies, segment bounds, selected counts, and artifact-presence
flags. A callback receives one ``ProgressEvent`` per completed target-frequency
pass; standard mode accepts the callback but emits no events.

``fit`` and ``transform`` alone are deliberately unavailable in adaptive mode:
there is no single global filter to fit or reuse because the filters are
segment-specific. Adaptive mode is therefore transductive over the recording
passed to ``fit_transform``.

## Parameters that change interpretation

* ``n_select`` is a component count or ``"auto"``; it is not a percentage of
  line-noise power.
* ``nfft`` controls the spectral resolution used by the line-noise bias.
* ``rank``, ``nkeep``, and ``reg`` affect DSS whitening and conditioning.
* ``threshold``, ``knee_rel_floor``, and ``knee_min_ratio`` control package
  selection heuristics in the automatic path.
* ``adaptive_params`` controls detection bounds, minimum segment duration,
  harmonic processing, per-segment removal caps, QA tolerances, and the
  optional hybrid fallback. Record the complete dictionary used in an
  analysis.

Inspect ``filters_``, ``patterns_``, ``eigenvalues_``, ``n_removed_``,
``n_harmonics_``, and adaptive segment diagnostics. A warning or a fallback
does not establish that the selected result is scientifically preferable.

## Assumptions and validation

ZapLine assumes that line noise has a spatial structure that DSS can isolate
and that the period-locked residual contains enough samples to estimate it.
Short records, changing line frequency, non-integer ``sfreq / line_freq``,
rank reduction, mixed sensor units, and neural activity at the target
frequency can all change the result. For adaptive mode, segment boundaries
also change the fitted operators.

Evaluate target-frequency suppression together with broadband and signal-of-
interest controls. Check neighboring spectral bands, time-domain responses,
and held-out or simulated signals rather than treating line-noise reduction as
evidence of neural preservation. The package QA utilities and forward-model
distortion metrics are described in :doc:`evaluation`.

## Original ZapLine, ZapLine-plus, and package additions

The period-locked smoothing, DSS line-noise bias, component subtraction, and
smooth-branch restoration are the core original ZapLine workflow
{footcite:p}`decheveigne2020_zapline`.

Automatic frequency detection, covariance-stationarity segmentation,
per-segment adaptation, and spectral QA follow the ZapLine-plus direction
{footcite:p}`klug_kloosterman2022_zapline_plus`. The exact estimator interface,
the DSS knee fallback, optional crossfade, mixed-unit whitening path, and
hybrid notch fallback are mne-denoise implementation details or extensions.
They should not be attributed to either publication without qualification.

## References

```{footbibliography}
```
