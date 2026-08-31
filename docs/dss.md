(dss)=
# Denoising Source Separation (DSS)

## Overview

Denoising Source Separation (DSS) finds spatial filters that maximize a
user-defined bias relative to a baseline covariance. The baseline describes
total data power; the biased covariance describes the reproducible, spectral,
temporal, or other structure to emphasize. This is the linear DSS formulation
described by Särelä and Valpola and its spatial-filtering treatment by de
Cheveigné and Simon {footcite:p}`sarela2005_dss,decheveigne_simon2008_spatial`.

DSS components are optimized subspaces, not individually identified neural
sources. A component selected by a reproducibility or artifact-related bias
can still contain signal of interest, so attenuation and preservation should be
evaluated together.

## Minimal estimator example

The estimator accepts a callable bias or one of the public bias classes:

```python
import numpy as np
from mne_denoise.dss import BandpassBias, DSS

rng = np.random.default_rng(0)
data = rng.standard_normal((8, 2000))  # (n_channels, n_times)
bias = BandpassBias(freq_band=(8.0, 12.0), sfreq=250.0)

dss = DSS(bias=bias, n_components=3, component_action="extract")
sources = dss.fit_transform(data)
```

For MNE ``Raw``, ``Epochs``, and ``Evoked`` objects, pass the object directly.
The estimator uses the object's sampling frequency where applicable, returns a
copy for sensor-space operations, selects one homogeneous data-channel type by
default, and preserves untouched channels and MNE metadata. Set ``whiten=True``
when a joint multi-type MNE decomposition is intentional. The API reference
and estimator docstrings define the exact fitted-channel and layout contracts.

## Component operations

``component_action`` makes the operation explicit:

* ``"extract"`` returns component time courses;
* ``"retain"`` reconstructs the selected leading components in sensor space;
* ``"subtract"`` removes those components from the input.

For NumPy input, arrays use ``(n_channels, n_times)`` or
``(n_channels, n_times, n_epochs)``. MNE ``Epochs`` uses its usual
``(n_epochs, n_channels, n_times)`` layout. Extraction returns
``(n_components, n_times)`` for continuous data,
``(n_components, n_times, n_epochs)`` for channel-first NumPy epochs, and
``(n_epochs, n_components, n_times)`` for MNE Epochs. Retention and subtraction
return the original array layout or a copied MNE container.

``fit`` learns the spatial filters, patterns, eigenvalues, and (when centering
is enabled) one global channel mean. ``transform`` reuses that state; it does
not recenter each transform batch. Standard ``fit_transform`` is composition
of ``fit`` and ``transform``. The adaptive DSS extension is different: its
``fit_transform`` fits and applies independent filters per segment and supports
subtraction only.

## Covariance-level interface

``compute_dss`` operates on two already-computed square covariance matrices,
not on raw data:

```python
import numpy as np
from mne_denoise.dss import compute_dss

rng = np.random.default_rng(0)
x = rng.standard_normal((8, 2000))
covariance_baseline = x @ x.T / x.shape[1]
covariance_biased = (x[:, ::4] @ x[:, ::4].T) / x[:, ::4].shape[1]

filters, patterns, eigenvalues = compute_dss(
    covariance_baseline,
    covariance_biased,
    n_components=3,
)
```

It returns exactly three arrays: filters with shape
``(n_components, n_channels)``, patterns with shape
``(n_channels, n_components)``, and eigenvalues with shape
``(n_components,)``. The estimator's ``bias`` object or callable is responsible
for constructing the biased data before the covariance is formed.

The baseline covariance is whitened, the biased covariance is rotated in that
whitened space, and the resulting filters are normalized against baseline
power. The optional estimator-level ``whiten=True`` is a separate sensor-space
pre-whitening step for joint MNE channel types; it is not a replacement for the
baseline-covariance whitening intrinsic to DSS.

## Linear bias functions

The public linear bias classes define different reproducibility or structure
criteria:

* ``AverageBias(axis="epochs")`` replaces each trial by its trial average and
  is useful for repeatable evoked structure. Its ``axis="datasets"`` operation
  is a low-level dataset-first averaging bias; the NumPy ``DSS`` estimator
  itself expects channel-first arrays and should not be given a dataset-first
  array as if it were a second input convention.
* ``CycleAverageBias`` averages fixed windows around supplied events. It is a
  package-level fixed-window operation, not a complete quasiperiodic cardiac
  implementation from the original DSS paper.
* ``BandpassBias`` emphasizes a frequency band; ``LineNoiseBias`` and
  ``PeakFilterBias`` target narrow-band periodic structure.
* ``CombFilterBias`` and ``QuasiPeriodicDenoiser`` provide periodic or
  quasiperiodic bias operations.
* ``LagAverageBias`` and ``SmoothingBias`` emphasize lagged or smoothed
  structure; ``SpectrogramBias`` operates on time-frequency masks.

For event-locked cardiac contrast, detect events with an MNE event detector
and fit the bias on training data only:

```python
from mne.preprocessing import find_ecg_events
from mne_denoise.dss import CycleAverageBias, DSS

events, _, _ = find_ecg_events(train_raw, ch_name="ECG")
eeg = train_raw.copy().pick("eeg")
bias = CycleAverageBias(
    event_samples=events[:, 0],
    window=(-0.15, 0.25),
    window_unit="seconds",
    sfreq=eeg.info["sfreq"],
    event_origin="raw",
    first_samp=eeg.first_samp,
)
cleaner = DSS(
    bias=bias,
    n_components=10,
    n_select=1,
    component_action="subtract",
)
cleaner.fit(eeg)
held_out = cleaner.transform(held_out_raw.copy().pick("eeg"))
```

The ECG-locked component is a reproducible component, not proof of a purely
cardiac source. The number removed is an explicit scientific choice and should
be checked on held-out data.

## Nonlinear DSS

``IterativeDSS`` uses fixed-point iteration with a public nonlinear denoiser,
such as ``KurtosisDenoiser``, ``RobustTanhDenoiser``, ``GaussDenoiser``, or
``SkewDenoiser``. The fixed-point structure follows the nonlinear DSS context
of Särelä and Valpola {footcite:p}`sarela2005_dss`; the available nonlinearities,
stopping rules, and convenience wrappers are package implementation choices.
Other public denoisers include ``VarianceMaskDenoiser``, ``WienerMaskDenoiser``,
``TanhMaskDenoiser``, ``DCTDenoiser``, and ``SpectrogramDenoiser``. They define
the bias or source operation; they do not turn component optimization into
source identification.

## Selection and variants

The public selection helpers ``iterative_outlier_removal``,
``auto_select_components``, ``detect_eigenvalue_knee``, and
``auto_select_components_robust`` implement explicit package heuristics. The
outlier rule follows the NoiseTools convention described in ZapLine-plus;
the eigenvalue-knee fallback and the combined maximum rule are
mne-denoise additions {footcite:p}`klug_kloosterman2022_zapline_plus`. Treat
automatic selection as a proposal to inspect, not as proof that the selected
components are artifact.

The public DSS variants are:

* ``TimeShiftDSS`` for repeated-trial lag-augmented decomposition; see
  :doc:`time_shift_dss`.
* ``smooth_dss`` and ``ssvep_dss`` for specialized smooth or periodic biases.
* ``narrowband_scan`` and ``narrowband_dss`` for frequency-specific scans.

ZapLine is a separate public estimator and has its own page; it is not a DSS
compatibility alias.

## Assumptions and evidence boundary

DSS assumes that a bias can be estimated from the available data and that the
resulting spatial subspace is useful for the stated purpose. Rank, centering,
filtering, trial boundaries, and the number of samples affect the fitted
covariances. A bias can enhance genuine neural activity as well as artifact.

The core decomposition is a published method. The package's bias classes,
adaptive segmentation, component-selection heuristics, cross-fade behavior,
and convenience wrappers are package extensions or implementation conventions.
Validate any subtraction or retention decision with controls appropriate to the
signal of interest, preferably using held-out data.

## References

```{footbibliography}
```
