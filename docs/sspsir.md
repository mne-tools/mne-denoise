(sspsir)=
# SSP-SIR

## Overview

Signal-space projection--source-informed reconstruction (SSP-SIR) targets
high-variance muscle artifacts in TMS-evoked EEG. It estimates an artifact
subspace from high-pass, average-referenced data, projects that subspace out,
and reconstructs the projected data through a forward model. The published
method is described by Mutanen et al. {footcite:p}`mutanen2016_sspsir`.

The projection is not a guarantee that the removed directions are purely
artifactual. A forward model makes the reconstruction source-informed, but an
inaccurate lead field or an overly large artifact subspace can distort the
neural response.

## Minimal API

For an MNE object, pass an object with EEG channel positions. When no
individual forward solution is supplied, the estimator builds a spherical
lead-field approximation from the montage:

```python
from mne_denoise.sspsir import SSPSIR

ssp = SSPSIR(n_components=3, art_window=(0.005, 0.050))
clean_epochs = ssp.fit_transform(epochs)
```

For a NumPy array, supply a lead field with the same channel order and an
explicit sampling frequency:

```python
from mne_denoise.sspsir import SSPSIR

ssp = SSPSIR(
    n_components=2,
    forward=forward,
    sfreq=1000.0,
    art_window=(0.005, 0.050),
)
clean = ssp.fit_transform(data)
```

NumPy input is two-dimensional ``(n_channels, n_times)`` or epoch-major
three-dimensional ``(n_epochs, n_channels, n_times)``. MNE ``Raw``, ``Epochs``,
and ``Evoked`` are supported. Fitting learns the artifact topographies, the
source-informed operators, and a time-dependent blend when requested;
``transform`` reuses those fitted quantities. MNE transformation returns a
copy, preserves the container metadata, and leaves channels outside the
selected homogeneous data type unchanged.

## How the reconstruction works

Let ``L`` be the average-referenced lead field and ``U_a`` the orthonormal
artifact topographies. The projected operator first applies

```text
P = I - U_a U_a.T
```

and then computes a rank-``M`` source-informed reconstruction from ``P L``.
The unprojected branch uses the same rank-``M`` reconstruction from ``L``.
Inside the artifact window, the projected branch is used; outside it, the
unprojected branch is used. This temporal blending limits the projection's
effect on baseline and later evoked activity. The package exposes the two
low-level operator constructors:

```python
from mne_denoise.sspsir import compute_sir, compute_sspsir

sir = compute_sir(leadfield, M=15)
sspsir = compute_sspsir(leadfield, artifact_topographies, M=15)
```

``compute_sir`` returns the unprojected rank-``M`` operator. It is not an
identity matrix. ``compute_sspsir`` returns the projected operator and expects
``artifact_topographies`` with shape ``(n_channels, n_components)`` and
orthonormal columns. Both functions return one square array, not cleaned
time-series data.

## Scientifically important choices

* ``n_components`` can be an explicit integer or a cumulative high-frequency
  variance fraction in ``(0, 1)``. The fitted ``singular_values_`` can be
  inspected when choosing an elbow, as in the published workflow.
* ``art_window=(tmin, tmax)`` estimates the artifact subspace from that time
  interval and creates a smooth crossfade around it. Without a window,
  ``blend="auto"`` derives the kernel from a sliding high-frequency envelope.
* ``blend="constant"`` applies the projected operator throughout the input.
  It is useful for an explicitly uniform projection, but it does not provide
  the time-local protection of the crossfaded mode.
* ``M`` is the source-informed reconstruction rank. If omitted, the estimator
  derives it from the fitted data rank and artifact-component count; numerical
  rank can reduce the effective fitted ``M_``.
* ``high_pass`` isolates the high-frequency artifact used for subspace
  estimation. It is an estimation choice, not a claim that the final output
  has been high-pass filtered.

## Forward model and MNE behavior

An individualized ``mne.Forward`` is preferred when available. For MNE input,
the supplied forward solution is matched to the selected channel names and
the gain matrix is average referenced. With no forward solution, the package
uses a three-layer spherical fallback constructed from the montage. Plain
arrays have no channel positions; they therefore require a compatible
forward solution.

The fitted ``leadfield_``, ``artifact_topographies_``, ``operator_``,
``operator_orig_``, ``kernel_``, ``M_``, and ``projs_`` are useful diagnostics.
For MNE fitting, ``projs_`` contains inactive ``mne.Projection`` objects for
the estimated artifact directions. Check the channel reference, lead-field
conditioning, selected rank, and topographies before interpreting the result.

## Published method versus package behavior

Artifact-subspace projection followed by source-informed reconstruction is the
published SSP-SIR method {footcite:p}`mutanen2016_sspsir`. The 2024 study
provides simulation evidence and compares SSP-SIR with ICA
{footcite:p}`mutanen2024_sspsir_simulation`. The spherical lead-field fallback,
the variance-fraction spelling of ``n_components``, the explicit low-level
operators, and the estimator/MNE interface are package implementation choices
or extensions. The broader source-based and TMS-EEG reviews provide context,
not validation of every package option {footcite:p}`mutanen2022_source_artifact,hernandez_pavon2022_tms_review`.

The current numerical core has reference parity and synthetic validation, but
the estimator has not been independently validated on a public real TMS-EEG
dataset. Treat real-data use as experimental and evaluate artifact attenuation
and preservation of the response of interest separately.

## References

```{footbibliography}
```
