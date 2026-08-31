(sound)=
# SOUND

## Overview

SOUND (Source-estimate-Utilizing Noise-discarding) estimates a separate noise
amplitude for each sensor and uses a forward-model-based Wiener operator to
reduce channel-specific noise. The published algorithm predicts each channel
from the other channels through a lead field and iteratively updates the noise
estimates {footcite:p}`mutanen2018_sound`. The covariance-form iteration used
by the implementation is also described by Mutanen et al.
{footcite:p}`mutanen2022_source_artifact`.

SOUND is not a generic channel interpolator. It assumes that the signal of
interest is represented by the forward model and that sensor-specific noise is
less consistent with that model. The method can therefore attenuate signal
that is poorly represented by the chosen lead field.

## Minimal API

With an MNE object, SOUND can use its montage to construct a spherical lead
field when no individual forward solution is supplied:

```python
from mne_denoise.sound import SOUND

sound = SOUND(reference="best", n_iter=5)
clean_raw = sound.fit_transform(raw)
```

For a NumPy array, supply a lead field in the same channel reference and order
as the data:

```python
from mne_denoise.sound import SOUND

sound = SOUND(forward=forward, reference="average", n_iter=5)
clean = sound.fit_transform(data)
```

The estimator accepts continuous arrays with shape ``(n_channels, n_times)``
and epoched arrays with shape ``(n_epochs, n_channels, n_times)``. MNE
``Raw``, ``Epochs``, and ``Evoked`` objects are supported. ``fit`` learns one
linear operator and per-channel noise estimates; ``transform`` applies that
operator to compatible data without refitting or mutating the input. MNE
outputs are copied and retain their container metadata and channels outside the
fitted homogeneous data selection.

## Forward model and leave-one-channel-out estimation

Let ``L`` be a lead field and ``Y`` the sensor data. For channel ``i``, SOUND
forms a leave-one-channel-out forward model and predicts that channel from the
others. The residual prediction error supplies an initial noise amplitude
``sigma_i``. It then alternates between:

1. weighting channels by the current inverse noise amplitudes;
2. solving the regularized forward-model prediction for each left-out channel;
3. estimating the residual noise amplitudes from the data covariance; and
4. stopping after ``n_iter`` iterations or when ``tol`` is reached.

The final operator is a sensor-space linear map. It is applied as
``cleaned = operator_ @ data`` for the plain array solver. ``sigmas_`` and
``convergence_`` expose the fitted noise estimates and iteration changes.

The lead field must have one row per data channel and must use the same
reference. Re-referencing changes both the data and the lead field. If a custom
forward model is supplied, perform the reference transformation explicitly
before fitting and document it as part of the analysis.

## Reference conventions

``reference="average"`` applies the plain all-channel SOUND solver and assumes
that the input is already average referenced. ``reference="best"`` follows the
reference bookkeeping of the TESA/``tesa_sound`` implementation: it chooses a
least-noisy reference channel using the initial data-driven estimate, drops that
channel during estimation, and maps the result back to an average-referenced
full montage. This reference choice is an implementation convention around the
published algorithm, not an anatomical channel-selection rule.

The functional interfaces expose the two corresponding operator constructions:
``compute_sound`` returns ``(operator, sigmas, convergence)`` and
``compute_sound_ref_best`` additionally returns the selected reference-channel
index.

## Important parameters and diagnostics

* ``lambda_`` regularizes the forward-model solves. It changes the
  noise-versus-model-fit trade-off and is not a universal optimum.
* ``n_iter`` and ``tol`` control iteration length. The fitted convergence array
  records the maximum relative change per completed iteration.
* ``sigma_source="evoked"`` estimates epoched noise from the trial average;
  ``"trials"`` estimates it from concatenated trials. These are different
  estimands, not interchangeable preprocessing labels.
* ``leadfield_``, ``operator_``, ``sigmas_``, ``best_channel_``, and
  ``convergence_`` describe the fitted operation. Inspect them for rank,
  conditioning, reference, and convergence problems.

## Assumptions and failure modes

SOUND requires finite, nonempty data, a compatible lead field, and enough
channels for the selected reference convention. It can fail or become
unstable when the lead field is poorly conditioned, the data reference does not
match the lead-field reference, the recording has too few samples, or the
channel noise is strongly correlated rather than channel-specific. A spherical
lead field is an approximation; an individual forward solution is preferable
when the acquisition and analysis support one.

Reduction in sensor noise or output variance does not establish neural-signal
preservation. Compare a cleaned result with a held-out signal control, source
model, or other domain-appropriate reference. The TMS-EEG review places SOUND
within a broader source-informed artifact-removal framework
{footcite:p}`hernandez_pavon2022_tms_review`.

## Published method versus implementation details

The leave-one-channel-out noise estimation and forward-model Wiener filtering
are the published SOUND method. Covariance-only iteration, the spherical
lead-field fallback, ``reference="best"`` montage bookkeeping, channel-wise
zero-estimate fallback, and the estimator/container interface are implementation
conventions or package extensions. The central references describe the method;
they do not validate every package option on every montage or artifact class.

## References

```{footbibliography}
```
