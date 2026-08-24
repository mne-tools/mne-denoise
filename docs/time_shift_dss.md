# TimeShiftDSS

```{warning}
`TimeShiftDSS` has substantially more free parameters than spatial DSS. Do not
interpret, retain, or remove a component without held-out and surrogate
validation.
```

## What it adds to trial-average DSS

Ordinary DSS learns an instantaneous spatial filter:

\[
y_k(t) = \sum_j a_{kj} x_j(t).
\]

Time-shift DSS first treats delayed copies of every sensor as additional
features and then fits the existing estimator as
`DSS(bias=AverageBias(axis="epochs"))`:

\[
y_k(t) = \sum_{j,d} a_{kjd} x_j(t-d).
\]

The resulting component is a multichannel finite-impulse-response filter. This
can separate a reproducible target from interference that overlaps it in the
instantaneous sensor space but differs in delay, propagation, or spectrum.
It is not the package's multi-dataset Joint DSS mode. Joint DSS uses
`AverageBias(axis="datasets")`; TimeShiftDSS uses trial averaging after lag
augmentation.

The initial package implementation deliberately supports only repeated-trial
data: a NumPy array shaped `(channels, times, epochs)` or MNE `Epochs`. Raw and
Evoked inputs and arbitrary DSS bias functions are outside the source-validated
contract.

## Explicit fit

```python
from mne_denoise.dss import TimeShiftDSS

tsdss = TimeShiftDSS(
    lag_samples=[0, 1],
    rank=16,
    n_components=2,
    n_select=1,
    component_action="extract",
)
sources = tsdss.fit_transform(epochs)
```

Positive lag `d` contributes `X(t - d)`. Lags are explicit, include zero, and
are applied separately inside each epoch. Only the time interval shared by all
lags is fitted or extracted. Sensor-valued `retain` and `subtract` operations
keep the original container and timeline and leave edge samples outside that
interval unchanged.

The 2010 paper defines delays as `0, ..., D - 1` but describes its one-sample
delayed-noise simulation as `D = 1`. The package therefore does not expose this
ambiguous shorthand: the reproduction declares `[0, 1]` explicitly.

## Source-derived algorithm

The implementation maps the paper's steps as follows.

| Source claim | Package implementation | Blocking tests/evidence |
| --- | --- | --- |
| 1. Add delayed sensor copies | Lag-major `_lag_augment` with equation `X(t-d)` | Exact lag sign, valid support, and epoch-isolation tests |
| 2. Concatenate trial observations | C-order `(feature, time, epoch)` flattening | Unequal weight-order regression tests |
| 3. PCA and normalize | Internal `DSS` estimator and its canonical covariance whitener | Existing DSS/NoiseTools parity plus full-rank reconstruction |
| 4. Average across trials | Shared `AverageBias(axis="epochs")`, including time-by-epoch weights | Uniform and time-by-epoch weight tests |
| 5. PCA of biased data | Internal `DSS` estimator's biased-covariance rotation | Component ordering and held-out score tests |
| 6. Apply rotation | Frozen lag-space filters | Transform-batch invariance tests |
| 7. Optional CCA | Shared training-only `canonical_correlation` inside the selected reproducible subspace | Shifted-waveform distortion fixture |
| Zero-weight outliers | Minimum weight across every sample touched by a lag window | Lag-window mask-dilation test |
| Relative PCA cutoff | Explicit rank and relative `reg` tolerance | Rank and source-simulation sensitivity sweep |
| Resampling and CV | Whole-trial bootstrap, grouped CV, nested CV, and max-null surrogates | Reduced CI fixtures and checked validation script |

Primary sources:

- [de Cheveigne (2010), Time-shift denoising source separation](https://doi.org/10.1016/j.jneumeth.2010.03.002)
- [de Cheveigne and Parra (2014), Joint denoising source separation](https://doi.org/10.1016/j.neuroimage.2014.05.068)

## Centering and covariance weights

The source/reference-aligned default is `center=False`: DSS is fitted from
uncentered second moments. This preserves meaningful baseline correction. It
also avoids the trial-wise mean removal that the 2014 review shows can create a
repeatable ramp.

`center=True` is a package extension. It fits one weighted global mean of the
valid augmented training observations, stores `feature_mean_`, and uses that
same mean for every later transform. It never computes a transform-batch or
per-epoch mean.

`fit(..., sample_weight=weights)` accepts either `(n_times,)` weights broadcast
over epochs or an exact `(n_times, n_epochs)` matrix. Weights must be finite and
nonnegative. For each valid reference observation, the effective weight is the
minimum across all source samples touched by its lag window. A zero-weight bad
sample therefore invalidates every augmented observation containing it.

For augmented observations \(z_{t,n}\), the repeated-trial contrast is

\[
\bar z_t =
\frac{\sum_n w_{t,n} z_{t,n}}{\sum_n w_{t,n}}.
\]

The baseline second moment uses all valid `(time, epoch)` observations. The
biased second moment uses \(\bar z_t\), weighted by the valid trial mass
\(\sum_n w_{t,n}\). The two matrices use identical lag support.

## Explicit rank and component choice

TimeShiftDSS does not offer `"auto"` selection. Callers provide `rank`,
`n_components`, and, for sensor operations, `n_select`. The fitted object
reports:

- `n_augmented_features_`;
- `positive_weight_observations_`; and
- `effective_observations_`, using the Kish weight formula.

A warning is raised when augmented dimension approaches the effective weighted
observation count. This ratio is only a heuristic: autocorrelated samples supply
fewer independent constraints than their count suggests.

The source simulation has a real rank cliff. In Simulation 2 with explicit
lags `[0, 1]`, five noise sources enter the augmented covariance at three time
states. A rank that keeps only those 15 noise directions omits the target;
admitting the next direction makes temporal cancellation possible. Rank and
lag sensitivity must therefore be reported, not tuned on the final test set.

## Held-out and null validation

Use sklearn splitters on explicit epoch indices. NumPy TSDSS data are
channel-first with epochs on the last axis, so passing the array directly to
`sklearn.model_selection.cross_validate` would incorrectly split channels.
For example:

```python
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

fold_scores = []
epoch_indices = np.arange(data.shape[2])
splitter = GroupKFold(n_splits=5)
for train, test in splitter.split(epoch_indices, groups=run_ids):
    fitted = clone(tsdss).fit(data[:, :, train])
    fold_scores.append(fitted.score(data[:, :, test]))

held_out_score = np.mean(fold_scores)
```

`TimeShiftDSS.score` requires an explicit `n_select` and returns one
trial-average-to-total power ratio for that leading component subspace. This
avoids comparing component ordinals that can reorder across folds and aligns
validation with the cumulative subspace used by `retain` and `subtract`.

For MNE Epochs, use the same splitter and replace `data[:, :, indices]` with
`epochs[indices]`. Means, weights, covariances, filters, and optional CCA must
be fitted on the training fold only.

The repository's checked scientific QA additionally calibrates one predeclared
model against circularly shifted trials. This procedure lives in
`scripts/validate_time_shift_dss.py`, not in the installed denoising API. It
shifts every complete multichannel epoch independently by more than the lag
span, preserving its within-epoch structure while destroying trial locking.

If lags, rank, or subspace size are tuned from the same data, use sklearn nested
cross-validation and repeat that exact search inside every surrogate; otherwise
the null distribution does not include search multiplicity. Continuous
event-marker randomization from the 2014 paper remains a future Raw/event-data
extension. Bootstrap uncertainty is likewise kept in the checked validation
script rather than duplicated as package API.

## Reconstruction and distortion

Standard sensor reconstruction is a weighted least-squares projection of
fitted component activity to the undelayed sensor block. It is tested directly
and reconstructs the valid interval at full rank.

The optional `distortion_control="cca"` implements the paper's step 7. It takes
the explicitly fitted reproducible subspace and learns the single combination
most correlated with undelayed training sensors. The rotation, component mean,
and sensor mean are frozen before held-out transformation. It can reduce FIR
latency and waveform distortion, but it trades repeatability for similarity and
adds another fitted operation that must stay inside validation folds.

## Scientific validation

Paper parity, efficacy, and null-calibration checks belong in the scientific
validation workflow rather than the package unit tests. Run it with:

```bash
python scripts/validate_time_shift_dss.py
```

It records seeds and dependency versions and writes JSON, CSV, and PNG evidence
under `reports/time_shift_dss/` by default. The script includes:

- the 2010 Simulation 2 dimensions: 10 channels, 1000 samples, 100 trials, and
  input SNRs of -20, -40, and -60 dB;
- held-out static-DSS versus TimeShiftDSS output SNR, gain, correlation, latency,
  distortion, and noise attenuation;
- rank and lag-grid sensitivity;
- 100 whole-training-set bootstrap refits with a fixed held-out test set;
- 1000 full-pipeline pure-noise surrogates; and
- the Figure 6 shifted-waveform failure with and without CCA step 7.

`--quick` reduces the repetitions for a local smoke run. It is marked as such
and is not parity evidence. The default run applies predeclared acceptance
gates and exits unsuccessfully if any gate fails. Generated reports are review
artifacts and are intentionally not embedded in the API documentation.

## Interpretation and limitations

- An evoked trial-average contrast retains reproducible activity. It cannot
  decide whether that activity is neural or artifactual.
- Retaining the leading target component intentionally suppresses off-target
  activity, including genuine non-target neural signals. The validation report
  quantifies this rather than calling the operation generally signal
  preserving.
- Components define optimized subspaces, not individual neural sources.
- Preserved edge samples are not FIR-filtered and are excluded from efficacy
  metrics.
- The epoched circular-shift null is a package extension, not the continuous
  random-event surrogate used in the 2014 paper.
- Arbitrary lag grids, optional centering, weighted/regularized CCA, and the
  high-dimensional warning threshold are package contracts rather than claims
  made by the 2010 paper.
- Apply one fitted transform to all study conditions. Fitting each
  condition separately can manufacture apparent condition differences.
