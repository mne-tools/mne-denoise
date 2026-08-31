(dss)=
# Denoising Source Separation (DSS)

DSS learns spatial filters that maximize a reproducible, spectral, temporal, or
other user-supplied bias relative to a baseline covariance. The family includes
linear, iterative, temporal, and lag-augmented variants
{footcite:p}`sarela2005_dss`.

## Usage

```python
import numpy as np
from mne_denoise.dss import BandpassBias, DSS

rng = np.random.default_rng(0)
data = rng.standard_normal((8, 2000))  # (n_channels, n_times)
bias = BandpassBias((8.0, 12.0), sfreq=250.0)

dss = DSS(bias=bias, n_components=3, component_action="extract")
sources = dss.fit_transform(data)
```

`DSS` accepts supported MNE `Raw`, `Epochs`, and `Evoked` objects. `extract`
returns component arrays, while `retain` and `subtract` return copied
sensor-space containers with metadata preserved. Exact channel and array-layout
rules are in the API reference.

## Component operations

The `component_action` parameter selects the sensor/source operation:

- `"extract"` returns component time courses.
- `"retain"` reconstructs the selected leading components in sensor space.
- `"subtract"` removes the selected components from the input.

NumPy input is channel-first: `(n_channels, n_times)` or
`(n_channels, n_times, n_epochs)`. The estimator learns filters and patterns in
`fit`; `transform` reuses that fitted operator. Components are optimized
directions, not necessarily isolated physical sources.

## Biases

Representative public biases include:

- `AverageBias` for repeated-epoch or dataset averaging;
- `CycleAverageBias` for fixed event-locked windows;
- `BandpassBias`, `LineNoiseBias`, `PeakFilterBias`, and `CombFilterBias` for
  spectral or periodic structure;
- `LagAverageBias`, `SmoothingBias`, and `SpectrogramBias` for temporal or
  time-frequency structure.

`AverageBias(axis="datasets")` is a low-level dataset-first bias operation; it
is not a second array layout accepted by the `DSS` estimator. Likewise,
`CycleAverageBias` is a fixed-window operation, not a complete quasiperiodic
cardiac procedure.

## Iterative DSS

`IterativeDSS` and `iterative_dss` use fixed-point updates with a nonlinear
denoiser such as `KurtosisDenoiser`, `RobustTanhDenoiser`, or a local-variance
mask. Stopping rules, denoiser choice, and component count are explicit user
choices {footcite:p}`sarela2005_dss`.

## DSS variants

### Time-shift DSS

`TimeShiftDSS` augments repeated-trial data with delayed sensor copies and
learns a spatiotemporal DSS subspace
{footcite:p}`decheveigne2010_time_shift`:

```python
from mne_denoise.dss import TimeShiftDSS

model = TimeShiftDSS(
    lag_samples=[0, 1, 2],
    n_components=2,
    rank=4,
    n_select=1,
    component_action="extract",
)
sources = model.fit_transform(epochs)
```

The input is repeated-trial NumPy data `(n_channels, n_times, n_epochs)` or
MNE `Epochs`. The explicit lag grid defines the temporal feature space; only
the common valid support is fitted. `extract`, `retain`, and `subtract` follow
the fitted lag-augmented operator. Optional CCA controls distortion of the
selected subspace.

### Specialized variants

`smooth_dss` creates ordinary DSS with `SmoothingBias`; `ssvep_dss` wraps
`CombFilterBias`; and `narrowband_dss` / `narrowband_scan` provide
frequency-specific DSS convenience functions. The scan returns one leading
DSS score per candidate frequency.

Automatic component-selection helpers are package heuristics. Inspect the
selected components and evaluate attenuation together with preservation of the
signal of interest.

## References

```{footbibliography}
```
