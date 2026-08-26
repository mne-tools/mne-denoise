# Reconciliation: scientific method → package symbol → math → status

Maps each benchmark method to its concrete `mne_denoise` symbol, the exact operation it performs, whether it
uses reference channels, its numerical-equivalence target, and its benchmark status. **Symbols flagged
`VERIFY` must be confirmed against the implementation in P1 before the name is used in the manuscript** — in
particular, do not label anything "TSPCA" unless it implements the TSPCA procedure.

## Methods under test (native)
| Scientific method | Package symbol | Exact operation | Uses refs? | Parity target | Benchmark status |
|---|---|---|---|---|---|
| Linear DSS | `mne_denoise.dss.DSS` / `compute_dss` | joint-decorrelation: generalized eig of biased vs baseline covariance | no | L2 vs NoiseTools/meegkit | active (evoked, ground-truth-exploratory) |
| Iterative (nonlinear) DSS | `mne_denoise.dss.IterativeDSS` / `iterative_dss` | fixed-point extraction w/ pluggable nonlinearity (tanh, kurtosis, gauss, Wiener mask) | no | L3 canonical | active (ground truth) |
| ZapLine | `mne_denoise.zapline.ZapLine` | spatial-subspace removal of line components (DSS on line vs broadband) | no | L2 vs NoiseTools | active (line) |
| ZapLine-plus | `mne_denoise.zapline.ZapLine(adaptive=True)` | adaptive/chunked ZapLine | no | L2/L3 vs reference | active (line) |
| ASR (Euclidean) | `mne_denoise.asr.ASR` | calibrate clean covariance, reconstruct components exceeding threshold | no (calibration) | L1/L2 vs EEGLAB clean_rawdata | active (muscle, compact-ASR) |
| rASR (windowed Riemannian) | `mne_denoise.asr.ASR(method="riemannian_windowed")` → ID `mne_denoise_riemannian_windowed_asr` | per-window Riemannian ASR | no (calibration) | L2/L3 vs Blum reference | active (compact-ASR) |
| AdaptiveASR | `mne_denoise.asr.AdaptiveASR` | continuously updated baseline | no | L2 | deferred ("implemented + parity-tested") |
| JugglerASR | `mne_denoise.asr.JugglerASR` | dynamic reference-sample selection | no | L2 | deferred ("implemented + parity-tested") |
| iCanClean | `mne_denoise.icanclean.ICanClean` / `compute_icanclean` | CCA between data and reference-noise channels; remove correlated subspace | **yes** | L2/L3 vs official (licensing permitting) | active (muscle, phantom) |
| Time-shift DSS | `mne_denoise.dss.time_shift_dss` (`variants/tsr.py`) **VERIFY** | DSS bias targeting temporal predictability across time-shifts | possibly no | L2/L3 | active (muscle, as time-shift DSS — see distinction below) |

## Reference / regression family — keep DISTINCT (do not conflate)
| Scientific method | Package symbol | Exact operation | Uses refs? | Notes |
|---|---|---|---|---|
| Reference regression | `VERIFY` (native lstsq / MNE) | OLS/ridge of data on reference channels | yes | blockwise cross-fit |
| TSR (time-shift regression) | `VERIFY` | regression on time-shifted reference regressors | usually yes | distinct from TSPCA |
| TSPCA | `VERIFY` (meegkit reference, or native if it matches) | PCA/regression in the time-shifted reference space | yes | label "TSPCA" **only if implementation matches** |
| Time-shift DSS | `mne_denoise.dss.time_shift_dss` | DSS bias on temporal predictability (not necessarily reference-based) | possibly no | the native symbol above |

## Comparators (external / baseline)
| Comparator | Source | Required? | Notes |
|---|---|---|---|
| none / no-correction | — | yes | baseline every arm |
| notch | `mne` `raw.notch_filter` | yes (line) | fixed temporal filter |
| non_spatial_line_noise | CleanLine **or** approved sinusoid-regression / spectrum-interpolation | yes (line) | report which family ran |
| rank-matched PCA | sklearn | yes (evoked) | critical control |
| `ica_rank_matched` | `mne.preprocessing.ICA` | yes (evoked 2°) | fixed rank reconstruction |
| `ica_iclabel_rejection` | `mne` + `mne-icalabel` | yes (ocular) | automatic ICLabel artifact removal |
| SSP-EOG / SSP-ECG | `mne.preprocessing.compute_proj_eog/ecg` | yes (ocular) | projection |
| EOG / ECG regression | `mne.preprocessing.EOGRegression` | yes (ocular) | |
| FastICA / Infomax | `mne.preprocessing.ICA` | yes (ground truth) | |
| Picard | `python-picard` | **yes** (ground truth) | strong modern ICA |
| Auto-CCA | `meegkit` / native | yes (phantom) | reference CCA |
| adaptive filtering | native LMS/RLS | yes (phantom) | published iCanClean comparator |
| dual-layer regression | native (blockwise cross-fit) | yes (muscle ref tier) | |
| oracle | derived from known mixing (`A_brain·S_brain`) | yes (ground truth) | upper bound; excluded from significance tests |
| AMICA / Autoreject / asrpy / PCA-OBS | external | optional | |

## ComparatorResult contract (P1)
```text
fit(train, ctx) -> state ; transform(eval, state, ctx) -> ComparatorResult
ComparatorResult( cleaned, model, diagnostics, status, error,
                  runtime_seconds, cpu_seconds, peak_memory_mb,
                  rank_before, rank_after, n_samples_before, n_samples_after,
                  parameters, random_seed, software_versions )
metadata: reference_aware, rank_reducing, requires_fit, manual_selection,
          optional_dependency, ground_truth_only, required_channels,
          supported_input_types, deterministic, information_tier, fit_scope
fit_scope ∈ { train_only | calibration_then_transform | per_recording_unsupervised | window_local }
```
A stable ID maps to exactly one implementation + parameter-selection policy; never reuse an ID for two behaviours.

## Parity evidence levels
- **L1** exact numerical (same fixture, aligned output within a declared tolerance).
- **L2** subspace/algorithmic (projector / principal-angle / repair-decision agreement after sign+permutation
  alignment) — the right level for ICA/DSS-like outputs.
- **L3** canonical-behaviour replication (reproduce a published qualitative/quantitative result).
Golden fixtures must be redistributable; CI must not require MATLAB/proprietary software.
