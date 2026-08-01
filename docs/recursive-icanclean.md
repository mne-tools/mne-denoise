# Experimental recursive iCanClean

`mne_denoise.icanclean.RecursiveICanClean` is a stateful, causal research
prototype for reference-based denoising. It recursively updates joint
primary/reference covariance statistics, periodically solves regularized CCA,
and then applies the component-selection and least-squares subtraction rule
used by iCanClean.

> **Evidence boundary:** this is not a published or author-provided “recursive
> iCanClean” implementation. Published iCanClean uses batch CCA in fixed or
> moving windows. Downey and Ferris (2023) identify recursive CCA as possible
> future work. Results obtained with ordinary iCanClean do not validate this
> class. The prototype currently has invariant and synthetic tests only.

The patent notice in the existing iCanClean implementation also applies: a
public U.S. patent application has been filed for the iCanClean method
(US20230363718A1). Patent rights may affect commercial use.

## What is recursive

For each accepted sample, the estimator updates primary and reference means,
within-view scatter matrices, and the cross-view scatter matrix using a stable
weighted online update. The forgetting factor is a **per-sample** value; it is
never interpreted per transport block.

- `forgetting_factor=None` retains all accepted samples and reproduces batch
  moments up to floating-point roundoff.
- `forgetting_factor=...` specifies a dimensionless per-sample decay.
- `memory_duration_s=...` specifies a physical exponential-memory time constant
  and requires a sampling frequency.

CCA models update after explicit accepted-sample counts. Sample-domain and
physical-time parameters have separate names:

```python
RecursiveICanClean(
    sfreq=250.0,
    warmup_samples=500,
    update_interval_samples=50,
)

RecursiveICanClean(
    sfreq=250.0,
    warmup_samples=None,
    warmup_duration_s=2.0,
    update_interval_samples=None,
    update_interval_s=0.2,
    memory_duration_s=4.0,
)
```

## Combined and separate reference assets

For a combined array or `mne.io.Raw`, select reference channels explicitly.
Unselected channels are preserved:

```python
from mne_denoise.icanclean import RecursiveICanClean

recursive = RecursiveICanClean(
    ref_channels=["REF1", "REF2", "REF3"],
    primary_channels=["EEG1", "EEG2", "EEG3", "EEG4"],
    warmup_samples=500,
    update_interval_samples=50,
)
recursive.fit(calibration_raw)
cleaned = recursive.transform(evaluation_raw)
```

A separate reference recording makes the consumed asset explicit:

```python
recursive = RecursiveICanClean(sfreq=250.0)
recursive.fit(primary_calibration, reference=reference_calibration)
cleaned = recursive.transform(primary_evaluation, reference=reference_evaluation)
```

The separate reference `Raw` is consumed in full; pick its intended reference
channels before passing it. Primary channels follow the package's homogeneous
MNE data-channel selection behavior. MNE channels are aligned by their
calibration names, and their channel types, physical units, and sampling
frequency are locked into the state. Separate primary/reference `Raw` objects
must also have the same `first_samp`. Raw subtype, annotations, first sample,
bad-channel metadata, and unselected primary channels are retained.

The streaming API currently accepts continuous `Raw` and two-dimensional NumPy
arrays; epochs and evoked averages have no unambiguous continuous state order
and are rejected. Raw and NumPy representations cannot be mixed after state is
initialized. Successive Raw calls to `partial_fit()` or `process()` must be
contiguous and ordered. NumPy arrays do not carry sample identities, so stream
order is necessarily the caller's responsibility.

## Causal streaming and contamination gates

`process()` cleans one transport block and, by default, updates after each
accepted sample. A sample therefore affects only future samples. Warm-up output
is passed through unchanged rather than being silently fit on insufficient
statistics.

```python
recursive = RecursiveICanClean(
    sfreq=250.0,
    warmup_samples=500,
    update_interval_samples=50,
    update_order="after",
)

for primary_block, reference_block, safe_to_adapt in stream:
    cleaned_block = recursive.process(
        primary_block,
        reference=reference_block,
        adaptation_mask=safe_to_adapt,
    )
```

`adaptation_mask=False` freezes moments for that sample: the sample is still
cleaned and logged, but it neither updates nor decays recursive covariance.
This is the explicit contamination gate. For a fixed-reference control,
calibrate first and set `adaptation_mode="frozen"`; `process()` then applies the
model without adaptation.

The default `update_order="after"` has zero algorithmic look-ahead and a
one-sample model-update delay. `process()` also records the transport-block size
and duration, because end-to-end deployment latency cannot be inferred from the
algorithm alone. `update_order="before"` allows the current sample to affect its
own filter and is intended only for controlled parity experiments.

`transform()` is always frozen: it never updates means, covariances, model
selection, or adaptation counters. For leakage-free assessment, call `fit()` on
an identified calibration partition and `transform()` on a disjoint evaluation
partition. `fit_transform()` is only a convenience reconstruction of the
calibration samples and must not be scored as held-out effectiveness evidence.

## Replay and diagnostics

Every model update records sample counts, effective weight, covariance ranks,
rank ceiling, squared canonical correlations, selected components, relative
operator change, convergence state, and a SHA-256 checksum of numerical state.
Every `process()` call records transport size, look-ahead, update delay, gated
sample count, warm-up pass-through count, and model versions before and after
the block. An inadmissible numerical update is recorded and retains the last
valid model rather than silently replacing it.

```python
checkpoint = recursive.state_dict()

replay = RecursiveICanClean(**recursive.get_params())
replay.load_state_dict(checkpoint)
replayed = replay.process(next_primary, reference=next_reference)
```

For storage or transfer, use the lossless, canonical UTF-8 JSON form:

```python
payload = recursive.state_json()
replay = RecursiveICanClean(**recursive.get_params()).load_state_json(payload)
```

The checkpoint includes recursive moments, the current operator, update
counters, Raw timeline identity, channel identity/unit signatures, and decision
logs. Arrays retain their exact dtype and bytes in JSON. Loading rejects schema,
version, configuration, geometry, non-finite-value, and checksum mismatches;
loading is transactional, so a rejected payload cannot destroy an existing
valid state.

## Locked invariants and remaining evidence gaps

The tests lock these implementation properties:

- recursive moments equal batch population moments when forgetting is disabled;
- a no-forgetting, near-unregularized frozen model reproduces the existing
  global batch iCanClean subtraction on the locked synthetic bridge;
- recursive state and causal output are invariant to transport-block boundaries
  for a fixed ordered stream and adaptation mask;
- frozen full-array and chunked evaluation are equivalent;
- state checkpoints replay future output and model checksums exactly;
- contamination gates and frozen mode do not mutate adaptation state;
- rank and cleaning decisions are invariant to a common physical-unit scale;
- MNE channel/unit and Raw sample-timeline alignment are explicit.

These are implementation invariants, not effectiveness evidence. Real phantom
or human validation, behavior bridges to an independently implemented recursive
CCA, reference-value sweeps, neural-preservation endpoints, sustained-
contamination controls, adaptation-lag experiments, and deployment latency/
memory audits remain outstanding.

## References

1. Downey, R. J., & Ferris, D. P. (2022). *The iCanClean Algorithm: How to
   Remove Artifacts using Reference Noise Recordings*. arXiv:2201.11798.
2. Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion, Muscle,
   Eye, and Line-Noise Artifacts from Phantom EEG. *Sensors*, 23(19), 8214.
   https://doi.org/10.3390/s23198214
3. Zhao, H., Sun, D., & Luo, Z. (2020). Incremental Canonical Correlation
   Analysis. *Applied Sciences*, 10(21), 7827.
   https://doi.org/10.3390/app10217827
