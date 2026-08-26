# MNE-Denoise M/EEG Benchmark Protocol

**Status:** frozen P0 protocol. This document records *final design decisions only* — the
review/amendment history lives in git, not here. Companion documents:
[ANALYSIS_PLAN](ANALYSIS_PLAN.md) · [CLAIM_EVIDENCE_MATRIX](CLAIM_EVIDENCE_MATRIX.md) ·
[PUBLICATION_GATE](PUBLICATION_GATE.md) · [DEVELOPMENT_AND_TEST_SETS](DEVELOPMENT_AND_TEST_SETS.md) ·
[RECONCILIATION](RECONCILIATION.md) · [STATUS](STATUS.md) · [RUNS_LOG](RUNS_LOG.md).

## 1. Purpose & scope
Empirically validate **selected** native `mne_denoise` denoisers across **EEG and MEG** benchmark
arms, on the Compute Canada **Fir** cluster, **CPU-only**, charged to **`rrg-kjerbi`**.

**Claim wording, used everywhere (title/abstract/discussion):** *"We validate selected denoising
methods across EEG and MEG benchmark arms."* Never *"all `mne-denoise` methods are validated for M/EEG."*
What is supported by direct evidence is fixed in [CLAIM_EVIDENCE_MATRIX](CLAIM_EVIDENCE_MATRIX.md).

Methods-under-test are the **native** implementations in `mne_denoise/{asr,dss,zapline,icanclean}`.
External packages appear only as comparators or numerical-equivalence references.

## 2. Structure: 4 scientific tracks → 10 dataset arms (5 runner programs)
Evoked and ocular are separate experiments (hence 5 runner programs, not 4). MEG enters through
**dataset configs, not new runners** — the runners are modality-agnostic (operate on MNE objects).

| Arm | Dataset | Methods + required comparators | Primary target | Primary preservation | Preservation reference (independent of fit) |
|---|---|---|---|---|---|
| `line_injection` | synthetic / forward-model base + `L_known` | ZapLine, ZapLine-plus, notch, non_spatial_line_noise | residual projection onto `L_known` | off-target waveform/spectral equivalence | known pre-injection base |
| `line_ds003620` | ds003620 (raw rate, 50 Hz) | none, notch, non_spatial_line_noise, ZapLine, ZapLine-plus | held-out R(f0) (not test-tuned) | sideband continuity (proxy) | sideband continuity + task response |
| `line_ds000117` | ds000117 **raw FIF** (50 Hz) | same | held-out line/harmonic residual | M170 / off-target spectral equivalence | held-out low-artifact trials |
| `evoked_erp_core` | ERP CORE N170 | none, rank-matched PCA, DSS; `ica_rank_matched` (2°); event-destroyed null | held-out face−car N170 effect size | N170 amplitude/topography equivalence | high-trial-count, low-contamination held-out ERP |
| `evoked_ds000117` | ds000117 (SSS, evoked) | same | held-out face−scrambled M170 effect | M170 amplitude/topography equivalence | held-out low-artifact trials, common MEG preproc |
| `ocular_erp_core` | ERP CORE N170 | none, EOG regression, `ica_iclabel_rejection`, SSP-EOG, EOG-DSS | held-out scalp blink residual | N170 amplitude equivalence | blink-free held-out trials |
| `muscle_ds004505` | ds004505 | free: none, fixed-ICA(/AMICA*), ASR, rASR; ref: regression, TSPCA, dual-layer, iCanClean; + compact ASR cutoff{10,20,30} | independent neck-EMG contamination metric | one **frozen** task-band contrast | low-motion intervals + conservative-rejection contrast |
| `ground_truth_generic` | generic random-mixing simulation | PCA, FastICA, Infomax, Picard, IterativeDSS (tanh/symmetric), oracle | neural-source RRMSE | clean-source correlation | known clean neural sources |
| `ground_truth_forward` | forward-model simulation (EEG lead field) | PCA, FastICA, Infomax, Picard, IterativeDSS, oracle | sensor/source reconstruction error | known neural-signal preservation | known clean neural sources |
| `phantom_ds004784` | iCanClean phantom (independent re-analysis) | none, iCanClean, TSPCA/Auto-CCA, adaptive filtering, ASR | known artifact-source attenuation | known brain-source preservation | known phantom brain sources |

`*` AMICA optional. The **phantom** is an *independent re-analysis of the publicly released iCanClean
phantom* (author-created data): it is **not** reported in the same statistical family as the random-mixture
BSS arms, and it is required for the reference-aware/iCanClean claim only (waiver only on a verified
licensing/technical barrier → `track_status=partial` + narrowed claim).

## 3. Comparator registry v2 (frozen; stable IDs — one ID = one implementation + policy)
Adapter contract (leakage barrier): `state = fit(train, ctx)`, then
`result = transform(eval, state, ctx)` returning a `ComparatorResult` (see RECONCILIATION/contracts).
Each comparator declares a **`fit_scope`**: `train_only` (DSS, rank-matched PCA, ICA, EOG-DSS) ·
`calibration_then_transform` (ASR, rASR) · `per_recording_unsupervised` (ZapLine, ZapLine-plus) ·
`window_local` (iCanClean); TSPCA / dual-layer = blockwise cross-fit.

- **Required deps:** `mne-icalabel`, `python-picard`, `meegkit`, a non-spatial line-noise comparator
  (CleanLine or an approved sinusoid-regression / spectrum-interpolation substitute), and **Auto-CCA +
  adaptive-filtering** adapters (phantom). **Optional:** AMICA, Autoreject, asrpy, PCA-OBS. A missing
  **required** comparator forces that arm to `partial`, never `done`.
- **rASR naming:** `mne_denoise_riemannian_windowed_asr` (= `method="riemannian_windowed"`) is kept
  distinct from `published_rASR_reference`; the term "rASR" is used only after demonstrated equivalence,
  and ASR-vs-rASR is compared at **matched attenuation / reconstruction fraction**, not equal raw cutoffs.
- **Linear DSS with a matched bias** in the ground-truth arms is **exploratory** (labelled, not primary).

## 4. Metrics & validity gating
QA modules under `mne_denoise/qa/` (reuse line-noise `metrics.py`): `ground_truth.py`, `preservation.py`,
`coupling.py`, `structural.py`, `computational.py`. **Validity gating is enforced in code:** RRMSE / clean
correlation need a known clean target; SIR/SDR/SAR need known matched sources; **Amari is computed only for
a known, square, identifiable mixing** (skipped with an explicit reason otherwise); mixing recovery needs
compatible dimensions; sample precision/recall need a known contamination mask.

**Objective-vs-independent rule:** any metric a method optimizes (R(f0) when auto-tuned;
SME/reproducibility for DSS; reference coupling for reference methods) is **secondary**; the per-arm
*independent* endpoint in §2 is primary.

## 5. Simulation & numerical-equivalence frameworks
- `simulation.py`: (a) generic random well-conditioned `A`; (b) forward-model (cortical sources via an EEG
  lead field, realistic ocular/muscle topographies, spatially-correlated sources); optional (c)
  time-varying/convolutive. **Source-recovery ground truth is EEG-only** (MEG evidence is empirical +
  parity). Same `A` within a replicate (`X_train=A·S_train`, `X_test=A·S_test`), independent `A` across
  replicates. **Source matching is solved on the training split and applied to test**; unmatched / duplicated
  / collapsed sources are reported. Line-injection scores residual projection onto `L_known`, off-target
  distortion, and removed-signal similarity — never plain reconstruction error against a base that already
  contains native line noise.
- `parity.py` + `tests/parity/`: three evidence levels — **L1** exact (within tolerance), **L2**
  subspace/algorithmic (projector / principal-angle / repair-decision agreement after sign+permutation
  alignment), **L3** canonical-behaviour replication. Uses redistributable golden fixtures; **CI never
  requires MATLAB or other proprietary software**.

## 6. Datasets & staging
`scripts/config.py` resolves each dataset by its registry `project_relative_path`
(`openneuro/ds003620`, `openneuro/ds004505`, `openneuro/ds000117`, `osf/erp-core/n170`,
`zenodo/eegdenoisenet`, `openneuro/ds004784`) in order `$DATASETS_ROOT/<rel>` →
`/project/rrg-kjerbi/datasets/<rel>` → `$SCRATCH/mne-denoise/datasets/<rel>/<version>`. A dedicated
`scripts/cc/stage_dataset.py` does lock + completion-sentinel + content validation (subject count, files,
sfreq, channel types, events, license, checksum/size, BIDS) — **array jobs are read-only and never
download**. `ds000117` = **19 participants** (16 = published subset; use 19 unless a pre-comparison raw-QC
rule excludes some).

**MEG line-noise preprocessing (frozen):** the line-noise arm operates on the **raw, unprocessed FIF** (full
rank for ZapLine) and reports rank; the supplied MaxFilter/SSS files are **forbidden** for that arm because
they already had 50 Hz + harmonics + HPI removed. A sensitivity branch recomputes SSS **without**
line-frequency removal and then cleans. HPI exclusion bands are derived from the raw metadata/sidecars, not
hard-coded.

## 7. Configs, runners, provenance
One YAML **per dataset arm** under `configs/benchmarks/` (see [STATUS](STATUS.md) for the list), each freezing
dataset+version, subjects/runs/exclusions, channel handling, filtering, splits (fit/selection/eval),
`preservation_reference`, primary+secondary metrics, equivalence margin, and storage policy. A submission
**validator refuses** any required field left `null`, `auto`-without-a-rule, `TBD`, or an unresolved
alternative. The five runners are config-driven (`--subject/--all/--slurm-array/--group-only`), write
per-subject results atomically (temp+rename), and aggregate via the deferred-group workflow. Provenance:
`run_fingerprint = hash(git_sha, config_hash, dataset_manifest_hash, environment_hash)` and
`execution_id = <timestamp>_<slurm_job_id>` → `results/<arm>/<run_fingerprint>/<execution_id>/`.

## 8. Trade-off curves (central)
Each method family is swept across its aggressiveness parameter (DSS components; ZapLine components;
ZapLine-plus threshold; ASR cutoff; iCanClean R²; ICA threshold; EOG-DSS components) to produce
attenuation–preservation curves + a Pareto front + `{default, validation-selected}` operating points, with
tuning budget recorded. **Final group-level curves are outputs of the full run, not a precondition for it.**

## 9. Phasing
- **P0** — this protocol + config skeletons (frozen).
- **P1** — foundation: comparator/`ComparatorResult` contracts, config schema+validator, metric gating,
  generic + forward simulation, source matching, parity framework, computational instrumentation,
  MNE-object integrity tests, unit tests.
- **P2** — dataset registry + staging + downloaders; ds004505 feasibility → freeze the muscle contrast.
- **P3** — per-arm configs + 5 runners + sweep/curve machinery + orchestration. Implementation order:
  line-injection → line-EEG → line-MEG → evoked → ocular → ground_truth_generic + forward →
  phantom_ds004784 → muscle (ds004505) last.
- **P4** — Fir pilots (env build + smoke inside `salloc`, never login): config+dataset validation →
  1-subject smoke → `--array=1-3` pilot curves → group aggregation → log to [RUNS_LOG](RUNS_LOG.md).

## 10. Gates
See [PUBLICATION_GATE](PUBLICATION_GATE.md) for the full checklist. Summary: each arm has its own
**per-track launch gate**; **full multi-subject arrays** additionally require the 10 freeze conditions
(preservation references, dev/test freeze + protocol tag, MEG SSS order, registry sync, frozen muscle
contrast, predefined confirmatory comparisons, justified margins+precision, passed pilots, required-comparator
evidence). The **software/methods publication gate** = Essential-10 + applicable reporting items; the
deferred Higher-impact-7 (OMEGA, second datasets) belongs to the higher-impact target, **not** the minimum.

## 11. Deferred (documented, scaffolded, not built)
SSVEP (Tsinghua/Wang + RESS/CCA), cardiac (MEG+ECG dataset + PCA-OBS), full ASR cutoff/rASR sensitivity,
mobile-ERP generalization (ds003620 walking), OMEGA second MEG dataset, full parameter-sensitivity grids,
comprehensive AdaptiveASR/JugglerASR benchmark (currently "implemented + parity-tested" only), and the
remaining ~15 DSS bias operators as documented examples.
