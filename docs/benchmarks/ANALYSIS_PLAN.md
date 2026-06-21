# Analysis Plan

Pre-specified statistical analysis for the benchmark suite. Each arm has **one** primary target endpoint and
**one** primary preservation endpoint (from [BENCHMARK_PLAN](BENCHMARK_PLAN.md) §2); everything else is
secondary. Metrics a method optimizes are never its own primary endpoint (objective-vs-independent rule).

## 1. Primary endpoints (per dataset arm)
| Arm | Primary target | Primary preservation |
|---|---|---|
| line_injection | residual projection onto `L_known` | off-target waveform/spectral equivalence |
| line_ds003620 | held-out R(f0) (not used for test-tuning) | sideband continuity |
| line_ds000117 | held-out line/harmonic residual | M170 / off-target spectral equivalence |
| evoked_erp_core | held-out face−car N170 effect size | N170 mean-amplitude equivalence |
| evoked_ds000117 | held-out face−scrambled M170 effect | M170 amplitude/topography equivalence |
| ocular_erp_core | held-out scalp blink-locked residual amplitude | N170 mean-amplitude equivalence |
| muscle_ds004505 | independent neck-EMG contamination metric | one frozen task-band contrast |
| ground_truth_generic | neural-source RRMSE | permutation-aligned clean-source correlation |
| ground_truth_forward | sensor/source reconstruction error | known neural-signal preservation |
| phantom_ds004784 | known artifact-source attenuation | known brain-source preservation |

Evoked "target" is **enhancement** (held-out split-half reproducibility / SME), not "attenuation". Evoked-power
SNR is descriptive only (DSS optimizes reproducible evoked structure → circular).

## 2. Confirmatory comparisons (one family per arm; everything else exploratory)
| Arm | Confirmatory comparison(s) |
|---|---|
| Line noise | ZapLine-plus vs ZapLine; ZapLine-plus vs the non-spatial comparator |
| Evoked DSS | DSS vs none; DSS vs rank-matched PCA |
| Ocular | EOG-DSS vs EOG regression; EOG-DSS vs ICA+ICLabel |
| Reference-aware | iCanClean vs TSPCA / Auto-CCA at equal reference access |
| ASR (compact) | Euclidean ASR vs windowed-Riemannian at matched attenuation |
| Ground truth | IterativeDSS vs FastICA / Infomax / Picard |

**~4 manuscript confirmatory claims:** (1) numerical/algorithmic fidelity; (2) ZapLine-family
attenuation–preservation; (3) DSS target enhancement EEG **and** MEG; (4) reference-coupled + transient cleaning
(real + controlled). All other comparisons (IterativeDSS nonlinearity ablations, ref-count sweep, all ASR cutoffs,
every ocular variant, extra ICA algorithms, scalability grids) are exploratory → supplement.

## 3. Unit of inference & models
- Real data: **subject** is the independent unit; repeated conditions nested within subject (mixed-effects).
- Simulation: nested over source-set / mixing matrix `A` / SNR / artifact condition / seed (mixed models or
  hierarchical summaries). **No window-level pseudoreplication.**
- Phantom: repeated trials are **technical repeats**, not subjects; reported separately from the BSS arms.
- Paired within-subject contrasts; bootstrap confidence intervals; permutation tests where distributional
  assumptions are weak; Holm/FDR correction across the secondary family.

## 4. Equivalence / non-inferiority (preservation)
"Non-significant" is **never** reported as "preserved." Each preservation claim uses a pre-specified equivalence /
non-inferiority margin with a declared direction, justified by:
- measurement scale + expected test–retest or split-half variability,
- a smallest effect size of interest (domain-based; ERP CORE supports measurement-based margins),
- sensitivity analyses at stricter and looser margins.
Margins are **not** chosen as 5%/10% defaults, and never after seeing results.

## 5. Precision / power (computed before fixing margins)
For each primary paired contrast, report: expected CI width; attainable equivalence bound at the available N;
minimum detectable within-subject effect; simulation replicate count needed for stable estimates; allowance for
failed methods / excluded recordings.

## 6. Missingness & failures (intention-to-benchmark)
Method runs carry an explicit terminal status (`success | unavailable_dependency | skipped_missing_channels |
failed_convergence | failed_numerical | timeout | excluded_dataset_qc`). For every method report: success rate,
performance conditional on success, contamination severity of failed vs successful cases, and a sensitivity
analysis assigning failures a pre-specified worst-rank penalty. Failed runs stay in the result set.

## 7. Secondary / structural / computational outcomes
Effective-rank change, principal angles, removed-component counts, rejected-window / calibration fractions;
ICLabel probability **distributions** (not only thresholded counts) and decoding (fixed classifier, subject-safe
splits) as **secondary** — better decoding is not automatically better denoising. Computational reporting in two
modes: algorithmic-fairness (1 core, fixed BLAS threads, cached input, warm-up excluded) and practical-deployment
(recommended allocation, multithread, wall time, peak RAM, throughput).

## 8. No global score
Conclusions are per-arm, with Pareto plots and regime-eligibility statements. No normalized cross-arm leaderboard
and no single "best denoiser" claim.
