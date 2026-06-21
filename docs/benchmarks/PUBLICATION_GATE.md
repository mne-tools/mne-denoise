# Publication Gate & Launch Checklist

Master checklist for what blocks **Fir compute** vs **publication**. Every item is tagged
`[arm/all] [phase] [blocks: full-run | publication | submission]`. Per-arm cross-references live in each
config and in [STATUS](STATUS.md).

## A. Per-track launch gate (each arm waits only on ITS prerequisites)
A parity failure in one method must not block an unrelated arm.

| Arm | Launch prerequisites |
|---|---|
| line_* | ZapLine L1/L2 evidence · non-spatial comparator available · injection pilot |
| evoked_* | DSS evidence · rank-matched PCA · event-destroyed null pilot |
| ocular_erp_core | frozen EOG-DSS definition · trial-count matching · ocular metrics |
| muscle_ds004505 | frozen ds004505 contrast · phantom wired · reference-tier fairness · ASR calibration |
| ground_truth_* | generic + forward sim · source matching · oracle defined |
| phantom_ds004784 | iCanClean + Auto-CCA + adaptive-filtering adapters · held-out tuning |

**Sweep/curve gate (all arms):** verify sweep implementation, parameter grids, validation-selection,
**pilot 1–3-subject trade-off curves**, runtime/storage estimate, valid Pareto direction, serialised
operating points. Final group-level curves are **outputs** of the full run, not preconditions.

## B. Full multi-subject array gate (10 — `[all][P4][blocks: full-run]`)
1. No contradictory legacy text in the protocol docs.
2. Preservation references defined per arm (independent of fit).
3. Dev/test history frozen + protocol tagged/preregistered ([DEVELOPMENT_AND_TEST_SETS](DEVELOPMENT_AND_TEST_SETS.md)).
4. MEG SSS/cleaning order fixed (raw for line-noise; SSS-without-notch sensitivity).
5. Comparator registry synced to the final methods (stable IDs).
6. ds004505 muscle contrast frozen (feasibility done; no fallback list in config).
7. Confirmatory comparisons predefined ([ANALYSIS_PLAN](ANALYSIS_PLAN.md) §2).
8. Equivalence margins + precision calculations justified (§4–5).
9. Per-track pilot gates passed (§A).
10. Required-comparator evidence met (L1/L2/L3 per [RECONCILIATION](RECONCILIATION.md)).

## C. Essential-10 (software/methods paper — `[all][blocks: publication]`)
1. Numerical-equivalence (parity) tests for native implementations.
2. Real multi-subject MEG benchmark (ds000117) with SSS/rank held constant.
3. Trade-off (attenuation–preservation) curves for each method family.
4. Controlled 50/60-Hz line-noise injection.
5. Phantom ds004784 reference-aware ground truth (independent re-analysis).
6. Forward-model simulation (EEG).
7. Justified equivalence margins.
8. Preregistered/frozen analysis + locked test set.
9. Per-method failure + tuning-budget reporting.
10. EMG-contamination vs genuine neuro-muscular-coupling distinction.

## D. Higher-impact-7 (validation paper — `[blocks: higher-impact target only, NOT minimum]`)
1. Second real dataset for line + ocular. 2. Channel-density sensitivity. 3. Reference-count/quality
sensitivity. 4. Second MEG dataset (OMEGA resting-MEG). 5. Picard already in ground-truth (extend usage).
6. Blinded expert QC subset. 7. Scalability curves (channels × duration).
*(OMEGA and second datasets are deferred — they do not block the minimum software/methods paper.)*

## E. Minor items — classify each as `required_before_submission` / `required_before_archival_release` /
`recommended` / `optional`
- `required_before_submission`: define all acronyms; remove placeholders; report exact dataset versions + access
  dates; report all seeds; PSD units; mean (not peak) N170; shared topo color scales; never call a single
  non-inferiority result "preserved"; separate author-developed methods from independent comparators; state
  secondary-dataset limitations; blinded rule-based manual QC.
- `required_before_archival_release`: flow + inclusion/exclusion diagrams; removed-signal plots; success+failure
  exemplars; covariance/mixing condition numbers; ICA convergence + iteration counts; HP-train/apply note;
  re-reference order; predefined sideband ranges; per-channel median + distribution (not only mean); full-rank
  vs nominal-rank reporting; singular-channel/interpolation handling.
- `recommended` / `optional`: remaining stylistic/supplementary items.

## F. Gate hierarchy (authoritative)
- **Software/methods publication gate** = Essential-10 + all `required_before_submission` minor items + every
  confirmatory claim supported by its arm + no unresolved licensing/provenance/reproducibility problem.
- **Higher-impact validation target** = Essential-10 + Higher-impact-7 + expanded external generalization.
- **Nonblocking** = `recommended`/`optional` minor items. None of these block Fir compute or the minimum suite.
