# Claim–Evidence Matrix

This table **governs all manuscript wording** (title, abstract, results, discussion, conclusion). A claim may
be made only where this matrix lists direct supporting evidence. Use *"selected denoising methods across EEG
and MEG"* — never *"all `mne-denoise` methods are validated for M/EEG."*

| Claim | Modality | Evidence arm(s) | Evidence type |
|---|---|---|---|
| DSS evoked enhancement | EEG | `evoked_erp_core` (ERP CORE N170) | empirical, multi-subject |
| DSS evoked enhancement | MEG | `evoked_ds000117` (M170, ds000117) | empirical, multi-subject |
| ZapLine / ZapLine-plus line-noise removal | EEG (mobile) | `line_ds003620` | empirical, multi-subject |
| ZapLine / ZapLine-plus line-noise removal | MEG (lab) | `line_ds000117` (raw) | empirical, multi-subject |
| Line-noise removal — preservation under known truth | EEG/sim | `line_injection` | controlled injection |
| Ocular artifact correction (EOG-DSS) | EEG | `ocular_erp_core` | empirical, multi-subject |
| Reference-coupled muscle/movement cleaning (iCanClean, TSPCA, dual-layer) | EEG | `muscle_ds004505` | empirical, multi-subject |
| Reference-aware cleaning under known truth | phantom | `phantom_ds004784` | independent re-analysis (author-created data) |
| Nonlinear BSS / source recovery (IterativeDSS) | EEG/sim | `ground_truth_generic`, `ground_truth_forward` | controlled simulation |
| Numerical/algorithmic fidelity of native implementations | — | `parity.py` (L1/L2/L3) | equivalence testing |
| All other DSS biases / variants / ASR variants | — | parity + canonical examples | implementation/example only — **no efficacy claim** |

## Hard limits on wording
- **One MEG dataset (ds000117)** = a MEG benchmark *arm*, not universal MEG validation. Say so.
- **Phantom (ds004784)** is author-created; describe our use as an *independent re-analysis*, paired with
  `muscle_ds004505` + our own simulation + held-out tuning + non-author metrics — not a fully independent
  external challenge.
- **Source-recovery ground truth (SIR/SDR/SAR/Amari/RRMSE)** exists only for the simulation arms (EEG); MEG
  evidence is empirical + parity. Do not present an EEG-only forward model as "M/EEG ground truth."
- **AdaptiveASR / JugglerASR** = "implemented and parity-tested, not yet comprehensively benchmarked."
- **No single overall winner** — conclusions are per-arm + Pareto + regime-eligibility; no cross-arm leaderboard.
