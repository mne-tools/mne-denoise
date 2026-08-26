# Benchmark Suite — Live Status

Per-arm status across sessions. Status vocabulary: `not_started` → `scaffolded` → `runnable` → `verified`
(pilots pass) → `partial` (a required comparator/dataset is unavailable) → `done` (all 10 "done"-gate
conditions met; see [PUBLICATION_GATE](PUBLICATION_GATE.md)).

> **4 scientific tracks / 5 runner programs / 10 dataset arms.** Evoked and ocular are separate experiments,
> so there are 5 runners; MEG enters via dataset configs, not new runners.

## Phase status
| Phase | State | Notes |
|---|---|---|
| P0 docs + config skeletons | done | commit 531a758 |
| P1 foundation (contracts, metrics, simulation, parity, tests) | done | comparators/config/parity/simulation + 5 qa modules; 68 unit tests passing |
| P2 dataset registry + staging + downloaders | code done + **verified on Fir** | datasets.py + stage_dataset.py + downloaders + feasibility script. **Fir: venv_fir built (offline wheelhouse incl. pyyaml), 68 benchmark tests pass, ds004505 staged+validated (sentinel ds004505.ok).** Bench clone at /scratch/sesma/mne-denoise-bench. ERP-CORE OSF node id & EEGdenoiseNet URL/hash still PENDING; ds003620/ds000117/ds004784 not yet downloaded. |
| P3 per-arm configs + 5 runners + sweep machinery | not_started | |
| P4 Fir pilots | not_started | gated on per-track launch gates |

## Per-arm status
| Arm | Runner | Dataset | Config | Status |
|---|---|---|---|---|
| line_injection | line_noise | synthetic/forward | line_noise_injection.yaml | not_started |
| line_ds003620 | line_noise | ds003620 (raw) | line_noise_ds003620.yaml | not_started |
| line_ds000117 | line_noise | ds000117 (raw FIF) | line_noise_ds000117.yaml | not_started |
| evoked_erp_core | evoked | ERP CORE N170 | evoked_erp_core.yaml | not_started |
| evoked_ds000117 | evoked | ds000117 (SSS) | evoked_ds000117.yaml | not_started |
| ocular_erp_core | ocular | ERP CORE N170 | ocular_erp_core.yaml | not_started |
| muscle_ds004505 | muscle | ds004505 | muscle_ds004505.yaml | not_started (contrast `pending_feasibility`) |
| ground_truth_generic | ground_truth | simulation | ground_truth_generic.yaml | not_started |
| ground_truth_forward | ground_truth | simulation (forward) | ground_truth_forward.yaml | not_started |
| phantom_ds004784 | ground_truth | ds004784 | phantom_ds004784.yaml | not_started |

## Dataset acquisition status
| Dataset | project_relative_path | Located | Validated | Notes |
|---|---|---|---|---|
| ds003620 | openneuro/ds003620 | no | — | needs login-node download (small EEG) |
| ds004505 | openneuro/ds004505/raw_bids | **yes (/project)** | **yes (250 Hz, 25 subj, eeg+emg)** | staged+validated 2026-06-21; git-annex content present (61 GB) |
| ds000117 | openneuro/ds000117 | no | — | 19 subjects; large MEG; raw FIF required for line-noise arm |
| ERP CORE N170 | osf/erp-core/n170 | ? | ? | new OSF downloader |
| EEGdenoiseNet | zenodo/eegdenoisenet | ? | ? | new Zenodo downloader |
| ds004784 (phantom) | openneuro/ds004784 | ? | ? | OpenNeuro helper |
