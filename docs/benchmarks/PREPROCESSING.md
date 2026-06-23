# Baseline preprocessing — paper-grounded, per dataset

**Purpose.** The artifact-method-under-test and every comparator must be applied to the **same, dataset-appropriate** data. This document fixes that common stage for each arm, grounded in each dataset's source paper (full texts/PDFs filed under `D:\mne-denoise-reports\documents\dataset_papers\<dataset>\`), not guessed.

**The load-bearing principle.** We adopt each paper's **acquisition / montage / reference / filter conventions** as the *baseline*. We do **NOT** adopt the paper's **artifact-removal** steps (ICA, SSP, ASR, AMICA, iCanClean, regression, Maxwell/SSS line removal) into the baseline — those are the very operations under test, so baking them in would pre-remove the artifacts we benchmark, or contaminate the method-vs-comparator comparison. **Baseline = the minimal common stage at which our method and every comparator plug in identically; artifact removal happens *after* the branch.**

Line-noise guards (already in the configs) remain authoritative: no low-pass below the highest scored harmonic, no resampling before line evaluation, no cleaner before the method/comparator branch, identical input to every method.

Mains frequency is stated explicitly only for ds000117 (UK, 50 Hz) and ds004505 (USA, 60 Hz, via Cleanline); elsewhere it is inferred from recording country and flagged.

---

## Summary table (baseline applied identically to method + comparators)

| Arm / dataset | System | Channels | $f_s$ | Online ref | Mains | **Baseline (adopt)** | **Branch point** (method + comparators apply here) | Excluded from baseline (= under test) |
|---|---|---|---|---|---|---|---|---|
| line_ds003620 | LiveAmp 32 | 32 EEG | 500 | FCz | 50 (AU, inf.) | native 500 Hz; HP 0.5 Hz drift only; ref as-recorded; **no resample/LP/ASR/ICA before line eval** | ZapLine / ZapLine+ vs notch, CleanLine | resample 250, LP 40, ASR, ICA |
| line_ds000117 | Neuromag 306 | 102 mag, 204 grad (+70 EEG) | 1100 | nose | **50 (UK)** | **raw FIF** (no SSS); native 1100 Hz; mag/grad separate; HPI bands (293–328 Hz) excluded from scoring | ZapLine / ZapLine+ vs notch, CleanLine | MaxFilter/SSS (already removed 50 Hz+HPI), resample |
| evoked_ds000117 | Neuromag 306 | 102 mag, 204 grad | 1100 | nose | 50 | SSS data; epoch −100/+800 ms (from −500/+1200 trim); baseline −100/0; mag/grad endpoints separate | DSS vs none, rank-matched PCA, ICA | the artifact-removal comparators themselves |
| evoked_erp_core / ocular_erp_core | Biosemi ActiveTwo | 30 EEG + 3 EOG | 1024 | CMS/DRL | 60 (US) | resample 256; **offline ref = avg(all 33)** for N170; HP 0.1 Hz (non-causal Butterworth); bipolar HEOG/VEOG; epoch −200/+800 ms, baseline −200/0 | evoked: DSS vs PCA/ICA · ocular: EOG-DSS vs EOG-reg / ICA+ICLabel / SSP-EOG | **ICA ocular correction** (under test for ocular), trial-level rejection |
| muscle_ds004505 | LiveAmp 64 ×4 (dual-layer) | 120 scalp + 120 noise + 8 EMG | 500 | CPz | **60 (US)** | HP 1 Hz; resample 250; **common-avg ref per layer (full-rank)**; 60 Hz CleanLine; bad-chan interp; **noise layer + neck EMG preserved as reference** | ASR/rASR/IterativeDSS/time-shift-DSS/iCanClean vs TSPCA, dual-layer reg, fixed-ICA | **AMICA, iCanClean, ASR, time-rejection** (these ARE the methods) |
| phantom_ds004784 | conductive phantom | EEG + injected sources | — | — | — | minimal HP; no resample; known brain/artifact sources preserved as reference | iCanClean / ASR vs Auto-CCA, adaptive filtering | the reference-aware methods themselves |
| ground_truth_{generic,forward} | simulation | synthetic | — | — | — | none (clean sources are generated); mixing applied in-sim | all BSS methods on identical mixed data | — |
| eegdenoisenet (sim source) | — | 1-ch clean/EOG/EMG segs | — | — | — | amplitude-preserving normalization; injected into multichannel sim | n/a (artifact source library) | — |

**Generalization tier (spec-grounded; papers optional refinement):**

| Arm / dataset | Channels | $f_s$ | Mains | Baseline (adopt) | Notes / source |
|---|---|---|---|---|---|
| ssvep tsinghua / beta | 64 EEG | 250 | 50 (CN) | native 250 Hz; bandpass around stim harmonics; per-trial epoch; ref as-recorded | beta paper (arXiv 1911.13045) filed; Tsinghua paper IEEE (optional) |
| cardiac ds007554 | EEG + ECG (+fNIRS) | per BIDS | 50? (verify) | read BIDS native; HP 0.5–1 Hz; ECG as QRS reference | OpenNeuro descriptor (read BIDS sidecars on Fir) |
| cardiac capslpdb | PSG: EEG+EOG+EMG+ECG | per-record (varies) | 50 (IT, inf.) | read EDF native montage/rate; minimal HP; ECG as reference | PhysioNet docs (CAP atlas = scoring, not preprocessing) |
| meg_scaling hcp_meg | 248 mag (4D/BTi) | 2034.5 (HCP manual) | 60 (US, inf.) | raw 4D via `read_raw_bti`; native rate; ref-channel noise-cancel context only | Larson-Prior 2013 (design paper; low-level specs in HCP MEG manual) |
| meg_scaling meg_masc | 208 KIT | 1000 | 50 (NYUAD, inf.) | distributed RAW; native 1000 Hz; supply own 50 Hz notch/HP for scaling test | Gwilliams 2023 (RAW, no author preprocessing) |
| mobile_scalp_ear | 32 scalp + 14 ear + 4 EOG | 500 (raw sourcedata) | 50 (KR, inf.) | **use raw sourcedata, NOT 100 Hz derivatives** (already EOG-removed/line-cleaned/interp/reref) | OSF r7s9b registry note |
| mobile ds004475 | mobile EEG | per BIDS | — | read BIDS native; HP; gait events | OpenNeuro descriptor (read sidecars on Fir) |
| sleep_edfx | 2 EEG (Fpz-Cz, Pz-Oz) + EOG + EMG | 100 | 50 (NL/EU, inf.) | native 100 Hz EDF montage; minimal HP; long-duration streaming | PhysioNet Sleep-EDF docs; Kemp 2000 IEEE (optional) |
| clinical chbmit | 23 EEG | 256 | 60 (US, inf.) | native 256 Hz EDF; minimal HP; stress/robustness only | PhysioNet docs; Shoeb thesis (optional) |
| eegmmidb | 64 EEG | 160 | 60 (US, inf.) | native 160 Hz EDF; minimal HP | PhysioNet docs (BCI2000 paper has no acquisition specifics) |
| p300 brain_invaders / vr_p300 | 16 EEG | 512 / 128 | 50 (FR, inf.) | native rate; HP 1 Hz; P300 epoch + baseline | Zenodo/HAL descriptors (optional) |
| lemon (line replication) | 62 EEG (61 + VEOG) | 2500 | 50 (DE, inf.) | for line-noise replication: native 2500/resample 250; HP drift only — **do NOT use the 1–45 Hz band-passed processed derivative** | Babayan 2019 |

---

## Per-dataset detail (confirmatory core — paper-grounded)

### ds003620 — mobile EEG line noise (`line_ds003620`)
Liebherr et al. 2021, *Sci Rep* [10.1038/s41598-021-01772-8]. BrainVision **LiveAmp 32**, 32 actiCAP Ag/AgCl, **500 Hz**, online ref **FCz**, ground AFz, built-in accelerometer, **no EOG/ECG/EMG**. Auditory oddball across Lab/Field/Campus; site Adelaide AU → **50 Hz mains** (inferred; authors handle line noise only via ASR "line-noise criterion"). *Authors' pipeline (artifact removal — NOT baseline):* resample 250, reref linked-mastoid (TP9+TP10), HP 0.5 Hz, ASR (clean_rawdata: flatline 5 s, corr 0.80, line 4, burst 10), LP 40 Hz, extended-infomax ICA + ICLabel.
**Baseline for the line arm:** keep the **raw 500 Hz** signal, HP 0.5 Hz for drift only, reference as recorded; **no resample, no 40 Hz LP, no ASR, no ICA before line evaluation** (these alter the 50/100/150 Hz content we score). ZapLine/ZapLine-plus and the notch/CleanLine comparators all branch from this identical input.

### ds000117 — simultaneous MEG+EEG faces (`line_ds000117`, `evoked_ds000117`)
Wakeman & Henson 2015, *Sci Data* [10.1038/sdata.2015.1]. **Elekta Neuromag VectorView 306** (102 mag + 204 planar grad) + **70-ch EasyCap** EEG, online ref **nose**, **1100 Hz**, online LP 350 Hz / no HP, HPI coils 293–328 Hz, **mains 50 Hz (UK, explicit; notched inside MaxFilter)**. 19 subjects (16 validation), 6×7.5 min, Famous/Unfamiliar/Scrambled faces (34 ms trigger→stim delay).
- **Line arm baseline:** the **raw FIF** (the supplied MaxFilter/SSS files already removed 50 Hz + harmonics + HPI), native 1100 Hz, **magnetometers and gradiometers denoised separately**, HPI/cHPI bands excluded from scoring. ZapLine/+ + notch/CleanLine branch here.
- **Evoked arm baseline:** SSS-processed data; epoch **−100/+800 ms** (the authors epoch −500/+1200 then trim 400 ms/side), baseline −100/0; M170 measured **separately for mags (ROI-mean/GFP) and planar grads (RMS of pair)**. DSS + rank-matched PCA + ICA branch from the common epoched evoked. *(Authors' example pipeline adds LP 32 Hz + 100 µV VEOG/HEOG trial rejection — apply uniformly as part of the common evoked state, not as a method.)*

### ERP CORE N170 — `evoked_erp_core`, `ocular_erp_core`
Kappenman et al. 2021, *NeuroImage* [10.1016/j.neuroimage.2020.117465]. **Biosemi ActiveTwo**, 30 scalp + HEOG/VEOG, **1024 Hz** (5th-order sinc, 204.8 Hz cutoff), single-ended **CMS@PO1/DRL@PO2**, site UC Davis → **60 Hz mains** (no notch). 40 subjects. Authors: event-code shift, **resample 256**, bipolar HEOG/VEOG, **HP 0.1 Hz non-causal Butterworth**, offline ref **avg(P9,P10)** generally but **average of all 33 sites for N170**.
**Baseline (both arms):** resample 256, **offline reference = average of all 33** (N170 convention), HP 0.1 Hz, bipolar EOG, epoch −200/+800 ms, baseline −200/0. *Evoked:* DSS vs none / rank-matched PCA branch here. *Ocular:* EOG-DSS vs EOG-regression / **ICA+ICLabel** / SSP-EOG branch here — **the authors' ICA ocular correction is itself a comparator (under test), so it is NOT in the baseline.** **GAP:** the exact per-paradigm epoch/measurement window lives in a paper table not in the body text; the standard N170 measurement window (≈110–150 ms, stimulus-locked) and a −200/+800 ms epoch are used pending confirmation from the BIDS sidecars on Fir.

### ds004505 — dual-layer table tennis (`muscle_ds004505`)
Studnicki & Ferris 2024 *Data in Brief* [10.1016/j.dib.2023.110024] + Studnicki et al. 2022 *Sensors* [10.3390/s22155867]. 4× **LiveAmp 64**, dual-layer actiCAP: **120 scalp + 120 noise** + **8 cap electrodes repurposed as neck EMG** (TP9/P9/PO9/O9/O10/PO10/P10/TP10 → sternocleidomastoid + trapezius), accelerometers + Cometa IMU, **500 Hz**, online ref **CPz**, ground Fpz, site Florida USA → **60 Hz (explicit; CleanLine)**. Free behaviour — **no ERP epoching** (segmented by play condition; 5-min standing baselines).
**Baseline:** HP 1 Hz, resample 250, **common-average reference computed separately per layer, full-rank**, **60 Hz CleanLine** (line ≠ the artifact under test here), bad-channel interpolation (>3 SD per layer), **noise layer + neck EMG preserved unmodified as the contamination reference**. ASR/rASR/IterativeDSS/time-shift-DSS/iCanClean (methods) and TSPCA/dual-layer-regression/fixed-ICA (comparators) all branch here. *The authors' AMICA / iCanClean (r>0.85) / ASR(SD 30) / time-rejection ARE the operations under test — excluded from baseline.*

### ds004784 — conductive phantom (`phantom_ds004784`)
Independent re-analysis of the iCanClean phantom (no dataset descriptor; method source filed: `ds004784/icanclean_method_local.*`). Known brain and contaminating sources injected into a conductive phantom. **Baseline:** minimal HP, no resample, known sources preserved as ground-truth reference; iCanClean/ASR (methods) vs Auto-CCA/adaptive-filtering (comparators) branch from identical phantom data. Unit = technical repeats (not subjects).

### EEGdenoiseNet — simulation source library
Zhang et al. 2021, *J Neural Eng* [10.1088/1741-2552/ac2bf8] (arXiv 2009.11662, PDF+markdown filed). 4514 clean EEG + 3400 EOG + 5598 EMG **single-channel** segments. Not benchmarked directly: feeds `ground_truth_*` as artifact/clean sources under known multichannel mixing with amplitude-preserving normalization. No "baseline preprocessing" — mixing and SNR are set in-simulation.

---

## Provenance
PDFs / full texts under `D:\mne-denoise-reports\documents\dataset_papers\<dataset>\`:
- **PDF + markdown:** eegdenoisenet (arXiv), beta_ssvep (arXiv), ds004784 (iCanClean method).
- **PMC full text (.txt):** ds003620, ds000117, ds004505 (×2), meg_masc, lemon, erp_core_n170, hcp_meg.
- **Spec-grounded (PhysioNet/registry/BIDS; paper optional):** eegmmidb, capslpdb, sleep_edfx, chbmit, tsinghua_ssvep, ds007554, ds004475, mobile_scalp_ear, brain_invaders_2012, vr_p300, lemon-line.

**Optional PDF refinements the user may add** (paywalled; baseline already grounded from specs): Tsinghua SSVEP (Wang 2017, IEEE TNSRE 10.1109/TNSRE.2016.2627556), Sleep-EDF (Kemp 2000, IEEE TBME 10.1109/10.867928). Drop into the matching `dataset_papers/<id>/` folder and re-run markitdown to refine those generalization cards.
