## [0.0.2] - 2026-09-14

#### Added

- Refactored ``mne_denoise.viz`` into content-oriented modules for themes,
  components, signals, spectra, statistics, and summaries. The former
  method-scoped plotting surface was replaced by explicit, study-agnostic public
  inputs.

  Added the canonical ``mne_denoise.qa`` namespace with array-based and Raw-based
  quality-assurance metrics for artifact suppression and signal preservation. ([#24](https://github.com/mne-tools/mne-denoise/issues/24))
- Added ``mne_denoise.icanclean.ICanClean`` for reference-based artifact removal
  with Canonical Correlation Analysis (CCA). The scikit-learn-style transformer
  supports NumPy and MNE Raw/Epochs inputs, four cleaning modes (``global``,
  ``sliding``, ``calibrated``, and ``hybrid``), configurable cleaning bases, and
  re-referencing. Added quality-control plots for window scores and component
  counts. ([#26](https://github.com/mne-tools/mne-denoise/issues/26))
- Added adaptive ``DSS`` processing for non-stationary artifacts. With
  ``adaptive=True``, ``fit_transform`` fits per-segment operators using a
  ``CovarianceSegmenter`` or ``FixedWindowSegmenter`` and supports smoothing,
  automatic component selection, per-segment caps/floors, and optional
  raised-cosine cross-fading; global ``fit`` remains available for pipeline use.

  Adaptive ``ZapLine`` now reuses the DSS segmentation and processing path and
  gains the corresponding cross-fade and selection behavior. ``CombFilterBias``
  gained ``q_mode``, and ``narrowband_scan`` now rejects unsupported adaptive
  use. ([#28](https://github.com/mne-tools/mne-denoise/issues/28))
- Added ``mne_denoise.asr`` with standard and Riemannian ASR,
  ``AdaptiveASR``, and ``JugglerASR`` estimators. The module follows the
  package's NumPy/MNE fit-transform conventions and includes calibration helpers,
  diagnostics and annotations, memory-bounded covariance processing, and
  reference-backed validation examples. ([#36](https://github.com/mne-tools/mne-denoise/issues/36))
- **DSS**:
  - Centered, unweighted ``DSS`` fits on :class:`mne.Evoked` now use the public
    :func:`mne.compute_covariance` API available in MNE-Python 1.13, while
    weighted and explicitly uncentered fits retain the NumPy covariance path
    (Issue #39). ([#42](https://github.com/mne-tools/mne-denoise/issues/42))
- Added ``mne_denoise.spectrum_interpolation.SpectrumInterpolation``, an FFT-based
  power-line noise remover following Leske & Dalal (2019) and FieldTrip's
  ``ft_preproc_dftfilter`` ``neighbour_fft`` mode. It replaces spectral amplitudes
  around the line frequency and its harmonics with the mean amplitude of
  neighbouring bins while preserving phase. It supports MNE ``Raw``, ``Epochs``,
  and ``Evoked`` objects and NumPy arrays. The lower-level
  ``interpolate_spectrum`` function is also exposed. ([#43](https://github.com/mne-tools/mne-denoise/issues/43))
- **DSS and ZapLine**:
  - ``DSS`` and ``ZapLine`` gained an optional ``whiten`` mode for jointly
    decomposing multiple data channel types (for example, magnetometers,
    gradiometers, and EEG) in one run. A supplied ``noise_cov`` uses MNE's
    covariance whitener; otherwise MNE inputs use channel-type scaling and NumPy
    arrays use per-channel scaling. The default homogeneous-channel behavior is
    unchanged. ([#44](https://github.com/mne-tools/mne-denoise/issues/44))
- Added ``mne_denoise.viz.plot_component_selector``, an interactive Matplotlib
  dashboard for selecting components from ``DSS``, ``IterativeDSS``, and standard
  ``ZapLine``. The returned ``ComponentSelector`` applies selections while
  preserving NumPy layouts or MNE containers and their metadata; adaptive
  ``ZapLine`` is rejected because its component basis varies by segment (Issue
  #38). ([#45](https://github.com/mne-tools/mne-denoise/issues/45))
- Added ``SOUND`` and ``SSPSIR``, forward-model denoisers from the TMS-EEG
  literature for sensor noise and TMS-evoked muscle artifacts. Both provide
  scikit-learn-style ``fit``/``transform`` support for NumPy and MNE
  Raw/Epochs/Evoked inputs and accept an optional ``mne.Forward`` (with an EEG
  spherical fallback where supported). Added ``quantify_overcorrection`` to
  assess source-model signal attenuation and distortion (Issue #32). ([#46](https://github.com/mne-tools/mne-denoise/issues/46))
- Added :class:`mne_denoise.bss_cca.BSSCCA` and
  :func:`mne_denoise.bss_cca.compute_bss_cca` implementing reference-free BSS-CCA
  for broadband muscle-artifact attenuation, with an explicit lag in samples or
  physical time, paper-aligned ``n_remove`` component selection, optional
  contiguous block-wise operation, signed autocorrelation diagnostics, and
  leakage-safe ``fit``/``transform`` semantics. ([#49](https://github.com/mne-tools/mne-denoise/issues/49))
- Added additive Basic SSA decomposition, frequency-guided per-channel cleaning,
  W-correlation diagnostics, and the clustering/PCA/MDL local SSA algorithm of
  Teixeira et al. through :mod:`mne_denoise.ssa`. The scikit-learn transformers
  support NumPy and MNE containers with explicit transductive semantics. The
  functional multichannel entry points are ``compute_basic_ssa`` and
  ``compute_local_ssa``. ([#50](https://github.com/mne-tools/mne-denoise/issues/50))
- Added :class:`mne_denoise.sns.SNS` with fitted batch-invariant centering,
  weighted and robust fitting, iterative projections, chunked execution, shared
  MNE container integration, rank diagnostics, and a documentation gallery. ([#52](https://github.com/mne-tools/mne-denoise/issues/52))
- Added experimental ``GuidedASR``, a DSS-guided ASR variant with artifact and
  preserve bias covariances and soft component reconstruction. It follows ASR's
  NumPy/MNE estimator workflows and applies guidance only to ASR-flagged
  components; soft reconstruction requires ``experimental=True``. The method is
  unpublished and not validated for scientific use. ([#54](https://github.com/mne-tools/mne-denoise/issues/54))
- Replaced the ambiguous DSS ``return_type`` behavior with explicit
  ``component_action`` operations for extracting, retaining, or subtracting
  selected components while preserving segmented adaptive subtraction. ([#69](https://github.com/mne-tools/mne-denoise/issues/69))
- Added ``TimeShiftDSS`` for explicit lag-augmented repeated-trial DSS, with
  weighted fitting, held-out scoring, sensor reconstruction, optional CCA
  distortion control, and a scientific validation script. ([#70](https://github.com/mne-tools/mne-denoise/issues/70))
- Added an explicit cardiac-cleaning composition from MNE ECG detection,
  ``CycleAverageBias``, and subtractive ``DSS``, with a held-out gallery example. ([#71](https://github.com/mne-tools/mne-denoise/issues/71))
- Added ``pseudo_ref`` and ``filter_ref`` to
  :class:`mne_denoise.icanclean.ICanClean`. With ``pseudo_ref=True``, a filtered
  copy of the primary channels supplies the CCA reference for recordings without
  physical reference electrodes; ``filter_ref`` can also filter supplied
  reference channels. Pseudo-reference mode requires ``ref_channels=None`` and a
  filter, while invalid, non-positive, or Nyquist-edge specifications now raise
  during construction (Downey & Ferris, 2023). ([#74](https://github.com/mne-tools/mne-denoise/issues/74))
- Added ``reject={'low', 'high'}`` and ``threshold_on={'rho', 'rsq'}`` to
  :func:`mne_denoise.bss_cca.compute_bss_cca` and
  :class:`mne_denoise.bss_cca.BSSCCA`. ``reject='high'`` drops the most
  autocorrelated components -- slow drift and movement artifact -- instead of
  the least autocorrelated ones -- muscle; ``threshold_on='rsq'`` thresholds the
  squared canonical correlation instead of the correlation itself. Both
  parameters default to the package's existing behaviour. ([#75](https://github.com/mne-tools/mne-denoise/issues/75))
- Added ``threshold='null'`` to :class:`mne_denoise.icanclean.ICanClean`. It
  estimates a data-dependent maximum squared canonical correlation attributable
  to sampling noise using circularly shifted reference surrogates, accounting for
  window length and channel counts, and rejects components above that threshold.
  Use ``null_random_state`` for reproducible surrogates; the null tests shared
  variance with the reference, not whether that variance is artifact.

  The fitted ``max_r2_``, ``thresholds_``, and ``samples_per_variable_``
  attributes record the evidence behind each window's threshold. ([#76](https://github.com/mne-tools/mne-denoise/issues/76))
- Reconciled structured callbacks around the immutable ``ProgressEvent``
  contract across the supported iterative, segmented, and channel-wise methods.
  Callbacks are runtime observers and remain independent from package logging. ([#89](https://github.com/mne-tools/mne-denoise/issues/89))
- Added an optional ``TqdmProgress`` adapter. ``tqdm`` remains optional, and the
  adapter consumes the existing structured progress callbacks. ([#90](https://github.com/mne-tools/mne-denoise/issues/90))
- Separated optional integrations from the base install. The base package
  supports NumPy/SciPy/scikit-learn workflows without MNE-Python, Matplotlib,
  Seaborn, or tqdm; install ``mne-denoise[mne]`` for MNE-Python, ``[viz]`` for
  Matplotlib and Seaborn, and ``[progress]`` for tqdm. ([#95](https://github.com/mne-tools/mne-denoise/issues/95))

#### Fixed

- **ZapLine**:
  - Fixed a bug in `ZapLine` adaptive mode where sampling rate mismatch caused incorrect frequency detection and potential crashes (Issue #16). ([#17](https://github.com/mne-tools/mne-denoise/issues/17))
- **DSS**:
  - Fixed pattern normalization so ``DSS`` and ``IterativeDSS`` reconstructions
    preserve physical signal scaling.
  - Added ``get_normalized_patterns()`` to ``DSS`` and ``IterativeDSS`` for
    unit-normalized visualization patterns. ([#19](https://github.com/mne-tools/mne-denoise/issues/19))
- **ZapLine**:
  - Fixed adaptive ``ZapLine`` ignoring ``rank``, ``reg``, ``nfft``, ``nkeep``,
    ``whiten`` and ``noise_cov``. The per-chunk estimator in the quality-assurance
    retry loop was built from a hand-written argument list that omitted them, so
    every chunk was fitted with defaults regardless of configuration. It is now
    derived from ``get_params()``. ([#28](https://github.com/mne-tools/mne-denoise/issues/28))
- Fixed documentation links and CI failures by moving project URLs to
  ``mne.tools``. ([#33](https://github.com/mne-tools/mne-denoise/issues/33))
- Improved MNE channel handling across denoisers and PSD plots: heterogeneous
  data-channel inputs now select a homogeneous type with a warning, while
  non-data channels and metadata are preserved. PSD plots avoid non-positive
  power warnings.

  Improved ``ZapLine`` automatic selection for high-channel-count or co-equal
  line-noise components, and made DSS covariance checks scale-aware for SI-unit
  MEG data with clearer rank-reduction diagnostics. ([#34](https://github.com/mne-tools/mne-denoise/issues/34))
- :class:`mne_denoise.spectrum_interpolation.SpectrumInterpolation` no longer
  silently ignores an explicit ``sfreq`` when fitted on an MNE object. A declared
  sampling frequency that disagrees with ``info['sfreq']`` now raises instead of
  being discarded. ([#49](https://github.com/mne-tools/mne-denoise/issues/49))
- Fixed standard ``ZapLine`` with ``n_select="auto"`` returning zero components
  when line noise is distributed across several DSS components with a smoothly
  decaying eigenvalue spectrum. ZapLine now uses spectral artifact detection and
  cleaning QA as a fallback when eigenvalue-based selection returns zero. ([#63](https://github.com/mne-tools/mne-denoise/issues/63))
- Fixed ASR spectral shaping, filter-state continuity, physical-unit invariance,
  and ``min_clean_fraction`` calibration semantics. ``AdaptiveASR`` updates are
  now transactional, while Juggler uses continuous filtering, unit-stable GEV
  fitting, and memory-bounded Chebyshev DBSCAN. ``ASR`` and ``GuidedASR`` now
  default to ``filter_kind="asr"``; parity coverage and cutoff guidance were also
  updated. ([#64](https://github.com/mne-tools/mne-denoise/issues/64))
- Fixed DSS fitting and reconstruction for MNE objects with bad channels, and
  preserved Raw annotations, acquisition sample coordinates, projections, and
  sample rejection consistently between baseline and biased covariance estimates. ([#69](https://github.com/mne-tools/mne-denoise/issues/69))
- Fixed 3-D covariance and ``AverageBias`` weights to follow time-by-epoch
  observation order, and made ``DSS`` transforms reuse their fitted global mean. ([#70](https://github.com/mne-tools/mne-denoise/issues/70))
- Fixed ``CycleAverageBias`` so epochs cannot share windows, overlaps are
  event-order invariant, integer averages cannot truncate, and event coordinates,
  boundaries, origins, duplicates, and minimum counts have explicit contracts. ([#71](https://github.com/mne-tools/mne-denoise/issues/71))
- Made documentation builds resilient without dropping real-data gallery output:
  CI now prefetches and caches the complete MNE dataset inventory, ZapLine uses a
  maintained MNE recording instead of the unavailable NoiseTools host, and an
  incomplete build cannot replace the published site. ([#72](https://github.com/mne-tools/mne-denoise/issues/72))
- **iCanClean**:
  - ``threshold`` and ``global_threshold`` now validate values in ``[0, 1]`` and
    handle numeric inputs consistently instead of silently producing pass-through
    or all-component behavior.
  - Re-fitting clears stale global/sliding window-count QC attributes, and the
    ``'calibrated'`` documentation now correctly states that calibration and
    cleaning use the same data. ([#76](https://github.com/mne-tools/mne-denoise/issues/76))
- Restored the final runtime dependency contract: NumPy ``>=1.26,<3``, SciPy
  ``>=1.13``, scikit-learn ``>=1.5``, and joblib ``>=1.4``. MNE remains optional
  (``mne>=1.13.0``), while ``viz`` installs Matplotlib and Seaborn. SSP-SIR now
  uses the reference default ``M = rank(data) - artifact rank`` and errors when
  no positive reconstruction rank remains. ([#123](https://github.com/mne-tools/mne-denoise/issues/123))

#### Documentation

- Added narrative iCanClean documentation covering threshold scales, window
  conditioning, operating modes, and reference construction. ([#76](https://github.com/mne-tools/mne-denoise/issues/76))
- Streamlined repository guidance, contributor documentation, and GitHub contribution templates. ([#115](https://github.com/mne-tools/mne-denoise/issues/115))
- Audited scientific documentation and public docstrings, corrected remaining API and citation mismatches, and improved core API guidance with concise examples and cross-references. ([#116](https://github.com/mne-tools/mne-denoise/issues/116))
- Refocused the examples gallery around scientifically motivated denoising use
  cases, with curated examples across the supported methods and clearer
  preservation and validation controls. Improved gallery layout, thumbnails,
  figure readability, and documentation-data prefetching. ([#121](https://github.com/mne-tools/mne-denoise/issues/121))

#### Removed

- **ZapLine**:
  - Renamed ``n_remove`` to ``n_select``, which ``ZapLine`` now inherits from
    ``DSS`` rather than duplicating under a second name. Accepted values
    (``int`` or ``'auto'``) and the ``n_removed_`` attribute are unchanged.
  - Removed ``mne_denoise.zapline.adaptive.segment_data``, superseded by the
    shared public ``mne_denoise.dss.CovarianceSegmenter`` and
    ``FixedWindowSegmenter``. ([#28](https://github.com/mne-tools/mne-denoise/issues/28))
- Removed the ambiguous DSS ``return_type`` parameter in favor of the single
  ``component_action`` operation contract. ([#69](https://github.com/mne-tools/mne-denoise/issues/69))
- Renamed ``TimeShiftBias`` to ``LagAverageBias`` and removed the ambiguous
  ``time_shift_dss()`` convenience constructor. ([#70](https://github.com/mne-tools/mne-denoise/issues/70))
- Removed ``min_select`` and ``max_prop_remove`` from the public ``ZapLine``
  constructor because these bounds apply only to adaptive per-segment processing.
  Adaptive ``ZapLine`` continues to use a minimum removal of 1 and a maximum
  removal proportion of 0.2 by default; custom adaptive values can be configured
  through ``adaptive_params["n_remove_params"]``. The corresponding parameters
  remain available on the generic ``DSS`` estimator (Issue #63). ([#78](https://github.com/mne-tools/mne-denoise/issues/78))
- Clarified the pre-1.0 public namespace: single-method APIs use direct
  canonical modules, obsolete raw DSS variant aliases and ``dss.utils`` paths
  are removed, and shared blending remains private. QA and visualization are
  exposed through explicit ``mne_denoise.qa`` and ``mne_denoise.viz`` namespaces;
  scientific algorithms are unchanged. ([#84](https://github.com/mne-tools/mne-denoise/issues/84))
- Development-only ``test``, ``docs``, ``dev``, and ``all`` extras were replaced
  by standardized dependency groups. The user-facing ``mne``, ``viz``, and
  ``progress`` extras remain available. ([#96](https://github.com/mne-tools/mne-denoise/issues/96))
- Python 3.11 support has been removed. Python 3.12 is now the minimum
  supported Python version. ([#99](https://github.com/mne-tools/mne-denoise/issues/99))

#### Internal

- Consolidated duplicated internals under package-level ownership. New
  ``mne_denoise._validation`` helpers and new ``mne_denoise._spatial``
  epoch-reshaping helpers replace validation and reshape logic that had been
  copied across ASR, DSS, SNS, ZapLine, spectrum interpolation, the covariance
  utilities, and the visualization module. ([#49](https://github.com/mne-tools/mne-denoise/issues/49))
- Shared weighted CCA, sensor-mixing, and positive-integer validation helpers
  across DSS, BSS-CCA, and SSA. ([#70](https://github.com/mne-tools/mne-denoise/issues/70))
- Added deterministic cardiac DSS scientific validation with isolated-source
  metrics, negative controls, sensitivity sweeps, resampling, and optional ECG SSP
  comparison on a local recording. ([#71](https://github.com/mne-tools/mne-denoise/issues/71))
- Added ``mne_denoise._filtering.design_butter_sos``, replacing independently
  hand-rolled Butterworth SOS design in ICanClean, the DSS covariance segmenter,
  ZapLine's cleanline notch fallback, and DSS's ``BandpassBias``. Also removed
  the manual per-epoch ``sosfiltfilt`` loops in ``BandpassBias``,
  ``PeakFilterBias``, and ``CombFilterBias``: ``sosfiltfilt`` already filters
  along a given axis independently of the others, so a 3-D
  ``(n_channels, n_times, n_epochs)`` array needs no loop. ([#74](https://github.com/mne-tools/mne-denoise/issues/74))
- **SSP-SIR** now delegates Forward-based source-informed projection reconstruction
  to MNE-Python instead of maintaining a local reconstruction implementation. ([#91](https://github.com/mne-tools/mne-denoise/issues/91))
- Package builds now use Hatchling with Git-derived versioning and modern
  distribution metadata. ([#96](https://github.com/mne-tools/mne-denoise/issues/96))
- Continuous integration now validates the supported Python range, minimum
  dependencies, MNE-Python development compatibility, and strict documentation
  builds against both stable and development MNE-Python. ([#97](https://github.com/mne-tools/mne-denoise/issues/97))
- Release distributions are now built and validated once, then published to
  PyPI from the exact validated artifacts using Trusted Publishing. ([#98](https://github.com/mne-tools/mne-denoise/issues/98))
- Developer tooling and CI maintenance now use Spin, prek, immutable GitHub
  Actions, automated lower-bound dependency validation, dependency review, and
  security-focused repository checks. ([#99](https://github.com/mne-tools/mne-denoise/issues/99))

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **ICanClean pseudo-reference mode** (restores functionality removed in
  `e38e8f5` without a changelog entry; see issue #68)
  - `pseudo_ref=True` derives the CCA reference block from the primary
    channels themselves rather than from physical noise electrodes, for
    recordings with no dual-layer cap. Implements the pseudo-reference
    method of Downey & Ferris 2023, *Sensors* 23(19):8214.
  - `filter_ref=(btype, freqs)` shapes the reference block before CCA,
    using scipy's own filter-kind names: `'bandstop'`, `'bandpass'`,
    `'highpass'`, `'lowpass'`; zero-phase 4th-order Butterworth. Usable
    on its own to filter physical reference channels.
  - `ref_channels` must be left as `None` when `pseudo_ref=True`; the two
    are mutually exclusive.
  - `pseudo_ref=True` without `filter_ref` raises: an unfiltered copy of
    the primary block is perfectly correlated with itself and would remove
    the entire signal.

## [0.0.1] - 2026-01-23

### Added

- **DSS Module**: Complete implementation of Denoising Source Separation
  - `DSS` estimator with scikit-learn compatible API
  - `IterativeDSS` for nonlinear/iterative DSS
  - 20+ pluggable denoiser functions:
    - Spectral: `BandpassBias`, `LineNoiseBias`
    - Temporal: `TimeShiftBias`, `SmoothingBias`, `DCTDenoiser`
    - Periodic: `CombFilterBias`, `PeakFilterBias`, `CycleAverageBias`
    - ICA-style: `KurtosisDenoiser`, `SkewDenoiser`, `TanhMaskDenoiser`
  - Variants: `tsr`, `ssvep`, `narrowband`
  - Full MNE-Python integration (Raw, Epochs, Evoked)

- **ZapLine Module**: Line noise removal algorithms
  - `ZapLine` estimator for standard mode
  - `ZapLine` adaptive mode (ZapLine-plus) with automatic frequency detection
  - Per-chunk processing for non-stationary data
  - Quality assurance with spectral checks

- **Visualization**: Component and comparison plotting
  - `plot_dss_components`
  - `plot_dss_sources`
  - `plot_before_after`

- **Documentation**: Sphinx-based documentation with examples
  - 12 DSS examples
  - 5 ZapLine examples
  - API reference

- **Testing**: Comprehensive test suite with 91% coverage
  - Cross-platform: Ubuntu, macOS, Windows
  - Python 3.10, 3.11, 3.12, 3.13

### Changed

- Minimum Python version is now 3.10
