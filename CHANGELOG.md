# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **BSS-CCA**: `reject={'low', 'high'}` selects which end of the autocorrelation
  spectrum is treated as artifactual. The existing behaviour, `'low'`, drops the
  least autocorrelated components, where muscle concentrates (De Clercq et al.,
  2006). The new `'high'` mode drops the most autocorrelated components, where
  slow drift and movement artifact concentrate.

  This completes parity with `autoLagCCA.m`, the reference implementation
  distributed with the ds004784 phantom dataset, whose `rejHiLo` switch has both
  branches. That dataset's own published parameter sweep selects the high branch
  for its clean, eye and motion conditions and the low branch for its two muscle
  conditions, so a package offering only one branch cannot reproduce four of its
  six artifact classes.

  `reject` defaults to `'low'`, so existing code is unaffected.

- **BSS-CCA**: `threshold_on={'rho', 'rsq'}` sets the scale `rho_threshold` is
  expressed on. `mne-denoise` thresholds the canonical correlation; the
  ds004784 reference implementation thresholds its square (`R.^2 > rsq_thres`),
  so a published `rsq_thres` of 0.57 corresponds to a correlation of 0.755, not
  0.57. `threshold_on='rsq'` lets those values be used verbatim, and the
  selection is verified to match the MATLAB branch component-for-component.

  `threshold_on` defaults to `'rho'`, so existing code is unaffected.

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
