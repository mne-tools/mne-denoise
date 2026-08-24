# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
