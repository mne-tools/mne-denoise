# mne-denoise

[![CI](https://github.com/mne-tools/mne-denoise/actions/workflows/ci.yml/badge.svg)](https://github.com/mne-tools/mne-denoise/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mne-tools/mne-denoise/branch/main/graph/badge.svg)](https://codecov.io/gh/mne-tools/mne-denoise)
[![PyPI version](https://img.shields.io/pypi/v/mne-denoise.svg)](https://pypi.org/project/mne-denoise/)
[![Python versions](https://img.shields.io/pypi/pyversions/mne-denoise.svg)](https://pypi.org/project/mne-denoise/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://mne.tools/mne-denoise/)
[![Downloads](https://pepy.tech/badge/mne-denoise)](https://pepy.tech/project/mne-denoise)

**Artifact removal for M/EEG, matched to the structure of the contamination.**

There is no universally correct denoiser. Line noise, transient movement bursts,
reference-correlated artifacts and "enhance the response I care about" are
different problems, and they need different information to solve. `mne-denoise`
implements each as a scikit-learn-style estimator that works directly on MNE
objects, and — just as importantly — leaves behind a fitted state you can
inspect to check what it actually did.

## Choosing a method

| Your problem | Method | What information it uses |
|---|---|---|
| Power-line noise, possibly non-stationary | [`ZapLine`](#zapline--power-line-noise) | narrowband spatial structure at the line frequency |
| Line noise, conservative and phase-preserving | [`SpectrumInterpolation`](#spectrum-interpolation) | the spectral neighbourhood of the peak |
| Large transient / movement artifacts | [`ASR`](#asr--transient-artifacts) and variants | abnormal covariance relative to a clean baseline |
| You recorded noise reference channels | [`ICanClean`](#icanclean--reference-based-cleaning) | correlation between scalp and reference channels |
| Channel-specific sensor noise | [`SNS`](#sns--sensor-noise-suppression) | what neighbouring sensors agree on |
| Enhance a response you can define | [`DSS`](#dss--enhancing-a-target-response) | a bias you declare (trial average, band, period…) |

## Features

### ASR — transient artifacts

Artifact Subspace Reconstruction: calibrate a clean-covariance baseline, then
reconstruct segments whose variance leaves that subspace.

- **`ASR`** — the standard algorithm, with a Riemannian-robust calibration
  backend via `method="riemannian_windowed"`
- **`AdaptiveASR`** — tracks non-stationarity; supports streaming through
  `fit()` / `partial_fit()` / `transform()`
- **`JugglerASR`** — sample-wise calibration for extreme-motion recordings where
  the window-based clean-data selector collapses
- **`GuidedASR`** — experimental research prototype, not a validated method
- Validated against MATLAB reference implementations by parity fixtures under
  `tests/parity/`

### DSS — enhancing a target response

Denoising Source Separation finds spatial filters that maximise a property you
declare, rather than variance.

- **Linear and iterative DSS**, with reconstruction back into sensor space
- **20+ pluggable bias functions**: trial-average, spectral, temporal, periodic,
  spectrogram and ICA-style
- **Specialized variants**: time-shift repeatability, SSVEP, narrowband

### ZapLine — power-line noise

- **`ZapLine`** — removes 50/60 Hz and harmonics by spatial filtering, keeping
  the surrounding spectrum intact
- **ZapLine-plus** via `adaptive=True` — automatic frequency detection and
  per-chunk processing for non-stationary contamination
- Built-in spectral checks intended to guard against over-cleaning

### iCanClean — reference-based cleaning

- Removes subspaces shared between scalp channels and dedicated reference
  channels using canonical correlation analysis
- `global`, `sliding`, `calibrated` and `hybrid` operating modes
- Adaptive thresholding with a cap on the fraction of components removed

### Spectrum interpolation

- Replaces amplitude in a narrow band around the line frequency and its
  harmonics **while preserving phase**
- Controls for frequency, harmonics, bandwidth and neighbour width

### SNS — sensor noise suppression

- Reconstructs each sensor from the sensors that agree with it, removing
  channel-specific noise

### Quality assurance

`mne_denoise.qa` provides endpoints for both halves of the question a denoiser
has to answer: did contamination go down, and did neural signal survive —
peak-to-surround ratios, broadband distortion, variance removed, and more.

### Integration

- **MNE-Python**: works with `Raw`, `Epochs` and `Evoked`, or plain NumPy arrays
- **Scikit-learn API**: `fit()`, `transform()`, `fit_transform()`
- **Visualization**: `mne_denoise.viz` for components, spectra, repair
  timelines and calibration diagnostics

## Installation

### From PyPI (recommended)

```bash
pip install mne-denoise
```

### From source (development)

```bash
git clone https://github.com/mne-tools/mne-denoise.git
cd mne-denoise
pip install -e ".[dev]"
```

## Quick Start

### Removing line noise

```python
import mne
from mne_denoise.zapline import ZapLine

raw = mne.io.read_raw_fif("sample-raw.fif", preload=True)

# Standard mode: you know the line frequency
zapline = ZapLine(sfreq=raw.info["sfreq"], line_freq=50.0)
raw_clean = zapline.fit_transform(raw)
print(f"Removed {zapline.n_removed_} components")

# ZapLine-plus: detect the frequency and adapt per chunk.
# Adaptive mode calibrates and cleans together, so use fit_transform.
zapline_plus = ZapLine(sfreq=raw.info["sfreq"], line_freq=None, adaptive=True)
raw_clean = zapline_plus.fit_transform(raw)

results = zapline_plus.adaptive_results_
print(f"Detected {results['line_freq']} Hz over {len(results['chunk_info'])} chunks")
```

### Repairing transient artifacts

```python
from mne_denoise.asr import ASR

# ASR expects high-pass filtered data
raw.filter(1.0, None)

asr = ASR(cutoff=20.0)
raw_clean = asr.fit_transform(raw)

# What did it actually do?
print(f"{asr.sample_mask_.mean():.1%} of samples repaired")
print(f"calibrated on {asr.calibration_info_['calibration_samples']} samples")
```

The `cutoff` default of 20 exists for comparability with reference
implementations. It is not a universally validated operating point — validate it
for your recording regime before freezing it.

### Enhancing an evoked response

```python
import mne
from mne_denoise.dss import DSS, AverageBias

epochs = mne.read_epochs("sample-epo.fif")

# Bias the decomposition toward what repeats across trials
dss = DSS(bias=AverageBias(axis="epochs"), n_components=5)
dss.fit(epochs)

sources = dss.transform(epochs)  # component time courses
pattern = dss.patterns_[:, 0]  # topography of component 1

# To get denoised sensor-space data instead, set return_type on the estimator
dss_sensor = DSS(bias=AverageBias(axis="epochs"), n_components=5, return_type="epochs")
epochs_clean = dss_sensor.fit_transform(epochs)
```

### Extracting oscillations

```python
from mne_denoise.dss import DSS, BandpassBias

bias = BandpassBias(freq_band=(8.0, 12.0), sfreq=epochs.info["sfreq"])
alpha_sources = DSS(bias=bias, n_components=3).fit_transform(epochs)
```

### Cleaning with reference channels

```python
from mne_denoise.icanclean import ICanClean

icc = ICanClean(
    sfreq=raw.info["sfreq"],
    ref_channels=[ch for ch in raw.ch_names if ch.startswith("N-")],
)
raw_clean = icc.fit_transform(raw)
print(f"removed {icc.n_removed_.mean():.1f} components per window")
```

## Documentation

Full documentation is available at **[mne.tools/mne-denoise](https://mne.tools/mne-denoise/)**.

- [Getting Started Guide](https://mne.tools/mne-denoise/getting-started.html)
- [ASR user guide](https://mne.tools/mne-denoise/asr.html)
- [DSS user guide](https://mne.tools/mne-denoise/dss.html)
- [API Reference](https://mne.tools/mne-denoise/api.html)
- [Example Gallery](https://mne.tools/mne-denoise/auto_examples/index.html)

## Architecture

```
mne_denoise/
├── asr/                    # Artifact Subspace Reconstruction
│   ├── core.py             # ASR estimator
│   ├── adaptive.py         # AdaptiveASR (streaming / non-stationary)
│   ├── juggler.py          # JugglerASR (sample-wise calibration)
│   └── guided.py           # GuidedASR (experimental)
├── dss/                    # Denoising Source Separation
│   ├── linear.py           # Core DSS algorithm, DSS estimator
│   ├── nonlinear.py        # Iterative DSS, IterativeDSS estimator
│   ├── denoisers/          # 20+ pluggable bias functions
│   └── variants/           # TSR, SSVEP, narrowband
├── zapline/                # Line noise removal
│   ├── core.py             # ZapLine estimator
│   └── adaptive.py         # ZapLine-plus utilities
├── icanclean/              # Reference-based CCA cleaning
├── spectrum_interpolation/ # Phase-preserving spectral replacement
├── sns/                    # Sensor noise suppression
├── qa.py                   # Artifact and preservation metrics
└── viz/                    # Visualization tools
```

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=mne_denoise --cov-report=html
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/<your-username>/mne-denoise.git
cd mne-denoise
pip install -e ".[dev,docs]"
pre-commit install
```

## References

### ASR

> Kothe, C. A. E., & Jung, T. P. (2016). Artifact removal techniques with signal reconstruction. _U.S. Patent No. 9,474,467_.

> Chang, C.-Y., Hsu, S.-H., Pion-Tonachini, L., & Jung, T.-P. (2020). Evaluation of Artifact Subspace Reconstruction for Automatic Artifact Components Removal in Multi-Channel EEG Recordings. _IEEE Transactions on Biomedical Engineering_, 67(4), 1114-1121.

> Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019). A Riemannian Modification of Artifact Subspace Reconstruction for EEG Artifact Handling. _Frontiers in Human Neuroscience_, 13, 141.

> Kim, H., Chang, C., Kothe, C., Iversen, J. R., & Miyakoshi, M. (2025). Juggler's ASR: unpacking the principles of artifact subspace reconstruction for revision toward extreme MoBI. _Journal of Neuroscience Methods_, 420, 110465.

### DSS

> Särelä, J., & Valpola, H. (2005). Denoising source separation. _Journal of Machine Learning Research_, 6, 233-272.

> de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering. _Journal of Neuroscience Methods_, 171(2), 331-339.

### ZapLine

> de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove power line artifacts. _NeuroImage_, 207, 116356.

> Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for automatic and adaptive removal of frequency-specific noise artifacts in M/EEG. _Human Brain Mapping_, 43(9), 2743-2758.

### iCanClean

> Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion, Muscle, Eye, and Line-Noise Artifacts from Phantom EEG. _Sensors_, 23(19), 8214.

### SNS

> de Cheveigné, A., & Simon, J. Z. (2008). Sensor noise suppression. _Journal of Neuroscience Methods_, 168(1), 195-202.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
