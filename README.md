# mne-denoise

[![Tests](https://github.com/mne-tools/mne-denoise/actions/workflows/tests.yml/badge.svg)](https://github.com/mne-tools/mne-denoise/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mne-tools/mne-denoise/branch/main/graph/badge.svg)](https://codecov.io/gh/mne-tools/mne-denoise)
[![PyPI version](https://img.shields.io/pypi/v/mne-denoise.svg)](https://pypi.org/project/mne-denoise/)
[![Python versions](https://img.shields.io/pypi/pyversions/mne-denoise.svg)](https://pypi.org/project/mne-denoise/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://mne.tools/mne-denoise/)
[![Downloads](https://pepy.tech/badge/mne-denoise)](https://pepy.tech/project/mne-denoise)

**Artifact removal and signal denoising for EEG and MEG.**

`mne-denoise` provides spatial, spectral, and statistical methods for removing
artifacts and suppressing noise in EEG and MEG recordings.

## Features

### DSS Module

- **Linear DSS**: Extract components based on reproducibility across trials or characteristic frequencies
- **Iterative DSS**: Powerful nonlinear separation for complex non-Gaussian sources
- **20+ Pluggable Denoisers**: Spectral, temporal, periodic, and ICA-style bias functions
- **Specialized Variants**: TimeShiftDSS, SSVEP enhancement, and narrowband oscillation extraction

### ZapLine Module

- **ZapLine**: Efficient removal of power line noise (50/60 Hz) and harmonics
- **ZapLine-plus**: Fully adaptive mode with automatic frequency detection
- **Per-chunk Processing**: Handles non-stationary noise characteristics
- **Quality Assurance**: Built-in spectral checks to prevent over-cleaning

### Integration

- **MNE-Python**: Works directly with `Raw`, `Epochs`, and `Evoked` objects or `numpy` arrays.
- **Scikit-Learn API**: Standard `fit()`, `transform()`, `fit_transform()` interface
- **Visualization**: Built-in plotting for components and cleaning results

## Installation

### Base installation

```bash
pip install mne-denoise
```

### MNE-Python objects

```bash
pip install "mne-denoise[mne]"
```

### Visualization

```bash
pip install "mne-denoise[viz]"
```

### tqdm progress bars

```bash
pip install "mne-denoise[progress]"
```

Extras can be combined:

```bash
pip install "mne-denoise[mne,viz,progress]"
```

### From source (development)

```bash
git clone https://github.com/mne-tools/mne-denoise.git
cd mne-denoise
python -m pip install --upgrade pip
python -m pip install -e . --group dev
```

## Quick Start

### DSS: Enhancing Evoked Responses

DSS finds spatial filters that maximize the ratio of reproducible (evoked) to total power:

The example below uses the optional MNE-Python integration.

```python
import mne
from mne_denoise.dss import DSS, AverageBias

# Load your epoched data
epochs = mne.read_epochs("sample-epo.fif")

# Create DSS with trial-average bias
dss = DSS(bias=AverageBias(), n_components=5, component_action="extract")
dss.fit(epochs)

# Option 1: Extract source time courses
sources = dss.transform(epochs)

# Option 2: Retain the leading two reproducible components in sensor space
enhancer = DSS(
    bias=AverageBias(),
    n_components=5,
    n_select=2,
    component_action="retain",
)
enhanced_epochs = enhancer.fit_transform(epochs)
```

### DSS: Extracting Oscillations

Isolate specific frequency bands (e.g., alpha rhythm):

```python
from mne_denoise.dss import DSS, BandpassBias

# Create bandpass bias for alpha (8-12 Hz)
bias = BandpassBias(sfreq=epochs.info["sfreq"], freq=10, bandwidth=4)

dss = DSS(bias=bias, n_components=3)
alpha_sources = dss.fit_transform(epochs)
```

### ZapLine: Removing Line Noise

Remove 50/60 Hz power line artifacts:

```python
import mne
from mne_denoise.zapline import ZapLine

# Load continuous data
raw = mne.io.read_raw_fif("sample-raw.fif", preload=True)

# Standard mode: specify line frequency
zapline = ZapLine(sfreq=raw.info["sfreq"], line_freq=50.0)
cleaned_data = zapline.fit_transform(raw)

# Adaptive mode: automatic detection and per-chunk processing
zapline_plus = ZapLine(
    sfreq=raw.info["sfreq"],
    line_freq=None,  # Auto-detect
    adaptive=True,
)
cleaned = zapline_plus.fit_transform(raw)
print(f"Detected line frequency: {zapline_plus.detected_freq_} Hz")
```

## Documentation

Full documentation is available at **[mne.tools/mne-denoise](https://mne.tools/mne-denoise/)**.

- [Getting Started Guide](https://mne.tools/mne-denoise/getting-started.html)
- [API Reference](https://mne.tools/mne-denoise/api.html)
- [Example Gallery](https://mne.tools/mne-denoise/auto_examples/index.html)

## 🏗️ Architecture

```
mne_denoise/
├── dss/                    # Denoising Source Separation
│   ├── linear.py           # Core DSS algorithm, DSS estimator
│   ├── nonlinear.py        # Iterative DSS, IterativeDSS estimator
│   ├── denoisers/          # 20+ pluggable bias functions
│   │   ├── spectral.py     # BandpassBias, LineNoiseBias
│   │   ├── temporal.py     # LagAverageBias, SmoothingBias
│   │   ├── periodic.py     # CombFilterBias, PeakFilterBias
│   │   └── ...
│   └── variants/           # Pre-built applications
│       ├── tsr.py          # Time-shift DSS and temporal smoothing
│       ├── ssvep.py        # SSVEP enhancement
│       └── narrowband.py   # Oscillation extraction
├── zapline/                # Line noise removal
│   ├── core.py             # ZapLine estimator
│   └── adaptive.py         # ZapLine-plus utilities
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
python -m pip install --upgrade pip
python -m pip install -e . --group dev
prek install
```

## References

### DSS

> Särelä, J., & Valpola, H. (2005). Denoising source separation. _Journal of Machine Learning Research_, 6, 233-272.

> de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering. _Journal of Neuroscience Methods_, 171(2), 331-339.

### ZapLine

> de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove power line artifacts. _NeuroImage_, 207, 116356.

> Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A completely automatic and highly effective method for removing power line noise. _Human Brain Mapping_, 43(9), 2743-2758.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
