# mne-denoise

[![Tests](https://github.com/mne-tools/mne-denoise/actions/workflows/tests.yml/badge.svg)](https://github.com/mne-tools/mne-denoise/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mne-tools/mne-denoise/branch/main/graph/badge.svg)](https://codecov.io/gh/mne-tools/mne-denoise)
[![PyPI version](https://img.shields.io/pypi/v/mne-denoise.svg)](https://pypi.org/project/mne-denoise/)
[![Python versions](https://img.shields.io/pypi/pyversions/mne-denoise.svg)](https://pypi.org/project/mne-denoise/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://mne.tools/mne-denoise/)
[![Downloads](https://pepy.tech/badge/mne-denoise)](https://pepy.tech/project/mne-denoise)

`mne-denoise` provides artifact-suppression and signal-denoising methods for
EEG and MEG, with NumPy and MNE-Python integration.

The package contains several complementary methods for spatial, spectral,
statistical, and source-informed denoising. Many methods accept MNE `Raw`,
`Epochs`, and `Evoked` objects directly, and sklearn-style estimators are
provided where that interface fits the method.

## Methods

| Method | Typical purpose |
| --- | --- |
| ASR | Detect and reconstruct transient, high-variance artifact subspaces. |
| BSS-CCA | Suppress lagged-correlation components associated with muscle artifacts. |
| DSS | Extract or suppress components selected by reproducibility, spectral, temporal, or other bias functions. |
| iCanClean | Suppress components shared with reference or pseudo-reference channels. |
| SNS | Suppress sensor-specific noise using spatially correlated neighboring channels. |
| SOUND | Suppress channel-specific noise with a forward-model-based Wiener filter. |
| Spectrum interpolation | Attenuate narrow-band line noise by interpolating FFT amplitudes while preserving phase. |
| SSA | Decompose and reconstruct time series for frequency-guided or local artifact cleaning. |
| SSP-SIR | Project an artifact subspace and reconstruct through a forward model, especially for TMS-evoked muscle artifacts. |
| ZapLine | Remove power-line noise and harmonics with DSS, including an adaptive mode. |

See the [user guide](https://mne.tools/mne-denoise/getting-started.html) and
[API reference](https://mne.tools/mne-denoise/api.html) to choose and
understand a method. Experimental APIs are identified in the documentation.

## Installation

```bash
pip install mne-denoise
```

Optional integrations can be installed with extras:

```bash
pip install "mne-denoise[mne]"
pip install "mne-denoise[viz]"
pip install "mne-denoise[progress]"
```

## Quick start

The example assumes that `raw` is an MNE `Raw` object loaded with
`preload=True`; install the `mne` extra to use it.

```python
from mne_denoise.spectrum_interpolation import SpectrumInterpolation

# `raw` is an mne.io.Raw object loaded with preload=True.
# Set line_freq to the mains frequency in your recording.
cleaner = SpectrumInterpolation(line_freq=60.0, n_harmonics=3)
clean_raw = cleaner.fit_transform(raw)
```

## Documentation

- [Documentation](https://mne.tools/mne-denoise/)
- [Getting started](https://mne.tools/mne-denoise/getting-started.html)
- [API reference](https://mne.tools/mne-denoise/api.html)
- [Example gallery](https://mne.tools/mne-denoise/auto_examples/index.html)

## Citing

When using mne-denoise in scientific work, cite the primary publication(s) for
the method(s) used in your analysis. Method-specific references are provided
in the documentation.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
human contribution guide.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
