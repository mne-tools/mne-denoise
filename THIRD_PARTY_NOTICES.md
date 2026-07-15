# Third-party dependency and comparator boundaries

`mne-denoise` is distributed under BSD-3-Clause. Its mandatory runtime
dependencies are imported as external packages and are not vendored into the
wheel. At the audited integration environment their package metadata reports:

| Dependency | Role | Reported license expression |
|---|---|---|
| MNE-Python | M/EEG objects and analysis substrate | BSD-3-Clause |
| NumPy | numerical arrays | BSD-3-Clause and bundled permissive notices |
| SciPy | scientific algorithms | BSD-3-Clause and bundled dependency notices |
| Matplotlib | visualization | Matplotlib license and bundled notices |
| scikit-learn | machine-learning utilities | BSD-3-Clause |

Optional benchmark dependencies are installed only when their comparator is
requested:

| Dependency/comparator | Reported license | Packaging boundary |
|---|---|---|
| PyWavelets | MIT and BSD-3-Clause | optional benchmark dependency |
| EMD-signal | Apache-2.0 | optional benchmark dependency |
| python-picard | BSD-3-Clause | optional BSS comparator |
| amica 0.0.1 | BSD-3-Clause | optional BSS comparator |
| MARA resources | GPL-isolated comparator layer | excluded from the BSD wheel and public imports |

The release archive must retain the exact dependency lock and the complete
license texts supplied by each distribution. This inventory documents software
packaging boundaries; it does not replace method-specific academic citation or
offer a legal opinion.

The iCanClean implementation is tracked separately because the primary paper
discloses related patents. Release and manuscript declarations must report the
documented institutional/author determination without independently asserting
freedom to operate.
