# NeuroImage benchmark archive

Source branch:
`codex/neuroimage-rerun-fix`

Archived source tip:
`c6c197f982b27fbf0b7d10045e21f5d50609f34f`

Status:
Historical scientific benchmark / integration workspace. It is not part of the
`mne-denoise` release package. Do not merge this archive wholesale into
`main`.

Purpose:
Preserve the NeuroImage validation campaign, frozen benchmark protocols,
reproducibility infrastructure, provenance records, and experimental methods.

## Current-main status

Already merged / maintained elsewhere and deliberately not salvaged here:

- SSA
- CCA
- GuidedASR
- SOUND / SSP-SIR
- SpectrumInterpolation
- ordinary DSS / adaptive DSS / ZapLine

Current `main` is authoritative for those implementations. The current-main
cardiac DSS example was also removed from `archive/sandbox`; its benchmark
campaign records remain below as historical provenance.

Preserved here because they were not merged into `main`:

- MWF
- EMD/EEMD
- Wavelet/WICA
- ReferenceBias
- the latest ContinuousDSS adaptation

Benchmark-only material:

- ADJUST
- benchmark comparators, adapters, registries, runners, and QA
- MARA (GPL-derived; archive/research only)

## ContinuousDSS canonical version

`experimental/continuous_dss.py` is the canonical archived
ContinuousDSS prototype. This version was taken from
`codex/neuroimage-rerun-fix`; it is the latest adaptation and supersedes the
earlier `feature/continuous-dss` prototype for any future reimplementation.
It includes the later estimator/replay, Raw/NumPy, chronological, channel-order,
diagnostic, non-finite-block, causal-state, component-tracking, and cross-fade
work. The directly relevant source-tip test is beside it.

## Archive contents

- `experimental/` contains branch-only algorithm snapshots and directly
  relevant tests. The MWF test is the latest historical `tests/test_mwf.py`
  snapshot from commit `c05db1ac04b942b23db9f28f1e1e1693b5e0ffba`, because the
  final source tip retained the inventory reference but not that test file.
  The final source tree had no dedicated EMD or Wavelet test file; their
  benchmark context and inventory remain preserved.
- `provenance/configs/` contains benchmark configs, manifests, protocol/freeze
  files, and the software method inventory.
- `provenance/docs/benchmarks/` contains the benchmark and analysis plans,
  development/test split, preprocessing, reconciliation/status, publication
  gate, filled values, and inferential statistics.
- `provenance/containers/`, `provenance/scripts/`, and
  `provenance/requirements-neuroimage-lock.txt` contain the container,
  Compute Canada / Fir staging and runner infrastructure, download scripts,
  figure/statistics/reproduction scripts, and locked environment.
- `provenance/mne_denoise/benchmarks/` and `provenance/tests/` contain the
  benchmark-only implementation, comparator registry/adapters, and benchmark
  QA tests. They are reference material, not package infrastructure.

## MARA licensing boundary

MARA is retained only for historical benchmark reproducibility. It must not be
copied into the BSD `mne-denoise` package without a separate licensing
decision. Its implementation, vendored data, and license notice are isolated
under `experimental/mara/`; it is not exposed through package imports. The
broader third-party notice is preserved under `provenance/`.

## Notebook decisions

The source-tip notebook set is under `notebooks/source-tip/`:

- The source Paper 1–4 notebooks are byte-for-byte identical to the existing
  `archive/replications/` copies, so no duplicates were added.
- The source Paper 7 ERP CORE notebook is byte-for-byte identical to
  `archive/replications/erp_core_n170_robust_dss.ipynb`, so it was not added
  again.
- Paper 5, Paper 8, Paper 9, and `viz_showcase` are genuinely additional
  notebooks and are preserved here.
- The source-tip Paper 6 ERP and line-noise notebooks are preserved here. The
  existing `archive/runabout/` copies differ in substantive protocol/QA and
  publication-artifact code, not only execution output, so both records are
  retained rather than silently replacing one with the other.

## History and inertness

The complete 213-commit source history is preserved as an ancestor of
`archive/sandbox` through the history-preserving merge commit recorded in
`provenance/refs.txt`. This browsable archive is inert: its files live below
`archive/`, are not imported by `mne_denoise`, are outside normal pytest
discovery (`pyproject.toml` keeps `testpaths = ["tests"]`), and are not added
to Sphinx navigation, CI, package exports, dependencies, or
`pyproject.toml`. No package implementation was reconciled or refactored for
this archival operation.
