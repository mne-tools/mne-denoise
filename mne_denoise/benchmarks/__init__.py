"""Benchmark I/O helpers for deferred-group workflows.

Canonical home for ``benchmark_io`` and ``erp_io``.
Previous import paths (``mne_denoise.viz.benchmark_io`` /
``mne_denoise.viz.erp_io``) still work via thin re-export shims
but are deprecated.
"""

from .erp_io import (  # noqa: F401
    ERPGroupData,
    aggregate_erp_results,
    load_subject_erp_results,
    save_subject_erp_results,
)
from .io import (  # noqa: F401
    LineNoiseGroupData,
    aggregate_benchmark_results,
    load_subject_benchmark_results,
    save_subject_benchmark_results,
)
