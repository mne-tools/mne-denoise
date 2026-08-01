"""Execution policy for the documentation example gallery."""

from __future__ import annotations

import re

# Keep this list explicit. A test audits all gallery scripts for the network and
# process-exit primitives that make them unsafe to execute in an offline build.
EXTERNAL_DATA_EXAMPLES = (
    "asr/plot_07_riemannian_asr.py",
    "asr/plot_11_epochs_and_meg.py",
    "asr/plot_13_pipeline_filter_asr_ica.py",
    "dss/plot_01_dss_fundamentals.py",
    "dss/plot_02_artifact_correction.py",
    "dss/plot_03_evoked_responses.py",
    "dss/plot_04_spectral_dss.py",
    "dss/plot_05_periodic_dss.py",
    "dss/plot_06_temporal_dss.py",
    "dss/plot_07_spectrogram_dss.py",
    "dss/plot_08_blind_source_separation.py",
    "zapline/plot_02_parameter_tuning.py",
    "zapline/plot_03_epoched_data.py",
)


def gallery_filename_pattern(*, execute_external_data_examples: bool) -> str:
    """Return the Sphinx-Gallery execution pattern for this build."""
    if execute_external_data_examples:
        return r"plot_"

    path_patterns = [
        re.escape(path).replace("/", r"[\\/]") for path in EXTERNAL_DATA_EXAMPLES
    ]
    external_data_pattern = "|".join(path_patterns)
    return rf"^(?!.*(?:{external_data_pattern})$).*plot_"
