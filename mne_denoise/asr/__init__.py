"""Artifact Subspace Reconstruction (ASR) for MNE-Python.

This module provides an entire suite of Artifact Subspace Reconstruction algorithms
for cleaning continuous and epoched EEG/MEG data. The algorithms function as
scikit-learn compatible transformers, and operate by calibrating a clean spatial
covariance baseline, mapping incoming data into its principal subspace, and
reconstructing dimensions that exceed a statistical threshold.

Implementations
---------------
ASR :
    The standard algorithm. Calibrates a single baseline covariance matrix and
    applies a static subspace reconstruction matrix.

AdaptiveASR :
    Updates the baseline statistics continuously to track signal non-stationarity,
    making it well-suited for long recordings where electrode impedances shift.

JugglerASR :
    A specialized variant that selects and updates reference samples dynamically
    using a sliding window approach, designed for highly unstable environments.

GuidedASR :
    An experimental research prototype that combines the
    ``riemannian_windowed`` ASR backbone with DSS-style bias operators and
    soft component reconstruction. It is not a validated or published method.

Core Workflow
-------------
1. **Calibration** (`calibrate_asr`): Automatically finds clean segments of data
   using robust statistics (geometric median of window covariances).
2. **Processing** (`process_asr`): Compares incoming data to the calibration
   baseline, reconstructing artifactual variance.

References
----------
.. [1] Kothe, C. A. E., & Jung, T. P. (2016). Artifact removal techniques with
       signal reconstruction. U.S. Patent No. 9,474,467.
.. [2] Chang, C. Y., Hsu, S. H., Pion-Tonachini, L., & Jung, T. P. (2018).
       Evaluation of Artifact Subspace Reconstruction for Automatic EEG
       Artifact Removal. Proceedings of the 40th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society,
       1242-1245. https://doi.org/10.1109/EMBC.2018.8512547
.. [3] Chang, C. Y., Hsu, S. H., & Jung, T. P. (2020). Real-time
       implementation of artifact subspace reconstruction for automatic
       artifact components removal from multichannel EEG recordings. IEEE
       Transactions on Biomedical Engineering, 67(4), 1114-1121.
       https://doi.org/10.1109/TBME.2019.2930186
.. [4] Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019).
       A Riemannian Modification of Artifact Subspace Reconstruction for EEG
       Artifact Handling. Frontiers in Human Neuroscience, 13, 141.
       https://doi.org/10.3389/fnhum.2019.00141
"""

# Core Algorithms
# Functions
from ._calibration import calibrate_asr
from ._distribution import fit_rms_distribution
from ._reconstruction import process_asr
from ._windowing import compute_clean_window_mask
from .adaptive import AdaptiveASR
from .core import ASR
from .guided import GuidedASR, process_guided_asr
from .juggler import JugglerASR, select_juggler_reference_samples

__all__ = [
    # Core Estimators
    "ASR",
    "AdaptiveASR",
    "JugglerASR",
    "GuidedASR",
    # Core Functions
    "calibrate_asr",
    "process_asr",
    "process_guided_asr",
    "fit_rms_distribution",
    "select_juggler_reference_samples",
    # QA and Metrics
    "compute_clean_window_mask",
]
