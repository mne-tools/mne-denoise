"""
Shared configuration for the ds003620 batch pipeline.

Auto-detects whether we are running on Narval (Compute Canada) or locally
and sets paths accordingly.
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  HPC Detection
# ═══════════════════════════════════════════════════════════════════════════════


def is_narval() -> bool:
    """Return True if running on a Compute Canada / Alliance cluster."""
    return "CC_CLUSTER" in os.environ


def is_slurm_job() -> bool:
    """Return True if we are inside a Slurm job (sbatch / srun)."""
    return "SLURM_JOB_ID" in os.environ


# ═══════════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════════

# Base data directory — matches download_openneuro.py convention
_DEFAULT_DATA_DIR = str(Path.home() / "scratch" / "mnedenoise" / "data")

if is_narval():
    _SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))
    _PROJECT = Path(os.environ.get("PROJECT", "/project"))
    _TMPDIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))

    # Persistent storage — matches download_openneuro.py layout:
    #   DATA_DIR / DATASET_ID / sub-XX / eeg / ...
    DATA_DIR = Path(os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR))
    OUTPUT_DIR = _SCRATCH / "mne_denoise_results"
    MNE_DENOISE_ROOT = _PROJECT / "mne-denoise"

    # Fast local SSD for the current job — copy data here for I/O
    JOB_TMPDIR = _TMPDIR
else:
    _ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = Path(
        os.environ.get(
            "DATA_DIR", str(_ROOT / "examples" / "tutorials" / "data" / "runabout")
        )
    )
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(_ROOT / "batch_results")))
    MNE_DENOISE_ROOT = _ROOT
    JOB_TMPDIR = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_ID = "ds003620"
DATASET_VERSION = "1.1.1"
TASK = "oddball"

# Resolved dataset root (DATA_DIR / DATASET_ID)
DATASET_ROOT = DATA_DIR / DATASET_ID

# Full subject list (44 subjects)
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 45)]

# OpenNeuro → GitHub raw URL for trigger-corrected events
GITHUB_RAW = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/main"


# ═══════════════════════════════════════════════════════════════════════════════
#  Preprocessing Parameters  (Tabi et al., 2021 — faithful replication)
# ═══════════════════════════════════════════════════════════════════════════════

LINE_FREQ = 50.0  # Hz — European recording
RESAMPLE_FREQ = 250.0  # Hz
HP_FREQ = 0.5  # Hz — passband edge
HP_TRANS = 0.5  # Hz — transition width
LP_FREQ = 40.0  # Hz — passband edge
LP_TRANS = 10.0  # Hz — transition width
ASR_CUTOFF = 10  # SD — burst criterion
ICA_THRESHOLD = 0.89  # > 90 % probability (tuned for IC0)
EXCLUDE_CLASSES = {"eye blink", "heart beat"}
MASTOID_CHS = ["TP9", "TP10"]
RANDOM_STATE = 42

# Bad-channel detection (clean_rawdata v2.3)
FLATLINE_CRIT = 5  # s
CORR_CRIT = 0.80
LINENOISE_CRIT = 4  # SD
WIN_LEN_CORR = 5.0  # s
MAX_BROKEN_FRAC = 0.6
IGNORED_QUANTILE = 0.1

# Trimming
TRIM_PAD = 2.0  # s before first / after last stimulus
GAP_THRESH = 8.0  # s — inter-block rest gaps
ORIG_SFREQ = 500.0  # Hz — recording sfreq (before resample)


# ═══════════════════════════════════════════════════════════════════════════════
#  Epoching Parameters
# ═══════════════════════════════════════════════════════════════════════════════

TMIN = -0.200  # s
TMAX = 0.800  # s
BASELINE = (-0.200, 0.0)
STIMTRAK_LAG = -0.200  # s — uniform event-marker shift
REJECT_THRESH = 125e-6  # V — 125 µV peak threshold
WIN_SIZE = 0.100  # s — moving-window artifact rejection
WIN_STEP = 0.050  # s
REJECT_TMIN = -0.200  # s
REJECT_TMAX = 0.600  # s


# ═══════════════════════════════════════════════════════════════════════════════
#  Condition Structure
# ═══════════════════════════════════════════════════════════════════════════════

EVENT_TYPE_MAP = {
    # trial_type        →  (tone, task, environment)
    "ntcount_lab": ("standard", "count", "lab"),
    "t_count_lab": ("deviant", "count", "lab"),
    "ntcount_oval": ("standard", "count", "oval"),
    "t_count_oval": ("deviant", "count", "oval"),
    "ntcount_campus": ("standard", "count", "campus"),
    "t_count_campus": ("deviant", "count", "campus"),
    "ntDontcount_lab": ("standard", "ignore", "lab"),
    "t_Dontcount_lab": ("deviant", "ignore", "lab"),
    "ntDontcount_oval": ("standard", "ignore", "oval"),
    "t_Dontcount_oval": ("deviant", "ignore", "oval"),
    "ntDontcount_campus": ("standard", "ignore", "campus"),
    "t_Dontcount_campus": ("deviant", "ignore", "campus"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Case 1 — Line Noise Removal (ZapLine / DSS+LineNoiseBias)
# ═══════════════════════════════════════════════════════════════════════════════

CASE1_N_HARMONICS = 3  # Number of harmonics to assess


# ═══════════════════════════════════════════════════════════════════════════════
#  Case 2 / Part III — ERP Enhancement / Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

ERP_WINDOWS = {
    "N1": (0.080, 0.130),
    "MMN": (0.100, 0.250),
    "P300": (0.250, 0.500),
}
BASELINE_WIN = (-0.200, 0.000)
PRIMARY_CH = "Cz"
P300_CH = "Pz"
DSS_N_COMPONENTS = 5
DSS_N_KEEP = 3
XDAWN_N_COMPONENTS = 3

PIPE_ORDER = ["Baseline", "xDAWN", "ICA", "DSS"]
