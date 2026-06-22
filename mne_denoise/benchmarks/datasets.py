"""Dataset registry, path resolution, and validation for the benchmark suite.

Resolution order (registry-relative; see ``docs/benchmarks/BENCHMARK_PLAN.md`` §6)::

    $DATASETS_ROOT/<project_relative_path>
    /project/rrg-kjerbi/datasets/<project_relative_path>
    $SCRATCH/mne-denoise/datasets/<project_relative_path>/<version>   (download target)

``validate_dataset`` does *content* checks (not "directory exists"): subject count,
BIDS metadata, and — when ``deep=True`` and data are present — sampling rate and
channel types via MNE.

Facts marked ``verified_date=None`` must be confirmed against the BIDS sidecars on
Fir before the protocol is frozen (see ``DEVELOPMENT_AND_TEST_SETS.md``).
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    repository: str                       # openneuro | osf | zenodo | synthetic
    project_relative_path: str            # e.g. "openneuro/ds003620"
    download_source: str | None           # how stage_dataset fetches it
    version: str | None = None
    doi: str | None = None
    license: str | None = None
    citation: str | None = None
    expected_subjects: int | None = None
    expected_channels: dict | None = None  # {"eeg": 30, ...}
    expected_sfreq: float | None = None
    expected_line_frequency: float | None = None
    required_channel_types: tuple[str, ...] = ()
    raw_or_processed: str = "raw"
    known_exclusions: tuple[str, ...] = ()
    download_exclude: tuple[str, ...] = ()   # openneuro-py exclude globs (e.g. skip MRI "*/anat/*")
    verified_date: str | None = None
    # access: "open" (auto-downloadable), "registration" (needs account/DUA — user
    # must provide), "confirm" (needs a verified durable URL/accession before download).
    access: str = "open"
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry  (facts to confirm on Fir are flagged verified_date=None)
# ---------------------------------------------------------------------------
REGISTRY: dict[str, DatasetSpec] = {
    "ds003620": DatasetSpec(
        dataset_id="ds003620", repository="openneuro",
        project_relative_path="openneuro/ds003620",
        download_source="openneuro-py:ds003620",
        version="1.1.1", license="CC0", citation="Liebherr et al. 2021 (Runabout)",
        expected_subjects=44, expected_channels={"eeg": 32}, expected_sfreq=500.0,
        expected_line_frequency=50.0, required_channel_types=("eeg",),
        raw_or_processed="raw", notes="mobile EEG oddball; raw rate for line-noise arm",
    ),
    "ds004505": DatasetSpec(
        dataset_id="ds004505", repository="openneuro",
        # On Fir the BIDS root is under raw_bids/ (admin/derivatives/raw_bids layout);
        # verified 2026-06-21: 25 subjects, 120 EEG + 8 EMG + 185 MISC (noise layer = MISC).
        project_relative_path="openneuro/ds004505/raw_bids",
        download_source="openneuro-py:ds004505",
        license="CC0", citation="Studnicki et al. 2023 (table tennis, dual-layer)",
        expected_subjects=25,
        # noise-layer (120) + IMU are typed MISC in channels.tsv, not a distinct type.
        expected_channels={"eeg": 120, "emg": 8, "misc": 185}, expected_sfreq=250.0,
        required_channel_types=("eeg",), raw_or_processed="raw", verified_date="2026-06-21",
        notes="dual-layer EEG + neck EMG + IMU; muscle/reference arm; BIDS under raw_bids/; "
              "noise-layer 120ch typed MISC (identify by name/electrodes in the muscle runner)",
    ),
    "ds000117": DatasetSpec(
        dataset_id="ds000117", repository="openneuro",
        project_relative_path="openneuro/ds000117",
        download_source="openneuro-py:ds000117",
        license="see dataset", citation="Wakeman & Henson 2015 (multimodal faces)",
        # OpenNeuro release = 16 subjects + an 'emptyroom' pseudo-subject (17 sub-* dirs);
        # study N=19 refers to the original paper. Skip the strict count; rely on deep-validate.
        expected_subjects=None, expected_channels={"mag": 102, "grad": 204, "eeg": 70},
        expected_sfreq=1100.0, expected_line_frequency=50.0,
        required_channel_types=("mag", "grad"), raw_or_processed="raw", verified_date="2026-06-21",
        notes="MEG faces (M170); OpenNeuro = 16 subj + emptyroom. LINE-NOISE ARM MUST USE RAW FIF "
              "(supplied SSS files already had 50 Hz+harmonics+HPI removed).",
    ),
    "erp_core_n170": DatasetSpec(
        dataset_id="erp_core_n170", repository="osf",
        # Full ERP CORE project (all 7 paradigms incl. N170); the N170 runner reads
        # its paradigm subdir within this tree.
        project_relative_path="osf/erp-core",
        download_source="osf:thsqg",   # ERP CORE OSF project node (osfclient clone)
        doi="10.18115/D5JW4R", license="CC-BY-SA 4.0",
        citation="Kappenman et al. 2021 (ERP CORE)",
        expected_subjects=40, expected_channels={"eeg": 30, "eog": 3}, expected_sfreq=1024.0,
        required_channel_types=("eeg",), raw_or_processed="raw",
        notes="faces vs cars N170; ROI/window adopted from the ERP CORE resource",
    ),
    "eegdenoisenet": DatasetSpec(
        dataset_id="eegdenoisenet", repository="gin",
        project_relative_path="gin/eegdenoisenet",
        download_source="gin:NCClab/EEGdenoiseNet",  # G-Node/GIN host (not Zenodo)
        doi="10.1088/1741-2552/ac2bf8", license="open",
        citation="Zhang et al. 2021 (EEGdenoiseNet)",
        raw_or_processed="processed",
        notes="clean EEG + EOG/EMG artifact .npy segments → feeds simulation.py mixtures",
    ),
    "ds004784": DatasetSpec(
        dataset_id="ds004784", repository="openneuro",
        project_relative_path="openneuro/ds004784",
        download_source="openneuro-py:ds004784",
        license="CC-BY", citation="Downey & Ferris 2023 (iCanClean phantom)",
        raw_or_processed="raw",
        notes="iCanClean phantom; independent re-analysis; trials are technical repeats",
    ),

    # ============================ extended datasets (user request 2026-06-21) ============================
    # ---- fully open, auto-downloadable (existing mechanisms) ----
    "ds004475": DatasetSpec(
        dataset_id="ds004475", repository="openneuro", project_relative_path="openneuro/ds004475",
        download_source="openneuro-py:ds004475", license="CC0", access="open",
        download_exclude=("*/anat/*",),  # skip MRI (not needed for EEG; a T1w.nii 404s on OpenNeuro S3)
        notes="split-belt treadmill / mobile EEG (verify channel types, raw rate, gait events). "
              "MRI anat excluded (unused + broken object upstream)"),
    "ds007554": DatasetSpec(
        dataset_id="ds007554", repository="openneuro", project_relative_path="openneuro/ds007554",
        download_source="openneuro-py:ds007554", license="CC0", access="open", expected_subjects=30,
        notes="EEG+fNIRS+ECG, motor/cognitive, 3 sessions; very new — verify BIDS/events before confirmatory use"),
    "sleep_edfx": DatasetSpec(
        dataset_id="sleep_edfx", repository="physionet", project_relative_path="physionet/sleep-edfx",
        download_source="physionet:sleep-edfx/1.0.0", license="ODC-BY", access="open", expected_sfreq=100.0,
        notes="Sleep-EDF Expanded; 197 PSG nights; EEG/EOG/chin-EMG; long-duration EOG/EMG + sleep-stage preservation"),
    "capslpdb": DatasetSpec(
        dataset_id="capslpdb", repository="physionet", project_relative_path="physionet/capslpdb",
        download_source="physionet:capslpdb/1.0.0", license="ODC-BY", access="open",
        notes="CAP Sleep; >=3 EEG + EOG + chin/tibial EMG + ECG; ocular/cardiac/muscle reference + long-data"),
    "eegmmidb": DatasetSpec(
        dataset_id="eegmmidb", repository="physionet", project_relative_path="physionet/eegmmidb",
        download_source="physionet:eegmmidb/1.0.0", license="ODC-BY", access="open", expected_subjects=109,
        notes="EEG Motor Movement/Imagery; 64ch; ERD/ERS preservation + large-N external test"),
    "chbmit": DatasetSpec(
        dataset_id="chbmit", repository="physionet", project_relative_path="physionet/chbmit",
        download_source="physionet:chbmit/1.0.0", license="ODC-BY", access="open",
        notes="CHB-MIT scalp EEG; long clinical; robustness/scaling stress test (large)"),

    # ---- need a verified durable URL / accession before download (access=confirm) ----
    "tsinghua_ssvep": DatasetSpec(
        dataset_id="tsinghua_ssvep", repository="http_portal", project_relative_path="ssvep/tsinghua-benchmark",
        download_source="thu:yijun", access="portal_unversioned",
        expected_subjects=35, license="not_stated",
        notes="Tsinghua 40-target SSVEP (primary SSVEP). THU portal upload/yijun (S1-S35.mat.7z + "
              "metadata); no DOI/version -> SHA256SUMS manifest saved on download"),
    "beta_ssvep": DatasetSpec(
        dataset_id="beta_ssvep", repository="http_portal", project_relative_path="ssvep/beta",
        download_source="thu:liubingchuan_eld_BETA_database", access="portal_unversioned",
        expected_subjects=70, license="not_stated",
        notes="BETA SSVEP (external replication). THU portal upload/liubingchuan_eld_BETA_database "
              "(S1-S10..S61-S70.tar.gz + Description); no DOI/version -> SHA256SUMS manifest"),
    "eegeyenet": DatasetSpec(
        dataset_id="eegeyenet", repository="osf", project_relative_path="osf/eegeyenet",
        download_source="osf:ktv7m", access="open", expected_subjects=356, license="verify_osf_metadata",
        notes="EEG + eye-tracking (external ocular). OSF node ktv7m; verify OSF license metadata"),
    "brain_invaders_2012": DatasetSpec(
        dataset_id="brain_invaders_2012", repository="zenodo", project_relative_path="zenodo/brain-invaders-2012",
        download_source="zenodo:2649006", access="open", expected_subjects=25,
        notes="P300 (BI.EEG.2012-GIPSA). Zenodo 10.5281/zenodo.2649006 — direct, no MOABB needed"),
    "vr_p300": DatasetSpec(
        dataset_id="vr_p300", repository="zenodo", project_relative_path="zenodo/vr-eeg-2018-gipsa",
        download_source="zenodo:2605204", access="open", expected_subjects=21,
        notes="VR vs PC P300 (VR.EEG.2018-GIPSA). Zenodo 10.5281/zenodo.2605204"),
    "meg_masc": DatasetSpec(
        dataset_id="meg_masc", repository="osf", project_relative_path="osf/meg-masc",
        download_source="osf:ag3kj", access="open", expected_subjects=27,
        notes="MEG-MASC natural speech (2nd MEG). OSF node ag3kj (BIDS) — NOT OpenNeuro"),
    "mobile_scalp_ear": DatasetSpec(
        dataset_id="mobile_scalp_ear", repository="osf", project_relative_path="osf/mobile-scalp-ear-eeg",
        download_source="osf:r7s9b", access="open", expected_subjects=24, license="CC-BY-4.0",
        doi="10.17605/OSF.IO/R7S9B", expected_sfreq=500.0,
        notes="scalp(32)+ear(14) EEG, 4 EOG, IMU; ERP+SSVEP under motion (0/0.8/1.6/2.0 m/s). OSF r7s9b, "
              "EEG-BIDS/BrainVision. BENCHMARK: use raw sourcedata (500 Hz), NOT the 100 Hz derivatives "
              "(already EOG-removed / line-cleaned / bad-chan-interpolated / rereferenced)"),
    "wearbci": DatasetSpec(
        dataset_id="wearbci", repository="onedrive", project_relative_path="mobile/wearbci",
        download_source="confirm:onedrive", access="confirm", expected_subjects=36, license="unknown",
        notes="EEG+IMU+video (HKUST-MINSys-Lab/WearBCI-Dataset). BLOCKED: no dataset license, OneDrive "
              "is not a durable versioned archive — awaiting author license clarification"),
    "japanese_eeg_emg": DatasetSpec(
        dataset_id="japanese_eeg_emg", repository="openneuro", project_relative_path="openneuro/ds007808",
        download_source="openneuro-py:ds007808", access="open", expected_subjects=3, license="CC0",
        notes="JapanEEG ~1020h EEG + facial EMG + audio (BIDS). OpenNeuro ds007808. STORAGE: ~1 TB — confirm before staging"),

    # ---- require registration / data-use agreement (user must obtain access) ----
    "omega": DatasetSpec(
        dataset_id="omega", repository="controlled_nextcloud", project_relative_path="meg/omega",
        download_source="registration:omega", access="registration",
        notes="Open MEG Archive (60-Hz MEG). Register: mcgill.ca/bic/omega-registration "
              "(ethics approval required). Landing: mcgill.ca/bic/neuroinformatics/omega. NOT automated"),
    "camcan": DatasetSpec(
        dataset_id="camcan", repository="controlled_sftp", project_relative_path="meg/camcan",
        download_source="registration:camcan", access="registration",
        notes="Cam-CAN lifespan MEG (multi-TB). Apply (supervisor required): "
              "opendata.mrc-cbu.cam.ac.uk/projects/camcan/request/ ; package phase2_arm1_raw_meg. NOT automated"),
    "hcp_meg": DatasetSpec(
        dataset_id="hcp_meg", repository="balsa", project_relative_path="meg/hcp",
        download_source="registration:hcp", access="registration", expected_subjects=95,
        notes="HCP Young Adult MEG (~95 subj, multi-TB) via BALSA balsa.wustl.edu; accept terms. "
              "Auto-stage possible AFTER terms acceptance (no AWS creds needed). NOT automated yet"),
    "tuh_eeg": DatasetSpec(
        dataset_id="tuh_eeg", repository="controlled_nedc", project_relative_path="clinical/tuh-eeg",
        download_source="registration:tuh", access="registration",
        notes="TUH EEG corpus. Request access: isip.piconepress.com/projects/tuh_eeg/ . NOT automated"),
    "tuh_artifact": DatasetSpec(
        dataset_id="tuh_artifact", repository="controlled_nedc", project_relative_path="clinical/tuh-artifact",
        download_source="registration:tuh", access="registration",
        notes="TUH EEG Artifact Corpus. Request: isip.piconepress.com/projects/nedc/html/resources.shtml. NOT automated"),
    "deap": DatasetSpec(
        dataset_id="deap", repository="controlled_http", project_relative_path="ocular/deap",
        download_source="registration:deap", access="registration", expected_subjects=32,
        notes="DEAP 32ch EEG + peripheral. Register: eecs.qmul.ac.uk/mmv/datasets/deap/ . NOT automated"),
    "lemon": DatasetSpec(
        dataset_id="lemon", repository="s3", project_relative_path="s3/lemon-eeg",
        download_source="s3:fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/",
        access="open", license="PDDL",
        notes="MPI Leipzig LEMON resting EEG (raw BIDS) via anonymous INDI S3 — no registration. "
              "MRI/preprocessed/localizer subsets skipped (raw only)"),
}


def get_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in REGISTRY:
        raise KeyError(f"unknown dataset {dataset_id!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[dataset_id]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def _scratch_root(env: dict) -> pathlib.Path:
    if env.get("SCRATCH"):
        return pathlib.Path(env["SCRATCH"])
    return pathlib.Path(env.get("HOME", ".")) / "scratch"


def candidate_roots(dataset_id: str, *, version: str | None = None,
                    env: dict | None = None) -> list[pathlib.Path]:
    """Ordered candidate roots (existing or not) for a dataset."""
    env = dict(os.environ if env is None else env)
    spec = get_spec(dataset_id)
    rel = spec.project_relative_path
    out: list[pathlib.Path] = []
    if env.get("DATASETS_ROOT"):
        out.append(pathlib.Path(env["DATASETS_ROOT"]) / rel)
    out.append(pathlib.Path("/project/rrg-kjerbi/datasets") / rel)
    # Scratch download target — same <rel> layout as the shared roots (no version
    # leaf) so openneuro-py's DATA_DIR/<id> download lands exactly here.
    out.append(_scratch_root(env) / "mne-denoise" / "datasets" / rel)
    return out


def resolve_dataset_root(dataset_id: str, *, version: str | None = None,
                         env: dict | None = None) -> pathlib.Path:
    """First existing candidate root; else the scratch download target (last candidate)."""
    cands = candidate_roots(dataset_id, version=version, env=env)
    for c in cands[:-1]:
        if c.exists():
            return c
    return cands[-1]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(root, dataset_id: str, *, deep: bool = False) -> list[str]:
    """Return a list of validation issues; empty list ⇒ valid (not just present)."""
    spec = get_spec(dataset_id)
    root = pathlib.Path(root)
    issues: list[str] = []
    if not root.exists():
        return [f"root does not exist: {root}"]

    if spec.repository == "openneuro":
        if not (root / "dataset_description.json").is_file():
            issues.append("missing dataset_description.json (BIDS)")
        if spec.expected_subjects is not None:
            n = sum(1 for p in root.glob("sub-*") if p.is_dir())
            if n != spec.expected_subjects:
                issues.append(
                    f"subject count {n} != expected {spec.expected_subjects}"
                )
    else:
        # osf (full OSF project) / gin (git-annex) / zenodo: custom layouts —
        # just confirm the download produced content (no BIDS subject convention).
        if not any(root.iterdir()):
            issues.append("directory is empty (download produced nothing)")

    # deep (read a recording, check sfreq/channels) only applies to BIDS/openneuro.
    if deep and spec.repository == "openneuro":
        issues.extend(_deep_validate(root, spec))
    return issues


def _deep_validate(root: pathlib.Path, spec: DatasetSpec) -> list[str]:
    """Read one recording (if present) and check sfreq/channel types. Needs MNE + data."""
    issues: list[str] = []
    # Primary BIDS recordings only: skip git-annex/datalad internals, sourcedata/
    # (raw vendor files, often incomplete), and derivatives/.
    _skip = {".git", "sourcedata", "derivatives"}
    hits: list[pathlib.Path] = []
    for pat in ("*.vhdr", "*.fif", "*.edf", "*.set"):
        hits = sorted(p for p in root.rglob(pat)
                      if p.is_file() and not (_skip & set(p.parts)))
        if hits:
            break
    if not hits:
        return ["deep: no readable recording found"]
    try:
        import mne

        readers = {".vhdr": mne.io.read_raw_brainvision, ".fif": mne.io.read_raw_fif,
                   ".edf": mne.io.read_raw_edf, ".set": mne.io.read_raw_eeglab}
        raw = readers[hits[0].suffix](str(hits[0]), preload=False, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001
        return [f"deep: could not read {hits[0].name}: {exc}"]
    if spec.expected_sfreq and abs(raw.info["sfreq"] - spec.expected_sfreq) > 1.0:
        issues.append(f"deep: sfreq {raw.info['sfreq']} != expected {spec.expected_sfreq}")
    types = set(raw.get_channel_types())
    for ct in spec.required_channel_types:
        if ct not in types:
            issues.append(f"deep: required channel type '{ct}' absent (found {sorted(types)})")
    return issues
