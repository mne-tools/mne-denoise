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
    verified_date: str | None = None
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
        expected_channels={"eeg": 120, "emg": 8, "misc": 185}, expected_sfreq=500.0,
        required_channel_types=("eeg",), raw_or_processed="raw", verified_date="2026-06-21",
        notes="dual-layer EEG + neck EMG + IMU; muscle/reference arm; BIDS under raw_bids/; "
              "noise-layer 120ch typed MISC (identify by name/electrodes in the muscle runner)",
    ),
    "ds000117": DatasetSpec(
        dataset_id="ds000117", repository="openneuro",
        project_relative_path="openneuro/ds000117",
        download_source="openneuro-py:ds000117",
        license="see dataset", citation="Wakeman & Henson 2015 (multimodal faces)",
        expected_subjects=19, expected_channels={"mag": 102, "grad": 204, "eeg": 70},
        expected_sfreq=1100.0, expected_line_frequency=50.0,
        required_channel_types=("mag", "grad"), raw_or_processed="raw",
        notes="MEG faces (M170); 16 = published subset. LINE-NOISE ARM MUST USE RAW FIF "
              "(supplied SSS files already had 50 Hz+harmonics+HPI removed).",
    ),
    "erp_core_n170": DatasetSpec(
        dataset_id="erp_core_n170", repository="osf",
        project_relative_path="osf/erp-core/n170",
        download_source="osf:PENDING_NODE_ID",   # verify the N170 OSF component on Fir
        doi="10.18115/D5JW4R", license="CC-BY-SA 4.0",
        citation="Kappenman et al. 2021 (ERP CORE)",
        expected_subjects=40, expected_channels={"eeg": 30, "eog": 3}, expected_sfreq=1024.0,
        required_channel_types=("eeg",), raw_or_processed="raw",
        notes="faces vs cars N170; ROI/window adopted from the ERP CORE resource",
    ),
    "eegdenoisenet": DatasetSpec(
        dataset_id="eegdenoisenet", repository="zenodo",
        project_relative_path="zenodo/eegdenoisenet",
        download_source="pooch:PENDING_URL",     # verify Zenodo/GitHub URL+hash on Fir
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
    ver = version or spec.version or "latest"
    out: list[pathlib.Path] = []
    if env.get("DATASETS_ROOT"):
        out.append(pathlib.Path(env["DATASETS_ROOT"]) / rel)
    out.append(pathlib.Path("/project/rrg-kjerbi/datasets") / rel)
    out.append(_scratch_root(env) / "mne-denoise" / "datasets" / rel / ver)
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
    elif spec.repository in ("osf",):
        if spec.expected_subjects is not None:
            n = sum(1 for p in root.glob("sub-*") if p.is_dir())
            if n == 0:
                issues.append("no sub-* directories found")

    if deep:
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
