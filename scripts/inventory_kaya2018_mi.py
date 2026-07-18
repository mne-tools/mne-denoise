#!/usr/bin/env python
"""Inventory hand-motor-imagery events in the public Kaya et al. archive.

The Tsai et al. adaptive-ASR paper reports 9,224 left/right-hand trials from
13 participants but does not list the included Kaya recording files.  This
script records event counts for every public MATLAB file and evaluates the
natural paper-cohort rule: marker codes 1 and 2 in all CLA and HaLT sessions.
No denoising or outcome analysis is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
COLLECTION_DOI = "10.6084/m9.figshare.c.3917698"
PAPER_REPORTED_TRIALS = 9224
PAPER_PARADIGMS = frozenset({"CLA", "HaLT"})
HAND_CODES = (1, 2)
NAME_PATTERN = re.compile(
    r"^(?P<paradigm>5F|CLA|FREEFORM|HaLT|NoMT)-Subject"
    r"(?P<subject>[A-Z])-(?P<session>.+)\.mat$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _parse_name(path: Path) -> dict[str, str]:
    match = NAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unrecognized Kaya recording name: {path.name}")
    return match.groupdict()


def _event_onsets(marker: np.ndarray, code: int) -> np.ndarray:
    """Return indices where a marker enters ``code`` from another value."""
    marker = np.asarray(marker).reshape(-1)
    selected = marker == code
    return np.flatnonzero(selected & np.r_[True, ~selected[:-1]])


def _recording_inventory(path: Path, root: Path) -> dict[str, Any]:
    identity = _parse_name(path)
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    recording = content.get("o")
    if recording is None:
        raise ValueError(f"MATLAB variable 'o' is absent: {path}")
    marker = np.asarray(recording.marker).reshape(-1)
    data_shape = tuple(int(value) for value in np.shape(recording.data))
    sfreq = float(recording.sampFreq)
    counts = {str(code): int(_event_onsets(marker, code).size) for code in range(1, 7)}
    return {
        **identity,
        "relative_path": path.relative_to(root).as_posix(),
        "sampling_frequency_hz": sfreq,
        "n_samples": int(marker.size),
        "duration_s": float(marker.size / sfreq),
        "data_shape": data_shape,
        "event_counts": counts,
        "hand_trial_count": int(sum(counts[str(code)] for code in HAND_CODES)),
    }


def build_inventory(dataset_root: Path, *, locked: bool = False) -> dict[str, Any]:
    """Build a complete recording and hand-trial inventory."""
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "figshare_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Figshare manifest: {manifest_path}")
    commit, dirty = _git_state()
    if locked and dirty:
        raise RuntimeError("refusing locked inventory from a dirty worktree")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = sorted(dataset_root.rglob("*.mat"))
    recordings: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        try:
            row = _recording_inventory(path, dataset_root)
            recordings.append(row)
            print(
                f"[{index:02d}/{len(paths):02d}] {path.name}: "
                f"hand={row['hand_trial_count']}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"relative_path": path.relative_to(dataset_root).as_posix(), "error": str(exc)}
            )
            print(f"[{index:02d}/{len(paths):02d}] {path.name}: FAILED: {exc}")

    by_paradigm: dict[str, int] = defaultdict(int)
    by_subject: dict[str, int] = defaultdict(int)
    paper_rows = []
    for row in recordings:
        by_paradigm[row["paradigm"]] += row["hand_trial_count"]
        if row["paradigm"] in PAPER_PARADIGMS:
            paper_rows.append(row)
            by_subject[row["subject"]] += row["hand_trial_count"]

    candidate_count = int(sum(row["hand_trial_count"] for row in paper_rows))
    subjects = sorted(by_subject)
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "collection_doi": COLLECTION_DOI,
        "dataset_root": str(dataset_root),
        "figshare_manifest_sha256": _sha256(manifest_path),
        "figshare_manifest_file_count": int(manifest["file_count"]),
        "repository_commit": commit,
        "dirty_worktree": dirty,
        "matlab_file_count": len(paths),
        "loaded_recording_count": len(recordings),
        "failure_count": len(failures),
        "failures": failures,
        "trial_counts_by_paradigm": dict(sorted(by_paradigm.items())),
        "paper_candidate": {
            "paradigms": sorted(PAPER_PARADIGMS),
            "marker_codes": list(HAND_CODES),
            "subjects": subjects,
            "subject_count": len(subjects),
            "recording_count": len(paper_rows),
            "trial_count": candidate_count,
            "reported_trial_count": PAPER_REPORTED_TRIALS,
            "matches_reported_trial_count": candidate_count == PAPER_REPORTED_TRIALS,
            "trial_counts_by_subject": dict(sorted(by_subject.items())),
            "relative_paths": [row["relative_path"] for row in paper_rows],
        },
        "recordings": recordings,
    }


def main() -> None:
    """Run the command-line inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--locked", action="store_true")
    args = parser.parse_args()

    inventory = build_inventory(args.dataset_root, locked=args.locked)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    candidate = inventory["paper_candidate"]
    print(
        "paper candidate: "
        f"{candidate['trial_count']} trials, {candidate['subject_count']} subjects, "
        f"match={candidate['matches_reported_trial_count']}"
    )
    if inventory["failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
