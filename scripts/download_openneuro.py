#!/usr/bin/env python3
"""Download a subset of an OpenNeuro dataset.

Download EEG + BIDS metadata and optionally
trigger-corrected derivative event files from the OpenNeuro GitHub mirror.

Narval-friendly defaults:
- DATA_DIR defaults to ~/scratch/mnedenoise/data
- dataset is stored in DATA_DIR / DATASET_ID

Usage examples:
    export DATASET_ID=ds003620
    export TASK=oddball
    export SUBJECTS="sub-01 sub-02 sub-03"
    export DATA_DIR=~/scratch/mnedenoise/data
    python ~/scratch/mnedenoise/scripts/download_openneuro.py

Or with CLI:
    python download_openneuro.py --dataset ds003620 --task oddball \
        --subjects sub-01 sub-02 sub-03

Download all 44 ds003620 subjects:
    python download_openneuro.py --dataset ds003620 --task oddball \
        --subjects $(printf "sub-%02d " $(seq 1 44))
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

try:
    import openneuro
except ImportError as e:
    raise SystemExit(
        "❌ Could not import 'openneuro'. Activate your environment and install it first:\n"
        "   pip install openneuro-py\n"
    ) from e


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET_ID", "ds003620"),
        help="OpenNeuro dataset ID (default from DATASET_ID env or ds003620)",
    )
    parser.add_argument(
        "--task",
        default=os.environ.get("TASK", "oddball"),
        help="BIDS task label without 'task-' prefix (default from TASK env or oddball)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="List of BIDS subject IDs (e.g., sub-01 sub-02). "
        "If omitted, reads SUBJECTS env (space/comma separated).",
    )
    parser.add_argument(
        "--skip-derivatives",
        action="store_true",
        help="Skip downloading trigger_corrected derivative event files from GitHub mirror",
    )
    return parser.parse_args()


def get_subjects_from_env() -> list[str]:
    """Return subject list from SUBJECTS environment variable."""
    raw = os.environ.get("SUBJECTS", "").strip()
    if not raw:
        return []
    # Supports both comma-separated and space-separated
    raw = raw.replace(",", " ")
    return [tok for tok in raw.split() if tok]


def resolve_subjects(args: argparse.Namespace) -> list[str]:
    """Resolve subject list from CLI args or environment."""
    subjects = args.subjects if args.subjects else get_subjects_from_env()
    if not subjects:
        raise SystemExit(
            "❌ No subjects provided.\n"
            "Provide them either as CLI args:\n"
            "   --subjects sub-01 sub-02\n"
            "or set env var:\n"
            "   export SUBJECTS='sub-01 sub-02'\n"
        )
    # basic validation
    for s in subjects:
        if not s.startswith("sub-"):
            print(f"⚠️ Warning: subject '{s}' does not start with 'sub-' (continuing).")
    return subjects


def build_include_patterns(subjects: list[str], task: str) -> list[str]:
    """Build include patterns for openneuro download."""
    if not task:
        raise SystemExit(
            "❌ TASK is empty.\n" "Set TASK env var or use --task oddball (example)."
        )

    include_patterns: list[str] = []
    for sub in subjects:
        include_patterns.extend(
            [
                f"{sub}/eeg/{sub}_task-{task}_eeg.*",
                f"{sub}/eeg/{sub}_task-{task}_events.tsv",
                f"{sub}/eeg/{sub}_task-{task}_channels.tsv",
                f"{sub}/eeg/{sub}_task-{task}_electrodes.tsv",
            ]
        )

    # Top-level BIDS metadata
    include_patterns.extend(
        [
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
        ]
    )
    return include_patterns


def urlretrieve_with_retry(
    url: str, dst: Path, retries: int = 3, sleep_s: float = 2.0
) -> None:
    """Download a URL with retry logic."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception as e:
            last_err = e
            print(f"    Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(sleep_s * attempt)
    raise RuntimeError(
        f"Failed to download after {retries} attempts: {url}"
    ) from last_err


def fix_encoding_issues(root: Path) -> int:
    """Fix known encoding issue with Latin-1 µ (0xB5) instead of UTF-8.

    Operates on .tsv/.vhdr/.json only.
    """
    n_fixed = 0
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix in (".tsv", ".vhdr", ".json"):
            raw_bytes = fp.read_bytes()
            if b"\xb5" in raw_bytes:
                fp.write_bytes(raw_bytes.replace(b"\xb5", "µ".encode()))
                print(f"  Fixed encoding in {fp}")
                n_fixed += 1
    return n_fixed


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    dataset_id = args.dataset
    task = args.task
    subjects = resolve_subjects(args)

    # --- Narval-friendly base path ---
    DATA_DIR = Path(
        os.environ.get("DATA_DIR", str(Path.home() / "scratch" / "mnedenoise" / "data"))
    )
    # Store each dataset under its own folder
    dataset_root = DATA_DIR / dataset_id
    dataset_root.mkdir(parents=True, exist_ok=True)

    print("=== OpenNeuro downloader ===")
    print(f"Dataset ID : {dataset_id}")
    print(f"Task       : {task}")
    print(f"Subjects   : {', '.join(subjects)}")
    print(f"Base dir   : {DATA_DIR}")
    print(f"Target dir : {dataset_root}")
    print()

    include_patterns = build_include_patterns(subjects, task)

    # Sentinel check (top-level metadata)
    sentinel = dataset_root / "dataset_description.json"
    if sentinel.exists():
        print(
            "ℹ️ dataset_description.json already exists (continuing; download is idempotent for requested files)."
        )

    # ── Download selected files from OpenNeuro ──────────────────────────────────
    print("Downloading selected OpenNeuro files...")
    openneuro.download(
        dataset=dataset_id,
        target_dir=str(dataset_root),
        include=include_patterns,
        verify_size=False,  # often faster / less fragile on clusters
    )

    # ── Download trigger-corrected events from derivatives (GitHub mirror) ─────
    if not args.skip_derivatives:
        github_raw = (
            f"https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/main"
        )
        print("\nDownloading trigger_corrected derivative events (if available)...")
        for sub in subjects:
            trig_relpath = (
                f"derivatives/trigger_corrected/{sub}/eeg/"
                f"{sub}_task-{task}_desc-trig_events.tsv"
            )
            trig_local = dataset_root / trig_relpath

            if trig_local.exists():
                print(f"  {sub}: trigger_corrected events.tsv already present")
                continue

            trig_local.parent.mkdir(parents=True, exist_ok=True)
            url = f"{github_raw}/{trig_relpath}"
            print(f"  {sub}: downloading {url}")
            try:
                urlretrieve_with_retry(url, trig_local, retries=3, sleep_s=2.0)
                print(
                    f"    → saved to {trig_local} ({trig_local.stat().st_size:,} bytes)"
                )
            except Exception as e:
                # Don't crash entire download if one subject derivative is missing
                print(f"    ⚠️ Could not fetch derivative for {sub}: {e}")

    # ── Fix known encoding issue ────────────────────────────────────────────────
    print("\nChecking for Latin-1 micro sign encoding issue...")
    n_fixed = fix_encoding_issues(dataset_root)
    print(f"Encoding fixes applied: {n_fixed}")

    print("\n✅ Download complete.")
    print(f"Dataset root: {dataset_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
