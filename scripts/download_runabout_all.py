#!/usr/bin/env python
"""Download ALL data for the Runabout (ds003620) notebook.

This script consolidates every download step from
  examples/tutorials/paper5_runabout_ds003620.ipynb
into a single, standalone script you can run from the terminal.

What it downloads
-----------------
1. Raw EEG files (BrainVision .vhdr/.eeg/.vmrk) for each subject from OpenNeuro.
2. BIDS sidecar files (events, channels, electrodes .tsv + .json).
3. Top-level BIDS metadata (dataset_description, participants, etc.).
4. Trigger-corrected events from the OpenNeuro GitHub mirror
   (these live under derivatives/ which openneuro-py doesn't traverse).
5. Fixes the known Latin-1 encoding issue (µ byte 0xB5 → UTF-8).

Usage
-----
    # Download only sub-01 (quick test):
    python scripts/download_runabout_all.py

    # Download specific subjects:
    python scripts/download_runabout_all.py --subjects 1 2 5 10

    # Download ALL 44 subjects:
    python scripts/download_runabout_all.py --all

    # Custom output directory:
    python scripts/download_runabout_all.py --all --data-dir D:/runabout
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# ──────────────────────────── Configuration ──────────────────────────────────
DATASET_ID = "ds003620"
DATASET_VERSION = "1.1.1"
TASK = "oddball"

# Default data directory (same as the notebook uses)
DEFAULT_DATA_DIR = Path("./data/runabout")

GITHUB_RAW = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/main"


def ensure_openneuro():
    """Import openneuro-py, installing it first if necessary."""
    try:
        import openneuro  # noqa: F811

        return openneuro
    except ImportError:
        print("[!] openneuro-py not installed. Installing now …")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])
        import openneuro  # noqa: F811

        return openneuro


MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds between retries

# Top-level BIDS metadata files to always include
BIDS_METADATA = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "CHANGES",
    "README",
]


def _download_with_retry(
    openneuro, include: list[str], data_dir: Path, label: str
) -> bool:
    """Run openneuro.download() with retry logic. Returns True on success."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            openneuro.download(
                dataset=DATASET_ID,
                target_dir=str(data_dir),
                include=include,
                verify_size=False,
            )
            return True
        except KeyboardInterrupt:
            print("\n  [!] Interrupted by user. Stopping.")
            sys.exit(1)
        except Exception as exc:
            print(f"\n    [!] {label} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                print(f"        Retrying in {RETRY_DELAY}s …")
                time.sleep(RETRY_DELAY)
    return False


def download_openneuro(subjects: list[str], data_dir: Path):
    """Step 1: Download raw EEG + BIDS metadata via openneuro-py.

    Downloads each subject individually so that a missing file for one
    subject does not block all others.
    """
    openneuro = ensure_openneuro()

    print("\n" + "=" * 65)
    print(f"  STEP 1 — OpenNeuro download  ({len(subjects)} subject(s))")
    print("=" * 65)
    print(f"  Dataset      : {DATASET_ID} v{DATASET_VERSION}")
    print(f"  Target dir   : {data_dir.resolve()}")
    print()

    # 1a. Download top-level BIDS metadata (once)
    print("  [1a] Downloading BIDS metadata …")
    _download_with_retry(openneuro, BIDS_METADATA, data_dir, "metadata")

    # 1b. Download each subject individually using a broad glob
    #     (sub-XX/eeg/*) so we don't fail on subjects missing channels.tsv
    succeeded = []
    failed = []
    for i, sub in enumerate(subjects, 1):
        print(f"\n  [{i}/{len(subjects)}] {sub} …")

        # Check if already downloaded (has .vhdr)
        existing = list(data_dir.resolve().rglob(f"{sub}_task-{TASK}_eeg.vhdr"))
        if existing:
            print("    Already present — skipped")
            succeeded.append(sub)
            continue

        # Use broad glob: grab everything under sub-XX/eeg/
        include = [f"{sub}/eeg/*"]
        ok = _download_with_retry(openneuro, include, data_dir, sub)
        if ok:
            print(f"    [OK] {sub}")
            succeeded.append(sub)
        else:
            print(f"    [FAIL] {sub} — will skip")
            failed.append(sub)

    print("\n" + "-" * 45)
    print(f"  OpenNeuro: {len(succeeded)} OK, {len(failed)} failed")
    if failed:
        print(f"  Failed subjects: {failed}")
    print("-" * 45)


def download_trigger_corrected(subjects: list[str], data_dir: Path):
    """Step 2: Fetch trigger-corrected events.tsv from GitHub mirror."""
    print("\n" + "=" * 65)
    print("  STEP 2 — Trigger-corrected events (GitHub mirror)")
    print("=" * 65)

    ok_count = 0
    skip_count = 0
    fail_count = 0
    for sub in subjects:
        trig_relpath = (
            f"derivatives/trigger_corrected/{sub}/eeg/"
            f"{sub}_task-{TASK}_desc-trig_events.tsv"
        )
        trig_local = data_dir / trig_relpath

        if trig_local.exists() and trig_local.stat().st_size > 0:
            print(f"  {sub}: already present — skipped")
            skip_count += 1
            continue

        trig_local.parent.mkdir(parents=True, exist_ok=True)
        url = f"{GITHUB_RAW}/{trig_relpath}"
        print(f"  {sub}: downloading …", end=" ")
        try:
            urllib.request.urlretrieve(url, trig_local)
            size = trig_local.stat().st_size
            print(f"OK ({size:,} bytes)")
            ok_count += 1
        except Exception as exc:
            # Clean up empty placeholder file on failure
            if trig_local.exists():
                trig_local.unlink()
            if "404" in str(exc):
                print("not available (no derivatives for this subject)")
            else:
                print(f"FAILED ({exc})")
            fail_count += 1

    print(
        f"\n  Trigger-corrected: {ok_count} downloaded, "
        f"{skip_count} already present, {fail_count} not available"
    )
    if fail_count:
        print(
            "  (Some subjects have no trigger-corrected derivatives "
            "on OpenNeuro — this is expected.)"
        )


def fix_encoding(data_dir: Path):
    """Step 3: Replace Latin-1 µ (0xB5) with proper UTF-8 in text files."""
    print("\n" + "=" * 65)
    print("  STEP 3 — Fix Latin-1 µ encoding")
    print("=" * 65)

    fixed = 0
    for tsv in data_dir.resolve().rglob("*"):
        if tsv.suffix in (".tsv", ".vhdr", ".json"):
            raw_bytes = tsv.read_bytes()
            if b"\xb5" in raw_bytes:
                tsv.write_bytes(raw_bytes.replace(b"\xb5", "µ".encode()))
                print(f"  Fixed encoding in {tsv.name}")
                fixed += 1

    if fixed:
        print(f"\n  [OK] Fixed {fixed} file(s).")
    else:
        print("  No encoding fixes needed.")


def report(subjects: list[str], data_dir: Path):
    """Step 4: Print a summary of what landed on disk."""
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    resolved = data_dir.resolve()
    if not resolved.exists():
        print("  Data directory does not exist!")
        return

    for sub in subjects:
        vhdr = list(resolved.rglob(f"{sub}_task-{TASK}_eeg.vhdr"))
        trig = list(resolved.rglob(f"{sub}_task-{TASK}_desc-trig_events.tsv"))
        status_eeg = "OK" if vhdr else "MISSING"
        status_trig = "OK" if trig else "MISSING"
        print(f"  {sub}:  EEG={status_eeg}  TrigCorrected={status_trig}")

    total_bytes = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
    print(f"\n  Total size on disk: {total_bytes / 1e9:.2f} GB")
    print("=" * 65)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download all Runabout (ds003620) data for the notebook."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        help="Subject numbers to download (e.g. 1 2 5 10). Default: 1",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download ALL 44 subjects (sub-01 through sub-44)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Target directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--skip-openneuro",
        action="store_true",
        help="Skip the openneuro download (only fetch trigger-corrected + fix encoding)",
    )
    return parser.parse_args()


def main():
    """CLI entry point."""
    args = parse_args()

    # Resolve subjects
    if args.all:
        subject_nums = list(range(1, 45))
    elif args.subjects:
        subject_nums = args.subjects
    else:
        subject_nums = [1]  # default: sub-01 only

    subjects = [f"sub-{n:02d}" for n in subject_nums]
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR

    print("=" * 65)
    print("  Runabout (ds003620) — Full Data Download")
    print("=" * 65)
    print(f"  Subjects   : {len(subjects)}  ({subjects[0]} … {subjects[-1]})")
    print(f"  Data dir   : {data_dir.resolve()}")
    print(f"  Task       : {TASK}")

    # Step 1: OpenNeuro download
    if not args.skip_openneuro:
        download_openneuro(subjects, data_dir)
    else:
        print("\n  [SKIP] OpenNeuro download (--skip-openneuro)")

    # Step 2: Trigger-corrected events from GitHub
    download_trigger_corrected(subjects, data_dir)

    # Step 3: Encoding fix
    fix_encoding(data_dir)

    # Step 4: Report
    report(subjects, data_dir)

    print("\n  All done! You can now run the notebook.")


if __name__ == "__main__":
    main()
