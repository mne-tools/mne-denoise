#!/usr/bin/env python
r"""Download ds003620 (Runabout) from OpenNeuro — resume-aware.

This script:
1. Moves any existing data from D:\\runabout into D:\\runabout\\ds003620
   (the path the notebook expects) so the ~40 GB already downloaded is kept.
2. Uses openneuro-py to download the full dataset, skipping files that
   already exist on disk.

Usage
-----
    python scripts/download_ds003620.py          # download everything
    python scripts/download_ds003620.py --eeg    # download raw EEG only
"""

import argparse
import pathlib
import shutil
import sys

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_ID = "ds003620"
DATA_ROOT = pathlib.Path(r"D:\runabout")
DS_PATH = DATA_ROOT / DATASET_ID  # D:\runabout\ds003620


# Files / dirs at D:\runabout that belong to the dataset and should be moved
# into the ds003620 subfolder (not the derivatives from mne-denoise).
DATASET_ITEMS = [
    "CHANGES",
    "LICENSE",
    "README",
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "code",
    "derivatives",
    "sourcedata",
    "stimuli",
] + [f"sub-{i:02d}" for i in range(1, 45)]


def move_existing_data():
    r"""Move already-downloaded files from D:\\runabout → D:\\runabout\\ds003620."""
    if DS_PATH.exists():
        print(f"[✓] {DS_PATH} already exists — checking for stray items …")
    else:
        DS_PATH.mkdir(parents=True, exist_ok=True)
        print(f"[+] Created {DS_PATH}")

    moved = 0
    for name in DATASET_ITEMS:
        src = DATA_ROOT / name
        dst = DS_PATH / name
        if not src.exists():
            continue
        if dst.exists():
            # Already in the right place; remove the empty stub if that's
            # what the source is (don't overwrite real data with empty dirs).
            if src.is_dir() and not any(src.rglob("*")):
                # source is empty, destination exists → just delete source
                shutil.rmtree(src)
                continue
            elif src.is_dir():
                # Both exist and source has content → merge
                _merge_tree(src, dst)
                shutil.rmtree(src)
                moved += 1
                continue
            else:
                continue  # file already at destination
        # Move
        print(f"  Moving {name} → {DS_PATH / name}")
        shutil.move(str(src), str(dst))
        moved += 1

    if moved:
        print(f"[✓] Moved {moved} items into {DS_PATH}")
    else:
        print("[✓] No items to move (already in place or nothing found)")


def _merge_tree(src: pathlib.Path, dst: pathlib.Path):
    """Recursively merge src into dst, preserving existing files at dst."""
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            target.mkdir(exist_ok=True)
            _merge_tree(item, target)
        else:
            if not target.exists():
                shutil.move(str(item), str(target))


def download_dataset(eeg_only: bool = False):
    """Download ds003620 using openneuro-py (skips existing files)."""
    try:
        import openneuro
    except ImportError:
        print("[!] openneuro-py is not installed. Installing …")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])
        import openneuro

    kwargs = {
        "dataset": DATASET_ID,
        "target_dir": str(DATA_ROOT),
        "verify_size": False,
    }

    if eeg_only:
        # Download only the raw EEG files (the ones the notebook needs)
        kwargs["include"] = ["sub-*/eeg/*"]
        print(f"\n[↓] Downloading RAW EEG files only for {DATASET_ID} …")
    else:
        print(f"\n[↓] Downloading full dataset {DATASET_ID} …")

    print(f"    target_dir : {DATA_ROOT}")
    print(f"    data lands : {DS_PATH}")
    print()

    openneuro.download(**kwargs)

    print("\n[✓] Download complete.")


def report():
    """Print a summary of what's on disk."""
    print("\n" + "=" * 60)
    print(f"  POST-DOWNLOAD REPORT — {DS_PATH}")
    print("=" * 60)

    if not DS_PATH.exists():
        print("  Dataset directory does not exist yet.")
        return

    # Count raw .vhdr files
    vhdr_files = sorted(DS_PATH.glob("sub-*/eeg/*.vhdr"))
    subs_with_eeg = {f.parent.parent.name for f in vhdr_files}
    all_sub_dirs = sorted(
        [d.name for d in DS_PATH.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    )

    print(f"  Subject dirs        : {len(all_sub_dirs)}")
    print(f"  Subjects with .vhdr : {len(subs_with_eeg)}")
    if subs_with_eeg:
        print(f"    → {sorted(subs_with_eeg)[:5]} … ")

    missing = set(all_sub_dirs) - subs_with_eeg
    if missing:
        print(f"  MISSING raw EEG for : {sorted(missing)}")
    else:
        print("  [✓] All subjects have raw EEG files")

    # Total size
    total = sum(f.stat().st_size for f in DS_PATH.rglob("*") if f.is_file())
    print(f"  Total size          : {total / 1e9:.2f} GB")

    # Derivative .vhdr
    deriv_vhdr = list(DS_PATH.glob("derivatives/**/*.vhdr"))
    print(f"  Derivative .vhdr    : {len(deriv_vhdr)}")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download ds003620 from OpenNeuro (resume-aware)"
    )
    parser.add_argument(
        "--eeg",
        action="store_true",
        help="Download only raw EEG files (sub-*/eeg/*), skip derivatives",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only reorganise existing data and print report (no download)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ds003620 Download Script")
    print("=" * 60)

    # Step 1: Reorganise existing data
    print("\n── Step 1: Reorganise existing files ──")
    move_existing_data()

    # Step 2: Download
    if not args.skip_download:
        print("\n── Step 2: Download from OpenNeuro ──")
        download_dataset(eeg_only=args.eeg)
    else:
        print("\n── Step 2: Skipped (--skip-download) ──")

    # Step 3: Report
    print("\n── Step 3: Summary ──")
    report()


if __name__ == "__main__":
    main()
