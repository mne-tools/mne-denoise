#!/usr/bin/env python
"""Download EEGdenoiseNet from G-Node/GIN via git + git-annex (the proper way).

The dataset is hosted on GIN (https://gin.g-node.org/NCClab/EEGdenoiseNet) as a
git-annex repository — so we ``git clone`` then ``git annex get .`` to fetch the
annexed ``.npy`` segment files.  Run on a Fir **login node** (internet) with
git-annex available (``module load git-annex``).

    python scripts/download_eegdenoisenet.py --dest <dir> --repo NCClab/EEGdenoiseNet
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

EXPECTED_FILES = ("EEG_all_epochs.npy", "EOG_all_epochs.npy", "EMG_all_epochs.npy")


def download(dest: pathlib.Path, repo: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    url = f"https://gin.g-node.org/{repo}"
    if dry_run:
        print(f"[dry-run] git clone {url} {dest} && git -C {dest} annex get .")
        return 0
    if shutil.which("git-annex") is None:
        print("ERROR: git-annex not on PATH. On Fir: module load git-annex", file=sys.stderr)
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        print(f"[gin] git clone {url} -> {dest}", flush=True)
        if subprocess.run(["git", "clone", url, str(dest)]).returncode:
            return 1
    subprocess.run(["git", "-C", str(dest), "annex", "init"])
    print("[gin] git annex get . (fetching annexed content)", flush=True)
    rc = subprocess.run(["git", "-C", str(dest), "annex", "get", "."]).returncode
    found = [f for f in EXPECTED_FILES if list(dest.rglob(f))]
    print(f"[gin] expected files present: {found or 'NONE (check repo layout)'}")
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--repo", default="NCClab/EEGdenoiseNet")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.repo, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
