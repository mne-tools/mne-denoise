#!/usr/bin/env python
"""Download a PhysioNet database via wget (PhysioNet's recommended method).

Run on a Fir **login node** (internet). ``--db`` is ``<slug>/<version>``::

    python scripts/download_physionet.py --dest <dir> --db sleep-edfx/1.0.0
    python scripts/download_physionet.py --dest <dir> --db chbmit/1.0.0
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

BASE = "https://physionet.org/files"


def download(dest: pathlib.Path, db: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    if "/" not in db:
        print("ERROR: --db must be '<slug>/<version>' (e.g. sleep-edfx/1.0.0)", file=sys.stderr)
        return 2
    url = f"{BASE}/{db}/"
    # /files/<slug>/<version>/... -> cut 3 leading path dirs so content lands in <dest>/
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "-nv",
           "-P", str(dest), url]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    if shutil.which("wget") is None:
        print("ERROR: wget not on PATH.", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[physionet] {url} -> {dest}", flush=True)
    return subprocess.run(cmd).returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--db", required=True, help="<slug>/<version>, e.g. sleep-edfx/1.0.0")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.db, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
