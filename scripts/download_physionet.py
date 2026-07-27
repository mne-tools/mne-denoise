#!/usr/bin/env python
"""Download a PhysioNet database via wget (PhysioNet's recommended method).

Run on a Fir **login node** (internet). ``--db`` is ``<slug>/<version>``::

    python scripts/download_physionet.py --dest <dir> --db sleep-edfx/1.0.0
    python scripts/download_physionet.py --dest <dir> --db chbmit/1.0.0

``--subjects`` restricts the pull to named per-subject directories plus the
database's root-level metadata. CHB-MIT is ~43 GB whole and the robustness arm
reads six cases, so pulling all of it would spend shared /project quota on data
no arm opens.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

BASE = "https://physionet.org/files"


def download(dest: pathlib.Path, db: str, dry_run: bool = False,
             subjects: list[str] | None = None) -> int:
    dest = pathlib.Path(dest)
    if "/" not in db:
        print("ERROR: --db must be '<slug>/<version>' (e.g. sleep-edfx/1.0.0)", file=sys.stderr)
        return 2
    url = f"{BASE}/{db}/"
    # /files/<slug>/<version>/... -> cut 3 leading path dirs so content lands in <dest>/
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "-nv",
           "-P", str(dest), url]
    if subjects:
        # Keep anything under a wanted subject directory, plus root-level files
        # (RECORDS, SUBJECT-INFO, checksums) which every loader expects to exist.
        version_path = re.escape(f"/{db}/")
        wanted = "|".join(re.escape(s) for s in subjects)
        cmd[1:1] = ["--regex-type", "posix",
                    "--accept-regex", f"{version_path}({wanted})/|{version_path}[^/]*$"]
        print(f"[physionet] restricting to {len(subjects)} subjects: {', '.join(subjects)}",
              flush=True)
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
    p.add_argument("--subjects", nargs="+",
                   help="per-subject directory names, e.g. chb01 chb02 (default: whole database)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.db, a.dry_run, a.subjects)


if __name__ == "__main__":
    raise SystemExit(main())
