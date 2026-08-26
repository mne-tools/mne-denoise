#!/usr/bin/env python
"""Download a THU BCI portal dataset by scraping the download page (stdlib only).

The Tsinghua BCI portal (https://bci.med.tsinghua.edu.cn/download.html) has no API
or versioned DOI; it links archives as ``./upload/<subdir>/<file>``.  This fetches
all files under a given subdir and writes a ``SHA256SUMS`` manifest (the source is
unversioned, so we record checksums).  Run on a Fir login node (internet).

    python scripts/download_thu_portal.py --dest <dir> --subdir yijun                       # Tsinghua Benchmark
    python scripts/download_thu_portal.py --dest <dir> --subdir liubingchuan_eld_BETA_database  # BETA
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys
import urllib.parse
import urllib.request

LANDING = "https://bci.med.tsinghua.edu.cn/download.html"
UA = {"User-Agent": "Mozilla/5.0 mne-denoise-benchmark/0.1"}


def list_files(landing: str, subdir: str) -> list[tuple[str, str]]:
    req = urllib.request.Request(landing, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    rels = re.findall(r'href="(\./upload/%s/[^"]+)"' % re.escape(subdir), html)
    base = landing.rsplit("/", 1)[0] + "/"
    seen = list(dict.fromkeys(rels))  # de-dup, keep order
    return [(urllib.parse.urljoin(base, rel), rel.split("/")[-1]) for rel in seen]


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: pathlib.Path, subdir: str, landing: str = LANDING, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    try:
        files = list_files(landing, subdir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: portal fetch failed: {exc}", file=sys.stderr)
        return 2
    print(f"[thu] subdir {subdir}: {len(files)} file(s)")
    if dry_run:
        for _, name in files:
            print(f"   {name}")
        return 0
    if not files:
        print(f"ERROR: no files found under upload/{subdir}", file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    rc = 0
    with open(dest / "SHA256SUMS", "a") as man:
        for url, name in files:
            out = dest / name
            if out.exists() and out.stat().st_size > 0:
                print(f"   skip {name} (present)"); continue
            print(f"   downloading {name} ...", flush=True)
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=600) as resp, open(out, "wb") as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                man.write(f"{_sha256(out)}  {name}\n"); man.flush()
            except Exception as exc:  # noqa: BLE001
                rc = 1
                print(f"   FAIL {name}: {exc}", file=sys.stderr)
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--subdir", required=True)
    p.add_argument("--landing", default=LANDING)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.subdir, a.landing, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
