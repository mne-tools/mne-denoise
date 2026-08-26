#!/usr/bin/env python
"""Download all files of a Zenodo record via the public API (stdlib only).

``--record`` is the numeric Zenodo id (the tail of the DOI, e.g. DOI
10.5281/zenodo.2605204 -> record 2605204).  Run on a Fir login node (internet).

    python scripts/download_zenodo.py --dest <dir> --record 2605204
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

UA = {"User-Agent": "mne-denoise-benchmark/0.1", "Accept": "application/json"}


def _file_url(rec: str, f: dict) -> str:
    links = f.get("links") or {}
    return (links.get("self") or links.get("download")
            or f"https://zenodo.org/records/{rec}/files/{f['key']}?download=1")


def download(dest: pathlib.Path, record: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    api = f"https://zenodo.org/api/records/{record}"
    try:
        req = urllib.request.Request(api, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            meta = json.load(r)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Zenodo API {api} failed: {exc}", file=sys.stderr)
        return 2
    files = meta.get("files", [])
    print(f"[zenodo] record {record}: {len(files)} file(s)")
    if dry_run:
        for f in files:
            print(f"   {f.get('key')}  ({f.get('size')} bytes)")
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    rc = 0
    for f in files:
        key = f["key"]
        out = dest / key
        if out.exists() and f.get("size") and out.stat().st_size == f["size"]:
            print(f"   skip {key} (already complete)"); continue
        url = _file_url(record, f)
        print(f"   downloading {key} ...", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
            with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            rc = 1
            print(f"   FAIL {key}: {exc}", file=sys.stderr)
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--record", required=True, help="numeric Zenodo record id")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.record, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
