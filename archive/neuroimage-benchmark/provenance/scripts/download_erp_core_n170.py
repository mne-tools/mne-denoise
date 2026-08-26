#!/usr/bin/env python
"""Robustly download the ERP CORE **N170** BIDS raw files (all 40 subjects) from OSF.

Why not osfclient: it drives the rate-limited OSF *API* (``api.osf.io``) with roughly one
call per file and stalls after a few subjects. This instead:

  1. Lists the file tree via the API *once* (the throttle-prone part) with retry+backoff,
     **pruning to ``ses-N170``** so the other six paradigms are never enumerated; then
  2. Downloads each file via its ``files.osf.io`` **waterbutler** link -- a *different*
     service that is not subject to the API's per-IP rate limiting.

Resumable: any file already on disk (non-empty) is skipped, so re-running fills the gaps.
Run on a Fir **login node** (internet); compute nodes are offline.

    python scripts/download_erp_core_n170.py \
        --dest /project/rrg-kjerbi/datasets/osf/erp-core/BIDS-Compatible_Raw_Files/osfstorage

Files land at ``<dest>/ERP_CORE_BIDS_Raw_Files/sub-NNN/ses-N170/eeg/...`` -- the same layout
the run_evoked/ocular loaders already glob, so the existing 7 subjects are simply skipped.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

API = "https://api.osf.io/v2"
BIDS_NODE = "9f5w7"  # OSF "BIDS-Compatible Raw Files" component (all paradigms; pruned to N170)


def _get_json(url, attempts=12):
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/vnd.api+json", "User-Agent": "mne-denoise/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"  [api retry {i + 1}/{attempts}] {type(exc).__name__}: {str(exc)[:60]}\n")
            time.sleep(min(120, 10 * (i + 1)))
    return None


def _list(node, fid=None):
    """List one osfstorage folder (paginated, retried): [(name, kind, id, download_url)]."""
    url = f"{API}/nodes/{node}/files/osfstorage/" + (f"{fid}/" if fid else "") + "?page%5Bsize%5D=100"
    items = []
    while url:
        d = _get_json(url)
        if not d:
            break
        for f in d.get("data", []):
            a = f["attributes"]
            items.append((a["name"], a["kind"], f["id"], (f.get("links") or {}).get("download")))
        url = (d.get("links") or {}).get("next")
    return items


def _download(url, dest, attempts=8):
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mne-denoise/1.0"})
            with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.replace(dest)
            return "ok"
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 8 * (i + 1)))
    if tmp.exists():
        tmp.unlink()
    return "fail"


def walk(node, fid, relpath, dest_root, stats):
    for name, kind, item_id, dl in _list(node, fid):
        rp = f"{relpath}/{name}" if relpath else name
        if kind == "folder":
            if name.startswith("ses-") and name != "ses-N170":
                continue  # prune: only the N170 session of each subject
            walk(node, item_id, rp, dest_root, stats)
        elif dl:
            res = _download(dl, pathlib.Path(dest_root) / rp)
            stats[res] = stats.get(res, 0) + 1
            if res != "skip":
                print(f"  {res:4} {rp}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True, help="osfstorage dir (files land under <dest>/ERP_CORE_BIDS_Raw_Files/...)")
    p.add_argument("--node", default=BIDS_NODE)
    a = p.parse_args(argv)
    stats = {}
    walk(a.node, None, "", a.dest, stats)
    print(f"[done] {stats}", flush=True)
    return 0 if stats.get("fail", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
