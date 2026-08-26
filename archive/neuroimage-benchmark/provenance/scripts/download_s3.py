#!/usr/bin/env python
"""Download a public (anonymous) S3 prefix via the REST API (stdlib only).

No aws CLI / boto3 needed.  Lists ListObjectsV2 (paginated) under a prefix on a
public bucket and downloads each key over HTTPS.  Used for LEMON on the INDI
``fcp-indi`` bucket.  Run on a Fir login node (internet).

    python scripts/download_s3.py --dest <dir> \
        --bucket-host fcp-indi.s3.amazonaws.com \
        --prefix data/Projects/INDI/MPI-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "mne-denoise-benchmark/0.1"}


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def list_objects(bucket_host: str, prefix: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = f"https://{bucket_host}/?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            root = ET.fromstring(r.read())
        trunc, token = False, None
        for el in root:
            ln = _localname(el.tag)
            if ln == "Contents":
                key = size = None
                for c in el:
                    if _localname(c.tag) == "Key":
                        key = c.text
                    elif _localname(c.tag) == "Size":
                        size = int(c.text)
                if key and not key.endswith("/"):
                    out.append((key, size or 0))
            elif ln == "IsTruncated":
                trunc = (el.text == "true")
            elif ln == "NextContinuationToken":
                token = el.text
        if not trunc:
            break
    return out


def download(dest: pathlib.Path, bucket_host: str, prefix: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    try:
        keys = list_objects(bucket_host, prefix)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: S3 list failed: {exc}", file=sys.stderr)
        return 2
    total = sum(s for _, s in keys)
    print(f"[s3] {len(keys)} objects under {prefix} (~{total/1e9:.1f} GB)")
    if dry_run:
        for k, s in keys[:15]:
            print(f"   {k}  ({s})")
        return 0
    if not keys:
        print("ERROR: no objects found under prefix", file=sys.stderr)
        return 1
    rc = 0
    for key, size in keys:
        rel = key[len(prefix):] if key.startswith(prefix) else key
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and size and out.stat().st_size == size:
            continue
        url = f"https://{bucket_host}/{urllib.parse.quote(key)}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as resp, \
                 open(out, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            rc = 1
            print(f"   FAIL {rel}: {exc}", file=sys.stderr)
    print(f"[s3] done (rc={rc})")
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--bucket-host", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.bucket_host, a.prefix, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
