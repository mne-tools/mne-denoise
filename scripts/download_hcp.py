#!/usr/bin/env python
"""Download HCP Young Adult (HCP1200) **MEG** data from s3://hcp-openaccess (signed, boto3).

MEG only: keys under ``HCP_1200/<subject>/MEG/`` (processed) and
``HCP_1200/<subject>/unprocessed/MEG/`` (raw 4D/BTi).  Needs AWS credentials
(``~/.aws/credentials`` default profile or env) tied to an accepted HCP data-use
agreement.  Run on a Fir login node (internet); compute nodes are offline.

    python scripts/download_hcp.py --dest <dir> --dry-run            # count subjects + size
    python scripts/download_hcp.py --dest <dir> --scope unprocessed  # raw MEG only (denoising input)
    python scripts/download_hcp.py --dest <dir>                      # all MEG (raw + processed)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

BUCKET = "hcp-openaccess"
ROOT = "HCP_1200/"
SUBPREFIXES = {"unprocessed": "unprocessed/MEG/", "processed": "MEG/"}


def _client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", region_name="us-east-1",
                        config=Config(retries={"max_attempts": 10, "mode": "standard"},
                                      max_pool_connections=16))


def list_subjects(s3) -> list[str]:
    subs, tok = [], None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=ROOT, Delimiter="/")
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        for cp in r.get("CommonPrefixes", []):
            subj = cp["Prefix"][len(ROOT):].strip("/")
            if subj.isdigit():
                subs.append(subj)
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return subs


def _list_prefix(s3, prefix: str) -> list[tuple[str, int]]:
    out, tok = [], None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out.extend((o["Key"], o["Size"]) for o in r.get("Contents", []))
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return out


def meg_keys(s3, subj: str, scopes) -> list[tuple[str, int]]:
    keys = []
    for sc in scopes:
        keys += _list_prefix(s3, f"{ROOT}{subj}/{SUBPREFIXES[sc]}")
    return keys


def download(dest: pathlib.Path, scope: str = "all", dry_run: bool = False, subjects=None) -> int:
    dest = pathlib.Path(dest)
    scopes = list(SUBPREFIXES) if scope == "all" else [scope]
    s3 = _client()
    allsubs = subjects or list_subjects(s3)
    print(f"[hcp] {len(allsubs)} HCP_1200 subjects; scanning for MEG (scope={scope}) ...", flush=True)
    meg, sz = {}, {"unprocessed": 0, "processed": 0}
    for i, subj in enumerate(allsubs):
        ks = meg_keys(s3, subj, scopes)
        if ks:
            meg[subj] = ks
            for k, s in ks:
                sz["unprocessed" if "/unprocessed/MEG/" in k else "processed"] += s
        if (i + 1) % 200 == 0:
            print(f"  scanned {i+1}/{len(allsubs)} | MEG subjects {len(meg)}", flush=True)
    nkeys = sum(len(v) for v in meg.values())
    tot = sz["unprocessed"] + sz["processed"]
    print(f"[hcp] MEG subjects: {len(meg)} | files: {nkeys} | "
          f"raw {sz['unprocessed']/1e12:.2f} TB + processed {sz['processed']/1e12:.2f} TB "
          f"= ~{tot/1e12:.2f} TB")
    if dry_run:
        for s in list(meg)[:8]:
            print("   e.g.", s, "->", len(meg[s]), "files")
        return 0
    if not meg:
        print("ERROR: no MEG subjects found (check creds / terms acceptance)", file=sys.stderr)
        return 1
    rc, done = 0, 0
    for subj, ks in meg.items():
        for key, size in ks:
            out = dest / key[len(ROOT):]
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size == size:
                continue
            try:
                s3.download_file(BUCKET, key, str(out))
                done += 1
            except Exception as exc:  # noqa: BLE001
                rc = 1
                print(f"   FAIL {key}: {exc}", file=sys.stderr)
        print(f"   [{subj}] done ({len(ks)} files)", flush=True)
    print(f"[hcp] downloaded {done} new files (rc={rc})")
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--scope", choices=["all", "unprocessed", "processed"], default="all")
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.scope, a.dry_run, a.subjects)


if __name__ == "__main__":
    raise SystemExit(main())
