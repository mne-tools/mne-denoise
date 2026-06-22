#!/usr/bin/env python
"""Stage + validate a benchmark dataset on Fir (lock + sentinel + validation).

Run this ONCE per dataset (inside an `salloc` allocation, or as a small prep job)
BEFORE submitting any benchmark array.  Array jobs must be read-only and call
:func:`assert_staged` — they must never download (see BENCHMARK_PLAN.md §6).

    python scripts/cc/stage_dataset.py ds003620
    python scripts/cc/stage_dataset.py erp_core_n170 --dry-run

Resolution + registry come from ``mne_denoise.benchmarks.datasets``.  A dataset
already present under ``/project`` (or ``$DATASETS_ROOT``) is validated in place;
otherwise it is downloaded to ``$SCRATCH/mne-denoise/datasets/...``.  Success is
recorded with a sentinel ``$SCRATCH/mne-denoise/staged/<id>.ok`` (JSON manifest).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from mne_denoise.benchmarks import datasets as D  # noqa: E402


def _scratch() -> pathlib.Path:
    return pathlib.Path(os.environ.get("SCRATCH", str(pathlib.Path.home() / "scratch")))


def sentinel_path(dataset_id: str) -> pathlib.Path:
    return _scratch() / "mne-denoise" / "staged" / f"{dataset_id}.ok"


def assert_staged(dataset_id: str) -> pathlib.Path:
    """For array jobs: ensure a dataset was staged + validated; return its root."""
    s = sentinel_path(dataset_id)
    if not s.is_file():
        raise RuntimeError(
            f"dataset {dataset_id!r} is not staged ({s} missing). "
            f"Run: python scripts/cc/stage_dataset.py {dataset_id}"
        )
    return pathlib.Path(json.loads(s.read_text())["root"])


def _download(spec: D.DatasetSpec, root: pathlib.Path, subjects: list[str] | None) -> None:
    src = spec.download_source or ""
    root.mkdir(parents=True, exist_ok=True)
    if src.startswith("openneuro-py:"):
        # Accession comes from the source (registry id may differ, e.g.
        # japanese_eeg_emg -> ds007808). Full dataset by default; subset via include=[].
        import openneuro

        accession = src.split(":", 1)[1]
        kw = {"dataset": accession, "target_dir": str(root)}
        if subjects:
            kw["include"] = list(subjects)
        if spec.download_exclude:
            kw["exclude"] = list(spec.download_exclude)  # e.g. skip MRI "*/anat/*"
        print(f"[openneuro] {accession} -> {root} "
              + (f"(subjects: {subjects})" if subjects else "(ALL subjects)"), flush=True)
        openneuro.download(**kw)
    elif src.startswith("osf-components:"):
        node = src.split(":", 1)[1]
        subprocess.run([sys.executable, str(_REPO / "scripts" / "download_erp_core.py"),
                        "--dest", str(root), "--osf-node", node], check=True)
    elif src.startswith("osf:"):
        node = src.split(":", 1)[1]
        subprocess.run([sys.executable, str(_REPO / "scripts" / "download_osf.py"),
                        "--dest", str(root), "--node", node], check=True)
    elif src.startswith("zenodo:"):
        rec = src.split(":", 1)[1]
        subprocess.run([sys.executable, str(_REPO / "scripts" / "download_zenodo.py"),
                        "--dest", str(root), "--record", rec], check=True)
    elif src.startswith("figshare:"):
        art = src.split(":", 1)[1]
        subprocess.run([sys.executable, str(_REPO / "scripts" / "download_figshare.py"),
                        "--dest", str(root), "--article", art], check=True)
    elif src.startswith("gin:"):
        repo = src.split(":", 1)[1]
        subprocess.run([sys.executable, str(_REPO / "scripts" / "download_eegdenoisenet.py"),
                        "--dest", str(root), "--repo", repo], check=True)
    elif src.startswith("thu:"):
        subdir = src.split(":", 1)[1]
        subprocess.run([sys.executable, str(_REPO / "scripts" / "download_thu_portal.py"),
                        "--dest", str(root), "--subdir", subdir], check=True)
    elif src.startswith("s3:"):
        bucket_host, prefix = src.split(":", 1)[1].split("/", 1)
        subprocess.run([sys.executable, str(_REPO / "scripts" / "download_s3.py"),
                        "--dest", str(root), "--bucket-host", bucket_host, "--prefix", prefix], check=True)
    elif src.startswith("physionet:"):
        db = src.split(":", 1)[1]
        subprocess.run([sys.executable, str(_REPO / "scripts" / "download_physionet.py"),
                        "--dest", str(root), "--db", db], check=True)
    elif src.startswith("manual:"):
        raise RuntimeError(
            f"{spec.dataset_id}: no headless download ({src}). Stage MANUALLY: place the data at "
            f"{root}, then re-run this command to validate + sentinel it. See the registry note."
        )
    elif src.startswith(("confirm:", "registration:", "portal:")):
        raise RuntimeError(
            f"{spec.dataset_id}: access='{spec.access}' is not automated ({src}). "
            "Provide a durable URL/accession or registration credentials, or use the "
            "dataset's custom downloader (see the registry note)."
        )
    else:
        raise ValueError(f"no downloader for source {src!r}")


def stage(dataset_id: str, *, version=None, subjects=None, dry_run=False, force=False) -> dict:
    spec = D.get_spec(dataset_id)
    # Persist datasets under the shared /project root by default (firm user preference);
    # override with DATASETS_ROOT. (/scratch is only a fallback in resolve_dataset_root.)
    datasets_root = os.environ.get("DATASETS_ROOT", "/project/rrg-kjerbi/datasets")
    root = pathlib.Path(datasets_root) / spec.project_relative_path
    sent = sentinel_path(dataset_id)
    plan = {"dataset": dataset_id, "root": str(root), "source": spec.download_source}

    if sent.is_file() and not force:
        print(f"[staged] {dataset_id} already staged -> {sent}")
        return {**plan, "status": "already_staged"}
    if dry_run:
        print(f"[dry-run] would ensure {dataset_id} at {root} via {spec.download_source}")
        return {**plan, "status": "dry_run"}

    # acquire a coarse lock (atomic mkdir) so concurrent prep jobs don't collide
    lock = _scratch() / "mne-denoise" / "locks" / f"{dataset_id}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"another staging job holds the lock {lock}; retry later")
    try:
        # Download if forced, missing, or failing validation. --force re-triggers the
        # downloader (which resumes/skips existing files) to complete a partial dataset.
        if force or not root.exists() or D.validate_dataset(root, dataset_id):
            print(f"[download] {dataset_id} -> {root}")
            _download(spec, root, subjects)
        issues = D.validate_dataset(root, dataset_id, deep=True)
        manifest = {**plan, "issues": issues,
                    "status": "ok" if not issues else "validation_failed"}
        if issues:
            print(f"[FAIL] {dataset_id} validation issues:")
            for i in issues:
                print("   -", i)
            return manifest
        sent.parent.mkdir(parents=True, exist_ok=True)
        sent.write_text(json.dumps(manifest, indent=2))
        print(f"[OK] {dataset_id} staged + validated -> sentinel {sent}")
        return manifest
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_id", choices=sorted(D.REGISTRY))
    p.add_argument("--version")
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    res = stage(a.dataset_id, version=a.version, subjects=a.subjects,
                dry_run=a.dry_run, force=a.force)
    return 0 if res.get("status") in ("ok", "already_staged", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
