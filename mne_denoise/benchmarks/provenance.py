"""Immutable benchmark provenance and terminal-status records.

The paper's evidence chain is deliberately file based: frozen YAML produces one
record per attempted method/subject/seed, and all later tables and figures consume
those records.  This module has no dependency on MNE and is safe on a login node.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import traceback
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import assert_submission_ready, load_arm_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    """Hash a JSON-serializable scientific specification."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha256_file(path: str | pathlib.Path) -> str:
    """Stream a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as fid:
        for block in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_root: str | pathlib.Path) -> dict[str, Any]:
    """Return commit, branch, and dirty paths for a Git worktree."""
    root = pathlib.Path(repo_root).resolve()

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    status = run("status", "--porcelain", "--untracked-files=normal")
    return {
        "root": str(root),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current") or "detached",
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def environment_record() -> dict[str, Any]:
    """Capture the runtime versions that can change numerical results."""
    packages: dict[str, str] = {}
    for name in (
        "mne-denoise",
        "mne",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "pymatreader",
        "PyWavelets",
        "EMD-signal",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def environment_hash() -> tuple[str, dict[str, Any]]:
    """Return an environment fingerprint and its human-readable record."""
    record = environment_record()
    return sha256_value(record), record


def dataset_manifest_hash(path: str | pathlib.Path | None) -> str:
    """Hash a dataset manifest without hashing multi-terabyte raw data.

    The manifest itself must contain the dataset DOI/version and file hashes or
    immutable upstream identifiers.  Missing manifests are submission blockers.
    """
    if path is None:
        raise ValueError("a dataset manifest is required for a frozen run")
    manifest = pathlib.Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest}")
    return sha256_file(manifest)


def protocol_fingerprint(
    *, git_commit: str, config_hash: str, dataset_hash: str, environment_hash_: str
) -> str:
    """Hash the four immutable inputs defining a scientific run."""
    return sha256_value(
        {
            "git_commit": git_commit,
            "config_hash": config_hash,
            "dataset_manifest_hash": dataset_hash,
            "environment_hash": environment_hash_,
        }
    )


def freeze_protocol(
    config_paths: list[str | pathlib.Path],
    *,
    repo_root: str | pathlib.Path,
    protocol_id: str,
    output_path: str | pathlib.Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Validate configurations and write an immutable protocol manifest."""
    root = pathlib.Path(repo_root).resolve()
    configs = []
    for path in sorted(pathlib.Path(p).resolve() for p in config_paths):
        cfg = load_arm_config(path)
        assert_submission_ready(cfg, source=str(path))
        try:
            portable_path = path.relative_to(root).as_posix()
        except ValueError:
            portable_path = str(path)
        configs.append(
            {
                "arm": cfg["arm"],
                "path": portable_path,
                "sha256": sha256_file(path),
            }
        )
    state = git_state(root)
    if state["dirty"] and not allow_dirty:
        raise RuntimeError(
            "refusing to freeze a dirty worktree; commit the protocol first:\n"
            + "\n".join(state["dirty_paths"])
        )
    env_hash, env = environment_hash()
    state["root"] = "."
    record = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "frozen_at_utc": _utc_now(),
        "git": state,
        "environment_hash": env_hash,
        "environment": env,
        "configs": configs,
    }
    record["manifest_hash"] = sha256_value(record)
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def _peak_memory_mb() -> float | None:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            value /= 1024.0
        return value / 1024.0
    except (ImportError, AttributeError, ValueError):
        try:
            import psutil

            memory = psutil.Process().memory_info()
            # Windows exposes the process peak working set.  On platforms where
            # psutil has no peak field, RSS is still a useful explicitly bounded
            # fallback for an attempted-run record.
            value = float(getattr(memory, "peak_wset", memory.rss))
            return value / (1024.0**2)
        except (ImportError, AttributeError, OSError, ValueError):
            return None


@dataclass
class RunRecord:
    """Serializable record for one attempted benchmark unit."""

    run_id: str
    arm: str
    method: str
    unit_id: str
    status: str
    started_at_utc: str
    completed_at_utc: str | None = None
    runtime_seconds: float | None = None
    peak_memory_mb: float | None = None
    seed: int | None = None
    information_tier: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    config_hash: str | None = None
    dataset_manifest_hash: str | None = None
    environment_hash: str | None = None
    run_fingerprint: str | None = None
    slurm_job_id: str | None = None
    slurm_array_task_id: str | None = None
    effective_rank_before: int | None = None
    effective_rank_after: int | None = None
    warnings: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None


class AttemptRecorder(AbstractContextManager[RunRecord]):
    """Always write ``terminal_status.json``, including failed attempts."""

    def __init__(self, output_dir: str | pathlib.Path, record: RunRecord):
        self.output_dir = pathlib.Path(output_dir)
        self.record = record
        self._started = time.perf_counter()

    @property
    def path(self) -> pathlib.Path:
        """Path of the atomic terminal-status record."""
        return self.output_dir / "terminal_status.json"

    def _write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(asdict(self.record), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def __enter__(self) -> RunRecord:
        self.record.status = "running"
        self._write()
        return self.record

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.record.completed_at_utc = _utc_now()
        self.record.runtime_seconds = time.perf_counter() - self._started
        self.record.peak_memory_mb = _peak_memory_mb()
        if exc is None:
            if self.record.status == "running":
                self.record.status = "completed"
        else:
            self.record.status = "failed"
            self.record.error_type = exc_type.__name__ if exc_type else type(exc).__name__
            self.record.error_message = str(exc)
            self.record.traceback = "".join(traceback.format_exception(exc_type, exc, tb))
        self._write()
        return False


def build_run_record(
    *,
    arm: str,
    method: str,
    unit_id: str,
    config_path: str | pathlib.Path,
    dataset_manifest: str | pathlib.Path,
    repo_root: str | pathlib.Path,
    seed: int | None = None,
    information_tier: str | None = None,
    allow_dirty: bool = False,
) -> RunRecord:
    """Create a fingerprinted record after enforcing the submission preflight."""
    cfg = load_arm_config(config_path)
    assert_submission_ready(cfg, source=str(config_path))
    state = git_state(repo_root)
    if state["dirty"] and not allow_dirty:
        raise RuntimeError("submission runs require a clean Git worktree")
    cfg_hash = sha256_file(config_path)
    data_hash = dataset_manifest_hash(dataset_manifest)
    env_hash, _ = environment_hash()
    fingerprint = protocol_fingerprint(
        git_commit=state["commit"],
        config_hash=cfg_hash,
        dataset_hash=data_hash,
        environment_hash_=env_hash,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{arm}_{unit_id}_{method}_{stamp}_{fingerprint[:10]}"
    return RunRecord(
        run_id=run_id,
        arm=arm,
        method=method,
        unit_id=unit_id,
        status="created",
        started_at_utc=_utc_now(),
        seed=seed,
        information_tier=information_tier,
        git_commit=state["commit"],
        git_dirty=state["dirty"],
        config_hash=cfg_hash,
        dataset_manifest_hash=data_hash,
        environment_hash=env_hash,
        run_fingerprint=fingerprint,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
    )


def inventory_results(root: str | pathlib.Path) -> dict[str, Any]:
    """Summarize terminal records without treating missing records as success."""
    result_root = pathlib.Path(root)
    records = []
    for path in sorted(result_root.rglob("terminal_status.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rec = {"status": "invalid_record", "error_message": str(exc)}
        rec["path"] = str(path)
        records.append(rec)
    counts: dict[str, int] = {}
    for record in records:
        status = record.get("status", "missing_status")
        counts[status] = counts.get(status, 0) + 1
    return {"root": str(result_root.resolve()), "counts": counts, "records": records}
