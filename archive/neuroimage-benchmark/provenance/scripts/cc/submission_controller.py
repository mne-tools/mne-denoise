#!/usr/bin/env python
"""Submission-safe wrapper for local and Slurm benchmark commands.

Examples
--------
Validate a frozen arm before allocating compute::

    python scripts/cc/submission_controller.py preflight \
        --config configs/benchmarks/ground_truth_generic.yaml

Run one immutable attempt (everything after ``--`` is the wrapped command)::

    python scripts/cc/submission_controller.py run \
        --config configs/benchmarks/ground_truth_generic.yaml \
        --dataset-manifest manifests/eegdenoisenet.json \
        --output-dir results/ground_truth_generic/seed-000/iterative_dss \
        --method iterative_dss --unit-id seed-000 -- \
        python scripts/run_ground_truth_arm.py --config \
        configs/benchmarks/ground_truth_generic.yaml --all

The wrapper never substitutes synthetic data for a missing real dataset.  A smoke
benchmark must be requested explicitly from the underlying runner and is not a
submission attempt.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from mne_denoise.benchmarks.config import assert_submission_ready, load_arm_config
from mne_denoise.benchmarks.provenance import (
    AttemptRecorder,
    build_run_record,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset-manifest", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--method", required=True)
    run.add_argument("--unit-id", required=True)
    run.add_argument("--seed", type=int)
    run.add_argument("--information-tier")
    run.add_argument("--repo-root", default=str(_REPO))
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    cfg = load_arm_config(args.config)
    assert_submission_ready(cfg, source=args.config)
    if args.action == "preflight":
        print(f"PASS {args.config}: submission-ready configuration")
        return 0

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("run requires a wrapped command after --")
    record = build_run_record(
        arm=cfg["arm"],
        method=args.method,
        unit_id=args.unit_id,
        config_path=args.config,
        dataset_manifest=args.dataset_manifest,
        repo_root=args.repo_root,
        seed=args.seed,
        information_tier=args.information_tier,
        allow_dirty=args.allow_dirty,
    )
    with AttemptRecorder(args.output_dir, record):
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
