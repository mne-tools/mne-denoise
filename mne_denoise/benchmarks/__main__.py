"""Command-line entry point for frozen benchmark protocols."""

from __future__ import annotations

import argparse
import json
import pathlib

from .config import load_arm_config, validate_for_submission
from .provenance import freeze_protocol, inventory_results


def _expand_config_paths(paths: list[str]) -> list[str]:
    expanded: list[pathlib.Path] = []
    for raw in paths:
        path = pathlib.Path(raw)
        if path.is_dir():
            expanded.extend(path.glob("*.yaml"))
            expanded.extend(path.glob("*.yml"))
        else:
            expanded.append(path)
    return [str(path) for path in sorted(set(expanded))]


def _validate(paths: list[str]) -> int:
    failed = False
    for raw in _expand_config_paths(paths):
        path = pathlib.Path(raw)
        issues = validate_for_submission(load_arm_config(path))
        if issues:
            failed = True
            print(f"FAIL {path}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"PASS {path}")
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate frozen benchmark YAML")
    validate.add_argument("configs", nargs="+")

    freeze = sub.add_parser("freeze", help="write an immutable protocol manifest")
    freeze.add_argument("configs", nargs="+")
    freeze.add_argument("--protocol-id", required=True)
    freeze.add_argument("--repo-root", default=".")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--allow-dirty", action="store_true")

    inventory = sub.add_parser("inventory", help="summarize terminal run records")
    inventory.add_argument("results_root")
    inventory.add_argument("--output")

    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.configs)
    if args.command == "freeze":
        record = freeze_protocol(
            _expand_config_paths(args.configs),
            repo_root=args.repo_root,
            protocol_id=args.protocol_id,
            output_path=args.output,
            allow_dirty=args.allow_dirty,
        )
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    report = inventory_results(args.results_root)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        pathlib.Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
