#!/usr/bin/env python3
"""Rename an unnumbered Towncrier fragment to the current pull request."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-number",
        type=int,
        help="Pull request number; otherwise derive it from GITHUB_EVENT_PATH.",
    )
    return parser


def _event_pr_number() -> int | None:
    """Return the pull request number from a GitHub event, if applicable."""
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_name.startswith("pull_request"):
        print(f"No-op: {event_name or 'local execution'} is not a pull request.")
        return None
    if not event_path:
        raise RuntimeError("GITHUB_EVENT_PATH is required for pull request events")
    with Path(event_path).open(encoding="utf-8") as file:
        event: dict[str, Any] = json.load(file)
    number = event.get("number") or event.get("pull_request", {}).get("number")
    if not isinstance(number, int) or number <= 0:
        raise RuntimeError("could not find a positive pull request number in the event")
    return number


def _towncrier_config() -> tuple[Path, tuple[str, ...]]:
    """Return the configured fragment directory and supported fragment types."""
    with Path("pyproject.toml").open("rb") as file:
        config = tomllib.load(file)
    towncrier = config["tool"]["towncrier"]
    directory = Path(towncrier["directory"])
    types = tuple(entry["directory"] for entry in towncrier["type"])
    if directory.is_absolute() or not types:
        raise RuntimeError("Towncrier directory and types must be configured locally")
    return directory, types


def rename(pr_number: int) -> int:
    """Rename all supported unnumbered fragments and return a process status."""
    directory, types = _towncrier_config()
    operations = [
        (
            directory / f"{fragment_type}.rst",
            directory / f"{pr_number}.{fragment_type}.rst",
        )
        for fragment_type in types
        if (directory / f"{fragment_type}.rst").is_file()
    ]
    if not operations:
        print(f"No unnumbered Towncrier fragments found in {directory}.")
        return 0
    collisions = [target for _, target in operations if target.exists()]
    if collisions:
        names = ", ".join(str(path) for path in collisions)
        print(
            f"Refusing to overwrite existing Towncrier fragment(s): {names}",
            file=sys.stderr,
        )
        return 1
    for source, target in operations:
        source.rename(target)
        print(f"Renamed {source} -> {target}")
    return 0


def main() -> int:
    """Run the Towncrier fragment renamer."""
    args = _parser().parse_args()
    try:
        pr_number = args.pr_number if args.pr_number is not None else _event_pr_number()
        return 0 if pr_number is None else rename(pr_number)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"Towncrier fragment renaming failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
