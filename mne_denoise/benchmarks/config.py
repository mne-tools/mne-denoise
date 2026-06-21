"""Benchmark arm configuration loading + submission validation.

P0 ships per-arm YAML *skeletons* with placeholder values
(``pending``/``pending_feasibility``/``TBD``/``null``/``auto``).  Those are fine
for scaffolding but must never reach a full Fir run.  :func:`validate_for_submission`
recursively flags any unresolved placeholder so the controller can refuse to
submit (see ``docs/benchmarks/PUBLICATION_GATE.md``).

This intentionally does **not** import heavy deps (only PyYAML), so it can run on
a login node as a pre-submission check.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

# Case-insensitive placeholder sentinels that must be resolved before a full run.
PLACEHOLDER_STRINGS = {"pending", "pending_feasibility", "tbd", "null", "auto"}


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    value: Any
    reason: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.path}: {self.reason} (value={self.value!r})"


def load_arm_config(path: str | pathlib.Path) -> dict:
    """Load and YAML-parse a single arm config."""
    import yaml

    p = pathlib.Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p} did not parse to a mapping")
    return data


def _is_placeholder(value: Any) -> str | None:
    """Return a reason string if ``value`` is an unresolved placeholder."""
    if value is None:
        return "null / None placeholder"
    if isinstance(value, str) and value.strip().lower() in PLACEHOLDER_STRINGS:
        return f"unresolved placeholder '{value}'"
    return None


def _walk(node: Any, prefix: str, issues: list[ConfigIssue]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _walk(v, f"{prefix}.{k}" if prefix else str(k), issues)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _walk(v, f"{prefix}[{i}]", issues)
    else:
        reason = _is_placeholder(node)
        if reason:
            issues.append(ConfigIssue(prefix, node, reason))


def validate_for_submission(cfg: dict) -> list[ConfigIssue]:
    """Return all unresolved-placeholder issues; empty list ⇒ submission-ready.

    Detects ``null``/``pending``/``pending_feasibility``/``TBD``/``auto`` anywhere
    in the config tree.  (Single-choice freezing of specific fields is enforced
    per-runner in P3; this is the generic gate.)
    """
    issues: list[ConfigIssue] = []
    _walk(cfg, "", issues)
    # A couple of always-required top-level keys.
    for key in ("arm", "runner"):
        if not cfg.get(key):
            issues.append(ConfigIssue(key, cfg.get(key), "missing required top-level key"))
    return issues


def is_submission_ready(cfg: dict) -> bool:
    return not validate_for_submission(cfg)
