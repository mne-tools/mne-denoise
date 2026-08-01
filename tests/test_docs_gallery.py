from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
POLICY_PATH = REPO_ROOT / "docs" / "_gallery_execution.py"


def _load_gallery_policy():
    spec = importlib.util.spec_from_file_location("_gallery_execution", POLICY_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gallery_scripts():
    return {
        path.relative_to(EXAMPLES_DIR).as_posix()
        for path in EXAMPLES_DIR.rglob("plot_*.py")
    }


def test_external_data_gallery_audit_is_complete():
    policy = _load_gallery_policy()
    scripts = _gallery_scripts()
    external_data_markers = (
        "mne.datasets",
        "from mne.datasets",
        "urlretrieve(",
    )
    detected = {
        path
        for path in scripts
        if any(
            marker in (EXAMPLES_DIR / path).read_text(encoding="utf-8")
            for marker in external_data_markers
        )
    }

    assert detected == set(policy.EXTERNAL_DATA_EXAMPLES)
    assert all(
        "sys.exit(" not in (EXAMPLES_DIR / path).read_text(encoding="utf-8")
        for path in scripts
    )


def test_offline_gallery_executes_only_audited_safe_scripts():
    policy = _load_gallery_policy()
    scripts = _gallery_scripts()
    external = set(policy.EXTERNAL_DATA_EXAMPLES)
    offline_pattern = policy.gallery_filename_pattern(
        execute_external_data_examples=False
    )
    local_pattern = policy.gallery_filename_pattern(execute_external_data_examples=True)

    assert all(re.search(local_pattern, path) for path in scripts)
    assert all(not re.search(offline_pattern, path) for path in external)
    assert all(re.search(offline_pattern, path) for path in scripts - external)
