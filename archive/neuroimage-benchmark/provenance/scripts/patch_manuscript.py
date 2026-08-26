#!/usr/bin/env python
r"""Fill ``\bval{arm.metric.comparator}`` placeholders in the manuscript from a flat
values JSON emitted by the benchmark.

The manuscript carries greppable ``\bval{key}`` macros (red ``[X.XX]`` + key superscript).
This rewrites every ``\bval{key}`` whose ``key`` appears in the values JSON to the literal
number; unmatched placeholders are left untouched so the remaining work stays greppable.

    # see what would change, and which keys are still pending
    python scripts/patch_manuscript.py --tex template.tex --values line_ds003620.json --report

    # apply in place (writes a .bak first)
    python scripts/patch_manuscript.py --tex template.tex --values line_ds003620.json --write

The values JSON is ``{"line_ds003620.rf0.zlp": 1.32, ...}``; numbers are formatted with
sensible precision, strings are inserted verbatim (let the producer control formatting).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

KEY_RE = re.compile(r"\\bval\{([^}]+)\}")


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    av = abs(v)
    if av != 0 and (av < 1e-2 or av >= 1e4):
        return f"{v:.2e}"
    return f"{v:.3g}"


def patch(tex: str, values: dict) -> tuple[str, list[str], list[str]]:
    """Return ``(new_tex, patched_keys, remaining_keys)``."""
    patched, remaining = [], []

    def _sub(m):
        key = m.group(1)
        if key in values:
            patched.append(key)
            return _fmt(values[key])
        remaining.append(key)
        return m.group(0)

    new = KEY_RE.sub(_sub, tex)
    return new, patched, remaining


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tex", required=True)
    p.add_argument("--values", required=True, help="flat JSON {key: number|string}")
    p.add_argument("--write", action="store_true", help="rewrite --tex in place (.bak backup)")
    p.add_argument("--out", default=None, help="write patched tex here instead of stdout/inplace")
    p.add_argument("--report", action="store_true", help="print patched/remaining summary")
    a = p.parse_args(argv)

    tex_path = pathlib.Path(a.tex)
    tex = tex_path.read_text(encoding="utf-8")
    values = json.loads(pathlib.Path(a.values).read_text(encoding="utf-8"))
    new, patched, remaining = patch(tex, values)

    unused = sorted(set(values) - set(patched))
    if a.report or not (a.write or a.out):
        total = len(patched) + len(remaining)
        print(f"placeholders: {total}  patched: {len(patched)}  remaining: {len(remaining)}")
        if patched:
            print("  patched: " + ", ".join(sorted(set(patched))))
        if remaining:
            print("  remaining: " + ", ".join(sorted(set(remaining))[:60]))
        if unused:
            print(f"  WARNING values with no placeholder ({len(unused)}): " + ", ".join(unused))

    if a.write:
        tex_path.with_suffix(tex_path.suffix + ".bak").write_text(tex, encoding="utf-8")
        tex_path.write_text(new, encoding="utf-8")
        print(f"[written] {tex_path} ({len(set(patched))} keys; backup .bak)")
    elif a.out:
        pathlib.Path(a.out).write_text(new, encoding="utf-8")
        print(f"[written] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
