"""Attenuation--preservation trade-off helpers: Pareto front + operating points.

A method family is swept over its aggressiveness parameter (e.g. ZapLine
``n_remove``, ASR ``cutoff``, DSS ``n_components``); each grid value yields one
(attenuation, preservation) point.  These helpers pick the non-dominated front
and the ``{default, validation-selected}`` operating points the manuscript
reports (see ``docs/benchmarks/BENCHMARK_PLAN.md`` §8).
"""

from __future__ import annotations

from typing import Sequence


def pareto_front(points: Sequence[tuple[float, float]], *,
                 x_better: str = "max", y_better: str = "max") -> list[int]:
    """Indices of non-dominated points. Each axis improves toward ``min``/``max``.

    A point *b* is dominated by *a* if *a* is at least as good on both axes and
    strictly better on at least one.
    """
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)

    def ge_x(a, b):
        return pts[a][0] <= pts[b][0] if x_better == "min" else pts[a][0] >= pts[b][0]

    def ge_y(a, b):
        return pts[a][1] <= pts[b][1] if y_better == "min" else pts[a][1] >= pts[b][1]

    def gt_x(a, b):
        return pts[a][0] < pts[b][0] if x_better == "min" else pts[a][0] > pts[b][0]

    def gt_y(a, b):
        return pts[a][1] < pts[b][1] if y_better == "min" else pts[a][1] > pts[b][1]

    front = []
    for i in range(n):
        dominated = any(
            i != j and ge_x(j, i) and ge_y(j, i) and (gt_x(j, i) or gt_y(j, i))
            for j in range(n)
        )
        if not dominated:
            front.append(i)
    return front


def select_operating_point(points: Sequence[tuple[float, float]], *,
                           min_preservation: float | None = None,
                           x_better: str = "max", y_better: str = "max") -> int | None:
    """Validation-selected operating point: the best attenuation (x) among grid
    values whose preservation (y) meets ``min_preservation`` (if given), else the
    knee of the front.  Returns an index into ``points`` or None."""
    if not points:
        return None
    idx = list(range(len(points)))
    if min_preservation is not None:
        ok = [i for i in idx if (points[i][1] >= min_preservation if y_better == "max"
                                 else points[i][1] <= min_preservation)]
        if ok:
            return max(ok, key=lambda i: points[i][0]) if x_better == "max" \
                else min(ok, key=lambda i: points[i][0])
    # knee: maximise normalised (x + y) on the front
    fr = pareto_front(points, x_better=x_better, y_better=y_better)
    xs = [points[i][0] for i in fr]
    ys = [points[i][1] for i in fr]
    rx = (max(xs) - min(xs)) or 1.0
    ry = (max(ys) - min(ys)) or 1.0
    sx = 1.0 if x_better == "max" else -1.0
    sy = 1.0 if y_better == "max" else -1.0
    return max(fr, key=lambda i: sx * points[i][0] / rx + sy * points[i][1] / ry)


def method_params(cfg: dict, method_id: str) -> dict:
    """Return a config's top-level parameter block for one method, if it declares one.

    A config may pin a method's operating point in a block named after it, e.g.::

        iterative_dss: {n_components: 3, polarity: subtract}

    Runners that never read this block fall through to adapter defaults, so a declared
    value is honoured only when it happens to equal the default. That is what happened to
    ocular_erp_core's ``eog_dss: {n_components: 2}``: the reported rank matched the
    declaration by coincidence, not because the config was read
    (documents/method_identity_correction_20260727_eog_dss.json in mne-denoise-reports).
    Reading it makes the config authoritative instead of decorative.
    """
    block = cfg.get(method_id)
    if not isinstance(block, dict):
        return {}
    # Only scalars are operating-point parameters. Nested structures in these blocks are
    # design metadata (event sources, windows, selection rules) that adapters do not take.
    return {k: v for k, v in block.items() if isinstance(v, (int, float, str, bool))}


def method_runs(cfg: dict, method_id: str):
    """Yield ``(suffix, params)`` for a method: its sweep grid, or one default run.

    ``suffix`` is a filesystem-safe tag (``""`` for the default operating point);
    ``params`` are passed to ``comparators.get(method_id, **params)``.

    The config's own parameter block for the method is merged in first, so a swept
    parameter overrides the pinned one for that axis and everything else stays as declared.
    """
    base = method_params(cfg, method_id)
    sweep = (cfg.get("sweep") or {}).get(method_id)
    grid = (sweep or {}).get("grid")
    if sweep and isinstance(grid, list) and grid:
        param = sweep["param"]
        for v in grid:
            yield f"{param}-{v}", {**base, param: v}
    else:
        yield "", dict(base)
