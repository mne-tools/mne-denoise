#!/usr/bin/env python
"""mne-denoise CPU benchmark runner — Compute Canada (Fir) scaffolding.

This is the *thin* runner that makes the branch runnable on Fir.  It proves the
merged denoisers (ASR / ZapLine / DSS / ICanClean) import and execute on a CPU
node, resolves per-subject paths, and writes a JSON result per task.  The actual
benchmark metrics / dataset matrix are wired in a follow-up step — search for
"TODO(benchmark)" below.

Modes
-----
    python scripts/cc/run_benchmark.py --smoke
        Build a synthetic Raw and run every registered denoiser; report per
        method status + timing.  No dataset needed.  Exit non-zero if the
        headline method (ASR) fails.  Run this in `salloc` after building the env.

    python scripts/cc/run_benchmark.py --subject sub-03
    python scripts/cc/run_benchmark.py --slurm-array      # subject from $SLURM_ARRAY_TASK_ID
    python scripts/cc/run_benchmark.py --all              # every subject, in series

For real subjects the runner loads the subject's raw if present (falling back to
the synthetic smoke path otherwise) and runs the same registry — a real-data
sanity pass.  Replace `run_subject()` body with the benchmark proper later.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ── Reuse the existing shared config (scripts/config.py); scripts/ isn't a pkg ──
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import config  # type: ignore
except Exception:  # pragma: no cover - config is optional for --smoke
    config = None


# ══════════════════════════════════════════════════════════════════════════════
#  Denoiser registry — name → factory(sfreq, line_freq) → (estimator, apply_fn)
#
#  Each factory imports lazily so a missing/old symbol degrades to a reported
#  "error" rather than crashing the whole run.  apply_fn(est, raw) runs it.
#  `required=True` methods gate the smoke exit code.
# ══════════════════════════════════════════════════════════════════════════════

def _asr(sfreq, line_freq):
    from mne_denoise.asr import ASR

    cutoff = getattr(config, "ASR_CUTOFF", 20.0) if config else 20.0
    return ASR(sfreq=sfreq, cutoff=float(cutoff), picks="eeg"), \
        lambda est, raw: est.fit_transform(raw.copy())


def _asr_adaptive(sfreq, line_freq):
    from mne_denoise.asr import AdaptiveASR

    return AdaptiveASR(sfreq=sfreq, picks="eeg"), \
        lambda est, raw: est.fit_transform(raw.copy())


def _zapline(sfreq, line_freq):
    from mne_denoise.zapline import ZapLine

    return ZapLine(sfreq=sfreq, line_freq=line_freq), \
        lambda est, raw: est.fit_transform(raw.copy())


def _dss_bandpass(sfreq, line_freq):
    from mne_denoise.dss import DSS, BandpassBias

    bias = BandpassBias(freq_band=(8.0, 12.0), sfreq=sfreq)
    return DSS(bias=bias, n_components=4), \
        lambda est, raw: est.fit_transform(raw.copy().get_data())


def _icanclean(sfreq, line_freq):
    from mne_denoise.icanclean import ICanClean

    return ICanClean(), lambda est, raw: est.fit_transform(raw.copy())


# (name, factory, required)
REGISTRY = [
    ("asr", _asr, True),                 # headline method — gates smoke exit code
    ("asr_adaptive", _asr_adaptive, False),
    ("zapline", _zapline, False),
    ("dss_bandpass", _dss_bandpass, False),
    ("icanclean", _icanclean, False),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_raw(sfreq=250.0, n_ch=16, dur=30.0, line_freq=50.0, seed=42):
    """A small EEG-like Raw with line noise, an alpha rhythm, and burst artifacts."""
    import mne

    rng = np.random.default_rng(seed)
    n = int(sfreq * dur)
    t = np.arange(n) / sfreq
    data = rng.standard_normal((n_ch, n)) * 1e-5           # ~10 µV pink-ish noise
    data += (2e-5 * np.sin(2 * np.pi * line_freq * t))[None, :]  # line noise
    data += (1e-5 * np.sin(2 * np.pi * 10.0 * t))[None, :]       # alpha
    for ch in (0, 1):                                       # inject bursts
        for i in rng.integers(0, n - 30, size=6):
            data[ch, i:i + 25] += rng.standard_normal(25) * 1e-4
    info = mne.create_info([f"EEG{i:03d}" for i in range(n_ch)], sfreq, "eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def _run_registry(raw, line_freq, label):
    """Run every registered denoiser on `raw`; return a list of result dicts."""
    sfreq = float(raw.info["sfreq"])
    results = []
    for name, factory, required in REGISTRY:
        rec = {"method": name, "required": required, "status": "error"}
        t0 = time.perf_counter()
        try:
            est, apply_fn = factory(sfreq, line_freq)
            out = apply_fn(est, raw)
            shape = getattr(out, "shape", None)
            if shape is None and hasattr(out, "get_data"):
                shape = out.get_data().shape
            rec.update(status="ok", out_shape=list(shape) if shape is not None else None)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
            rec["traceback"] = traceback.format_exc(limit=3)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        flag = {"ok": "OK ", "error": "ERR"}[rec["status"]]
        print(f"  [{flag}] {label}:{name:<14s} {rec['seconds']:>7.3f}s"
              + (f"  ({rec.get('error')})" if rec["status"] == "error" else ""))
        results.append(rec)
    return results


def _output_dir():
    base = getattr(config, "OUTPUT_DIR", None) if config else None
    base = Path(base) if base else Path("batch_results")
    out = base / "cc_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(payload, name):
    out = _output_dir() / name
    out.write_text(json.dumps(payload, indent=2))
    print(f"=== wrote {out} ===")
    return out


def _env_stamp():
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "on_cluster": bool(config and config.is_narval()) if config else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Modes
# ══════════════════════════════════════════════════════════════════════════════

def run_smoke():
    line_freq = getattr(config, "LINE_FREQ", 50.0) if config else 50.0
    print("=== SMOKE: synthetic Raw through the denoiser registry ===")
    raw = _synthetic_raw(line_freq=line_freq)
    results = _run_registry(raw, line_freq, label="smoke")
    payload = {"mode": "smoke", "env": _env_stamp(), "results": results}
    _write(payload, "smoke_result.json")

    required_failed = [r["method"] for r in results if r["required"] and r["status"] != "ok"]
    ok = sum(r["status"] == "ok" for r in results)
    print(f"=== smoke: {ok}/{len(results)} methods OK ===")
    if required_failed:
        print(f"FAILED required method(s): {required_failed}", file=sys.stderr)
        return 1
    return 0


def _load_subject_raw(sub):
    """Best-effort load of a subject's raw EEG; None if not found."""
    if config is None:
        return None
    root = Path(getattr(config, "DATASET_ROOT", ""))
    eeg_dir = root / sub / "eeg"
    if not eeg_dir.is_dir():
        return None
    import mne

    for pat, reader in ((".vhdr", mne.io.read_raw_brainvision),
                        (".edf", mne.io.read_raw_edf),
                        (".fif", mne.io.read_raw_fif),
                        (".set", mne.io.read_raw_eeglab)):
        hits = sorted(eeg_dir.glob(f"*{pat}"))
        if hits:
            return reader(str(hits[0]), preload=True, verbose="ERROR")
    return None


def run_subject(sub):
    line_freq = getattr(config, "LINE_FREQ", 50.0) if config else 50.0
    print(f"=== subject {sub} ===")
    raw = _load_subject_raw(sub)
    if raw is None:
        print(f"  no data for {sub} → synthetic fallback (env/denoiser sanity only)")
        raw = _synthetic_raw(line_freq=line_freq)
        source = "synthetic"
    else:
        print(f"  loaded {sub}: {len(raw.ch_names)} ch, {raw.times[-1]:.1f}s")
        source = "real"
        raw.pick("eeg")

    # TODO(benchmark): replace the registry sanity-pass below with the real
    # benchmark (preprocess → per-method clean → QA metrics via mne_denoise.qa).
    results = _run_registry(raw, line_freq, label=sub)
    payload = {"mode": "subject", "subject": sub, "data_source": source,
               "env": _env_stamp(), "results": results}
    _write(payload, f"{sub}_result.json")
    return 0 if all(r["status"] == "ok" for r in results if r["required"]) else 1


# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="synthetic registry smoke test")
    g.add_argument("--subject", metavar="sub-XX", help="run one subject")
    g.add_argument("--slurm-array", action="store_true",
                   help="subject from $SLURM_ARRAY_TASK_ID (sub-NN)")
    g.add_argument("--all", action="store_true", help="every subject, in series")
    args = p.parse_args(argv)

    if args.smoke:
        return run_smoke()
    if args.slurm_array:
        task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        return run_subject(f"sub-{task:02d}")
    if args.subject:
        return run_subject(args.subject)
    if args.all:
        subs = getattr(config, "ALL_SUBJECTS", []) if config else []
        if not subs:
            print("No ALL_SUBJECTS in config; nothing to do.", file=sys.stderr)
            return 1
        rc = 0
        for sub in subs:
            rc |= run_subject(sub)
        return rc
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
