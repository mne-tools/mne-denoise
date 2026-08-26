"""Tier-C AASR ICLabel scoring on Klados-cleaned EEG.

For each AASR variant (init / mw / psp / psw):

1. Loads the 41 paired Klados trials.
2. Applies 1-50 Hz bandpass + AASR cleaning at cutoff=20.
3. Concatenates the 41 cleaned trials into one continuous MNE Raw,
   separated by 1-s zero-padding gaps to avoid filter-ringing at trial
   boundaries.
4. Attaches the 19-channel standard 10-20 montage (the Klados channel
   order, per the Data in Brief paper).
5. Applies 1-100 Hz bandpass + common-average reference (ICLabel
   preprocessing).
6. Fits extended-infomax ICA and runs ``mne_icalabel.iclabel_label_components``.
7. Emits per-variant ``tier_C_iclabel_<variant>.{json,csv}`` plus a
   combined ``tier_C_iclabel_comparison.csv``.

For comparison, the contaminated baseline is also scored (no AASR cleaning)
as the "before" reference. The dipolar-IC proxy is ICLabel brain
probability >= 0.5 (same convention as the Standard ASR and rASR sprints).
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import mne
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import AdaptiveASR  # noqa: E402

# Standard 19-channel 10-20 layout used by Klados (per Data in Brief paper).
# Channel order matches the BIOSEMI-resampled order in Pure_Data.mat.
KLADOS_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]

ICLABEL_CLASSES = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)

VARIANTS = ("contaminated", "init", "mw", "psp", "psw")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "refs" / "asr" / "datasets" / "mendeley_klados_eog" / "data",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "paper_validation" / "aasr",
    )
    p.add_argument("--cutoff", type=float, default=20.0)
    p.add_argument("--sfreq", type=float, default=200.0)
    p.add_argument("--aasr-highpass", type=float, default=1.0)
    p.add_argument("--aasr-lowpass", type=float, default=50.0)
    p.add_argument("--iclabel-lowpass", type=float, default=100.0)
    p.add_argument("--init-window-s", type=float, default=20.0)
    p.add_argument("--psw-window-s", type=float, default=20.0)
    p.add_argument("--mw-window-s", type=float, default=20.0)
    p.add_argument("--random-state", type=int, default=97)
    p.add_argument("--ica-n-components", type=float, default=0.99)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument(
        "--brain-prob-threshold",
        type=float,
        default=0.5,
        help="ICLabel brain-class threshold used as dipolar-IC proxy.",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=VARIANTS,
    )
    p.add_argument("--max-trials", type=int, default=41)
    p.add_argument("--gap-seconds", type=float, default=1.0)
    return p.parse_args()


def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def _lowpass(data: np.ndarray, sfreq: float, high: float) -> np.ndarray:
    sos = butter(4, high, btype="lowpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def _load_trial(pure_path: Path, contam_path: Path, idx: int):
    pure_var = f"sim{idx}_resampled"
    contam_var = f"sim{idx}_con"
    try:
        pure = loadmat(pure_path, squeeze_me=False, variable_names=[pure_var])[
            pure_var
        ].astype(np.float64)
        contam = loadmat(contam_path, squeeze_me=False, variable_names=[contam_var])[
            contam_var
        ].astype(np.float64)
    except (KeyError, ValueError, OSError) as exc:
        # Sporadic OSError("could not read bytes") on Klados zlib streams --
        # skip the trial gracefully.
        print(f"  [trial {idx}] load failed: {type(exc).__name__}: {exc}")
        return None
    n = min(pure.shape[1], contam.shape[1])
    return pure[:, :n], contam[:, :n]


def _run_aasr_variant(
    variant: str,
    contam_filt: np.ndarray,
    *,
    sfreq: float,
    cutoff: float,
    init_window_s: float,
    psw_window_s: float,
    mw_window_s: float,
) -> np.ndarray:
    if variant == "contaminated":
        return contam_filt
    if variant == "init":
        asr = AdaptiveASR(sfreq=sfreq, cutoff=cutoff, variant="psp", verbose=False)
        first = contam_filt[
            :, : min(int(round(init_window_s * sfreq)), contam_filt.shape[1])
        ]
        asr.fit(first)
        return asr.transform(contam_filt)
    if variant == "psp":
        asr = AdaptiveASR(sfreq=sfreq, cutoff=cutoff, variant="psp", verbose=False)
        return asr.fit_transform(contam_filt)
    if variant == "mw":
        asr = AdaptiveASR(
            sfreq=sfreq,
            cutoff=cutoff,
            variant="mw",
            mw_window_length=mw_window_s,
            verbose=False,
        )
        return asr.fit_transform(contam_filt)
    if variant == "psw":
        asr = AdaptiveASR(sfreq=sfreq, cutoff=cutoff, variant="psw", verbose=False)
        win = int(round(psw_window_s * sfreq))
        asr.fit(contam_filt[:, : min(win, contam_filt.shape[1])])
        cursor = win
        while cursor < contam_filt.shape[1]:
            stop = min(cursor + win, contam_filt.shape[1])
            chunk = contam_filt[:, cursor:stop]
            if chunk.shape[1] >= asr.blocksize:
                with contextlib.suppress(Exception):
                    asr.partial_fit(chunk)
            cursor = stop
        return asr.transform(contam_filt)
    raise ValueError(f"Unknown variant: {variant}")


def _build_raw(
    cleaned_segments: list[np.ndarray],
    sfreq: float,
    gap_samples: int,
) -> mne.io.RawArray:
    n_ch = cleaned_segments[0].shape[0]
    pieces: list[np.ndarray] = []
    for i, seg in enumerate(cleaned_segments):
        pieces.append(seg)
        if i != len(cleaned_segments) - 1:
            pieces.append(np.zeros((n_ch, gap_samples), dtype=seg.dtype))
    big = np.concatenate(pieces, axis=1)
    info = mne.create_info(KLADOS_CHANNELS, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(big, info, verbose=False)
    # Standard 10-20 montage (assume Klados labels match the 19-ch standard set).
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    return raw


def _classify_iclabel(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> dict:
    from mne_icalabel.iclabel import iclabel_label_components

    proba = np.asarray(iclabel_label_components(raw, ica, inplace=False))
    if proba.ndim != 2 or proba.shape[1] != len(ICLABEL_CLASSES):
        raise RuntimeError(f"Unexpected ICLabel proba shape {proba.shape}")
    argmax = np.argmax(proba, axis=1)
    labels = [ICLABEL_CLASSES[i] for i in argmax]
    counts = {cls: int(sum(1 for L in labels if cls == L)) for cls in ICLABEL_CLASSES}
    return {
        "proba": proba.tolist(),
        "labels": labels,
        "counts": counts,
        "brain_prob_per_ic": proba[:, 0].tolist(),
    }


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pure_path = args.data_dir / "Pure_Data.mat"
    contam_path = args.data_dir / "Contaminated_Data.mat"
    if not pure_path.exists() or not contam_path.exists():
        raise SystemExit(
            f"Klados files missing under {args.data_dir}; "
            "see refs/asr/datasets/mendeley_klados_eog/README.md."
        )

    try:
        from mne_icalabel.iclabel import iclabel_label_components  # noqa: F401
    except ImportError:
        raise SystemExit(
            "mne_icalabel is not installed. Run: pip install mne-icalabel onnxruntime"
        ) from None

    pure_vars = {n for n, _, _ in whosmat(pure_path)}
    trial_indices = [
        i for i in range(1, args.max_trials + 1) if f"sim{i}_resampled" in pure_vars
    ]
    print(f"[klados-iclabel] paired trials: {len(trial_indices)}")

    # Pre-load + filter the contaminated EEG trials once -- AASR will be re-run
    # per variant, but the filtering is variant-independent.
    contam_filtered: list[np.ndarray] = []
    skipped = 0
    for idx in trial_indices:
        pair = _load_trial(pure_path, contam_path, idx)
        if pair is None:
            skipped += 1
            continue
        _, contam = pair
        contam_filt = _bandpass(
            contam, args.sfreq, args.aasr_highpass, args.aasr_lowpass
        )
        contam_filtered.append(contam_filt)
    print(f"[klados-iclabel] loaded {len(contam_filtered)} trials (skipped {skipped})")

    gap_samples = max(0, int(round(args.gap_seconds * args.sfreq)))

    overall: list[dict] = []
    for variant in args.variants:
        print(f"\n[klados-iclabel] variant={variant}")
        cleaned_segments: list[np.ndarray] = []
        for trial_idx, contam_filt in zip(trial_indices, contam_filtered):
            try:
                cleaned = _run_aasr_variant(
                    variant,
                    contam_filt,
                    sfreq=args.sfreq,
                    cutoff=args.cutoff,
                    init_window_s=args.init_window_s,
                    psw_window_s=args.psw_window_s,
                    mw_window_s=args.mw_window_s,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [trial {trial_idx}] FAILED ({variant}): {type(exc).__name__}: {exc}"
                )
                continue
            cleaned_segments.append(cleaned)

        if not cleaned_segments:
            print(f"  -> no usable trials for variant={variant}, skipping ICLabel")
            continue

        raw = _build_raw(cleaned_segments, args.sfreq, gap_samples)
        # ICLabel preprocessing: low-pass at args.iclabel_lowpass Hz, common-average reference.
        # Skip the LP if the data's Nyquist is already <= requested h_freq
        # (e.g. Klados @ 200 Hz has Nyquist = 100 Hz == iclabel_lowpass default).
        nyquist = raw.info["sfreq"] / 2.0
        if args.iclabel_lowpass < nyquist - 1e-6:
            raw.filter(
                l_freq=None,
                h_freq=args.iclabel_lowpass,
                picks="eeg",
                verbose=False,
            )
        raw.set_eeg_reference("average", projection=False, verbose=False)

        ica = mne.preprocessing.ICA(
            n_components=args.ica_n_components,
            method="infomax",
            random_state=args.random_state,
            max_iter=args.max_iter,
            fit_params={"extended": True},
        )
        ica.fit(raw, picks="eeg", verbose=False)
        out = _classify_iclabel(raw, ica)

        dipolar = int(
            sum(1 for p in out["brain_prob_per_ic"] if p >= args.brain_prob_threshold)
        )
        payload = {
            "variant": variant,
            "cutoff": args.cutoff,
            "n_trials_included": len(cleaned_segments),
            "n_components": len(out["labels"]),
            "counts": out["counts"],
            "dipolar_count": dipolar,
            "brain_prob_threshold": args.brain_prob_threshold,
        }
        overall.append(
            {
                **payload,
                "labels": out["labels"],
                "brain_prob_per_ic": out["brain_prob_per_ic"],
            }
        )
        print(
            f"  -> n_ic={payload['n_components']} "
            f"brain={out['counts']['brain']} eye={out['counts']['eye blink']} "
            f"muscle={out['counts']['muscle artifact']} dipolar={dipolar}"
        )

        # Per-variant outputs
        (args.output_dir / f"tier_C_iclabel_{variant}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        with (args.output_dir / f"tier_C_iclabel_{variant}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            w = csv.writer(fh)
            w.writerow(["class", "count"])
            for cls in ICLABEL_CLASSES:
                w.writerow([cls, payload["counts"][cls]])
            w.writerow(["dipolar (brain prob >= threshold)", dipolar])

        del raw, ica, cleaned_segments

    # Comparison CSV: rows = variants, cols = class counts + dipolar + n_ic
    cmp_path = args.output_dir / "tier_C_iclabel_comparison.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = (
            ["variant", "n_trials_included", "n_components"]
            + list(ICLABEL_CLASSES)
            + ["dipolar"]
        )
        w.writerow(header)
        for rec in overall:
            row = [rec["variant"], rec["n_trials_included"], rec["n_components"]]
            row.extend(rec["counts"][cls] for cls in ICLABEL_CLASSES)
            row.append(rec["dipolar_count"])
            w.writerow(row)

    (args.output_dir / "tier_C_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "cutoff": args.cutoff,
                "brain_prob_threshold": args.brain_prob_threshold,
                "records": [
                    {
                        k: v
                        for k, v in r.items()
                        if k not in ("labels", "brain_prob_per_ic")
                    }
                    for r in overall
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[klados-iclabel] wrote {cmp_path}")
    print(f"                {args.output_dir / 'tier_C_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
