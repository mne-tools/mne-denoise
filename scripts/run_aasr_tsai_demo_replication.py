"""Reproduce the public Tsai et al. AASR demonstration protocol.

This runner requires the authors' ``data_extract_cell.mat`` file referenced by
the public AASR repository. It preserves the demonstration's FFT filter,
2--26 second crop, 240-second alternating sequence, inclusive 20-second update
windows, final-state MW reconstruction, and channel-wise RMSE/SNR definitions.

The linked prepared-data file is no longer publicly downloadable. A run made
from a separately obtained copy must record its SHA-256 hash; it must not be
substituted silently with the raw Klados archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import AdaptiveASR  # noqa: E402
from scripts.asr_paper_protocols import (  # noqa: E402
    build_tsai_demo_sequence,
    paper_rmse_and_snr,
    tsai_demo_update_slices,
    tsai_fft_bandpass,
)

VARIANTS = ("offline", "init", "mw", "psp", "psw")
CONTAMINATIONS = ("eog", "emg")
CHANNEL_NAMES = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
)
SCORING_CHANNELS = {
    "eog": ("Fp1", "Fp2", "F3", "F4"),
    "emg": ("T3", "T4", "T5", "T6"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_items(value: np.ndarray) -> list[np.ndarray]:
    value = np.asarray(value, dtype=object)
    items = []
    for item in value.ravel(order="F"):
        array = np.asarray(item, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Each prepared-data cell must contain a 2D array")
        items.append(array)
    return items


def _load_prepared_data(path: Path) -> dict[str, list[np.ndarray]]:
    payload = loadmat(
        path,
        variable_names=["pure_data_cell", "con_eog_cell", "con_emg_cell"],
        squeeze_me=True,
    )
    required = {"pure_data_cell", "con_eog_cell", "con_emg_cell"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Prepared-data file is missing variables: {missing}")
    out = {name: _cell_items(payload[name]) for name in sorted(required)}
    n_subjects = len(out["pure_data_cell"])
    if any(len(items) != n_subjects for items in out.values()):
        raise ValueError("Prepared clean, EOG, and EMG cell arrays differ in length")
    return out


def _run_variant(
    data: np.ndarray,
    *,
    sfreq: float,
    cutoff: float,
    variant: str,
    update_window_s: float,
) -> tuple[np.ndarray, AdaptiveASR, float]:
    update_samples = int(round(sfreq * update_window_s))
    slices = tsai_demo_update_slices(data.shape[1], sfreq, update_window_s)
    if not slices:
        raise ValueError("The prepared stream is shorter than two update intervals")

    t0 = time.perf_counter()
    if variant == "offline":
        model = AdaptiveASR(
            sfreq=sfreq, cutoff=cutoff, variant="psw", verbose=False
        ).fit(data)
    elif variant == "init":
        model = AdaptiveASR(
            sfreq=sfreq, cutoff=cutoff, variant="psw", verbose=False
        ).fit(data[:, :update_samples])
    elif variant == "mw":
        # Repeated ``subspace`` calls in the public notebook replace the state
        # without adaptive carry-over. Only the final call determines the
        # reconstruction state, so fitting its final inclusive slice is exact.
        model = AdaptiveASR(
            sfreq=sfreq, cutoff=cutoff, variant="psw", verbose=False
        ).fit(data[:, slices[-1]])
    elif variant in ("psp", "psw"):
        model = AdaptiveASR(
            sfreq=sfreq, cutoff=cutoff, variant=variant, verbose=False
        ).fit(data[:, slices[0]])
        for update_slice in slices[1:]:
            model.partial_fit(data[:, update_slice])
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model.reset_process_state()
    cleaned = model.transform(data)
    return cleaned, model, time.perf_counter() - t0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "paper_replications" / "tsai_aasr_demo",
    )
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument("--update-window-s", type=float, default=20.0)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--contaminations",
        nargs="+",
        choices=CONTAMINATIONS,
        default=list(CONTAMINATIONS),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prepared_path = args.prepared_data.resolve()
    if not prepared_path.exists():
        raise SystemExit(f"Prepared AASR data not found: {prepared_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = _load_prepared_data(prepared_path)
    clean_cells = prepared["pure_data_cell"]
    rows: list[dict[str, Any]] = []
    for subject_index, clean_raw in enumerate(clean_cells, start=1):
        clean_filtered = tsai_fft_bandpass(clean_raw, args.sfreq)
        for contamination in args.contaminations:
            contaminated_raw = prepared[f"con_{contamination}_cell"][subject_index - 1]
            contaminated_filtered = tsai_fft_bandpass(contaminated_raw, args.sfreq)
            sequence = build_tsai_demo_sequence(
                clean_filtered,
                contaminated_filtered,
                sfreq=args.sfreq,
            )
            score_indices = [
                CHANNEL_NAMES.index(name) for name in SCORING_CHANNELS[contamination]
            ]
            for variant in args.variants:
                cleaned, model, wall_time = _run_variant(
                    sequence.contaminated,
                    sfreq=args.sfreq,
                    cutoff=args.cutoff,
                    variant=variant,
                    update_window_s=args.update_window_s,
                )
                rmse, snr = paper_rmse_and_snr(sequence.clean, cleaned)
                rows.append(
                    {
                        "subject_index": subject_index,
                        "contamination": contamination,
                        "variant": variant,
                        "status": "passed",
                        "mean_target_rmse": float(np.mean(rmse[score_indices])),
                        "mean_target_snr_db": float(np.mean(snr[score_indices])),
                        "mean_all_channel_rmse": float(np.mean(rmse)),
                        "mean_all_channel_snr_db": float(np.mean(snr)),
                        "wall_time_s": wall_time,
                        "rank": int(model.rank_),
                        "n_samples": int(cleaned.shape[1]),
                    }
                )

    csv_path = args.output_dir / "per_subject_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "tsai_aasr_public_demo_v1",
        "prepared_data_path": str(prepared_path),
        "prepared_data_sha256": _sha256(prepared_path),
        "n_subjects": len(clean_cells),
        "sfreq": args.sfreq,
        "cutoff": args.cutoff,
        "update_window_s": args.update_window_s,
        "variants": list(args.variants),
        "contaminations": list(args.contaminations),
        "scoring_channels": SCORING_CHANNELS,
        "n_rows": len(rows),
        "output": str(csv_path),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
