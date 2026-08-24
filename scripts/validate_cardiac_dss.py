"""Validate the explicit CycleAverageBias plus DSS cardiac recipe.

This script is intentionally outside the package unit-test surface. It fits on
synthetic training events, applies one frozen spatial operator to isolated
held-out contributions, runs negative controls and sensitivity analyses, and
writes machine-readable JSON. An optional local FIF pathway provides a
plausibility check against unchanged data and ECG SSP; it is not evidence of
universal or clinical superiority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import compute_proj_ecg, find_ecg_events
from scipy.signal import resample_poly

from mne_denoise import compute_covariance
from mne_denoise.dss import DSS, CycleAverageBias, compute_dss


def _rms(values):
    """Return root-mean-square amplitude as a Python float."""
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _db_ratio(before, after):
    """Return positive attenuation in decibels."""
    tiny = np.finfo(float).tiny
    return float(20 * np.log10(max(_rms(before), tiny) / max(_rms(after), tiny)))


def _locked_average(data, events, window):
    """Average complete half-open windows around data-relative events."""
    start, stop = window
    return np.mean(
        np.stack([data[:, event + start : event + stop] for event in events]), axis=0
    )


def _spectral_power(data, sfreq, frequencies):
    """Return summed sensor power at the nearest FFT bins."""
    spectrum = np.fft.rfft(data, axis=1)
    bins = np.fft.rfftfreq(data.shape[1], 1 / sfreq)
    powers = {}
    for frequency in frequencies:
        index = int(np.argmin(np.abs(bins - frequency)))
        powers[f"{frequency:.4g}_hz"] = float(np.sum(np.abs(spectrum[:, index]) ** 2))
    return powers


def _waveform_metrics(before, after, sfreq):
    """Quantify power, correlation, gain, latency, and RMS distortion."""
    x = np.asarray(before, dtype=float).ravel()
    y = np.asarray(after, dtype=float).ravel()
    denominator = float(x @ x)
    gain = float((x @ y) / denominator) if denominator else 0.0
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
    relative_rms = _rms(y - x) / max(_rms(x), np.finfo(float).tiny)
    max_lag = max(1, round(0.05 * sfreq))
    lag_scores = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = x[-lag:], y[:lag]
        elif lag > 0:
            left, right = x[:-lag], y[lag:]
        else:
            left, right = x, y
        lag_scores.append(float(left @ right))
    latency_samples = int(np.argmax(lag_scores) - max_lag)
    return {
        "retained_power": float((y @ y) / denominator) if denominator else 0.0,
        "waveform_correlation": correlation,
        "gain": gain,
        "latency_samples": latency_samples,
        "latency_ms": float(1_000 * latency_samples / sfreq),
        "relative_rms_distortion": float(relative_rms),
    }


def _make_events(rng, n_times, sfreq):
    """Generate complete, jittered heartbeat coordinates."""
    event = round(0.8 * sfreq)
    events = []
    while event < n_times - round(0.5 * sfreq):
        events.append(event)
        interval = round(rng.normal(0.8, 0.045) * sfreq)
        event += max(round(0.55 * sfreq), interval)
    return np.asarray(events, dtype=int)


def _qrs_train(data, events, rng, sfreq, topography):
    """Add variable-amplitude and slightly variable-width QRS morphology."""
    artifact = np.zeros_like(data)
    for event in events:
        half_width = round(rng.uniform(0.08, 0.11) * sfreq)
        offsets = np.arange(-half_width, half_width + 1)
        width = rng.uniform(0.012, 0.018) * sfreq
        qrs = np.exp(-0.5 * (offsets / width) ** 2)
        qrs -= 0.35 * np.exp(-0.5 * ((offsets - 0.04 * sfreq) / (1.7 * width)) ** 2)
        amplitude = rng.lognormal(mean=0.0, sigma=0.12)
        artifact[:, event - half_width : event + half_width + 1] += (
            amplitude * topography[:, np.newaxis] * qrs
        )
    return artifact


def make_synthetic(seed=97, sfreq=200.0, duration=45.0, n_channels=8):
    """Create isolated train/test cardiac, neural, and noise contributions."""
    rng = np.random.default_rng(seed)
    n_times = round(duration * sfreq)
    heart_frequency = 1.25
    topography = np.array([8, 5.6, -4, 2.8, -2, 1.2, -0.8, 0.4], dtype=float)[
        :n_channels
    ]
    neural_mix = rng.normal(size=(n_channels, 4))

    def one_split(split_seed):
        split_rng = np.random.default_rng(split_seed)
        times = np.arange(n_times) / sfreq
        events = _make_events(split_rng, n_times, sfreq)
        cardiac = _qrs_train(
            np.zeros((n_channels, n_times)), events, split_rng, sfreq, topography
        )
        target_sources = np.vstack(
            [
                np.sin(2 * np.pi * 10 * times + split_rng.uniform(0, 2 * np.pi)),
                0.3
                * np.sin(
                    2 * np.pi * heart_frequency * times
                    + split_rng.uniform(0, 2 * np.pi)
                ),
            ]
        )
        non_target_sources = np.vstack(
            [
                0.6 * np.sin(2 * np.pi * 18 * times + 0.3),
                0.35 * np.sin(2 * np.pi * 3.7 * times + 1.2),
            ]
        )
        target = neural_mix[:, :2] @ target_sources
        non_target = neural_mix[:, 2:] @ non_target_sources
        scale = np.std(np.vstack([target, non_target]))
        target /= scale
        non_target /= scale
        noise = 0.08 * split_rng.standard_normal((n_channels, n_times))
        return {
            "events": events,
            "cardiac": cardiac,
            "target_neural": target,
            "non_target_neural": non_target,
            "noise": noise,
        }

    return {
        "sfreq": sfreq,
        "heart_frequency": heart_frequency,
        "train": one_split(rng.integers(0, 2**32)),
        "held_out": one_split(rng.integers(0, 2**32)),
    }


def _mixture(split):
    """Sum all isolated synthetic contributions."""
    return sum(
        split[key] for key in ("cardiac", "target_neural", "non_target_neural", "noise")
    )


def _fit(events, data, window, rank, n_components, n_select):
    """Fit the explicit public fixed-window cardiac composition."""
    return DSS(
        bias=CycleAverageBias(
            events,
            window,
            window_unit="samples",
            event_origin="data",
        ),
        rank=rank,
        n_components=n_components,
        n_select=n_select,
        component_action="subtract",
        normalize_input=False,
        center=False,
    ).fit(data)


def _evaluate(estimator, held_out, window, sfreq, heart_frequency):
    """Apply a frozen estimator to each isolated held-out contribution."""
    mixture = _mixture(held_out)
    clean = mixture - held_out["cardiac"]
    transformed = {
        key: estimator.transform(held_out[key])
        for key in ("cardiac", "target_neural", "non_target_neural", "noise")
    }
    output = estimator.transform(mixture)
    component_sum = sum(transformed.values())
    before_locked = _locked_average(mixture, held_out["events"], window)
    after_locked = _locked_average(output, held_out["events"], window)
    cardiac_before_locked = _locked_average(
        held_out["cardiac"], held_out["events"], window
    )
    cardiac_after_locked = _locked_average(
        transformed["cardiac"], held_out["events"], window
    )
    harmonic_frequencies = [heart_frequency * multiplier for multiplier in (1, 2, 3)]
    before_harmonics = _spectral_power(held_out["cardiac"], sfreq, harmonic_frequencies)
    after_harmonics = _spectral_power(
        transformed["cardiac"], sfreq, harmonic_frequencies
    )
    harmonic_attenuation = {
        name: float(10 * np.log10(before_harmonics[name] / after_harmonics[name]))
        for name in before_harmonics
    }
    snr_before = 20 * np.log10(_rms(clean) / _rms(held_out["cardiac"]))
    snr_after = 20 * np.log10(_rms(clean) / _rms(output - clean))
    return {
        "cardiac": {
            "locked_rms_attenuation_db": _db_ratio(
                cardiac_before_locked, cardiac_after_locked
            ),
            "mixture_locked_rms_attenuation_db": _db_ratio(before_locked, after_locked),
            "locked_peak_to_peak_attenuation_db": _db_ratio(
                np.ptp(cardiac_before_locked, axis=1),
                np.ptp(cardiac_after_locked, axis=1),
            ),
            "global_rms_attenuation_db": _db_ratio(
                held_out["cardiac"], transformed["cardiac"]
            ),
            "harmonic_attenuation_db": harmonic_attenuation,
        },
        "target_neural": _waveform_metrics(
            held_out["target_neural"], transformed["target_neural"], sfreq
        ),
        "non_target_neural": _waveform_metrics(
            held_out["non_target_neural"], transformed["non_target_neural"], sfreq
        ),
        "background_noise": _waveform_metrics(
            held_out["noise"], transformed["noise"], sfreq
        ),
        "total_output": {
            "relative_rms_error": float(_rms(output - clean) / _rms(clean)),
            "held_out_snr_before_db": float(snr_before),
            "held_out_snr_after_db": float(snr_after),
            "held_out_snr_improvement_db": float(snr_after - snr_before),
            "isolated_sum_max_abs_error": float(np.max(np.abs(output - component_sum))),
        },
    }


def _controlled_events(events, n_times, window, rng):
    """Return shuffled, reversed, and circular event negative controls."""
    low, high = -window[0], n_times - window[1]
    shuffled = np.sort(rng.choice(np.arange(low, high + 1), len(events), replace=False))
    reversed_events = np.sort(n_times - 1 - events)
    shift = round(0.37 * np.median(np.diff(events)))
    span = high - low + 1
    circular = np.sort(low + ((events - low + shift) % span))
    return {
        "shuffled_events": shuffled,
        "time_reversed_events": reversed_events,
        "circularly_shifted_events": circular,
    }


def run_negative_controls(dataset, window, rank, n_components):
    """Run event, pure-noise, neural-only, and wrong-origin controls."""
    train = dataset["train"]
    held_out = dataset["held_out"]
    train_mixture = _mixture(train)
    rng = np.random.default_rng(2025)
    controls = {}
    for name, events in _controlled_events(
        train["events"], train_mixture.shape[1], window, rng
    ).items():
        estimator = _fit(events, train_mixture, window, rank, n_components, 1)
        metrics = _evaluate(
            estimator,
            held_out,
            window,
            dataset["sfreq"],
            dataset["heart_frequency"],
        )
        controls[name] = {
            "true_qrs_locked_attenuation_db": metrics["cardiac"][
                "mixture_locked_rms_attenuation_db"
            ],
            "broad_power_retained": float(
                np.sum(estimator.transform(_mixture(held_out)) ** 2)
                / np.sum(_mixture(held_out) ** 2)
            ),
        }

    neural_train = train["target_neural"] + train["non_target_neural"]
    neural_test = held_out["target_neural"] + held_out["non_target_neural"]
    neural_estimator = _fit(
        train["events"], neural_train, window, rank, n_components, 1
    )
    controls["neural_only"] = _waveform_metrics(
        neural_test, neural_estimator.transform(neural_test), dataset["sfreq"]
    )
    noise_estimator = _fit(
        train["events"], train["noise"], window, rank, n_components, 1
    )
    controls["pure_noise"] = _waveform_metrics(
        held_out["noise"],
        noise_estimator.transform(held_out["noise"]),
        dataset["sfreq"],
    )

    try:
        wrong = CycleAverageBias(
            train["events"] + 10_000,
            window,
            window_unit="samples",
            event_origin="data",
        )
        wrong.apply(train_mixture)
    except ValueError as error:
        controls["wrong_event_origin"] = {
            "rejected": True,
            "reason": str(error),
        }
    else:  # pragma: no cover - a regression should make this visible in JSON
        controls["wrong_event_origin"] = {"rejected": False}
    return controls


def run_sensitivity(dataset, base_window, rank, n_components):
    """Sweep timing, event quality, window, rank, selection, and resampling."""
    train = dataset["train"]
    held_out = dataset["held_out"]
    mixture = _mixture(train)
    rng = np.random.default_rng(404)
    cases = []

    def evaluate_case(name, events, window, case_rank, case_components, n_select):
        estimator = _fit(events, mixture, window, case_rank, case_components, n_select)
        metrics = _evaluate(
            estimator,
            held_out,
            base_window,
            dataset["sfreq"],
            dataset["heart_frequency"],
        )
        cases.append(
            {
                "case": name,
                "event_count": len(events),
                "window": list(window),
                "rank": case_rank,
                "n_select": n_select,
                "cardiac_locked_attenuation_db": metrics["cardiac"][
                    "locked_rms_attenuation_db"
                ],
                "target_retained_power": metrics["target_neural"]["retained_power"],
                "target_correlation": metrics["target_neural"]["waveform_correlation"],
            }
        )

    events = train["events"]
    evaluate_case("baseline", events, base_window, rank, n_components, 1)
    for jitter in (1, 3, 6, 12):
        jittered = np.sort(events + rng.integers(-jitter, jitter + 1, len(events)))
        evaluate_case(
            f"event_jitter_{jitter}_samples",
            jittered,
            base_window,
            rank,
            n_components,
            1,
        )
    evaluate_case(
        "duplicate_25_percent",
        events[::4].tolist() + events.tolist(),
        base_window,
        rank,
        n_components,
        1,
    )
    missed = np.delete(events, np.arange(0, len(events), 4))
    evaluate_case(
        "missed_25_percent_unique", missed, base_window, rank, n_components, 1
    )
    false_events = _controlled_events(events, mixture.shape[1], base_window, rng)[
        "shuffled_events"
    ][: max(2, len(events) // 4)]
    evaluate_case(
        "false_25_percent",
        np.unique(np.concatenate([events, false_events])),
        base_window,
        rank,
        n_components,
        1,
    )
    for seconds in ((-0.1, 0.2), (-0.15, 0.25), (-0.25, 0.4)):
        window = tuple(round(value * dataset["sfreq"]) for value in seconds)
        evaluate_case(
            f"window_{seconds[0]}_{seconds[1]}_s", events, window, rank, n_components, 1
        )
    for case_rank in (max(2, rank // 2), rank):
        for n_select in (1, 2):
            evaluate_case(
                f"rank_{case_rank}_select_{n_select}",
                events,
                base_window,
                case_rank,
                min(n_components, case_rank),
                n_select,
            )

    # Resampling is evaluated separately because every isolated contribution
    # and coordinate must use the same new sampling grid.
    new_sfreq = dataset["sfreq"] / 2
    resampled = {"sfreq": new_sfreq, "heart_frequency": dataset["heart_frequency"]}
    for split_name in ("train", "held_out"):
        source = dataset[split_name]
        split = {
            key: resample_poly(source[key], 1, 2, axis=1)
            for key in ("cardiac", "target_neural", "non_target_neural", "noise")
        }
        split["events"] = np.unique(np.rint(source["events"] / 2).astype(int))
        resampled[split_name] = split
    resampled_window = tuple(round(value / 2) for value in base_window)
    resampled_estimator = _fit(
        resampled["train"]["events"],
        _mixture(resampled["train"]),
        resampled_window,
        rank,
        n_components,
        1,
    )
    metrics = _evaluate(
        resampled_estimator,
        resampled["held_out"],
        resampled_window,
        new_sfreq,
        resampled["heart_frequency"],
    )
    cases.append(
        {
            "case": "resampled_100_hz",
            "event_count": len(resampled["train"]["events"]),
            "window": list(resampled_window),
            "rank": rank,
            "n_select": 1,
            "cardiac_locked_attenuation_db": metrics["cardiac"][
                "locked_rms_attenuation_db"
            ],
            "target_retained_power": metrics["target_neural"]["retained_power"],
            "target_correlation": metrics["target_neural"]["waveform_correlation"],
        }
    )
    return cases


def _source_parity(estimator, train_data):
    """Compare the fitted composition with its public numerical constituents."""
    biased = estimator.bias.apply(train_data)
    filters, patterns, eigenvalues = compute_dss(
        compute_covariance(train_data, assume_centered=True),
        compute_covariance(biased, assume_centered=True),
        n_components=estimator.n_components,
        rank=estimator.rank,
        reg=estimator.reg,
    )
    expected = train_data - patterns[:, :1] @ filters[:1] @ train_data
    return {
        "filters_max_abs_error": float(np.max(np.abs(estimator.filters_ - filters))),
        "patterns_max_abs_error": float(np.max(np.abs(estimator.patterns_ - patterns))),
        "eigenvalues_max_abs_error": float(
            np.max(np.abs(estimator.eigenvalues_ - eigenvalues))
        ),
        "output_max_abs_error": float(
            np.max(np.abs(estimator.transform(train_data) - expected))
        ),
    }


def run_real_data(raw_fif, ecg_channel, duration):
    """Run an optional local-recording plausibility check and ECG SSP comparator."""
    raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
    if duration is not None:
        raw.crop(tmax=min(duration, raw.times[-1]))
    events, _, _ = find_ecg_events(raw, ch_name=ecg_channel, verbose=False)
    midpoint = raw.times[-1] / 2
    train_all = raw.copy().crop(tmax=midpoint, include_tmax=False)
    held_all = raw.copy().crop(tmin=midpoint)
    train_events = events[
        (events[:, 0] >= train_all.first_samp) & (events[:, 0] <= train_all.last_samp)
    ]
    held_events = events[
        (events[:, 0] >= held_all.first_samp) & (events[:, 0] <= held_all.last_samp)
    ]
    if min(len(train_events), len(held_events)) < 2:
        raise ValueError(
            "real-data split requires at least two ECG events in each half"
        )

    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    sensor_type = "eeg"
    if len(picks) < 2:
        picks = mne.pick_types(raw.info, meg="grad", eeg=False, exclude="bads")
        sensor_type = "grad"
    if len(picks) < 2:
        picks = mne.pick_types(raw.info, meg="mag", eeg=False, exclude="bads")
        sensor_type = "mag"
    if len(picks) < 2:
        raise ValueError(
            "real-data check requires at least two good EEG or MEG channels"
        )
    train = train_all.copy().pick(picks)
    held = held_all.copy().pick(picks)
    sfreq = float(raw.info["sfreq"])
    window = (-round(0.2 * sfreq), round(0.4 * sfreq))
    rank = min(len(train.ch_names), 20)
    estimator = DSS(
        bias=CycleAverageBias(
            train_events[:, 0],
            window,
            event_origin="raw",
            first_samp=train.first_samp,
        ),
        rank={sensor_type: rank},
        n_components=rank,
        n_select=1,
        component_action="subtract",
        normalize_input=False,
    ).fit(train)
    cleaned = estimator.transform(held)
    held_relative_events = held_events[:, 0] - held.first_samp
    unchanged_locked = _locked_average(held.get_data(), held_relative_events, window)
    cleaned_locked = _locked_average(cleaned.get_data(), held_relative_events, window)
    result = {
        "dataset": str(Path(raw_fif).resolve()),
        "preprocessing": "preloaded; first half fit; second half held out; good homogeneous sensors; no filtering added by this script",
        "sensor_type": sensor_type,
        "n_channels": len(train.ch_names),
        "train_event_count": len(train_events),
        "held_out_event_count": len(held_events),
        "unchanged_locked_rms": _rms(unchanged_locked),
        "cardiac_dss_locked_rms": _rms(cleaned_locked),
        "cardiac_dss_attenuation_db": _db_ratio(unchanged_locked, cleaned_locked),
        "cardiac_dss_broad_power_retained": float(
            np.sum(cleaned.get_data() ** 2) / np.sum(held.get_data() ** 2)
        ),
    }
    try:
        projections, _ = compute_proj_ecg(
            train_all,
            ch_name=ecg_channel,
            n_grad=1,
            n_mag=1,
            n_eeg=1,
            verbose=False,
        )
        ssp = held_all.copy().add_proj(projections).apply_proj().pick(picks)
        ssp_locked = _locked_average(ssp.get_data(), held_relative_events, window)
        result["ecg_ssp"] = {
            "locked_rms": _rms(ssp_locked),
            "attenuation_db": _db_ratio(unchanged_locked, ssp_locked),
            "broad_power_retained": float(
                np.sum(ssp.get_data() ** 2) / np.sum(held.get_data() ** 2)
            ),
        }
    except Exception as error:  # comparator availability varies by recording
        result["ecg_ssp"] = {"available": False, "reason": str(error)}
    return result


def main():
    """Run validation and write deterministic JSON metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cardiac_dss_validation.json"),
        help="JSON metrics path.",
    )
    parser.add_argument("--real-raw-fif", type=Path, default=None)
    parser.add_argument("--ecg-channel", default=None)
    parser.add_argument("--real-duration", type=float, default=120.0)
    args = parser.parse_args()

    dataset = make_synthetic()
    sfreq = dataset["sfreq"]
    window = (-round(0.15 * sfreq), round(0.25 * sfreq))
    rank = _mixture(dataset["train"]).shape[0]
    n_components = min(6, rank)
    estimator = _fit(
        dataset["train"]["events"],
        _mixture(dataset["train"]),
        window,
        rank,
        n_components,
        1,
    )
    positive = _evaluate(
        estimator,
        dataset["held_out"],
        window,
        sfreq,
        dataset["heart_frequency"],
    )
    negative = run_negative_controls(dataset, window, rank, n_components)
    sensitivity = run_sensitivity(dataset, window, rank, n_components)
    parity = _source_parity(estimator, _mixture(dataset["train"]))

    gates = {
        "cardiac_locked_attenuation_at_least_6_db": positive["cardiac"][
            "locked_rms_attenuation_db"
        ]
        >= 6,
        "target_retained_power_between_0_8_and_1_2": 0.8
        <= positive["target_neural"]["retained_power"]
        <= 1.2,
        "target_waveform_correlation_at_least_0_95": positive["target_neural"][
            "waveform_correlation"
        ]
        >= 0.95,
        "non_target_retained_power_at_least_0_85": positive["non_target_neural"][
            "retained_power"
        ]
        >= 0.85,
        "held_out_snr_improvement_at_least_3_db": positive["total_output"][
            "held_out_snr_improvement_db"
        ]
        >= 3,
        "pure_noise_retained_power_at_least_0_8": negative["pure_noise"][
            "retained_power"
        ]
        >= 0.8,
        "neural_only_retained_power_at_least_0_8": negative["neural_only"][
            "retained_power"
        ]
        >= 0.8,
        "wrong_origin_rejected": negative["wrong_event_origin"]["rejected"],
        "public_operation_parity_below_1e_12": max(parity.values()) < 1e-12,
    }
    positive_locked = positive["cardiac"]["mixture_locked_rms_attenuation_db"]
    for control_name in (
        "shuffled_events",
        "time_reversed_events",
        "circularly_shifted_events",
    ):
        gates[f"{control_name}_at_least_6_db_below_positive"] = (
            negative[control_name]["true_qrs_locked_attenuation_db"]
            <= positive_locked - 6
        )
    gates["all_passed"] = all(gates.values())
    result = {
        "schema_version": 1,
        "method": "CycleAverageBias fixed-window composition with DSS subtraction",
        "seed": 97,
        "synthetic_design": {
            "sfreq": sfreq,
            "duration_per_split_seconds": 45.0,
            "train_event_count": len(dataset["train"]["events"]),
            "held_out_event_count": len(dataset["held_out"]["events"]),
            "window_samples": list(window),
            "rank": rank,
            "n_components": n_components,
            "n_select": 1,
        },
        "source_parity": parity,
        "positive_control": positive,
        "negative_controls": negative,
        "sensitivity": sensitivity,
        "predeclared_gates": gates,
    }
    if args.real_raw_fif is not None:
        if args.ecg_channel is None:
            parser.error("--ecg-channel is required with --real-raw-fif")
        result["real_data"] = run_real_data(
            args.real_raw_fif, args.ecg_channel, args.real_duration
        )
    else:
        result["real_data"] = {
            "status": "not_run",
            "reason": "Pass --real-raw-fif and --ecg-channel for an optional local-recording plausibility check.",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gates, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    if not gates["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
