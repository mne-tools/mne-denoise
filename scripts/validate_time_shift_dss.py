#!/usr/bin/env python
"""Run the checked scientific validation for TimeShiftDSS.

The default run uses the source-scale dimensions from de Cheveigne (2010), 100
whole-trial bootstrap refits, and 1000 full-pipeline null surrogates. Use
``--quick`` only for local smoke testing; quick output is marked accordingly
and is not suitable as PR evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from mne_denoise import __version__  # noqa: E402
from mne_denoise.dss import (  # noqa: E402
    DSS,
    AverageBias,
    TimeShiftDSS,
)
from mne_denoise.dss.variants.tsr import (  # noqa: E402
    _lag_augment,
    _resolve_lags,
)


def _rms(data):
    """Return root-mean-square amplitude."""
    return float(np.sqrt(np.mean(np.asarray(data) ** 2)))


def _snr_db(target, interference):
    """Return RMS target-to-interference ratio in decibels."""
    return float(20 * np.log10(_rms(target) / _rms(interference)))


def _whitening_rank(data, lags, requested_rank, reg=1e-9):
    """Compute the costly numerical-rank diagnostic only for validation."""
    augmented, _, _ = _lag_augment(data, tuple(lags))
    singular_values = np.linalg.svd(
        augmented.reshape(augmented.shape[0], -1), compute_uv=False
    )
    numerical_rank = np.count_nonzero(
        singular_values**2 > reg * singular_values[0] ** 2
    )
    return min(requested_rank, int(numerical_rank))


def _time_shift_dss_surrogate_test(
    estimator,
    data,
    *,
    n_surrogates,
    n_splits,
    alpha=0.05,
    random_state=0,
):
    """Calibrate one fixed TSDSS model for this validation experiment."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] < n_splits:
        raise ValueError("data must provide enough complete epochs for n_splits")
    if estimator.n_select is None:
        raise ValueError("surrogate validation requires an explicit n_select")
    lags, _, _ = _resolve_lags(
        lag_samples=estimator.lag_samples,
        lag_times=estimator.lag_times,
        sfreq=estimator.sfreq,
    )
    min_shift = max(lags) - min(lags) + 1
    shifts = np.arange(min_shift, data.shape[1] - min_shift + 1)
    if shifts.size == 0:
        raise ValueError("Epochs are too short for a lag-destroying surrogate")
    folds = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            np.arange(data.shape[2])
        )
    )
    if any(train.size < 2 or test.size < 2 for train, test in folds):
        raise ValueError("Every fold must contain at least two complete epochs")

    def _statistic(candidate):
        scores = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The augmented feature count approaches"
            )
            for train, test in folds:
                fitted = clone(estimator).fit(candidate[:, :, train])
                scores.append(fitted.score(candidate[:, :, test]))
        return float(np.mean(scores))

    observed = _statistic(data)
    rng = np.random.default_rng(random_state)
    null_scores = np.empty(n_surrogates)
    for index in range(n_surrogates):
        surrogate = np.empty_like(data)
        for epoch in range(data.shape[2]):
            surrogate[:, :, epoch] = np.roll(
                data[:, :, epoch], int(rng.choice(shifts)), axis=1
            )
        null_scores[index] = _statistic(surrogate)
    threshold = float(np.quantile(null_scores, 1.0 - alpha, method="higher"))
    pvalue = float((1 + np.count_nonzero(null_scores >= observed)) / (n_surrogates + 1))
    return {
        "observed_score": observed,
        "null_scores": null_scores,
        "threshold": threshold,
        "pvalue": pvalue,
        "promoted": bool(pvalue <= alpha),
        "alpha": float(alpha),
    }


def _waveform_metrics(output, reference):
    """Quantify sensor-unit target gain, shape, latency, and residual error."""
    output = np.asarray(output).mean(axis=(0, 2))
    reference = np.asarray(reference).mean(axis=(0, 2))
    gain = float(np.dot(output, reference) / np.dot(reference, reference))
    aligned = output if gain >= 0 else -output
    gain = abs(gain)
    correlation = float(abs(np.corrcoef(output, reference)[0, 1]))
    cross = np.correlate(aligned, reference, mode="full")
    latency = int(np.argmax(cross) - (reference.size - 1))
    fitted = gain * reference
    distortion = float(_rms(aligned - fitted) / max(_rms(fitted), np.finfo(float).eps))
    return {
        "target_gain": gain,
        "target_waveform_correlation": correlation,
        "target_latency_error_samples": latency,
        "target_relative_rms_distortion": distortion,
    }


def _simulation_2(seed, *, n_times=1000, n_epochs=100):
    """Generate de Cheveigne (2010) Simulation 2 with isolated contributions."""
    rng = np.random.default_rng(seed)
    n_channels, n_noise = 10, 5
    waveform = np.sin(2 * np.pi * np.arange(n_times) / n_times)
    target = np.broadcast_to(
        waveform[np.newaxis, :, np.newaxis],
        (n_channels, n_times, n_epochs),
    ).copy()
    sources = rng.standard_normal((n_noise, n_times + 1, n_epochs))
    current_mixing = rng.standard_normal((n_channels, n_noise))
    delayed_mixing = rng.standard_normal((n_channels, n_noise))
    noise = np.einsum("jf,ftn->jtn", current_mixing, sources[:, 1:, :], optimize=True)
    noise += np.einsum("jf,ftn->jtn", delayed_mixing, sources[:, :-1, :], optimize=True)
    return target, noise


def _scale_noise(target, noise, input_snr_db):
    """Scale noise to an exact global RMS SNR."""
    return noise * (_rms(target) / _rms(noise)) / 10 ** (input_snr_db / 20)


def _fit_methods(train_data):
    """Fit static and time-shift DSS with source-defined ranks."""
    static = DSS(
        bias=AverageBias(axis="epochs"),
        n_components=1,
        rank=10,
        normalize_input=False,
        component_action="retain",
        n_select=1,
    ).fit(train_data)
    shifted = TimeShiftDSS(
        lag_samples=[0, 1],
        n_components=1,
        # Five noise sources enter the two lag blocks at three time states
        # (15 dimensions), plus one retained target direction. Rank sensitivity
        # around this explicit choice is reported separately.
        rank=16,
        n_select=1,
        component_action="retain",
    ).fit(train_data)
    return {"static_dss": static, "time_shift_dss": shifted}


def _valid_output(estimator, data):
    """Exclude package-preserved edge samples from FIR efficacy scoring."""
    if isinstance(estimator, TimeShiftDSS):
        return data[:, estimator.valid_slice_, :]
    return data


def _method_metrics(name, estimator, target, noise, input_snr_db):
    """Apply one frozen sensor reconstruction to isolated contributions."""
    target_output = _valid_output(estimator, estimator.transform(target))
    noise_output = _valid_output(estimator, estimator.transform(noise))
    target_reference = _valid_output(estimator, target)
    noise_reference = _valid_output(estimator, noise)
    row = {
        "method": name,
        "input_snr_db": float(input_snr_db),
        "output_snr_db": _snr_db(target_output, noise_output),
        "snr_improvement_db": _snr_db(target_output, noise_output) - input_snr_db,
        "noise_attenuation_db": float(
            20 * np.log10(_rms(noise_output) / _rms(noise_reference))
        ),
    }
    row.update(_waveform_metrics(target_output, target_reference))
    return row


def _bootstrap_snr_delta(target, noise, *, n_resamples, seed):
    """Refit both methods on resampled training trials and score fixed test trials."""
    rng = np.random.default_rng(seed)
    train_target, test_target = target[:, :, :80], target[:, :, 80:]
    train_noise, test_noise = noise[:, :, :80], noise[:, :, 80:]
    deltas = np.empty(n_resamples)
    for index in range(n_resamples):
        selection = rng.integers(0, train_target.shape[2], train_target.shape[2])
        train = train_target[:, :, selection] + train_noise[:, :, selection]
        methods = _fit_methods(train)
        scores = {}
        for name, estimator in methods.items():
            scores[name] = _snr_db(
                _valid_output(estimator, estimator.transform(test_target)),
                _valid_output(estimator, estimator.transform(test_noise)),
            )
        deltas[index] = scores["time_shift_dss"] - scores["static_dss"]
    return deltas


def _distortion_fixture():
    """Reproduce the shifted-waveform failure and optional CCA correction."""
    n_times = 200
    waveform = np.zeros(n_times)
    support = np.arange(60, 100)
    waveform[support] = np.sin(2 * np.pi * np.arange(support.size) / support.size)
    data = np.broadcast_to(waveform[None, :, None], (1, n_times, 10)).copy()
    params = {
        "lag_samples": [0, 1, 2, 3, 4],
        "n_components": 5,
        "rank": 5,
    }
    plain = TimeShiftDSS(**params).fit(data).transform(data)[0, :, 0]
    controlled = (
        TimeShiftDSS(**params, distortion_control="cca")
        .fit(data)
        .transform(data)[0, :, 0]
    )
    reference = waveform[4:]
    return {
        "reference": reference,
        "plain": plain,
        "cca": controlled,
        "plain_correlation": float(abs(np.corrcoef(reference, plain)[0, 1])),
        "cca_correlation": float(abs(np.corrcoef(reference, controlled)[0, 1])),
    }


def _rank_lag_sensitivity(target, noise):
    """Report the expected rank cliff and sensitivity to a longer lag grid."""
    train = np.arange(80)
    test = np.arange(80, 100)
    rows = []
    training_data = target[:, :, train] + noise[:, :, train]
    for lags, ranks in (([0, 1], [15, 16, 17]), ([0, 1, 2], [20, 21, 22])):
        for rank in ranks:
            estimator = TimeShiftDSS(
                lag_samples=lags,
                n_components=1,
                rank=rank,
                n_select=1,
                component_action="retain",
            ).fit(training_data)
            target_output = _valid_output(
                estimator, estimator.transform(target[:, :, test])
            )
            noise_output = _valid_output(
                estimator, estimator.transform(noise[:, :, test])
            )
            rows.append(
                {
                    "lag_samples": " ".join(str(value) for value in lags),
                    "requested_rank": rank,
                    "fitted_rank": _whitening_rank(training_data, lags, rank),
                    "held_out_output_snr_db": _snr_db(target_output, noise_output),
                }
            )
    return rows


def _preservation_probe(estimator, *, n_times, n_epochs, seed):
    """Measure how target retention treats an off-target neural oscillation."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2 * np.pi, n_epochs)
    topography = rng.standard_normal(10)
    time = np.arange(n_times) / 1000.0
    waveform = np.sin(2 * np.pi * 10 * time[:, None] + phases[None, :])
    neural = topography[:, None, None] * waveform[None, :, :]
    retained = _valid_output(estimator, estimator.transform(neural))
    neural = _valid_output(estimator, neural)
    return {
        "nontarget_neural_retained_power": float(
            np.mean(retained**2) / np.mean(neural**2)
        ),
        "nontarget_neural_waveform_correlation": float(
            abs(np.corrcoef(retained.ravel(), neural.ravel())[0, 1])
        ),
        "nontarget_neural_relative_rms_change": float(
            _rms(retained - neural) / _rms(neural)
        ),
    }


def _write_csv(path, rows):
    """Write scalar method metrics with stable columns."""
    with path.open("w", newline="", encoding="utf-8") as fid:
        writer = csv.DictWriter(fid, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, rows, distortion, bootstrap_delta):
    """Save source simulation, distortion, and bootstrap evidence."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for method, label in (
        ("static_dss", "Static DSS"),
        ("time_shift_dss", "TimeShiftDSS"),
    ):
        selected = [row for row in rows if row["method"] == method]
        axes[0].plot(
            [row["input_snr_db"] for row in selected],
            [row["output_snr_db"] for row in selected],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="Input SNR (dB)", ylabel="Held-out output SNR (dB)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(distortion["reference"], color="black", label="Undelayed")
    axes[1].plot(distortion["plain"], label="Plain TSDSS")
    axes[1].plot(distortion["cca"], label="CCA step 7")
    axes[1].set(xlabel="Valid sample", ylabel="Arbitrary component units")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].hist(bootstrap_delta, bins=min(20, max(5, bootstrap_delta.size // 5)))
    axes[2].axvline(0.0, color="black", linestyle="--")
    axes[2].set(
        xlabel="Held-out SNR gain over static DSS (dB)",
        ylabel="Bootstrap refits",
    )
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    """Run validation and write machine-readable and visual artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/time_shift_dss"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-surrogates", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_surrogates = min(args.n_surrogates, 20)
        args.n_bootstrap = min(args.n_bootstrap, 20)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target, base_noise = _simulation_2(args.seed)
    train = np.arange(80)
    test = np.arange(80, 100)
    rows = []
    fitted_at_minus_40 = None
    noise_at_minus_40 = None
    for input_snr_db in (-20.0, -40.0, -60.0):
        noise = _scale_noise(target, base_noise, input_snr_db)
        methods = _fit_methods(target[:, :, train] + noise[:, :, train])
        for name, estimator in methods.items():
            rows.append(
                _method_metrics(
                    name,
                    estimator,
                    target[:, :, test],
                    noise[:, :, test],
                    input_snr_db,
                )
            )
        if input_snr_db == -40.0:
            fitted_at_minus_40 = methods["time_shift_dss"]
            noise_at_minus_40 = noise

    bootstrap_delta = _bootstrap_snr_delta(
        target,
        noise_at_minus_40,
        n_resamples=args.n_bootstrap,
        seed=args.seed + 1,
    )
    distortion = _distortion_fixture()
    preservation = _preservation_probe(
        fitted_at_minus_40,
        n_times=target.shape[1],
        n_epochs=test.size,
        seed=args.seed + 2,
    )
    sensitivity = _rank_lag_sensitivity(target, noise_at_minus_40)

    pure_noise = np.random.default_rng(args.seed + 3).standard_normal(target.shape)
    null_estimator = TimeShiftDSS(
        lag_samples=[0, 1],
        n_components=10,
        rank=20,
        n_select=1,
    )
    null = _time_shift_dss_surrogate_test(
        null_estimator,
        pure_noise,
        n_surrogates=args.n_surrogates,
        n_splits=5,
        random_state=args.seed + 4,
    )
    time_shift_rows = [row for row in rows if row["method"] == "time_shift_dss"]

    summary = {
        "status": "quick-smoke" if args.quick else "source-scale",
        "configuration": {
            "base_main_commit": "625f4d5",
            "seed": args.seed,
            "n_surrogates": args.n_surrogates,
            "n_bootstrap": args.n_bootstrap,
            "simulation_2_shape": list(target.shape),
            "train_epochs": 80,
            "test_epochs": 20,
            "explicit_lags": [0, 1],
            "time_shift_rank": 16,
        },
        "environment": {
            "python": platform.python_version(),
            "mne_denoise": __version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "mne": mne.__version__,
        },
        "bootstrap_snr_delta_db": {
            "mean": float(bootstrap_delta.mean()),
            "lower_95": float(np.quantile(bootstrap_delta, 0.025)),
            "upper_95": float(np.quantile(bootstrap_delta, 0.975)),
        },
        "pure_noise_max_null": {
            "observed_score": null["observed_score"],
            "threshold_95": null["threshold"],
            "pvalue": null["pvalue"],
            "promoted": null["promoted"],
        },
        "figure_6_distortion": {
            "plain_correlation": distortion["plain_correlation"],
            "cca_correlation": distortion["cca_correlation"],
        },
        "off_target_neural_probe": preservation,
        "rank_lag_sensitivity": sensitivity,
        "acceptance": {
            "time_shift_beats_static_at_all_snrs": all(
                next(
                    row["output_snr_db"]
                    for row in rows
                    if row["method"] == "time_shift_dss" and row["input_snr_db"] == snr
                )
                > next(
                    row["output_snr_db"]
                    for row in rows
                    if row["method"] == "static_dss" and row["input_snr_db"] == snr
                )
                for snr in (-20.0, -40.0, -60.0)
            ),
            "bootstrap_delta_ci_excludes_zero": bool(
                np.quantile(bootstrap_delta, 0.025) > 0
            ),
            "pure_noise_abstains": not null["promoted"],
            "cca_reduces_distortion": bool(
                distortion["cca_correlation"] > distortion["plain_correlation"]
            ),
            "target_shape_preserved": all(
                row["target_waveform_correlation"] > 0.999
                and row["target_latency_error_samples"] == 0
                and row["target_relative_rms_distortion"] < 0.01
                for row in time_shift_rows
            ),
        },
    }
    _write_csv(args.output_dir / "simulation_2_metrics.csv", rows)
    _write_csv(args.output_dir / "rank_lag_sensitivity.csv", sensitivity)
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savetxt(args.output_dir / "pure_noise_null_scores.csv", null["null_scores"])
    np.savetxt(args.output_dir / "bootstrap_snr_delta_db.csv", bootstrap_delta)
    _plot(
        args.output_dir / "time_shift_dss_validation.png",
        rows,
        distortion,
        bootstrap_delta,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all(summary["acceptance"].values()):
        raise SystemExit("One or more pre-registered validation gates failed")


if __name__ == "__main__":
    main()
