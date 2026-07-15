#!/usr/bin/env python
"""Run frozen factorial BSS recovery on known sources and mixing matrices."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import pathlib
import sys
from time import perf_counter

import numpy as np
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from mne_denoise.benchmarks import simulation as sim
from mne_denoise.benchmarks.config import assert_submission_ready
from mne_denoise.benchmarks.intended import locked_seed
from mne_denoise.benchmarks.provenance import AttemptRecorder, build_run_record
from mne_denoise.qa import ground_truth as gt


def _rjd(matrices, eps=1e-12, maxiter=200):
    """Real Jacobi joint approximate diagonalization."""
    matrices = np.asarray(matrices, dtype=float).copy()
    _, count, _ = matrices.shape
    rotation = np.eye(count)
    for _ in range(maxiter):
        changed = False
        for left in range(count - 1):
            for right in range(left + 1, count):
                h = np.vstack(
                    [
                        matrices[:, left, left] - matrices[:, right, right],
                        matrices[:, left, right] + matrices[:, right, left],
                    ]
                )
                hh = h @ h.T
                diagonal = hh[0, 0] - hh[1, 1]
                off_diagonal = hh[0, 1] + hh[1, 0]
                angle = 0.5 * np.arctan2(
                    off_diagonal,
                    diagonal + np.sqrt(diagonal**2 + off_diagonal**2),
                )
                cosine, sine = np.cos(angle), np.sin(angle)
                if abs(sine) <= eps:
                    continue
                changed = True
                row_l = matrices[:, left, :].copy()
                row_r = matrices[:, right, :].copy()
                matrices[:, left, :] = cosine * row_l + sine * row_r
                matrices[:, right, :] = -sine * row_l + cosine * row_r
                col_l = matrices[:, :, left].copy()
                col_r = matrices[:, :, right].copy()
                matrices[:, :, left] = cosine * col_l + sine * col_r
                matrices[:, :, right] = -sine * col_l + cosine * col_r
                vector_l = rotation[:, left].copy()
                vector_r = rotation[:, right].copy()
                rotation[:, left] = cosine * vector_l + sine * vector_r
                rotation[:, right] = -sine * vector_l + cosine * vector_r
        if not changed:
            break
    return rotation


def _whiten(data, n_components):
    centered = data - data.mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(centered @ centered.T / centered.shape[1])
    order = np.argsort(values)[::-1][:n_components]
    whitener = (
        vectors[:, order] / np.sqrt(np.maximum(values[order], 1e-18))
    ).T
    return whitener @ centered, whitener


def _sobi(data, n_components, n_lags=20):
    whitened, whitener = _whiten(data, n_components)
    covariances = []
    for lag in range(1, n_lags + 1):
        covariance = (
            whitened[:, lag:] @ whitened[:, :-lag].T
        ) / (whitened.shape[1] - lag)
        covariances.append((covariance + covariance.T) / 2.0)
    return _rjd(np.asarray(covariances)).T @ whitener


def _jade(data, n_components):
    whitened, whitener = _whiten(data, n_components)
    count, samples = whitened.shape
    cumulants = []
    for left in range(count):
        for right in range(left, count):
            matrix = (
                whitened * (whitened[left] * whitened[right])
            ) @ whitened.T / samples
            matrix -= np.eye(count) if left == right else 0.0
            matrix[left, right] -= 1.0
            matrix[right, left] -= 1.0
            scale = 1.0 if left == right else np.sqrt(2.0)
            cumulants.append((matrix + matrix.T) * scale / 2.0)
    return _rjd(np.asarray(cumulants)).T @ whitener


def _iterative_denoiser(name):
    from mne_denoise.dss.denoisers import (
        DCTDenoiser,
        GaussDenoiser,
        KurtosisDenoiser,
        QuasiPeriodicDenoiser,
        RobustTanhDenoiser,
        SkewDenoiser,
        SmoothTanhDenoiser,
        SpectrogramDenoiser,
        TanhMaskDenoiser,
        VarianceMaskDenoiser,
        WienerMaskDenoiser,
    )

    factories = {
        "tanh": TanhMaskDenoiser,
        "robust_tanh": RobustTanhDenoiser,
        "gauss": GaussDenoiser,
        "skew": SkewDenoiser,
        "smooth_tanh": SmoothTanhDenoiser,
        "kurtosis": KurtosisDenoiser,
        "wiener_mask": WienerMaskDenoiser,
        "variance_mask": VarianceMaskDenoiser,
        "dct": DCTDenoiser,
        "quasi_periodic": QuasiPeriodicDenoiser,
        "spectrogram": SpectrogramDenoiser,
    }
    if name not in factories:
        raise KeyError(name)
    return factories[name]()


def _fit_unmix_with_metadata(method, data, n_components, seed, *, max_iter):
    """Return transform, sensor-space unmixing, and convergence metadata."""
    train_mean = data.mean(axis=1, keepdims=True)
    if method in {"sobi", "jade"}:
        unmixing = (_sobi if method == "sobi" else _jade)(data, n_components)
        return lambda value: unmixing @ (value - train_mean), unmixing, {}
    if method == "amica":
        from amica import Amica, AmicaConfig

        model = Amica(
            AmicaConfig(pcakeep=int(n_components), max_iter=2000, do_newton=True),
            random_state=seed,
        )
        result = model.fit(np.asarray(data, dtype=float))
        unmixing = np.asarray(result.unmixing_matrix_sensor_)
        return lambda value: np.asarray(model.transform(value)), unmixing, {}
    if method == "pca":
        from sklearn.decomposition import PCA

        model = PCA(n_components=n_components, whiten=True, svd_solver="full")
        model.fit(data.T)
        return lambda value: model.transform(value.T).T, model.components_, {}
    if method == "fastica":
        from sklearn.decomposition import FastICA

        model = FastICA(
            n_components=n_components,
            whiten="unit-variance",
            max_iter=1000,
            random_state=seed,
        )
        model.fit(data.T)
        meta = {"converged": bool(model.n_iter_ < model.max_iter), "iterations": int(model.n_iter_)}
        return lambda value: model.transform(value.T).T, model.components_, meta
    if method == "infomax":
        import mne

        unmixing = mne.preprocessing.infomax(
            (data - train_mean).T,
            extended=True,
            random_state=seed,
            max_iter=1000,
        )[:n_components]
        return lambda value: unmixing @ (value - train_mean), unmixing, {}
    if method == "picard":
        from picard import picard

        whitening, unmixing_white, _ = picard(
            data - train_mean,
            n_components=n_components,
            ortho=False,
            random_state=seed,
            max_iter=500,
        )
        unmixing = unmixing_white @ whitening
        return lambda value: unmixing @ (value - train_mean), unmixing, {}
    if method.startswith("iterative_dss_"):
        from mne_denoise.dss import iterative_dss
        from mne_denoise.dss.denoisers import beta_tanh

        contrast = method.removeprefix("iterative_dss_")
        denoiser = _iterative_denoiser(contrast)
        unmixing, _, _, convergence = iterative_dss(
            data,
            denoiser,
            n_components,
            method="symmetric",
            beta=beta_tanh if contrast == "tanh" else None,
            max_iter=max_iter,
            random_state=seed,
        )
        convergence = np.asarray(convergence)
        meta = {
            "converged_fraction": float(np.mean(convergence[:, 1]))
            if convergence.ndim == 2
            else None,
            "iterations_mean": float(np.mean(convergence[:, 0]))
            if convergence.ndim == 2
            else None,
        }
        return lambda value: unmixing @ (value - train_mean), unmixing, meta
    raise KeyError(method)


def _fit_unmix(method, data, n_components, seed=0, *, max_iter=None):
    """Fit a BSS transform, preserving the historical two-value helper API.

    Benchmark execution passes ``max_iter`` and receives convergence metadata;
    older downstream code that calls the helper with three positional arguments
    continues to receive ``(transform, unmixing)``.
    """
    result = _fit_unmix_with_metadata(
        method,
        data,
        n_components,
        seed,
        max_iter=1000 if max_iter is None else max_iter,
    )
    return result[:2] if max_iter is None else result


def _align(recovered, matching, n_reference):
    applied = sim.apply_source_matching(recovered, matching)
    aligned = np.zeros((n_reference, applied.shape[1]))
    for recovered_index, reference_index in enumerate(matching.perm):
        if 0 <= reference_index < n_reference:
            aligned[reference_index] = applied[recovered_index]
    return aligned


def _method_list(cfg):
    contrasts = cfg["methods_under_test"]["iterative_dss"]["implemented_contrasts"]
    methods = [f"iterative_dss_{contrast}" for contrast in contrasts]
    methods.extend(cfg["comparators"]["required"])
    return list(dict.fromkeys(methods))


def _score_method(method, replicate, seed, *, max_iter):
    brain = np.asarray(replicate.brain_idx)
    truth_train = replicate.S_train[brain]
    truth_test = replicate.S_test[brain]
    if method == "oracle":
        estimate = truth_test.copy()
        unmixing = None
        matching = None
        meta = {"converged": True, "iterations": 0}
    else:
        n_components = min(replicate.A.shape[1], replicate.X_train.shape[0])
        transform, unmixing, meta = _fit_unmix(
            method,
            replicate.X_train,
            n_components,
            seed,
            max_iter=max_iter,
        )
        matching = sim.fit_source_matching(
            transform(replicate.X_train), truth_train, min_corr=0.5
        )
        estimate = _align(transform(replicate.X_test), matching, len(brain))

    sensor_truth = replicate.A[:, brain] @ truth_test
    sensor_estimate = replicate.A[:, brain] @ estimate
    bss_metrics = [
        gt.bss_eval(estimate[index], truth_test, index)
        for index in range(len(brain))
    ]
    metrics = {
        "method": method,
        "status": "success",
        "neural_source_rrmse": float(gt.rrmse(estimate, truth_test)),
        "permutation_aligned_clean_source_correlation": float(
            gt.correlation_with_clean(estimate, truth_test)
        ),
        "sensor_source_reconstruction_error": float(
            gt.rrmse(sensor_estimate, sensor_truth)
        ),
        "known_neural_signal_preservation": float(
            gt.correlation_with_clean(sensor_estimate, sensor_truth)
        ),
        "sir_db": float(np.mean([value["sir"] for value in bss_metrics])),
        "sdr_db": float(np.mean([value["sdr"] for value in bss_metrics])),
        "sar_db": float(np.mean([value["sar"] for value in bss_metrics])),
        "condition_number": float(np.linalg.cond(replicate.A)),
        **meta,
    }
    if matching is not None:
        metrics.update(
            {
                "unmatched_sources": len(matching.unmatched),
                "collapsed_sources": len(matching.collapsed),
                "mean_train_match_correlation": float(
                    np.mean(matching.matched_corr)
                ),
            }
        )
    if unmixing is not None:
        with contextlib.suppress(ValueError):
            metrics["mixing_recovery_error"] = float(
                gt.mixing_recovery_error(np.linalg.pinv(unmixing), replicate.A)
            )
        if replicate.is_square:
            metrics["amari"] = float(gt.amari_index(unmixing, replicate.A))
    return metrics


def _write_tsv(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args):
    config_path = pathlib.Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert_submission_ready(cfg, source=str(config_path))
    specification = cfg["simulation"]
    regime = "forward" if "forward" in str(specification["regime"]) else "generic"
    source_regimes = list(specification["source_regimes"])
    snrs = list(specification["snr_levels_db"])
    channels = list(specification["channel_counts"])
    n_sources = int(specification["source_counts"][0])
    repetitions = int(specification["n_replicates"])
    train_samples = int(specification["n_train_samples"])
    test_samples = int(specification["n_test_samples"])
    source_sets = int(specification["crossed_design"]["source_sets"])
    mixing_matrices = int(specification["crossed_design"]["mixing_matrices"])
    if source_sets * mixing_matrices != repetitions:
        raise ValueError("crossed_design cells must equal n_replicates")
    methods = _method_list(cfg)
    max_iter = int(
        cfg["methods_under_test"]["iterative_dss"]["fit"]["max_iter"]
    )
    if args.smoke:
        source_regimes = source_regimes[:1]
        snrs = [0]
        channels = [max(8, n_sources)]
        repetitions = source_sets = mixing_matrices = 1
        train_samples = test_samples = 1200
        max_iter = min(max_iter, 20)
    if args.methods:
        requested = args.methods.split(",")
        missing = sorted(set(requested) - set(methods))
        if missing:
            raise ValueError(f"unknown requested methods: {missing}")
        methods = requested

    condition_range = specification.get("mixing", {}).get(
        "condition_number_range", [1.0, 10.0]
    )
    global_seed = int(specification["seeds"]["global"])
    root = pathlib.Path(args.output_root).resolve()
    rows = []
    for source_regime in source_regimes:
        for snr in snrs:
            for n_channels in channels:
                for replicate_index in range(repetitions):
                    source_set = replicate_index % source_sets
                    mixing_matrix = (replicate_index // source_sets) % mixing_matrices
                    source_seed = locked_seed(
                        global_seed, cfg["arm"], source_regime, "source", source_set
                    )
                    mixing_seed = locked_seed(
                        global_seed, cfg["arm"], n_channels, "mixing", mixing_matrix
                    )
                    seed = locked_seed(source_seed, mixing_seed, snr)
                    replicate = sim.simulate_replicate(
                        regime=regime,
                        source_regime=source_regime,
                        n_channels=int(n_channels),
                        n_brain=n_sources // 2,
                        n_artifact=n_sources - n_sources // 2,
                        n_train=train_samples,
                        n_test=test_samples,
                        snr_db=float(snr),
                        seed=seed,
                        source_seed=source_seed,
                        mixing_seed=mixing_seed,
                        min_cond=float(condition_range[0]),
                        max_cond=float(condition_range[1]),
                    )
                    unit_id = (
                        f"{source_regime}_snr{snr}_ch{n_channels}_"
                        f"source{source_set:02d}_mix{mixing_matrix:02d}"
                    )
                    for method in methods:
                        method_dir = root / unit_id / method
                        record = build_run_record(
                            arm=cfg["arm"],
                            method=method,
                            unit_id=unit_id,
                            config_path=config_path,
                            dataset_manifest=args.dataset_manifest,
                            repo_root=_REPO,
                            seed=seed,
                            information_tier="oracle" if method == "oracle" else "blind",
                            allow_dirty=args.allow_dirty,
                        )
                        try:
                            with AttemptRecorder(method_dir, record) as active:
                                started = perf_counter()
                                try:
                                    metrics = _score_method(
                                        method,
                                        replicate,
                                        seed,
                                        max_iter=max_iter,
                                    )
                                except (ImportError, ModuleNotFoundError) as error:
                                    active.status = "unavailable_dependency"
                                    metrics = {
                                        "method": method,
                                        "status": "unavailable_dependency",
                                        "error": f"{type(error).__name__}: {error}",
                                    }
                                metrics.update(
                                    {
                                        "unit_id": unit_id,
                                        "source_regime": source_regime,
                                        "mixing_regime": regime,
                                        "snr_db": snr,
                                        "n_channels": n_channels,
                                        "source_set": source_set,
                                        "mixing_matrix": mixing_matrix,
                                        "seed": seed,
                                        "runtime_seconds_method": perf_counter() - started,
                                    }
                                )
                                (method_dir / "metrics.json").write_text(
                                    json.dumps(metrics, indent=2, sort_keys=True),
                                    encoding="utf-8",
                                )
                                (method_dir / "model.json").write_text(
                                    json.dumps(
                                        {
                                            "method": method,
                                            "train_only": True,
                                            "source_matching": cfg["source_matching"],
                                        },
                                        indent=2,
                                        sort_keys=True,
                                    ),
                                    encoding="utf-8",
                                )
                                rows.append(metrics)
                        except Exception as error:
                            rows.append(
                                {
                                    "method": method,
                                    "status": "failed",
                                    "unit_id": unit_id,
                                    "source_regime": source_regime,
                                    "seed": seed,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
    _write_tsv(root / "raw_metrics.tsv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset-manifest", default=str(_REPO / "configs/manifests/synthetic_v1.json")
    )
    parser.add_argument("--output-root", default=str(_REPO / "results/ground_truth"))
    parser.add_argument("--methods", help="comma-separated subset for preflight")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    rows = run(args)
    acceptable = {"success", "unavailable_dependency"}
    accepted = sum(row.get("status") in acceptable for row in rows)
    print(f"attempts={len(rows)} accepted={accepted} output={args.output_root}")
    return 0 if accepted == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
