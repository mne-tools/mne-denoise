"""Unit tests for whitening utilities."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from mne_denoise._spatial import apply_spatial_transform, fit_mixing_matrix
from mne_denoise.dss._whitening import (
    apply_covariance_transform,
    compute_data_covariance_whitener,
    compute_mne_sensor_whitener,
    map_spatial_matrices_to_sensor_space,
    whiten_from_data_covariance,
)


def test_apply_spatial_transform_contract():
    """Spatial transforms preserve layout, chunking, and reject bad matrices."""
    rng = np.random.default_rng(0)
    matrix = np.diag([2.0, 3.0, 4.0])
    for shape in ((3, 20), (3, 10, 2)):
        data = rng.standard_normal(shape)
        transformed = apply_spatial_transform(matrix, data)
        for index, scale in enumerate((2.0, 3.0, 4.0)):
            np.testing.assert_allclose(transformed[index], scale * data[index])

    data = rng.standard_normal((3, 7, 31))
    rectangular = rng.standard_normal((4, 3))
    np.testing.assert_allclose(
        apply_spatial_transform(rectangular, data),
        apply_spatial_transform(rectangular, data, chunk_size=19),
    )
    invalid = [
        (np.ones(3), np.ones((3, 10)), "matrix must be 2D"),
        (np.eye(2), np.ones((3, 10)), "channel dimensions do not match"),
    ]
    for bad_matrix, bad_data, message in invalid:
        with pytest.raises(ValueError, match=message):
            apply_spatial_transform(bad_matrix, bad_data)
    for chunk_size in (0, -1, 2.5, True):
        with pytest.raises((TypeError, ValueError), match="chunk_size"):
            apply_spatial_transform(np.eye(3), np.ones((3, 10)), chunk_size=chunk_size)


def test_fit_mixing_matrix_weighted_epoched_data():
    """Shared weighted regression recovers a sensor projection."""
    rng = np.random.default_rng(42)
    sources = rng.standard_normal((2, 20, 4))
    expected = np.array([[1.0, 2.0], [-0.5, 0.25], [3.0, -1.0]])
    data = np.einsum("cs,ste->cte", expected, sources)
    weights = np.ones((20, 4))
    weights[:3, 0] = 0.0
    np.testing.assert_allclose(
        fit_mixing_matrix(data, sources, sample_weight=weights), expected, atol=1e-12
    )


def test_apply_covariance_transform_contract():
    """Covariance transforms use congruence multiplication and validate shapes."""
    matrix = np.array([[1.0, 2.0], [-1.0, 0.5]])
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    transformed = apply_covariance_transform(matrix, covariance)
    np.testing.assert_allclose(transformed, matrix @ covariance @ matrix.T)
    np.testing.assert_allclose(transformed, transformed.T)

    invalid = [
        (np.ones(2), np.eye(2), "matrix must be 2D"),
        (np.eye(2), np.ones((2, 3)), "covariance must be square"),
        (np.eye(2), np.eye(3), "dimensions do not match"),
    ]
    for bad_matrix, bad_covariance, message in invalid:
        with pytest.raises(ValueError, match=message):
            apply_covariance_transform(bad_matrix, bad_covariance)


def test_map_spatial_matrices_to_sensor_space_contract():
    """Sensor mapping is numerically correct and rejects incompatible layouts."""
    filters = np.array([[1.0, 2.0], [3.0, 4.0]])
    patterns = np.array([[5.0, 6.0], [7.0, 8.0]])
    whitener = np.diag([0.5, 0.25])
    dewhitener = np.diag([2.0, 4.0])
    filters_sensor, patterns_sensor = map_spatial_matrices_to_sensor_space(
        filters, patterns, whitener=whitener, dewhitener=dewhitener
    )
    np.testing.assert_allclose(filters_sensor, filters @ whitener)
    np.testing.assert_allclose(patterns_sensor, dewhitener @ patterns)

    invalid = [
        (np.ones(2), np.eye(2), np.eye(2), np.eye(2), "must be 2D"),
        (np.ones((2, 3)), np.eye(2), np.eye(2), np.eye(2), "filter and whitener"),
        (np.eye(2), np.ones((3, 2)), np.eye(2), np.eye(2), "dewhitener and pattern"),
        (
            np.ones((3, 2)),
            np.ones((2, 2)),
            np.eye(2),
            np.eye(2),
            "same number of components",
        ),
    ]
    for bad_filters, bad_patterns, bad_whitener, bad_dewhitener, message in invalid:
        with pytest.raises(ValueError, match=message):
            map_spatial_matrices_to_sensor_space(
                bad_filters,
                bad_patterns,
                whitener=bad_whitener,
                dewhitener=bad_dewhitener,
            )


def test_compute_mne_sensor_whitener_contract():
    """Sensor scaling pairs channel types with the supplied channel-name order."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 1000)) * np.array([1.0, 10.0, 100.0])[:, None]
    whitener, colorer = compute_mne_sensor_whitener(data)
    np.testing.assert_allclose(apply_spatial_transform(whitener, data).std(axis=1), 1.0)
    np.testing.assert_allclose(colorer @ whitener, np.eye(3))

    info = mne.create_info(
        ["EEG0", "MAG0", "EEG1", "MAG1"],
        100.0,
        ["eeg", "mag", "eeg", "mag"],
    )
    ch_names = ["MAG1", "EEG0", "MAG0", "EEG1"]
    base = np.arange(1.0, 101.0)
    data = np.vstack([base, 2.0 * base, 3.0 * base, 4.0 * base])
    expected_scales = np.array(
        [
            np.std(data[[0, 2]]),
            np.std(data[[1, 3]]),
            np.std(data[[0, 2]]),
            np.std(data[[1, 3]]),
        ]
    )
    whitener, colorer = compute_mne_sensor_whitener(data, info=info, ch_names=ch_names)
    np.testing.assert_allclose(np.diag(whitener), 1.0 / expected_scales)
    np.testing.assert_allclose(np.diag(colorer), expected_scales)
    np.testing.assert_allclose(
        apply_spatial_transform(whitener, data), data / expected_scales[:, None]
    )
    equivalent_info = mne.create_info(ch_names, 100.0, ["mag", "eeg", "mag", "eeg"])
    equivalent = compute_mne_sensor_whitener(
        data, info=equivalent_info, ch_names=ch_names
    )
    np.testing.assert_allclose(equivalent[0], whitener)
    np.testing.assert_allclose(equivalent[1], colorer)

    bad_data = np.ones((2, 10))
    bad_data[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_mne_sensor_whitener(bad_data)


def test_whiten_from_data_covariance_contract():
    """Data whitening produces identity covariance and preserves supported layouts."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 16, 5000
    mixing = rng.standard_normal((n_channels, n_channels))
    data = mixing @ rng.standard_normal((n_channels, n_samples))
    whitened, _whitener, _dewhitener = whiten_from_data_covariance(data)
    covariance = whitened @ whitened.T / n_samples
    np.testing.assert_allclose(covariance, np.eye(whitened.shape[0]), atol=0.1)

    true_rank = 8
    rank_data = rng.standard_normal((true_rank, 1000))
    rank_data = rng.standard_normal((16, true_rank)) @ rank_data
    rank_whitened, _, _ = whiten_from_data_covariance(rank_data)
    assert rank_whitened.shape[0] <= true_rank + 1

    epoched = rng.standard_normal((8, 100, 20))
    epoched_whitened, _, _ = whiten_from_data_covariance(epoched)
    assert epoched_whitened.ndim == 3
    assert epoched_whitened.shape[1:] == epoched.shape[1:]


def test_compute_data_covariance_whitener_numerical_contract():
    """Whitening matrices are inverse pairs and honor explicit rank."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((10, 10))
    covariance = A @ A.T
    whitener, dewhitener, eigenvalues = compute_data_covariance_whitener(covariance)
    np.testing.assert_allclose(
        whitener @ dewhitener, np.eye(whitener.shape[0]), atol=1e-10
    )

    rank_whitener, _rank_dewhitener, rank_eigenvalues = (
        compute_data_covariance_whitener(covariance, rank=5)
    )
    assert rank_whitener.shape[0] == 5
    assert len(rank_eigenvalues) == 5
    assert len(eigenvalues) == 10


def test_compute_data_covariance_whitener_edge_contract():
    """Zero/no-component cases fail while tiny SI-unit covariance remains valid."""
    with pytest.raises(ValueError, match="no significant variance"):
        compute_data_covariance_whitener(np.zeros((5, 5)))
    with pytest.raises(ValueError, match="No components"):
        compute_data_covariance_whitener(np.eye(3), reg=1.0)

    covariance = np.diag([3.0, 2.0, 1.0]) * 1e-35
    whitener, _, eigenvalues = compute_data_covariance_whitener(covariance)
    np.testing.assert_allclose(
        apply_covariance_transform(whitener, covariance), np.eye(3)
    )
    assert eigenvalues.shape == (3,)


def test_compute_data_covariance_whitener_validation_contract():
    """Covariance, rank, regularization, and data dimensionality are validated."""
    covariance_cases = [
        (np.ones(3), "must be square"),
        (np.ones((2, 3)), "must be square"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.diag([1.0, -1.0]), "positive semi-definite"),
    ]
    for covariance, message in covariance_cases:
        with pytest.raises(ValueError, match=message):
            compute_data_covariance_whitener(covariance)
    parameter_cases = [
        ({"rank": 0}, "rank must be positive"),
        ({"rank": 1.5}, "rank must be an int"),
        ({"rank": True}, "rank must be an int"),
        ({"reg": -1.0}, "reg must be a finite non-negative"),
        ({"reg": np.inf}, "reg must be a finite non-negative"),
        ({"reg": "small"}, "reg must be a real number"),
    ]
    for kwargs, message in parameter_cases:
        with pytest.raises((TypeError, ValueError), match=message):
            compute_data_covariance_whitener(np.eye(3), **kwargs)
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        whiten_from_data_covariance(np.array([1, 2, 3, 4, 5]))
