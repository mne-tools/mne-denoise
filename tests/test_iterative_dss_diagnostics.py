"""Tests for descriptive IterativeDSS execution diagnostics."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from mne_denoise.dss import (
    IterativeDSS,
    IterativeDSSDiagnostics,
    iterative_dss,
)
from mne_denoise.dss.denoisers import KurtosisDenoiser


def _data(*, n_channels: int = 4, n_times: int = 400) -> np.ndarray:
    """Create deterministic, full-rank test data."""
    return np.random.default_rng(42).standard_normal((n_channels, n_times))


def test_functional_diagnostics_are_opt_in_and_json_safe():
    """The historical return stays unchanged and opt-in data serialize safely."""
    data = _data()
    denoiser = KurtosisDenoiser()

    historical = iterative_dss(data, denoiser, 2, max_iter=3, random_state=42)
    with_diagnostics = iterative_dss(
        data,
        denoiser,
        2,
        max_iter=3,
        random_state=42,
        return_diagnostics=True,
    )

    assert len(historical) == 4
    assert len(with_diagnostics) == 5
    diagnostics = with_diagnostics[-1]
    assert isinstance(diagnostics, IterativeDSSDiagnostics)
    assert json.loads(json.dumps(diagnostics.to_dict())) == diagnostics.to_dict()
    with pytest.raises(FrozenInstanceError):
        diagnostics.method = "symmetric"


def test_diagnostics_match_algorithm_outputs():
    """Convergence fields are exact summaries of the algorithm output."""
    result = iterative_dss(
        _data(),
        KurtosisDenoiser(),
        3,
        max_iter=1,
        tol=0.0,
        random_state=42,
        return_diagnostics=True,
    )
    _, sources, _, convergence_info, diagnostics = result

    expected_iterations = tuple(int(value) for value in convergence_info[:, 0])
    expected_converged = tuple(bool(value) for value in convergence_info[:, 1])
    expected_non_converged = tuple(
        index for index, converged in enumerate(expected_converged) if not converged
    )
    assert diagnostics.iteration_counts == expected_iterations
    assert diagnostics.converged == expected_converged
    assert diagnostics.non_converged_components == expected_non_converged
    assert diagnostics.convergence_fraction == pytest.approx(
        np.mean(expected_converged)
    )
    assert len(diagnostics.source_rms) == sources.shape[0]
    assert np.all(np.isfinite(diagnostics.source_rms))
    assert diagnostics.reconstruction_energy_fraction is not None
    assert np.isfinite(diagnostics.reconstruction_energy_fraction)
    assert "validation" not in diagnostics.to_dict()


def test_estimator_publishes_immutable_diagnostics():
    """Fitted convenience attributes mirror the canonical diagnostics value."""
    estimator = IterativeDSS(
        KurtosisDenoiser(), n_components=3, max_iter=2, random_state=42
    ).fit(_data())
    diagnostics = estimator.get_diagnostics()

    assert diagnostics is estimator.diagnostics_
    assert estimator.n_components_ == diagnostics.n_components_extracted
    assert estimator.n_iter_ == diagnostics.iteration_counts
    assert estimator.converged_ == diagnostics.converged
    assert estimator.convergence_fraction_ == diagnostics.convergence_fraction
    assert estimator.non_converged_components_ == diagnostics.non_converged_components
    with pytest.raises(ValueError, match="convergence_fraction"):
        replace(diagnostics, convergence_fraction=0.123456)


def test_rank_limited_epoch_transform_uses_extracted_component_count():
    """Epoch reshaping follows the fitted rank, not the requested component count."""
    rng = np.random.default_rng(42)
    epochs = rng.standard_normal((3, 3, 80))
    estimator = IterativeDSS(
        KurtosisDenoiser(),
        n_components=3,
        rank=2,
        max_iter=2,
        random_state=42,
    ).fit(epochs)

    transformed = estimator.transform(epochs)

    assert estimator.n_components_ == 2
    assert transformed.shape == (3, 2, 80)
    assert (
        "requested components were limited by the whitening rank"
        in estimator.get_diagnostics().notes
    )


def test_diagnostics_require_fit():
    """Diagnostics cannot be read before a decomposition exists."""
    estimator = IterativeDSS(KurtosisDenoiser(), n_components=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        estimator.get_diagnostics()


@pytest.mark.parametrize("n_components", [0, -1, 1.5, True])
def test_invalid_component_count_rejected(n_components):
    """A positive integer component count is required before data processing."""
    with pytest.raises(ValueError, match="positive integer"):
        iterative_dss(_data(), KurtosisDenoiser(), n_components)


def test_return_diagnostics_requires_bool():
    """Ambiguous truthy diagnostic flags are rejected."""
    with pytest.raises(TypeError, match="must be a bool"):
        iterative_dss(_data(), KurtosisDenoiser(), 2, return_diagnostics=1)


def test_numpy_integer_component_count_is_supported():
    """NumPy integer scalars retain the historical integer behavior."""
    result = iterative_dss(
        _data(), KurtosisDenoiser(), np.int64(2), max_iter=1, random_state=42
    )
    assert result[0].shape[0] == 2


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"input_channel_count": 0}, "positive integer"),
        ({"method": "parallel"}, "method must"),
        ({"whitening_rank": 3}, "cannot exceed"),
        ({"n_components_requested": 1}, "requested count"),
        ({"iteration_counts": [1, 2]}, "one entry per"),
        ({"iteration_counts": (0, 2)}, "positive integers"),
        ({"converged": (1, False)}, "bool values"),
        ({"source_rms": (-1.0, 0.5)}, "non-negative"),
        ({"non_converged_components": ()}, "inconsistent with converged"),
        ({"convergence_fraction": 0.75}, "convergence_fraction"),
        ({"near_zero_components": (2,)}, "unique, ordered"),
        ({"near_zero_components": (0.0,)}, "integer indices"),
        ({"max_abs_source_correlation": 1.1}, "correlation"),
        ({"reconstruction_energy_fraction": -1.0}, "non-negative"),
        ({"notes": ["not immutable"]}, "notes must"),
    ],
)
def test_public_diagnostics_reject_inconsistent_values(changes, message):
    """Every serialized diagnostic invariant is enforced at construction."""
    diagnostics = IterativeDSSDiagnostics(
        method="deflation",
        input_channel_count=2,
        input_sample_count=100,
        whitening_rank=2,
        n_components_requested=2,
        n_components_extracted=2,
        iteration_counts=(1, 2),
        converged=(True, False),
        convergence_fraction=0.5,
        non_converged_components=(1,),
        source_rms=(1.0, 0.5),
        near_zero_components=(),
        max_abs_source_correlation=0.1,
        reconstruction_energy_fraction=0.5,
    )
    with pytest.raises(ValueError, match=message):
        replace(diagnostics, **changes)
