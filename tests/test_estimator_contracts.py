"""Package-wide contracts shared by sklearn-style denoising estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tests._contract_cases import (
    CLONEABLE,
    ESTIMATOR_CASES,
    FIT_RETURNS_SELF,
    FIT_TRANSFORM_COMPOSES,
    FITTED_CHANNEL_COUNT,
    NOT_FITTED,
    NUMPY_LAYOUT,
    NUMPY_NO_MUTATION,
)


def _cases(capability: str):
    return tuple(case for case in ESTIMATOR_CASES if capability in case.capabilities)


def _same_parameter(left, right) -> bool:
    """Compare public constructor values without treating arrays as scalars."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _same_parameter(left[key], right[key]) for key in left
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return False
        return len(left) == len(right) and all(
            _same_parameter(item, other) for item, other in zip(left, right)
        )
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return bool(result)


@pytest.mark.parametrize("case", _cases(CLONEABLE), ids=lambda case: case.name)
def test_sklearn_clone_preserves_public_parameters(case):
    """Cloning preserves constructor parameters without inspecting fit state."""
    estimator = case.make_estimator()
    cloned = clone(estimator)

    original_params = estimator.get_params()
    cloned_params = cloned.get_params()
    assert cloned_params.keys() == original_params.keys()
    assert all(
        _same_parameter(cloned_params[name], original_params[name])
        for name in original_params
    )


@pytest.mark.parametrize("case", _cases(FIT_RETURNS_SELF), ids=lambda case: case.name)
def test_fit_returns_the_same_estimator(case):
    """Applicable public ``fit`` methods follow sklearn return semantics."""
    estimator = case.make_estimator()
    assert estimator.fit(case.make_array()) is estimator


@pytest.mark.parametrize(
    "case", _cases(FIT_TRANSFORM_COMPOSES), ids=lambda case: case.name
)
def test_fit_transform_is_fit_then_transform(case):
    """The shared composition contract holds for ordinary array estimators."""
    data = case.make_array()
    separate = case.make_estimator().fit(data).transform(data)
    composed = case.make_estimator().fit_transform(data)
    np.testing.assert_allclose(composed, separate, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("case", _cases(NOT_FITTED), ids=lambda case: case.name)
def test_transform_before_fit_raises_not_fitted(case):
    """Learned transforms reject use before the estimator has been fitted."""
    with pytest.raises(NotFittedError):
        case.make_estimator().transform(case.make_array())


@pytest.mark.parametrize("case", _cases(NUMPY_NO_MUTATION), ids=lambda case: case.name)
def test_public_array_operations_do_not_mutate_input(case):
    """Cleaning through the public estimator API leaves NumPy input untouched."""
    data = case.make_array()
    before = data.copy()
    result = case.make_estimator().fit_transform(data)

    assert isinstance(result, np.ndarray)
    assert result.shape == data.shape
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("case", _cases(NUMPY_LAYOUT), ids=lambda case: case.name)
def test_supported_numpy_layouts_are_preserved(case):
    """Same-shape estimators preserve both channel-first NumPy layouts."""
    data = case.make_array()
    epoched = data.reshape(data.shape[0], 2, -1).transpose(1, 0, 2)
    result = case.make_estimator().fit_transform(epoched)

    assert isinstance(result, np.ndarray)
    assert result.shape == epoched.shape


@pytest.mark.parametrize(
    "case", _cases(FITTED_CHANNEL_COUNT), ids=lambda case: case.name
)
def test_fitted_channel_count_mismatch_is_rejected(case):
    """Estimators that learn a spatial layout reject a different channel count."""
    data = case.make_array()
    estimator = case.make_estimator().fit(data)
    mismatched = np.vstack((data, np.zeros((1, data.shape[1]))))

    with pytest.raises(ValueError):
        estimator.transform(mismatched)
