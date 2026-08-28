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


@pytest.mark.parametrize("case", ESTIMATOR_CASES, ids=lambda case: case.name)
def test_shared_estimator_contracts(case):
    """Exercise every shared contract declared for one estimator case."""
    if CLONEABLE in case.capabilities:
        estimator = case.make_estimator()
        cloned = clone(estimator)

        original_params = estimator.get_params()
        cloned_params = cloned.get_params()
        assert cloned_params.keys() == original_params.keys(), (
            f"{case.name}: clone changed the public parameter names"
        )
        for name in original_params:
            assert _same_parameter(cloned_params[name], original_params[name]), (
                f"{case.name}: clone changed parameter {name!r}"
            )

    if FIT_RETURNS_SELF in case.capabilities:
        assert case.make_array is not None, f"{case.name}: fit needs array data"
        estimator = case.make_estimator()
        fitted = estimator.fit(case.make_array())
        assert fitted is estimator, f"{case.name}: fit must return self"

    if FIT_TRANSFORM_COMPOSES in case.capabilities:
        assert case.make_array is not None, (
            f"{case.name}: fit_transform composition needs array data"
        )
        data = case.make_array()
        separate = case.make_estimator().fit(data).transform(data)
        composed = case.make_estimator().fit_transform(data)
        np.testing.assert_allclose(
            composed,
            separate,
            rtol=1e-9,
            atol=1e-10,
            err_msg=f"{case.name}: fit_transform must compose fit then transform",
        )

    if NOT_FITTED in case.capabilities:
        assert case.make_array is not None, (
            f"{case.name}: pre-fit transform needs array data"
        )
        with pytest.raises(NotFittedError):
            case.make_estimator().transform(case.make_array())

    if NUMPY_NO_MUTATION in case.capabilities:
        assert case.make_array is not None, f"{case.name}: mutation check needs data"
        data = case.make_array()
        before = data.copy()
        result = case.make_estimator().fit_transform(data)

        assert isinstance(result, np.ndarray), f"{case.name}: result is not an array"
        assert result.shape == data.shape, f"{case.name}: result shape changed"
        np.testing.assert_array_equal(
            data,
            before,
            err_msg=f"{case.name}: public array operation mutated its input",
        )

    if NUMPY_LAYOUT in case.capabilities:
        assert case.make_array is not None, f"{case.name}: layout check needs data"
        data = case.make_array()
        epoched = data.reshape(data.shape[0], 2, -1).transpose(1, 0, 2)
        result = case.make_estimator().fit_transform(epoched)

        assert isinstance(result, np.ndarray), f"{case.name}: result is not an array"
        assert result.shape == epoched.shape, f"{case.name}: NumPy layout changed"

    if FITTED_CHANNEL_COUNT in case.capabilities:
        assert case.make_array is not None, (
            f"{case.name}: channel-count check needs array data"
        )
        data = case.make_array()
        estimator = case.make_estimator().fit(data)
        mismatched = np.vstack((data, np.zeros((1, data.shape[1]))))

        with pytest.raises(ValueError):
            estimator.transform(mismatched)
