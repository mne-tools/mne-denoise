"""Unit tests for denoiser base classes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_base_denoiser_protocols():
    """Both public base protocols expose one abstract operation and ``__call__``."""
    from mne_denoise.dss.denoisers.base import LinearDenoiser, NonlinearDenoiser

    for base, method, data, expected in [
        (LinearDenoiser, "apply", np.ones((2, 2)), np.full((2, 2), 2.0)),
        (NonlinearDenoiser, "denoise", np.array([1, 2, 3]), np.array([1, 4, 9])),
    ]:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            base()

        operation = (
            (lambda values: values * 2)
            if method == "apply"
            else (lambda values: values**2)
        )
        implementation = type(
            "MockDenoiser",
            (base,),
            {method: lambda self, values: operation(values)},
        )()
        assert_allclose(getattr(implementation, method)(data), expected)
        assert_allclose(implementation(data), expected)
