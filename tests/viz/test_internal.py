"""Tests for internal visualization utilities."""

from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from mne_denoise.viz import _seaborn as seaborn_utils
from mne_denoise.viz import _utils as viz_utils


def test_viz_utils_info(fitted_dss, synthetic_data):
    """Test resolving info."""
    assert isinstance(viz_utils._get_info(fitted_dss), mne.Info)
    assert viz_utils._get_info(fitted_dss, synthetic_data.info) is synthetic_data.info

    class MockEst:
        pass

    est = MockEst()
    assert viz_utils._get_info(est) is None
    est._mne_info = synthetic_data.info
    assert viz_utils._get_info(est) is synthetic_data.info


def test_viz_utils_picks(synthetic_data):
    """Test handle_picks handles all types."""
    info = synthetic_data.info

    # None -> defaults
    picks = viz_utils._handle_picks(info, picks=None)
    assert len(picks) == 5

    # List of indices
    picks = viz_utils._handle_picks(info, picks=[0, 1])
    assert np.array_equal(picks, [0, 1])

    # String 'all'
    picks = viz_utils._handle_picks(info, picks="all")
    assert len(picks) == len(info["ch_names"])

    # String type
    picks = viz_utils._handle_picks(info, picks="eeg")
    assert len(picks) == 5

    # List of channel names
    names = info["ch_names"][:2]
    picks = viz_utils._handle_picks(info, picks=names)
    assert len(picks) == 2
    assert picks[0] == 0

    # Fallback using a single integer
    picks = viz_utils._handle_picks(info, picks=2)
    assert np.asarray(picks) == 2


def test_viz_utils_gfp(synthetic_data):
    """Test GFP computation (lines 32-44)."""
    # From MNE object
    gfp = viz_utils._compute_gfp(synthetic_data)
    assert gfp.shape == (200,)

    # From 3D array
    data_3d = np.ones((10, 5, 200))
    gfp_3d = viz_utils._compute_gfp(data_3d)
    assert gfp_3d.shape == (200,)
    assert np.allclose(gfp_3d, 1.0)

    # From 2D array
    data_2d = np.ones((5, 200))
    gfp_2d = viz_utils._compute_gfp(data_2d)
    assert gfp_2d.shape == (200,)

    # Invalid shape
    with pytest.raises(ValueError, match="GFP input must be 2D"):
        viz_utils._compute_gfp(np.ones(10))


def test_viz_utils_estimators(fitted_dss):
    """Test patterns, filters, and scores."""
    # Patterns
    patterns = viz_utils._get_patterns(fitted_dss)
    assert patterns.shape == (5, 3)

    class MockEst:
        pass

    with pytest.raises(ValueError, match="patterns_"):
        viz_utils._get_patterns(MockEst())

    # Filters
    filters = viz_utils._get_filters(fitted_dss)
    assert filters.shape == (3, 5)
    with pytest.raises(ValueError, match="filters_"):
        viz_utils._get_filters(MockEst())

    # Scores (DSS eigenvalues)
    scores = viz_utils._get_scores(fitted_dss)
    assert len(scores) == 3

    # Scores (ZapLine scores_)
    est_zap = MockEst()
    est_zap.scores_ = np.array([10.0, 5.0])
    assert np.array_equal(viz_utils._get_scores(est_zap), [10.0, 5.0])

    # Empty scores check
    est_empty = MockEst()
    est_empty.eigenvalues_ = np.array([])
    est_empty.scores_ = np.array([])
    assert viz_utils._get_scores(est_empty) is None

    # Iterative DSS
    est_iter = MockEst()
    est_iter.convergence_info_ = {}
    assert viz_utils._get_scores(est_iter) is None

    # Global None
    assert viz_utils._get_scores(MockEst()) is None


def test_viz_utils_components(fitted_dss, synthetic_data):
    """Test source extraction."""
    # Compute from data (LinearDSS style)
    sources = viz_utils._get_components(fitted_dss, synthetic_data)
    assert sources.shape == (3, 200, 10)

    # Cached sources
    est_cached = MagicMock()
    est_cached.sources_ = np.zeros((3, 200, 10))
    assert viz_utils._get_components(est_cached, data=None) is est_cached.sources_

    with pytest.raises(ValueError, match="Data must be provided"):
        viz_utils._get_components(fitted_dss, data=None)

    # Raw data fallback
    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    sources_raw = viz_utils._get_components(fitted_dss, raw)
    assert sources_raw.ndim == 2

    # Fallback transform
    class BasicTransformer:
        def transform(self, d):
            return np.ones((5, 10))

    assert viz_utils._get_components(
        BasicTransformer(), data=np.zeros((5, 10))
    ).shape == (5, 10)


def test_seaborn_utils():
    """Test seaborn internal helpers."""
    # Test successful import
    sns = seaborn_utils._try_import_seaborn()
    assert sns.__name__ == "seaborn"

    # Test context manager
    with seaborn_utils._suppress_seaborn_plot_warnings():
        pass


def test_seaborn_import_error():
    """Explicitly test the seaborn import error path."""
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError, match="seaborn is required"):
            seaborn_utils._try_import_seaborn()
