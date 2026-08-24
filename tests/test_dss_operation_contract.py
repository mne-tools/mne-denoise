"""Tests for explicit DSS component-operation semantics."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from mne_denoise.dss import DSS
from mne_denoise.dss.utils.segmentation import FixedWindowSegmenter


def _ranked_bias(data):
    """Give channels distinct deterministic bias scores."""
    shape = (data.shape[0],) + (1,) * (data.ndim - 1)
    weights = np.linspace(2.0, 0.5, data.shape[0]).reshape(shape)
    return data * weights


def _array_data(seed=0, shape=(6, 600)):
    """Create nonzero-mean, differently scaled channel data."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape)
    channel_shape = (shape[0],) + (1,) * (len(shape) - 1)
    offsets = np.linspace(-3.0, 4.0, shape[0]).reshape(channel_shape)
    scales = np.geomspace(0.2, 5.0, shape[0]).reshape(channel_shape)
    return data * scales + offsets


@pytest.mark.parametrize("action", ["extract", "retain", "subtract"])
def test_explicit_standard_fit_transform_is_composition(action):
    """Every explicit standard operation follows the sklearn composition."""
    data = _array_data()
    params = {
        "bias": _ranked_bias,
        "n_components": 4,
        "n_select": 2,
        "component_action": action,
        "normalize_input": False,
    }

    expected = DSS(**params).fit(data).transform(data)
    actual = DSS(**params).fit_transform(data)

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_explicit_actions_have_distinct_algebra():
    """Extraction, retention, and subtraction implement their named algebra."""
    data = _array_data()
    est = DSS(
        bias=_ranked_bias,
        n_components=4,
        n_select=2,
        component_action="extract",
        normalize_input=False,
    ).fit(data)

    sources = est.transform(data)
    selected = est.mixing_[:, :2] @ sources[:2]
    mean = data.mean(axis=1, keepdims=True)

    retained = est.set_params(component_action="retain").transform(data)
    subtracted = est.set_params(component_action="subtract").transform(data)

    assert sources.shape == (4, data.shape[1])
    assert_allclose(retained, selected + mean, rtol=1e-12, atol=1e-12)
    assert_allclose(subtracted, data - selected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("normalize_input", [False, True])
def test_full_rank_retain_reconstructs_input(normalize_input):
    """Retention without a selection uses every fitted component."""
    data = _array_data()
    retained = DSS(
        bias=_ranked_bias,
        component_action="retain",
        normalize_input=normalize_input,
    ).fit_transform(data)

    assert_allclose(retained, data, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("shape", [(6, 600), (6, 150, 4)])
def test_subtract_without_selection_is_exact_copy(shape):
    """Explicit subtraction with no selection is a non-mutating no-op."""
    data = _array_data(shape=shape)
    original = data.copy()
    out = DSS(
        bias=_ranked_bias,
        component_action="subtract",
        normalize_input=False,
    ).fit_transform(data)

    assert out is not data
    assert_array_equal(out, data)
    assert_array_equal(data, original)


@pytest.mark.parametrize("shape", [(6, 600), (6, 150, 4)])
@pytest.mark.parametrize("action", ["retain", "subtract"])
def test_sensor_actions_preserve_array_layout(shape, action):
    """Sensor-space operations preserve 2-D and 3-D array orientation."""
    data = _array_data(shape=shape)
    out = DSS(
        bias=_ranked_bias,
        n_components=4,
        n_select=2,
        component_action=action,
        normalize_input=False,
    ).fit_transform(data)

    assert out.shape == data.shape


@pytest.mark.parametrize("action", [None, "replace", "sources", "raw"])
def test_invalid_component_action_is_rejected(action):
    """Unknown operations fail before fitting any spatial model."""
    with pytest.raises(ValueError, match="component_action"):
        DSS(bias=_ranked_bias, component_action=action).fit(_array_data())


def test_component_action_is_cloneable():
    """The operation remains a regular sklearn constructor parameter."""
    est = DSS(
        bias=_ranked_bias,
        n_select=2,
        component_action="subtract",
        normalize_input=False,
    )

    cloned = clone(est)

    assert cloned.component_action == "subtract"
    assert cloned.n_select == 2


@pytest.mark.parametrize("action", ["retain", "subtract"])
def test_raw_sensor_actions_preserve_metadata_and_bad_channels(action):
    """Explicit Raw operations preserve the corrected MNE input contract."""
    data = _array_data(shape=(4, 800)) * 1e-6
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 200.0, "eeg")
    raw = mne.io.RawArray(data, info, first_samp=321, verbose=False)
    raw.info["bads"] = ["EEG3"]
    raw.set_annotations(mne.Annotations([0.5], [0.25], ["marker"]))

    out = DSS(
        bias=_ranked_bias,
        n_components=3,
        n_select=1,
        component_action=action,
        normalize_input=False,
    ).fit_transform(raw)

    assert isinstance(out, mne.io.BaseRaw)
    assert out.first_samp == raw.first_samp
    assert out.annotations == raw.annotations
    assert out.info["bads"] == raw.info["bads"]
    assert_array_equal(out.get_data(picks=["EEG3"]), raw.get_data(picks=["EEG3"]))


def test_epochs_extract_and_subtract_layouts():
    """Explicit operations honor MNE Epochs source and sensor orientations."""
    data = np.transpose(_array_data(shape=(4, 100, 5)), (2, 0, 1))
    info = mne.create_info(4, 100.0, "eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)

    sources = DSS(
        bias=_ranked_bias,
        n_components=3,
        component_action="extract",
        normalize_input=False,
    ).fit_transform(epochs)
    cleaned = DSS(
        bias=_ranked_bias,
        n_components=3,
        n_select=1,
        component_action="subtract",
        normalize_input=False,
    ).fit_transform(epochs)

    assert sources.shape == (5, 3, 100)
    assert isinstance(cleaned, mne.BaseEpochs)
    assert cleaned.get_data().shape == data.shape


def test_subtract_with_smoothing_removes_only_selected_residual():
    """Subtraction keeps the smooth signal and removes the selected residual."""
    data = _array_data(shape=(6, 800))
    est = DSS(
        bias=_ranked_bias,
        n_components=4,
        n_select=1,
        smooth=5,
        component_action="extract",
        normalize_input=False,
    ).fit(data)
    sources = est.transform(data)
    selected = est.mixing_[:, :1] @ sources[:1]

    cleaned = est.set_params(component_action="subtract").transform(data)

    assert_allclose(cleaned, data - selected, rtol=1e-12, atol=1e-12)


def test_adaptive_explicit_subtract_runs_segments():
    """Canonical subtraction must not bypass adaptive segment processing."""
    data = _array_data(shape=(4, 800))
    info = mne.create_info(4, 100.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    est = DSS(
        bias=_ranked_bias,
        n_components=3,
        n_select=1,
        component_action="subtract",
        normalize_input=False,
        adaptive=True,
        segmenter=FixedWindowSegmenter(sfreq=100.0, window_len=2.0),
    )

    out = est.fit_transform(raw)

    assert isinstance(out, mne.io.BaseRaw)
    assert est.segment_results_ is not None
    assert len(est.segment_results_) == 4


@pytest.mark.parametrize("action", ["extract", "retain"])
def test_adaptive_rejects_incompatible_explicit_actions(action):
    """Segment-specific bases cannot produce one coherent extract/retain output."""
    data = _array_data(shape=(4, 800))
    info = mne.create_info(4, 100.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    est = DSS(
        bias=_ranked_bias,
        n_components=3,
        n_select=1,
        component_action=action,
        normalize_input=False,
        adaptive=True,
        segmenter=FixedWindowSegmenter(sfreq=100.0, window_len=2.0),
    )

    with pytest.raises(ValueError, match="supports only.*subtract"):
        est.fit_transform(raw)
