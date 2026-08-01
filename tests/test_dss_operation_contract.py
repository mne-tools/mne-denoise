"""Tests for the explicit DSS component-operation contract."""

from __future__ import annotations

import warnings

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from mne_denoise.dss import DSS


def _ranked_bias(data):
    """Give channels distinct deterministic bias scores."""
    shape = (data.shape[0],) + (1,) * (data.ndim - 1)
    weights = np.linspace(2.0, 0.5, data.shape[0]).reshape(shape)
    return data * weights


def _array_data(seed=0, shape=(6, 600)):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape)
    offsets = np.linspace(-3.0, 4.0, shape[0]).reshape(
        (shape[0],) + (1,) * (len(shape) - 1)
    )
    scales = np.geomspace(0.2, 5.0, shape[0]).reshape(offsets.shape)
    return data * scales + offsets


@pytest.mark.parametrize(
    ("action", "selection"),
    [("extract", 2), ("retain", 2), ("subtract", 2)],
)
def test_canonical_fit_transform_equals_fit_then_transform(action, selection):
    """Canonical fit_transform is exactly the sklearn composition."""
    data = _array_data()
    params = {
        "bias": _ranked_bias,
        "n_components": 4,
        "component_action": action,
        "component_selection": selection,
        "normalize_input": False,
    }

    expected = DSS(**params).fit(data).transform(data)
    actual = DSS(**params).fit_transform(data)

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_canonical_adaptive_fit_transform_is_inductive_composition():
    """Explicit operations keep the sklearn contract in adaptive mode too."""
    data = _array_data(shape=(6, 900))
    params = {
        "bias": _ranked_bias,
        "adaptive": True,
        "component_action": "subtract",
        "component_selection": 2,
        "normalize_input": False,
    }

    expected = DSS(**params).fit(data).transform(data)
    actual = DSS(**params).fit_transform(data)

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_component_selection_is_leading_sensor_operation_count():
    """Selection acts on a leading prefix while extraction exposes all sources."""
    data = _array_data()
    common = {
        "bias": _ranked_bias,
        "n_components": 4,
        "component_selection": 2,
        "normalize_input": False,
    }
    extractor = DSS(component_action="extract", **common).fit(data)
    sources = extractor.transform(data)
    centered = data - data.mean(axis=1, keepdims=True)
    selected = extractor.mixing_[:, :2] @ (extractor.filters_[:2] @ centered)

    retained = DSS(component_action="retain", **common).fit_transform(data)
    subtracted = DSS(component_action="subtract", **common).fit_transform(data)

    assert sources.shape == (4, data.shape[1])
    assert extractor.n_selected_ == 2
    assert_allclose(retained, selected + data.mean(axis=1, keepdims=True))
    assert_allclose(subtracted, data - selected)


@pytest.mark.parametrize("selection", [None, 0])
@pytest.mark.parametrize("adaptive", [False, True])
def test_subtract_without_selection_is_exact_noop(selection, adaptive):
    """No selected components means exact, not approximate, passthrough."""
    data = _array_data()
    out = DSS(
        bias=_ranked_bias,
        adaptive=adaptive,
        component_action="subtract",
        component_selection=selection,
    ).fit_transform(data)

    assert_array_equal(out, data)


def test_full_rank_retain_reconstructs_and_preserves_centering():
    """Full-rank retention restores the signal and its channel offsets."""
    data = _array_data()
    est = DSS(
        bias=_ranked_bias,
        component_action="retain",
        normalize_input=False,
    )
    retained = est.fit_transform(data)

    assert_allclose(retained, data, rtol=1e-9, atol=1e-9)
    assert_allclose(retained.mean(axis=1), data.mean(axis=1), atol=1e-12)

    sources = est.set_params(component_action="extract").transform(data)
    assert_allclose(sources.mean(axis=1), 0.0, atol=1e-12)


@pytest.mark.parametrize("shape", [(6, 600), (6, 150, 4)])
@pytest.mark.parametrize("action", ["retain", "subtract"])
def test_numpy_sensor_actions_preserve_2d_and_3d_layout(shape, action):
    """Sensor actions preserve NumPy layout for continuous and epoched data."""
    data = _array_data(shape=shape)
    out = DSS(
        bias=_ranked_bias,
        n_components=4,
        component_action=action,
        component_selection=2,
        normalize_input=False,
    ).fit_transform(data)

    assert isinstance(out, np.ndarray)
    assert out.shape == data.shape


def test_numpy_3d_extract_shape():
    """Three-dimensional NumPy extraction keeps channel-first epoch layout."""
    data = _array_data(shape=(6, 150, 4))
    sources = DSS(
        bias=_ranked_bias,
        n_components=3,
        component_action="extract",
        normalize_input=False,
    ).fit_transform(data)

    assert sources.shape == (3, 150, 4)


def _mixed_info(sfreq=200.0):
    return mne.create_info(
        ["EEG001", "EEG002", "EEG003", "EEG004", "EOG001"],
        sfreq,
        ["eeg", "eeg", "eeg", "eeg", "eog"],
    )


def test_raw_subtract_preserves_container_channels_and_metadata():
    """Raw output retains annotations, channel metadata, and untouched EOG."""
    data = _array_data(shape=(5, 800)) * 1e-6
    raw = mne.io.RawArray(data, _mixed_info(), first_samp=321, verbose=False)
    raw.info["bads"] = ["EEG004"]
    raw.set_annotations(mne.Annotations([0.5], [0.25], ["bad_motion"]))

    out = DSS(
        bias=_ranked_bias,
        component_action="subtract",
        component_selection=1,
        normalize_input=False,
    ).fit_transform(raw)

    assert isinstance(out, mne.io.BaseRaw)
    assert out.ch_names == raw.ch_names
    assert out.first_samp == raw.first_samp
    assert out.info["bads"] == raw.info["bads"]
    assert_array_equal(out.get_data(picks="eog"), raw.get_data(picks="eog"))
    assert list(out.annotations.description) == ["bad_motion"]


def test_raw_covariances_use_identical_annotation_sample_support(monkeypatch):
    """Baseline and biased Raw covariances reject the same annotated samples."""
    rng = np.random.default_rng(41)
    info = mne.create_info(["EEG 001", "EEG 002"], 100.0, "eeg")
    raw = mne.io.RawArray(
        rng.standard_normal((2, 1_000)),
        info,
        first_samp=1_000,
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([1.0], [2.0], ["BAD_motion"]))

    original = mne.compute_raw_covariance
    observed = []

    def capture_sample_support(inst, *args, **kwargs):
        observed.append(
            {
                "first_samp": inst.first_samp,
                "onset": inst.annotations.onset.copy(),
                "duration": inst.annotations.duration.copy(),
                "description": inst.annotations.description.copy(),
                "retained": inst.get_data(reject_by_annotation="omit").shape[-1],
            }
        )
        return original(inst, *args, **kwargs)

    monkeypatch.setattr(mne, "compute_raw_covariance", capture_sample_support)

    DSS(
        bias=_ranked_bias,
        n_components=2,
        component_action="retain",
        component_selection=1,
    ).fit(raw)

    assert len(observed) == 2
    assert observed[0]["first_samp"] == observed[1]["first_samp"] == raw.first_samp
    assert observed[0]["retained"] == observed[1]["retained"] == 800
    assert_array_equal(observed[0]["onset"], observed[1]["onset"])
    assert_array_equal(observed[0]["duration"], observed[1]["duration"])
    assert_array_equal(observed[0]["description"], observed[1]["description"])


def test_epochs_retain_preserves_container_channels_and_metadata():
    """Epoch retention preserves events, metadata, and untouched channels."""
    pd = pytest.importorskip("pandas")
    data = np.transpose(_array_data(shape=(5, 120, 4)), (2, 0, 1)) * 1e-6
    events = np.column_stack([np.arange(4) * 200, np.zeros(4, dtype=int), [1, 2, 1, 2]])
    epochs = mne.EpochsArray(
        data,
        _mixed_info(),
        events=events,
        event_id={"a": 1, "b": 2},
        tmin=-0.1,
        baseline=(None, 0),
        verbose=False,
    )
    epochs.metadata = pd.DataFrame({"trial": np.arange(4)})

    out = DSS(
        bias=_ranked_bias,
        n_components=4,
        component_action="retain",
        component_selection=2,
        normalize_input=False,
    ).fit_transform(epochs)

    assert isinstance(out, mne.BaseEpochs)
    assert out.ch_names == epochs.ch_names
    assert_array_equal(out.events, epochs.events)
    assert out.event_id == epochs.event_id
    assert out.baseline == epochs.baseline
    assert out.metadata.equals(epochs.metadata)
    assert_array_equal(out.get_data(picks="eog"), epochs.get_data(picks="eog"))


def test_evoked_subtract_preserves_container_channels_and_metadata():
    """Evoked subtraction preserves comment, nave, bads, and untouched EOG."""
    data = _array_data(shape=(5, 500)) * 1e-6
    evoked = mne.EvokedArray(
        data, _mixed_info(), tmin=-0.2, comment="auditory", nave=17, verbose=False
    )
    evoked.info["bads"] = ["EEG004"]

    out = DSS(
        bias=_ranked_bias,
        component_action="subtract",
        component_selection=1,
        normalize_input=False,
    ).fit_transform(evoked)

    assert isinstance(out, mne.Evoked)
    assert out.ch_names == evoked.ch_names
    assert out.comment == evoked.comment
    assert out.nave == evoked.nave
    assert out.info["bads"] == evoked.info["bads"]
    assert_array_equal(out.get_data(picks="eog"), evoked.get_data(picks="eog"))


@pytest.mark.parametrize(
    "extra",
    [
        {"normalize_input": True},
        {"normalize_input": False, "whiten": True},
        {"normalize_input": False, "smooth": 7},
    ],
)
def test_full_rank_retain_with_preprocessing_reconstructs(extra):
    """Normalization, whitening, and smoothing share reconstruction semantics."""
    data = _array_data()
    out = DSS(
        bias=_ranked_bias,
        component_action="retain",
        **extra,
    ).fit_transform(data)

    assert_allclose(out, data, rtol=1e-8, atol=1e-8)


def test_canonical_parameters_clone_get_params_and_set_params():
    """The new contract is a regular sklearn estimator parameter surface."""
    est = DSS(
        bias=_ranked_bias,
        n_components=4,
        component_action="subtract",
        component_selection=2,
        whitening_rank=3,
    )
    params = est.get_params()
    copied = clone(est)

    assert params["component_action"] == "subtract"
    assert params["component_selection"] == 2
    assert params["whitening_rank"] == 3
    assert copied.get_params()["component_action"] == "subtract"
    assert copied.get_params()["component_selection"] == 2
    assert copied.get_params()["whitening_rank"] == 3

    est.set_params(component_action="retain", component_selection=1, whitening_rank=2)
    assert est.component_action == "retain"
    assert est.component_selection == 1
    assert est.whitening_rank == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"component_action": "extract", "return_type": "sources"},
        {"component_selection": 1, "n_select": 1},
        {"whitening_rank": 3, "rank": 3},
    ],
)
def test_canonical_and_legacy_parameter_conflicts_raise(kwargs):
    """Even equal duplicate spellings are rejected to keep plans unambiguous."""
    with pytest.raises(ValueError, match="cannot be combined"):
        DSS(bias=_ranked_bias, **kwargs)


@pytest.mark.parametrize(
    ("legacy", "replacement"),
    [
        ({"return_type": "sources"}, "component_action"),
        ({"n_select": 1}, "component_selection"),
        ({"rank": 3}, "whitening_rank"),
    ],
)
def test_legacy_parameters_emit_future_warning(legacy, replacement):
    """All released compatibility names identify their canonical replacement."""
    with pytest.warns(FutureWarning, match=replacement):
        DSS(bias=_ranked_bias, **legacy)


def test_legacy_sensor_return_type_preserves_0x_behavior():
    """Legacy transform retains while legacy fit_transform still subtracts."""
    data = _array_data()
    with pytest.warns(FutureWarning, match="component_action"):
        transform_est = DSS(
            bias=_ranked_bias,
            n_components=3,
            n_select=1,
            return_type="raw",
            normalize_input=False,
        )
    retained = transform_est.fit(data).transform(data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        subtracted = DSS(
            bias=_ranked_bias,
            n_components=3,
            n_select=1,
            return_type="raw",
            normalize_input=False,
        ).fit_transform(data)

    assert retained.shape == data.shape
    assert subtracted.shape == data.shape
    assert not np.allclose(retained, subtracted)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"component_action": "invalid"},
        {"component_selection": -1},
        {"component_selection": "invalid"},
    ],
)
def test_invalid_canonical_contract_raises_on_fit(kwargs):
    """Invalid operations and selections fail before fitting covariance."""
    with pytest.raises(ValueError):
        DSS(bias=_ranked_bias, **kwargs).fit(_array_data())
