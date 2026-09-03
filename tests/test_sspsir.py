"""Tests for the SSP-SIR estimator."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from mne_denoise.sspsir import SSPSIR


@pytest.fixture(scope="module")
def eeg_info():
    ch = mne.channels.make_standard_montage("standard_1020").ch_names[:24]
    info = mne.create_info(ch, 1000.0, "eeg")
    info.set_montage("standard_1020")
    return info


@pytest.fixture(scope="module")
def tms_epochs(eeg_info):
    """Synthetic TMS-EEG: a clean evoked source + a focal high-frequency burst."""
    rng = np.random.default_rng(1)
    n_ch, n_t, n_ep = 24, 400, 16
    t = (np.arange(n_t) - 100) / 1000.0  # -100..300 ms
    btopo = rng.standard_normal(n_ch)
    btopo /= np.linalg.norm(btopo)
    brain = np.sin(2 * np.pi * 10 * t) * np.exp(-((t - 0.1) ** 2) / 0.01)
    mtopo = np.zeros(n_ch)
    mtopo[[8, 9, 10]] = [1.0, 0.8, 0.6]
    mtopo /= np.linalg.norm(mtopo)
    win = (t >= 0.005) & (t <= 0.050)
    data = np.zeros((n_ep, n_ch, n_t))
    for e in range(n_ep):
        art = np.zeros(n_t)
        art[win] = rng.standard_normal(win.sum()) * 8.0
        data[e] = (
            np.outer(btopo, brain)
            + np.outer(mtopo, art)
            + 0.1 * rng.standard_normal((n_ch, n_t))
        )
    epochs = mne.EpochsArray(data * 1e-6, eeg_info, tmin=t[0], verbose=False)
    return epochs, btopo, brain, win


def _assert_value_error(label, operation, expected):
    """Assert a labelled ValueError without making each case a pytest node."""
    try:
        operation()
    except ValueError as err:
        assert expected in str(err), (
            f"{label}: expected {expected!r} in the error, got {err}"
        )
    else:
        pytest.fail(f"{label}: expected ValueError containing {expected!r}")


def test_sspsir_removes_artifact_preserves_source(tms_epochs):
    epochs, btopo, brain, win = tms_epochs
    ss = SSPSIR(n_components=3).fit(epochs)
    cleaned = ss.transform(epochs).get_data()
    before = epochs.get_data()
    # Muscle-window variance on the artifact channels drops sharply.
    assert (
        cleaned[:, [8, 9, 10]][:, :, win].var()
        < 0.1 * before[:, [8, 9, 10]][:, :, win].var()
    )
    # The brain source is preserved outside the artifact window.
    proj = np.einsum("c,ct->t", btopo, cleaned.mean(0))
    corr = np.corrcoef(proj[~win], brain[~win])[0, 1]
    assert corr > 0.95


def test_sspsir_blend_protects_outside_artifact_window(tms_epochs):
    """Time-varying blending preserves the baseline outside the artifact window."""
    epochs = tms_epochs[0]
    win = tms_epochs[3]
    blended = SSPSIR(n_components=3, blend="auto").fit(epochs)
    constant = SSPSIR(n_components=3, blend="constant").fit(epochs)

    data = epochs.get_data().mean(0)
    reference = blended.operator_orig_ @ data  # rank-M, no artifact removed
    out_blend = blended.transform(epochs).get_data().mean(0)
    out_const = constant.transform(epochs).get_data().mean(0)

    def baseline_error(x):
        return np.linalg.norm(x[:, ~win] - reference[:, ~win]) / np.linalg.norm(
            reference[:, ~win]
        )

    assert baseline_error(out_blend) < 0.5 * baseline_error(out_const)


def test_sspsir_kernel_modes(tms_epochs):
    """The automatic, windowed, and constant blending modes are explicit."""
    epochs = tms_epochs[0]
    win = tms_epochs[3]
    auto = SSPSIR(n_components=3).fit(epochs)
    assert auto.kernel_.shape == (epochs.get_data().shape[-1],)
    assert auto.kernel_.min() >= 0.0 and auto.kernel_.max() <= 1.0
    assert win[int(np.argmax(auto.kernel_))]

    tmin, tmax = 0.005, 0.050
    manual = SSPSIR(n_components=2, art_window=(tmin, tmax)).fit(epochs)
    times = epochs.times
    kernel = manual.kernel_
    assert kernel.max() > 0.99
    assert kernel[times < tmin - 0.03].max() < 0.01
    assert kernel[times > tmax + 0.03].max() < 0.01
    above = times[kernel > 0.9 * kernel.max()]
    start = times[kernel > 0.1 * kernel.max()][0]
    assert 0.005 < (above[0] - start) < 0.020

    constant = SSPSIR(n_components=2, blend="constant").fit(epochs)
    np.testing.assert_array_equal(constant.kernel_, np.ones_like(constant.kernel_))
    data = epochs.get_data()
    expected = np.einsum("ij,ejt->eit", constant.operator_, data)
    np.testing.assert_allclose(constant.transform(epochs).get_data(), expected)


def test_sspsir_transform_compatibility(tms_epochs):
    """Time-varying fits stay time-locked; constant fits stay spatial-only."""
    epochs = tms_epochs[0]
    auto = SSPSIR(n_components=2).fit(epochs)
    manual = SSPSIR(n_components=2, art_window=(0.005, 0.050)).fit(epochs)

    short = epochs.get_data()[:, :, :100]
    for label, model in (("auto length", auto), ("manual length", manual)):
        _assert_value_error(
            label,
            lambda model=model: model.transform(short),
            "time points",
        )

    shifted = mne.EpochsArray(
        epochs.get_data(), epochs.info.copy(), tmin=epochs.tmin + 0.01, verbose=False
    )
    for label, model in (("auto time axis", auto), ("manual time axis", manual)):
        _assert_value_error(
            label,
            lambda model=model: model.transform(shifted),
            "different time axis",
        )

    info = mne.create_info(epochs.ch_names, epochs.info["sfreq"] / 2.0, "eeg")
    resampled_clock = mne.EpochsArray(
        epochs.get_data(), info, tmin=epochs.tmin, verbose=False
    )
    for label, model in (("auto sfreq", auto), ("manual sfreq", manual)):
        _assert_value_error(
            label,
            lambda model=model: model.transform(resampled_clock),
            "sampling frequency",
        )

    wrong_channels = epochs.get_data()[:, :-1, :]
    for label, model in (("auto channels", auto), ("manual channels", manual)):
        _assert_value_error(
            label,
            lambda model=model: model.transform(wrong_channels),
            "23 channels",
        )

    constant = SSPSIR(n_components=2, blend="constant").fit(epochs)
    out_short = constant.transform(short)
    np.testing.assert_allclose(
        out_short, np.einsum("ij,ejt->eit", constant.operator_, short)
    )
    out_shifted = constant.transform(shifted)
    np.testing.assert_allclose(
        out_shifted.get_data(),
        np.einsum("ij,ejt->eit", constant.operator_, shifted.get_data()),
    )
    np.testing.assert_array_equal(out_shifted.times, shifted.times)


def test_sspsir_component_selection(tms_epochs):
    """Counts, variance fractions, and the exposed singular spectrum agree."""
    epochs = tms_epochs[0]
    for label, value in (("python integer", 3), ("NumPy integer", np.int64(3))):
        model = SSPSIR(n_components=value).fit(epochs)
        assert model.n_components_ == 3, label

    model = SSPSIR(n_components=3).fit(epochs)
    singular_values = model.singular_values_
    assert singular_values.ndim == 1
    assert singular_values.size >= model.n_components_
    assert np.all(np.diff(singular_values) <= 1e-12)

    fractional = SSPSIR(n_components=np.float64(0.9)).fit(epochs)
    cumulative = np.cumsum(fractional.singular_values_**2) / np.sum(
        fractional.singular_values_**2
    )
    expected = int(np.searchsorted(cumulative, 0.9) + 1)
    assert fractional.n_components_ == expected


def test_sspsir_parameter_validation(tms_epochs):
    """Equivalent invalid spellings share one labelled public validation owner."""
    arr = np.random.default_rng(10).standard_normal((24, 400))
    base = {"n_components": 1, "sfreq": 1000.0}
    cases = [
        ("M zero", {"M": 0}, arr, "M must be a positive integer"),
        ("M noninteger", {"M": 2.5}, arr, "M must be a positive integer"),
        ("smooth_length zero", {"smooth_length": 0.0}, arr, "smooth_length"),
        ("smooth_length infinite", {"smooth_length": np.inf}, arr, "smooth_length"),
        ("n_dipoles zero", {"n_dipoles": 0}, arr, "n_dipoles"),
        ("n_dipoles noninteger", {"n_dipoles": 2.5}, arr, "n_dipoles"),
        ("high_pass zero", {"high_pass": 0.0}, arr, "high_pass"),
        ("high_pass NaN", {"high_pass": np.nan}, arr, "high_pass"),
        (
            "art_window reversed",
            {"art_window": (0.1, 0.0)},
            arr,
            "tmin < tmax",
        ),
        (
            "art_window non-tuple",
            {"art_window": [0.0, 0.1]},
            arr,
            "tuple",
        ),
        (
            "art_window infinite",
            {"art_window": (0.0, np.inf)},
            arr,
            "finite",
        ),
        ("invalid blend", {"blend": "nope"}, arr, "blend must be"),
        ("n_components zero", {"n_components": 0}, arr, "n_components"),
        (
            "n_components out of range",
            {"n_components": 1.1},
            arr,
            "variance fraction",
        ),
        ("n_components bool", {"n_components": True}, arr, "n_components"),
        ("n_components string", {"n_components": "2"}, arr, "n_components"),
        ("sfreq zero", {"sfreq": 0.0}, arr, "sfreq"),
        ("sfreq infinite", {"sfreq": np.inf}, arr, "sfreq"),
        ("sfreq NaN", {"sfreq": np.nan}, arr, "sfreq"),
        ("sfreq bool", {"sfreq": True}, arr, "sfreq"),
        (
            "high_pass above Nyquist",
            {"sfreq": 150.0, "high_pass": 100.0},
            arr,
            "Nyquist",
        ),
    ]
    for label, overrides, data, expected in cases:
        kwargs = base.copy()
        kwargs.update(overrides)
        _assert_value_error(
            label,
            lambda kwargs=kwargs, data=data: SSPSIR(**kwargs).fit(data),
            expected,
        )

    _assert_value_error(
        "n_components missing",
        lambda: SSPSIR().fit(arr),
        "n_components must be set",
    )
    _assert_value_error(
        "array sampling frequency missing",
        lambda: SSPSIR(n_components=2).fit(arr),
        "sampling frequency",
    )
    _assert_value_error(
        "artifact window without overlap",
        lambda: SSPSIR(n_components=2, art_window=(5.0, 9.0)).fit(tms_epochs[0]),
        "does not overlap",
    )


def test_sspsir_not_fitted_raises():
    with pytest.raises(NotFittedError):
        SSPSIR(n_components=2).transform(np.zeros((24, 100)))


def test_sspsir_clips_m_to_operator_rank(tms_epochs):
    with pytest.warns(RuntimeWarning, match="exceeds the numerical rank"):
        ss = SSPSIR(n_components=2, M=100).fit(tms_epochs[0])
    assert ss.M_ < 100
    assert ss.M_ == np.linalg.matrix_rank(ss.operator_)
    assert ss.M_ == np.linalg.matrix_rank(ss.operator_orig_)


@pytest.fixture(scope="module")
def forward(eeg_info):
    sphere = mne.make_sphere_model(
        r0="auto", head_radius="auto", info=eeg_info, verbose=False
    )
    src = mne.setup_volume_source_space(
        sphere=sphere, pos=25.0, sphere_units="mm", verbose=False
    )
    return mne.make_forward_solution(
        eeg_info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False
    )


def test_sspsir_forward_integrations(tms_epochs, forward):
    """Array, MNE, and insufficient-spatial-information paths stay coherent."""
    arr = np.random.default_rng(7).standard_normal((24, 400))
    array_model = SSPSIR(n_components=2, sfreq=1000.0, forward=forward).fit(arr)
    array_out = array_model.transform(arr)
    assert isinstance(array_out, np.ndarray)
    assert array_out.shape == arr.shape
    assert np.isfinite(array_out).all()
    assert array_model.projs_ == []

    epochs = tms_epochs[0]
    mne_model = SSPSIR(n_components=2, forward=forward).fit(epochs)
    assert mne_model.leadfield_.shape == (
        len(epochs.ch_names),
        forward["sol"]["data"].shape[1],
    )
    mne_out = mne_model.transform(epochs)
    assert isinstance(mne_out, mne.BaseEpochs)
    assert mne_out.ch_names == epochs.ch_names
    assert mne_out.get_data().shape == epochs.get_data().shape
    np.testing.assert_array_equal(mne_out.times, epochs.times)

    evoked = epochs.average()
    evoked_out = SSPSIR(n_components=2).fit_transform(evoked)
    assert isinstance(evoked_out, mne.Evoked)
    assert evoked_out.ch_names == evoked.ch_names
    assert evoked_out.data.shape == evoked.data.shape

    _assert_value_error(
        "array without forward",
        lambda: SSPSIR(n_components=2, sfreq=1000.0).fit(arr),
        "channel positions",
    )
    bad_arr = np.random.default_rng(9).standard_normal((30, 400))
    _assert_value_error(
        "forward channel mismatch",
        lambda: SSPSIR(n_components=2, sfreq=1000.0, forward=forward).fit(bad_arr),
        "same number of",
    )


def test_sspsir_exposes_mne_projections(tms_epochs):
    """Removed topographies are available as inactive, plottable MNE projections."""
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=3).fit(epochs)

    assert len(ss.projs_) == ss.n_components_
    assert all(isinstance(proj, mne.Projection) for proj in ss.projs_)
    for i, proj in enumerate(ss.projs_):
        np.testing.assert_allclose(
            proj["data"]["data"].ravel(), ss.artifact_topographies_[:, i]
        )
        assert proj["data"]["col_names"] == epochs.ch_names
        assert not proj["active"]
    mne.viz.plot_projs_topomap(ss.projs_, epochs.info, show=False)


def test_sspsir_verbose_logs_fit_summary(tms_epochs, caplog):
    """The package logging convention reports what was removed."""
    import logging

    epochs = tms_epochs[0]
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        SSPSIR(n_components=3, verbose=True).fit(epochs)
    assert any(
        "SSP-SIR:" in record.message
        and "channels=" in record.message
        and "removed 3 artifact component(s)" in record.message
        for record in caplog.records
    )
