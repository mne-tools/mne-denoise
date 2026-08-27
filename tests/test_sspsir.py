"""Tests for the SSP-SIR estimator."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from mne_denoise.sspsir import SSPSIR, _artifact_subspace, compute_sir, compute_sspsir


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
    """The crossfade must leave the baseline closer to the unprojected SIR.

    Without the TESA crossfade the artifact projection also strips brain signal
    from the baseline and from late components; this is the regression that
    motivated adding ``operator_orig_``.
    """
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


def test_sspsir_blend_kernel_shape_and_range(tms_epochs):
    epochs = tms_epochs[0]
    win = tms_epochs[3]
    ss = SSPSIR(n_components=3).fit(epochs)
    assert ss.kernel_.shape == (epochs.get_data().shape[-1],)
    assert ss.kernel_.min() >= 0.0 and ss.kernel_.max() <= 1.0
    # The automatic kernel peaks inside the muscle-artifact window.
    assert win[int(np.argmax(ss.kernel_))]


def test_sspsir_manual_kernel_is_smooth_step(tms_epochs):
    """The art_window kernel matches TESA's dsigmf crossfade."""
    epochs = tms_epochs[0]
    tmin, tmax = 0.005, 0.050
    ss = SSPSIR(n_components=2, art_window=(tmin, tmax)).fit(epochs)
    times = epochs.times
    kernel = ss.kernel_
    assert kernel.max() > 0.99
    # Flat baseline well before and well after the window.
    assert kernel[times < tmin - 0.03].max() < 0.01
    assert kernel[times > tmax + 0.03].max() < 0.01
    # 10-90% transition takes roughly smooth_length (10 ms by default).
    above = times[kernel > 0.9 * kernel.max()]
    start = times[kernel > 0.1 * kernel.max()][0]
    assert 0.005 < (above[0] - start) < 0.020


def test_sspsir_blend_constant_is_uniform(tms_epochs):
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2, blend="constant").fit(epochs)
    np.testing.assert_array_equal(ss.kernel_, np.ones_like(ss.kernel_))
    # With a constant kernel the output is exactly the projected operator.
    data = epochs.get_data()
    expected = np.einsum("ij,ejt->eit", ss.operator_, data)
    # The constant path must not waste time evaluating the unprojected branch.
    ss.operator_orig_[:] = np.nan
    np.testing.assert_allclose(ss.transform(epochs).get_data(), expected)


def test_sspsir_transform_length_mismatch_raises(tms_epochs):
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2).fit(epochs)
    with pytest.raises(ValueError, match="time points"):
        ss.transform(epochs.get_data()[:, :, :100])


def test_sspsir_transform_channel_count_mismatch_raises(tms_epochs):
    """SSP-SIR rejects data with a different fitted channel count."""
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2).fit(epochs)
    with pytest.raises(
        ValueError, match="SSPSIR: X has 23 channels; fitted data had 24"
    ):
        ss.transform(epochs.get_data()[:, :-1, :])


def test_sspsir_constant_blend_survives_length_change(tms_epochs):
    """blend='constant' is time-invariant, so any length transforms fine."""
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2, blend="constant").fit(epochs)
    short = epochs.get_data()[:, :, :100]
    out = ss.transform(short)
    assert out.shape == short.shape
    np.testing.assert_allclose(out, np.einsum("ij,ejt->eit", ss.operator_, short))


def test_sspsir_rejects_bad_blend(tms_epochs):
    with pytest.raises(ValueError, match="blend must be"):
        SSPSIR(n_components=2, blend="nope").fit(tms_epochs[0])


def test_sspsir_transform_time_axis_must_match(tms_epochs):
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2).fit(epochs)
    shifted = mne.EpochsArray(
        epochs.get_data(), epochs.info.copy(), tmin=epochs.tmin + 0.01, verbose=False
    )
    with pytest.raises(ValueError, match="different time axis"):
        ss.transform(shifted)

    # A constant spatial operator is independent of the time axis.
    constant = SSPSIR(n_components=2, blend="constant").fit(epochs)
    assert constant.transform(shifted).get_data().shape == shifted.get_data().shape


def test_sspsir_transform_sfreq_must_match(tms_epochs):
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2).fit(epochs)
    info = mne.create_info(epochs.ch_names, epochs.info["sfreq"] / 2.0, "eeg")
    resampled_clock = mne.EpochsArray(
        epochs.get_data(), info, tmin=epochs.tmin, verbose=False
    )
    with pytest.raises(ValueError, match="sampling frequency"):
        ss.transform(resampled_clock)


def test_sspsir_variance_fraction(tms_epochs):
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=0.9).fit(epochs)
    assert ss.n_components_ >= 1
    assert ss.transform(epochs).get_data().shape == epochs.get_data().shape


def test_sspsir_exposes_singular_values(tms_epochs):
    """singular_values_ lets users pick components by the spectrum elbow."""
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=3).fit(epochs)
    s = ss.singular_values_
    assert s.ndim == 1 and s.size >= ss.n_components_
    assert np.all(np.diff(s) <= 1e-12)  # descending
    # The variance criterion this class documents is reproducible from them.
    cum = np.cumsum(s**2) / np.sum(s**2)
    expected = int(np.searchsorted(cum, 0.9) + 1)
    assert SSPSIR(n_components=0.9).fit(epochs).n_components_ == expected


def test_sspsir_variance_criterion_differs_from_tesa():
    """Documented divergence: variance fraction vs TESA's nuclear-norm form."""
    s = np.array([10.0, 6.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2])
    ours = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), 0.9) + 1)
    tesa = int(np.searchsorted(np.cumsum(s) ** 2 / np.sum(s) ** 2, 0.9) + 1)
    assert ours < tesa  # ours is the more conservative, standard PCA criterion


def test_artifact_subspace_accepts_numpy_scalar_types():
    data = np.diag([3.0, 2.0, 1.0])
    topographies, n_pc, singular_values = _artifact_subspace(data, np.int64(2))
    assert topographies.shape == (3, 2)
    assert n_pc == 2
    np.testing.assert_allclose(singular_values, [3.0, 2.0, 1.0])

    _, n_pc, _ = _artifact_subspace(data, np.float64(0.8))
    assert n_pc == 2


@pytest.mark.parametrize("n_components", [True, 0, -1, 0.0, 1.0, -0.1, 1.1, "2"])
def test_artifact_subspace_rejects_invalid_n_components(n_components):
    with pytest.raises(ValueError, match="n_components"):
        _artifact_subspace(np.eye(3), n_components)


def test_artifact_subspace_rejects_too_many_components():
    with pytest.raises(ValueError, match="exceeds the 3 available"):
        _artifact_subspace(np.eye(3), 4)
    with pytest.raises(ValueError, match="leave at least one channel"):
        _artifact_subspace(np.eye(3), 3)


@pytest.mark.parametrize(
    "data, match",
    [
        (np.zeros((3, 4)), "all-zero"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), "finite"),
        (np.empty((0, 3)), "non-empty 2D"),
    ],
)
def test_artifact_subspace_rejects_invalid_input(data, match):
    with pytest.raises(ValueError, match=match):
        _artifact_subspace(data, 1)


def test_sspsir_manual_window(tms_epochs):
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2, art_window=(0.005, 0.050)).fit(epochs)
    assert ss.n_components_ == 2
    assert hasattr(ss, "operator_")


def test_sspsir_window_no_overlap_raises(tms_epochs):
    epochs = tms_epochs[0]
    with pytest.raises(ValueError, match="does not overlap"):
        SSPSIR(n_components=2, art_window=(5.0, 9.0)).fit(epochs)


def test_sspsir_runs_on_evoked(tms_epochs):
    evoked = tms_epochs[0].average()
    ss = SSPSIR(n_components=2).fit(evoked)
    out = ss.transform(evoked)
    assert out.data.shape == evoked.data.shape


def test_sspsir_requires_n_components(tms_epochs):
    epochs = tms_epochs[0]
    with pytest.raises(ValueError, match="n_components must be set"):
        SSPSIR().fit(epochs)


def test_sspsir_fit_preserves_constructor_parameters(tms_epochs):
    """Validation during fit does not normalize constructor parameters."""
    model = SSPSIR(n_components=2, high_pass=100, smooth_length=1)
    model.fit(tms_epochs[0])
    assert model.high_pass == 100
    assert model.smooth_length == 1
    assert type(model.high_pass) is int
    assert type(model.smooth_length) is int


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"M": 0}, "M must be a positive integer"),
        ({"M": 2.5}, "M must be a positive integer"),
        ({"smooth_length": 0.0}, "smooth_length"),
        ({"smooth_length": np.inf}, "smooth_length"),
        ({"n_dipoles": 0}, "n_dipoles"),
        ({"n_dipoles": 2.5}, "n_dipoles"),
        ({"high_pass": 0.0}, "high_pass"),
        ({"high_pass": np.nan}, "high_pass"),
        ({"art_window": (0.1, 0.0)}, "tmin < tmax"),
        ({"art_window": [0.0, 0.1]}, "tuple"),
        ({"art_window": (0.0, np.inf)}, "finite"),
    ],
)
def test_sspsir_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SSPSIR(n_components=1, sfreq=1000.0, **kwargs).fit(np.ones((3, 20)))


@pytest.mark.parametrize("sfreq", [0.0, np.inf, np.nan, True])
def test_sspsir_rejects_invalid_sfreq(sfreq):
    with pytest.raises(ValueError, match="sfreq"):
        SSPSIR(n_components=1, sfreq=sfreq).fit(np.ones((3, 20)))


def test_sspsir_array_requires_sfreq():
    arr = np.random.default_rng(0).standard_normal((24, 400))
    with pytest.raises(ValueError, match="sampling frequency"):
        SSPSIR(n_components=2).fit(arr)


def test_sspsir_high_pass_above_nyquist_raises():
    arr = np.random.default_rng(0).standard_normal((24, 400))
    with pytest.raises(ValueError, match="Nyquist"):
        SSPSIR(n_components=2, sfreq=150.0, high_pass=100.0).fit(arr)


def test_sspsir_not_fitted_raises():
    with pytest.raises(NotFittedError):
        SSPSIR(n_components=2).transform(np.zeros((24, 100)))


def test_compute_sspsir_shape(eeg_info):
    from mne_denoise._leadfield import make_spherical_leadfield

    leadfield = make_spherical_leadfield(eeg_info, n_dipoles=300)
    u = np.linalg.svd(np.random.default_rng(0).standard_normal((24, 24)))[0][:, :2]
    operator = compute_sspsir(leadfield, u, M=20)
    assert operator.shape == (24, 24)


def test_compute_sspsir_channel_mismatch_raises():
    leadfield = np.random.default_rng(0).standard_normal((24, 100))
    bad = np.linalg.svd(np.random.default_rng(1).standard_normal((30, 30)))[0][:, :2]
    with pytest.raises(ValueError, match="artifact"):
        compute_sspsir(leadfield, bad, M=10)


def test_compute_sir_uses_shared_leadfield_validator(monkeypatch):
    import mne_denoise.sspsir as sspsir_core

    leadfield = np.random.default_rng(12).standard_normal((5, 8))
    seen = []
    original = sspsir_core._validate_leadfield

    def wrapped(value, **kwargs):
        seen.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(sspsir_core, "_validate_leadfield", wrapped)
    compute_sir(leadfield, M=3)

    assert len(seen) == 1
    assert seen[0] is leadfield


def test_compute_sspsir_rejects_nonorthonormal_subspace():
    leadfield = np.random.default_rng(11).standard_normal((5, 10))
    with pytest.raises(ValueError, match="orthonormal"):
        compute_sspsir(leadfield, np.ones((5, 2)), M=3)


@pytest.mark.parametrize(
    ("leadfield", "topographies", "M", "message"),
    [
        (np.ones((4, 3)), np.ones(4), 2, "artifact_topographies must be 2D"),
        (np.ones((4, 3)), np.full((4, 1), np.nan), 2, "finite"),
        (np.ones((4, 3)), np.empty((4, 0)), 2, "between 1"),
        (np.ones((4, 3)), np.eye(4)[:, :1], 0, "positive integer"),
    ],
)
def test_compute_sspsir_input_contracts(leadfield, topographies, M, message):
    with pytest.raises(ValueError, match=message):
        compute_sspsir(leadfield, topographies, M)


def test_compute_sir_is_rank_m_not_identity():
    """orig_data_SIR restricts to the M leading topographies; it is not I."""
    leadfield = np.random.default_rng(2).standard_normal((24, 100))
    leadfield -= leadfield.mean(axis=0, keepdims=True)
    operator = compute_sir(leadfield, M=15)
    assert np.linalg.matrix_rank(operator) == 15
    assert not np.allclose(operator, np.eye(24))


def test_compute_sir_rejects_invalid_m_or_zero_rank():
    with pytest.raises(ValueError, match="positive integer"):
        compute_sir(np.eye(3), M=1.5)
    with pytest.raises(ValueError, match="rank is zero"):
        compute_sir(np.zeros((3, 4)), M=1)


def test_sspsir_warns_when_m_exceeds_rank():
    leadfield = np.random.default_rng(3).standard_normal((24, 100))
    leadfield -= leadfield.mean(axis=0, keepdims=True)  # rank 23
    u = np.linalg.svd(np.random.default_rng(4).standard_normal((24, 24)))[0][:, :2]
    with pytest.warns(RuntimeWarning, match="exceeds the numerical rank"):
        operator = compute_sspsir(leadfield, u, M=24)
    assert np.isfinite(operator).all()


def test_sspsir_records_effective_m(tms_epochs):
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


def test_sspsir_array_with_forward(forward):
    arr = np.random.default_rng(7).standard_normal((24, 400))
    ss = SSPSIR(n_components=2, sfreq=1000.0, forward=forward).fit(arr)
    assert ss.transform(arr).shape == (24, 400)


def test_sspsir_mne_object_with_forward(tms_epochs, forward):
    """Epochs + an individual forward: the main real-world path."""
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2, forward=forward).fit(epochs)
    assert ss.leadfield_.shape == (24, forward["sol"]["data"].shape[1])
    assert ss.transform(epochs).get_data().shape == epochs.get_data().shape


def test_sspsir_array_forward_channel_mismatch_raises(forward):
    arr = np.random.default_rng(9).standard_normal((30, 400))
    with pytest.raises(ValueError, match="same number of"):
        SSPSIR(n_components=2, sfreq=1000.0, forward=forward).fit(arr)


def test_sspsir_array_without_forward_raises():
    arr = np.random.default_rng(8).standard_normal((24, 400))
    with pytest.raises(ValueError, match="channel positions"):
        SSPSIR(n_components=2, sfreq=1000.0).fit(arr)


def test_sspsir_exposes_mne_projections(tms_epochs):
    """Removed topographies are available as MNE Projection objects."""
    mne_mod = pytest.importorskip("mne")
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=3).fit(epochs)

    assert len(ss.projs_) == ss.n_components_
    assert all(isinstance(p, mne_mod.Projection) for p in ss.projs_)
    # Each projection carries the corresponding artifact topography.
    for i, proj in enumerate(ss.projs_):
        np.testing.assert_allclose(
            proj["data"]["data"].ravel(), ss.artifact_topographies_[:, i]
        )
        assert proj["data"]["col_names"] == epochs.ch_names
        assert not proj["active"]
    # They are accepted by MNE's own plotting entry point.
    mne_mod.viz.plot_projs_topomap(ss.projs_, epochs.info, show=False)


def test_sspsir_projs_empty_for_array_input(forward):
    """A plain array has no channel names, so no projections can be built."""
    arr = np.random.default_rng(11).standard_normal((24, 400))
    ss = SSPSIR(n_components=2, sfreq=1000.0, forward=forward).fit(arr)
    assert ss.projs_ == []


def test_sspsir_verbose_logs_fit_summary(tms_epochs, caplog):
    """The package logging convention reports what was removed."""
    import logging

    epochs = tms_epochs[0]
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        SSPSIR(n_components=3, verbose=True).fit(epochs)
    assert any(
        "SSP-SIR:" in r.message
        and "channels=" in r.message
        and "removed 3 artifact component(s)" in r.message
        for r in caplog.records
    )
