"""Tests for the SOUND estimator."""

from __future__ import annotations

import logging

import mne
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from mne_denoise.sound import SOUND, _ddwiener, compute_sound, compute_sound_ref_best


@pytest.fixture(scope="module")
def eeg_info():
    ch = mne.channels.make_standard_montage("standard_1020").ch_names[:24]
    info = mne.create_info(ch, 500.0, "eeg")
    info.set_montage("standard_1020")
    return info


@pytest.fixture(scope="module")
def noisy_raw(eeg_info):
    """One badly corrupted channel over an otherwise clean dipolar source."""
    rng = np.random.default_rng(0)
    n_ch, n_t = 24, 1200
    topo = rng.standard_normal(n_ch)
    topo /= np.linalg.norm(topo)
    src = np.sin(2 * np.pi * 10 * np.arange(n_t) / 500.0)
    data = np.outer(topo, src) + 0.2 * rng.standard_normal((n_ch, n_t))
    data[7] += 5.0 * rng.standard_normal(n_t)  # corrupt channel 7
    data -= data.mean(axis=0, keepdims=True)  # average reference
    return mne.io.RawArray(data * 1e-6, eeg_info, verbose=False), 7


def test_sound_suppresses_noisy_channel(noisy_raw):
    raw, bad = noisy_raw
    sound = SOUND(n_iter=5, reference="average", random_state=0).fit(raw)
    cleaned = sound.transform(raw).get_data()
    before = raw.get_data()
    # The corrupted channel gets the largest noise estimate and is suppressed.
    assert int(np.argmax(sound.sigmas_)) == bad
    assert cleaned[bad].var() < before[bad].var()
    # Convergence improves over iterations.
    assert sound.convergence_[-1] < sound.convergence_[0]


def test_sound_ref_best_suppresses_noisy_channel(noisy_raw):
    """The default reference='best' path also cleans the bad channel."""
    raw, bad = noisy_raw
    sound = SOUND(n_iter=5, random_state=0).fit(raw)
    cleaned = sound.transform(raw).get_data()
    before = raw.get_data()
    assert cleaned[bad].var() < before[bad].var()
    # The bad channel is never chosen as the reference, and is dropped from
    # the noise estimate (so sigmas_ has one entry fewer than channels).
    assert sound.best_channel_ != bad
    assert sound.sigmas_.shape == (23,)
    # Output is average referenced regardless of the working reference.
    assert np.abs(cleaned.mean(axis=0)).max() < 1e-6 * np.abs(cleaned).max()


def test_sound_ref_best_matches_tesa_pipeline(eeg_info):
    """The folded operator equals the explicit tesa_sound reference path."""
    from mne_denoise._leadfield import make_spherical_leadfield

    rng = np.random.default_rng(11)
    leadfield = make_spherical_leadfield(eeg_info, n_dipoles=300)
    data = rng.standard_normal((24, 400))
    data -= data.mean(axis=0, keepdims=True)
    lambda_, n_iter, seed = 0.1, 4, 3

    operator, sigmas, _, best = compute_sound_ref_best(
        data, leadfield, lambda_=lambda_, n_iter=n_iter, random_state=seed
    )

    # Explicit tesa_sound path: SOUND on n-1 channels in best-channel
    # reference, then reconstruct through the average-referenced lead field.
    keep = np.array([i for i in range(24) if i != best])
    data_ref = (data - data[best])[keep]
    lf_ref = (leadfield - leadfield[best])[keep]
    _, sig_ref, _ = compute_sound(
        data_ref, lf_ref, lambda_=lambda_, n_iter=n_iter, random_state=seed
    )
    w = np.diag(1.0 / sig_ref)
    wl = w @ lf_ref
    wllw = wl @ wl.T
    lam = lambda_ * np.trace(wllw) / 23
    x = wl.T @ np.linalg.solve(wllw + lam * np.eye(23), w @ data_ref)
    expected = (leadfield - leadfield.mean(axis=0, keepdims=True)) @ x

    np.testing.assert_allclose(sigmas, sig_ref)
    np.testing.assert_allclose(operator @ data, expected, rtol=1e-9, atol=1e-12)


def test_sound_ref_best_too_few_channels_raises():
    with pytest.raises(ValueError, match="at least 4 channels"):
        compute_sound_ref_best(np.eye(3), np.eye(3))


def test_sound_warns_when_not_average_referenced(eeg_info):
    rng = np.random.default_rng(12)
    data = rng.standard_normal((24, 400)) + 10.0  # large common offset
    raw = mne.io.RawArray(data * 1e-6, eeg_info, verbose=False)
    with pytest.warns(RuntimeWarning, match="average referenced"):
        SOUND(n_iter=1, reference="average", random_state=0).fit(raw)


def test_sound_ref_best_does_not_warn_about_reference(eeg_info):
    """reference='best' handles the reference itself, so no warning."""
    import warnings

    rng = np.random.default_rng(13)
    data = rng.standard_normal((24, 400)) + 10.0
    raw = mne.io.RawArray(data * 1e-6, eeg_info, verbose=False)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        SOUND(n_iter=1, random_state=0).fit(raw)
    assert not any("average referenced" in str(w.message) for w in log)


def test_sound_tol_stops_early(eeg_info):
    """tol ends the iteration once sigmas stop moving (Mutanen 2022, step 3)."""
    from mne_denoise._leadfield import make_spherical_leadfield

    leadfield = make_spherical_leadfield(eeg_info, n_dipoles=300)
    data = np.random.default_rng(20).standard_normal((24, 500))
    data -= data.mean(axis=0, keepdims=True)

    _, _, fixed = compute_sound(data, leadfield, n_iter=25, random_state=0)
    _, _, stopped = compute_sound(data, leadfield, n_iter=25, tol=0.01, random_state=0)
    assert fixed.shape == (25,)
    assert stopped.size < fixed.size  # stopped early
    assert stopped[-1] < 0.01
    # Up to the stopping point the two runs are the same trajectory.
    np.testing.assert_allclose(stopped, fixed[: stopped.size])


def test_sound_tol_none_matches_reference_iteration_count(eeg_info):
    """Default tol=None runs exactly n_iter, as SOUND.m and TESA do."""
    from mne_denoise._leadfield import make_spherical_leadfield

    leadfield = make_spherical_leadfield(eeg_info, n_dipoles=300)
    data = np.random.default_rng(21).standard_normal((24, 400))
    _, _, convergence = compute_sound(data, leadfield, n_iter=7, random_state=0)
    assert convergence.shape == (7,)


def test_sound_rejects_bad_tol(eeg_info):
    with pytest.raises(ValueError, match="tol must be positive"):
        compute_sound(np.eye(6), np.eye(6), tol=0.0)


def test_sound_sigmas_independent_of_record_length_scaling():
    """The covariance-based noise update is an exact identity.

    Mutanen et al. (2022) Eqs. (35)-(36) replace the sample-wise residual with
    ``sqrt(w_N.T @ Cov(Y) @ w_N)``. Recomputing the residual explicitly must
    give the same sigmas.
    """
    rng = np.random.default_rng(22)
    n_ch, n_times = 12, 400
    leadfield = rng.standard_normal((n_ch, 60))
    leadfield -= leadfield.mean(axis=0, keepdims=True)
    data = rng.standard_normal((n_ch, n_times))
    data -= data.mean(axis=0, keepdims=True)

    _, sigmas, _ = compute_sound(data, leadfield, n_iter=3, random_state=0)

    # Independent check: the DDWiener seed via explicit time-domain residuals.
    cov = data @ data.T
    gamma = np.mean(np.diag(cov))
    expected = np.empty(n_ch)
    for i in range(n_ch):
        others = np.array([j for j in range(n_ch) if j != i])
        coef = np.linalg.solve(
            cov[np.ix_(others, others)] + gamma * np.eye(n_ch - 1), data[others]
        )
        pred = cov[i, others] @ coef
        expected[i] = np.sqrt(np.sum((data[i] - pred) ** 2) / n_times)
    np.testing.assert_allclose(_ddwiener(data), expected, rtol=1e-10)
    assert np.all(sigmas > 0)


def test_sound_runs_on_epochs(eeg_info):
    rng = np.random.default_rng(1)
    data = rng.standard_normal((6, 24, 300)) * 1e-6
    epochs = mne.EpochsArray(data, eeg_info, verbose=False)
    sound = SOUND(n_iter=2, random_state=0).fit(epochs)
    out = sound.transform(epochs)
    assert out.get_data().shape == (6, 24, 300)


def test_sound_sigma_source_evoked_vs_trials(eeg_info):
    """Both sigma sources run; 'evoked' is the tesa_sound default."""
    rng = np.random.default_rng(14)
    data = rng.standard_normal((8, 24, 300)) * 1e-6
    epochs = mne.EpochsArray(data, eeg_info, verbose=False)
    evoked_fit = SOUND(n_iter=2, sigma_source="evoked", random_state=0).fit(epochs)
    trials_fit = SOUND(n_iter=2, sigma_source="trials", random_state=0).fit(epochs)
    assert evoked_fit.sigmas_.shape == trials_fit.sigmas_.shape
    # They are genuinely different estimates, not an aliased code path.
    assert not np.allclose(evoked_fit.sigmas_, trials_fit.sigmas_)


def test_sound_rejects_bad_options(eeg_info):
    arr = np.random.default_rng(15).standard_normal((24, 200))
    with pytest.raises(ValueError, match="reference must be"):
        SOUND(reference="nope").fit(arr)
    with pytest.raises(ValueError, match="sigma_source must be"):
        SOUND(sigma_source="nope").fit(arr)


def test_sound_array_input_requires_forward(eeg_info):
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((24, 500))
    with pytest.raises(ValueError, match="channel positions"):
        SOUND().fit(arr)


def test_sound_array_input_with_leadfield(eeg_info):
    from mne_denoise._leadfield import make_spherical_leadfield

    leadfield = make_spherical_leadfield(eeg_info, n_dipoles=300)
    op, sigmas, conv = compute_sound(
        np.random.default_rng(3).standard_normal((24, 600)), leadfield, n_iter=2
    )
    assert op.shape == (24, 24)
    assert sigmas.shape == (24,)


def test_sound_not_fitted_raises():
    with pytest.raises(NotFittedError):
        SOUND().transform(np.zeros((24, 100)))


def test_compute_sound_channel_mismatch_raises():
    with pytest.raises(ValueError, match="channels"):
        compute_sound(np.zeros((10, 50)), np.zeros((8, 30)))


def test_compute_sound_uses_shared_leadfield_validator(monkeypatch):
    import mne_denoise.sound as sound_core

    data = np.random.default_rng(52).standard_normal((6, 50))
    leadfield = np.random.default_rng(53).standard_normal((6, 4))
    seen = []
    original = sound_core._validate_leadfield

    def wrapped(value, **kwargs):
        seen.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(sound_core, "_validate_leadfield", wrapped)
    compute_sound(data, leadfield, n_iter=1, random_state=0)

    assert len(seen) == 1
    assert seen[0] is leadfield


def test_compute_sound_too_few_channels_raises():
    with pytest.raises(ValueError, match="at least 3 channels"):
        compute_sound(np.zeros((2, 50)), np.zeros((2, 30)))


def test_compute_sound_all_zero_data_raises():
    """Degenerate input gives a clear error rather than silent NaNs."""
    with pytest.raises(ValueError, match="zero noise estimate"):
        compute_sound(np.zeros((6, 50)), np.zeros((6, 30)))


def test_ddwiener_flags_noisy_channel():
    rng = np.random.default_rng(4)
    data = rng.standard_normal((8, 400))
    data[3] += 6.0 * rng.standard_normal(400)
    sigmas = _ddwiener(data)
    assert int(np.argmax(sigmas)) == 3


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


def test_sound_array_with_forward(forward):
    arr = np.random.default_rng(5).standard_normal((24, 600))
    sound = SOUND(n_iter=2, forward=forward, random_state=0).fit(arr)
    assert sound.transform(arr).shape == (24, 600)


def test_sound_transform_checks_fitted_channel_count(forward):
    """SOUND rejects array transforms with a different channel count."""
    arr = np.random.default_rng(51).standard_normal((24, 600))
    sound = SOUND(n_iter=1, forward=forward, random_state=0).fit(arr)

    assert sound.transform(arr).shape == arr.shape
    with pytest.raises(
        ValueError, match="SOUND: X has 23 channels; fitted data had 24"
    ):
        sound.transform(arr[:-1])


def test_sound_mne_transform_uses_fitted_channel_order(noisy_raw, forward):
    """MNE transforms are extracted in the fitted channel order."""
    raw, _ = noisy_raw
    sound = SOUND(n_iter=1, forward=forward, random_state=0).fit(raw)
    reordered = raw.copy().reorder_channels(raw.ch_names[::-1])

    cleaned = sound.transform(reordered)
    np.testing.assert_allclose(
        cleaned.get_data()[::-1], sound.operator_ @ raw.get_data()
    )


def test_sound_mne_object_with_forward(noisy_raw, forward):
    """Raw + an individual forward: the main real-world path.

    Previously only the plain-array forward path was covered, so a regression
    in the MNE-object branch (which additionally aligns the forward's rows to
    the data by channel name) would have gone unnoticed.
    """
    raw, _ = noisy_raw
    sound = SOUND(n_iter=2, forward=forward, random_state=0).fit(raw)
    # The user's forward is used verbatim, not a spherical fallback.
    assert sound.leadfield_.shape == (24, forward["sol"]["data"].shape[1])
    cleaned = sound.transform(raw)
    assert cleaned.get_data().shape == raw.get_data().shape


def test_sound_forward_rows_follow_channel_order(eeg_info, forward):
    """Reordering the Raw's channels reorders the lead field to match."""
    names = list(eeg_info["ch_names"])
    flipped = mne.create_info(names[::-1], 500.0, "eeg")
    flipped.set_montage("standard_1020")
    rng = np.random.default_rng(30)
    data = rng.standard_normal((24, 400)) * 1e-6
    data -= data.mean(axis=0, keepdims=True)

    straight = SOUND(n_iter=1, forward=forward, random_state=0).fit(
        mne.io.RawArray(data, eeg_info, verbose=False)
    )
    reversed_ = SOUND(n_iter=1, forward=forward, random_state=0).fit(
        mne.io.RawArray(data[::-1], flipped, verbose=False)
    )
    np.testing.assert_allclose(
        reversed_.leadfield_, straight.leadfield_[::-1], atol=1e-12
    )


def test_sound_array_forward_channel_mismatch_raises(forward):
    arr = np.random.default_rng(6).standard_normal((30, 600))
    with pytest.raises(ValueError, match="same number of"):
        SOUND(forward=forward).fit(arr)


def test_sound_verbose_logs_fit_summary(noisy_raw, caplog):
    """The package logging convention reports the iteration outcome."""
    raw, _ = noisy_raw
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        SOUND(n_iter=2, random_state=0, verbose=True).fit(raw)
    summaries = [r for r in caplog.records if r.message.startswith("SOUND:")]
    assert len(summaries) == 1
    for token in (
        "2 iteration(s)",
        "channels=",
        "sources=",
        "final max relative sigma change",
        "reference=",
    ):
        assert token in summaries[0].message


def test_sound_ref_best_channel_mismatch_raises():
    """The ref_best path validates the lead field like the plain solver does."""
    with pytest.raises(ValueError, match="channels"):
        compute_sound_ref_best(np.zeros((10, 50)), np.zeros((8, 30)))


def test_sound_warns_on_custom_ref_applied(eeg_info):
    """A custom (non-average) reference is exactly what should warn.

    MNE sets ``custom_ref_applied`` when a *non*-average reference is applied
    and clears it for an average one, so a true flag means the montage is
    referenced to something ``reference='average'`` does not expect.
    """
    rng = np.random.default_rng(31)
    data = rng.standard_normal((24, 400))
    data -= data.mean(axis=0, keepdims=True)  # zero-mean: the numeric check stays quiet
    raw = mne.io.RawArray(data * 1e-6, eeg_info, verbose=False)
    with raw.info._unlock():
        raw.info["custom_ref_applied"] = True
    with pytest.warns(RuntimeWarning, match="average referenced"):
        SOUND(n_iter=1, reference="average", random_state=0).fit(raw)


def test_sound_average_referenced_input_is_quiet(eeg_info):
    """Genuinely average-referenced data must not warn."""
    import warnings

    rng = np.random.default_rng(32)
    data = rng.standard_normal((24, 400))
    data -= data.mean(axis=0, keepdims=True)
    raw = mne.io.RawArray(data * 1e-6, eeg_info, verbose=False)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        SOUND(n_iter=1, reference="average", random_state=0).fit(raw)
    assert not any("average referenced" in str(w.message) for w in log)


def _common_mode_raw(eeg_info, relative_offset, seed):
    """Zero-mean data plus a uniform common-mode term of a known relative size.

    Scaling to unit peak amplitude first makes the added constant *be* the
    relative offset the check computes, up to the ``1 + offset`` it adds to the
    peak. Returns the raw and the offset the estimator will actually see, so
    the test asserts against the implementation's own formula rather than
    against arithmetic repeated in the test.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((24, 400))
    data -= data.mean(axis=0, keepdims=True)
    data /= np.abs(data).max()
    data += relative_offset
    achieved = np.abs(data.mean(axis=0)).max() / np.abs(data).max()
    return mne.io.RawArray(data * 1e-6, eeg_info, verbose=False), achieved


def test_sound_common_mode_above_threshold_warns(eeg_info):
    """A common-mode term just over 1e-6 relative trips the check."""
    raw, achieved = _common_mode_raw(eeg_info, 1e-5, seed=33)
    assert achieved > 1e-6, f"test data does not straddle the threshold: {achieved}"
    with pytest.warns(RuntimeWarning, match="average referenced"):
        SOUND(n_iter=1, reference="average", random_state=0).fit(raw)


def test_sound_common_mode_below_threshold_is_quiet(eeg_info):
    """Rounding-level common mode is not a reference mismatch."""
    import warnings

    raw, achieved = _common_mode_raw(eeg_info, 1e-7, seed=34)
    assert achieved < 1e-6, f"test data does not straddle the threshold: {achieved}"
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        SOUND(n_iter=1, reference="average", random_state=0).fit(raw)
    assert not any("average referenced" in str(w.message) for w in log)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lambda_": -0.1}, "lambda_"),
        ({"lambda_": np.nan}, "lambda_"),
        ({"lambda_": True}, "lambda_"),
        ({"n_iter": 0}, "n_iter"),
        ({"n_iter": 1.5}, "n_iter"),
        ({"n_iter": True}, "n_iter"),
        ({"tol": 0.0}, "tol"),
        ({"tol": np.inf}, "tol"),
    ],
)
def test_compute_sound_validates_parameters(kwargs, match):
    rng = np.random.default_rng(35)
    with pytest.raises(ValueError, match=match):
        compute_sound(
            rng.standard_normal((4, 20)), rng.standard_normal((4, 8)), **kwargs
        )


@pytest.mark.parametrize(
    ("data", "leadfield", "message"),
    [
        (np.ones(20), np.ones((4, 3)), "data must be 2D"),
        (np.full((4, 20), np.nan), np.ones((4, 3)), "data.*finite"),
        (np.empty((4, 0)), np.ones((4, 3)), "at least one sample"),
        (np.zeros((4, 20)), np.ones((4, 3)), "all-zero"),
    ],
)
def test_compute_sound_input_contracts(data, leadfield, message):
    with pytest.raises(ValueError, match=message):
        compute_sound(data, leadfield)


def test_compute_sound_ref_best_shares_input_contracts():
    rng = np.random.default_rng(36)
    with pytest.raises(ValueError, match="n_iter"):
        compute_sound_ref_best(
            rng.standard_normal((4, 20)), rng.standard_normal((4, 8)), n_iter=0
        )
    with pytest.raises(ValueError, match="at least 4 channels"):
        compute_sound_ref_best(
            rng.standard_normal((3, 20)), rng.standard_normal((3, 8))
        )
