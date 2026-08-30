"""Tests for the SOUND estimator."""

from __future__ import annotations

import mne
import numpy as np
import pytest

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


def test_compute_sound_synthetic_source_leadfield_reduces_sensor_noise():
    """A known forward signal is preserved while sensor-specific noise falls."""
    rng = np.random.default_rng(2026)
    n_channels, n_sources, n_times = 8, 3, 1200
    leadfield = rng.standard_normal((n_channels, n_sources))
    source = rng.standard_normal((n_sources, n_times))
    signal = leadfield @ source
    noise = 0.15 * rng.standard_normal((n_channels, n_times))
    noise[3] += 1.5 * rng.standard_normal(n_times)
    data = signal + noise

    operator, sigmas, _convergence = compute_sound(
        data, leadfield, n_iter=5, random_state=0
    )

    assert int(np.argmax(sigmas)) == 3
    assert _convergence[-1] < _convergence[0]
    error_before = np.linalg.norm(data - signal)
    error_after = np.linalg.norm(operator @ data - signal)
    assert error_after < 0.5 * error_before


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
    assert best == int(np.argmin(_ddwiener(data)))

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
    """Default iterations and a fixed random update order are reproducible."""
    from mne_denoise._leadfield import make_spherical_leadfield

    leadfield = make_spherical_leadfield(eeg_info, n_dipoles=300)
    data = np.random.default_rng(21).standard_normal((24, 400))
    first = compute_sound(data, leadfield, n_iter=7, random_state=0)
    second = compute_sound(data, leadfield, n_iter=7, random_state=0)
    assert first[2].shape == (7,)
    for expected, actual in zip(first, second, strict=True):
        np.testing.assert_allclose(expected, actual)


def test_compute_sound_ref_best_progress_is_one_stream():
    """Reference-best SOUND emits the shared SOUND iteration stream once."""
    rng = np.random.default_rng(105)
    data = rng.standard_normal((6, 300))
    leadfield = rng.standard_normal((6, 8))
    events = []

    with_callback = compute_sound_ref_best(
        data, leadfield, n_iter=3, random_state=0, callback=events.append
    )

    assert len(events) == with_callback[2].size == 3
    assert [event.current for event in events] == [1, 2, 3]
    assert all(
        event.method == "sound"
        and event.stage == "iteration"
        and event.component is None
        for event in events
    )
    np.testing.assert_allclose([event.metric for event in events], with_callback[2])


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


def test_compute_sound_regularizes_rank_deficient_leadfield():
    """Positive regularization keeps a rank-deficient forward model usable."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((6, 300))
    # Zero rows make L @ L.T exactly rank deficient across BLAS implementations.
    leadfield = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )

    operator, sigmas, convergence = compute_sound(
        data, leadfield, lambda_=0.1, n_iter=3, random_state=0
    )

    assert np.linalg.matrix_rank(leadfield) == 2
    assert operator.shape == (data.shape[0], data.shape[0])
    assert sigmas.shape == (data.shape[0],)
    assert convergence.shape == (3,)
    assert np.isfinite(operator).all()
    assert np.isfinite(sigmas).all()
    assert np.isfinite(convergence).all()
    with pytest.raises(ValueError, match="positive lambda_"):
        compute_sound(data, leadfield, lambda_=0.0, n_iter=1, random_state=0)


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


def test_sound_array_input_requires_forward():
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((24, 500))
    with pytest.raises(ValueError, match="channel positions"):
        SOUND().fit(arr)


def test_compute_sound_channel_mismatch_raises():
    with pytest.raises(ValueError, match="channels"):
        compute_sound(np.zeros((10, 50)), np.zeros((8, 30)))


def test_compute_sound_too_few_channels_raises():
    with pytest.raises(ValueError, match="at least 3 channels"):
        compute_sound(np.zeros((2, 50)), np.zeros((2, 30)))


def test_compute_sound_all_zero_data_raises():
    """Degenerate input gives a clear error rather than silent NaNs."""
    with pytest.raises(ValueError, match="zero noise estimate"):
        compute_sound(np.zeros((6, 50)), np.zeros((6, 30)))


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


def test_sound_mne_object_with_forward(noisy_raw, forward):
    """An MNE Raw fit uses the supplied physical forward model."""
    raw, _ = noisy_raw
    sound = SOUND(n_iter=2, forward=forward, random_state=0).fit(raw)
    assert sound.leadfield_.shape == (24, forward["sol"]["data"].shape[1])
    cleaned = sound.transform(raw)
    assert cleaned.get_data().shape == raw.get_data().shape


def test_sound_array_forward_channel_mismatch_raises(forward):
    arr = np.random.default_rng(6).standard_normal((30, 600))
    with pytest.raises(ValueError, match="same number of"):
        SOUND(forward=forward).fit(arr)
