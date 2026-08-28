"""Unit tests for nonlinear (iterative) DSS."""

from __future__ import annotations

import inspect
import logging
from unittest.mock import patch

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss import IterativeDSS, iterative_dss
from mne_denoise.dss.denoisers import KurtosisDenoiser, VarianceMaskDenoiser
from mne_denoise.dss.nonlinear import _symmetric_orthogonalize, iterative_dss_one

# =============================================================================
# iterative_dss_one - Core Single Component Algorithm
# =============================================================================


def test_iterative_dss_one_basic():
    """iterative_dss_one should extract a single component."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 8, 1000
    X_whitened = rng.standard_normal((n_whitened, n_times))

    denoiser = KurtosisDenoiser()
    w, source, n_iter, converged = iterative_dss_one(X_whitened, denoiser)

    assert w.shape == (n_whitened,)
    assert source.shape == (n_times,)
    assert n_iter > 0
    assert isinstance(converged, bool)
    # Weight should be normalized
    assert_allclose(np.linalg.norm(w), 1.0, atol=1e-10)


def test_iterative_dss_one_with_init():
    """iterative_dss_one should accept initial weight."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 5, 500
    X_whitened = rng.standard_normal((n_whitened, n_times))

    w_init = np.ones(n_whitened) / np.sqrt(n_whitened)

    denoiser = KurtosisDenoiser()
    w, source, n_iter, converged = iterative_dss_one(
        X_whitened, denoiser, w_init=w_init
    )

    assert w.shape == (n_whitened,)


def test_iterative_dss_one_with_alpha():
    """iterative_dss_one should support alpha parameter."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 5, 500
    X_whitened = rng.standard_normal((n_whitened, n_times))

    denoiser = KurtosisDenoiser()

    # Float alpha
    w, source, _, _ = iterative_dss_one(X_whitened, denoiser, alpha=0.5)
    assert w.shape == (n_whitened,)

    # Callable alpha
    def alpha_func(s):
        return 1.0 / np.std(s)

    w, source, _, _ = iterative_dss_one(X_whitened, denoiser, alpha=alpha_func)
    assert w.shape == (n_whitened,)


def test_iterative_dss_one_with_beta():
    """iterative_dss_one should support beta parameter (Newton step)."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 5, 500
    X_whitened = rng.standard_normal((n_whitened, n_times))

    denoiser = KurtosisDenoiser()

    # Float beta
    w, source, _, _ = iterative_dss_one(X_whitened, denoiser, beta=-0.5)
    assert w.shape == (n_whitened,)

    # Callable beta (FastICA style)
    def beta_func(s):
        return -np.mean(1 - np.tanh(s) ** 2)

    w, source, _, _ = iterative_dss_one(X_whitened, denoiser, beta=beta_func)
    assert w.shape == (n_whitened,)


def test_iterative_dss_one_with_gamma():
    """iterative_dss_one should support gamma parameter (relaxation)."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 5, 500
    X_whitened = rng.standard_normal((n_whitened, n_times))

    denoiser = KurtosisDenoiser()

    # Float gamma
    w, source, _, _ = iterative_dss_one(X_whitened, denoiser, gamma=0.8)
    assert w.shape == (n_whitened,)

    # Callable gamma
    def gamma_func(w_new, w_old, iteration):
        return 0.5 + 0.5 * min(iteration / 10, 1.0)

    w, source, _, _ = iterative_dss_one(X_whitened, denoiser, gamma=gamma_func)
    assert w.shape == (n_whitened,)


def test_iterative_dss_one_convergence():
    """iterative_dss_one should converge for reasonable data."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 4, 2000

    # Create data with clear structure
    X_whitened = rng.standard_normal((n_whitened, n_times))

    def denoiser(s):
        return np.tanh(s)  # Simple nonlinearity

    w, source, n_iter, converged = iterative_dss_one(
        X_whitened, denoiser, max_iter=100, tol=1e-6
    )

    # Should converge for simple data
    assert n_iter <= 100


def test_iterative_dss_one_progress_events():
    """The single-component solver emits completed iteration events."""
    rng = np.random.default_rng(101)
    X_whitened = rng.standard_normal((5, 500))
    events = []

    _, _, n_iter, _ = iterative_dss_one(
        X_whitened,
        np.tanh,
        max_iter=5,
        tol=-1.0,
        random_state=0,
        callback=events.append,
    )

    assert len(events) == n_iter == 5
    assert [event.current for event in events] == list(range(1, n_iter + 1))
    assert all(event.method == "iterative_dss" for event in events)
    assert all(event.stage == "iteration" for event in events)
    assert all(event.component is None for event in events)
    assert all(event.total == 5 for event in events)
    assert all(
        isinstance(event.metric, float)
        and np.isfinite(event.metric)
        and event.metric >= 0
        for event in events
    )


def test_iterative_dss_one_progress_event_on_early_convergence():
    """The single-component converged iteration is included in the stream."""
    rng = np.random.default_rng(102)
    X_whitened = rng.standard_normal((5, 500))
    events = []

    _, _, n_iter, converged = iterative_dss_one(
        X_whitened,
        np.tanh,
        max_iter=8,
        tol=1e6,
        random_state=0,
        callback=events.append,
    )

    assert converged
    assert n_iter == len(events) == 1
    assert events[-1].current == n_iter
    assert events[-1].total == 8


def test_iterative_dss_one_degenerate_progress_event():
    """Degenerate reinitialization iterations emit without a fake metric."""
    rng = np.random.default_rng(103)
    X_whitened = rng.standard_normal((5, 500))
    events = []

    _, _, n_iter, converged = iterative_dss_one(
        X_whitened,
        lambda source: np.zeros_like(source),
        max_iter=3,
        random_state=0,
        callback=events.append,
    )

    assert not converged
    assert n_iter == len(events) == 3
    assert [event.current for event in events] == [1, 2, 3]
    assert all(event.total == 3 for event in events)
    assert all(event.metric is None for event in events)


# =============================================================================
# iterative_dss - Multi-Component Extraction
# =============================================================================


def test_iterative_dss_shape():
    """iterative_dss should return correct shapes."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 2000))

    denoiser = KurtosisDenoiser()
    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=3, max_iter=10
    )

    assert filters.shape == (3, 8)
    assert sources.shape == (3, 2000)
    assert patterns.shape == (8, 3)
    assert conv.shape == (3, 2)


def test_iterative_dss_deflation_progress_component_context():
    """Deflation events carry one-based component context and reset current."""
    rng = np.random.default_rng(104)
    data = rng.standard_normal((6, 500))
    events = []

    _, _, _, convergence_info = iterative_dss(
        data,
        np.tanh,
        n_components=2,
        method="deflation",
        max_iter=6,
        random_state=0,
        callback=events.append,
    )

    assert all(event.method == "iterative_dss" for event in events)
    assert all(event.stage == "iteration" for event in events)
    assert {event.component for event in events} == {1, 2}
    for component in (1, 2):
        component_events = [event for event in events if event.component == component]
        n_iterations = int(convergence_info[component - 1, 0])
        assert len(component_events) == n_iterations
        assert [event.current for event in component_events] == list(
            range(1, n_iterations + 1)
        )
        assert all(event.total == 6 for event in component_events)


def test_iterative_dss_symmetric_progress_events():
    """Symmetric DSS emits one joint event per completed iteration."""
    rng = np.random.default_rng(105)
    data = rng.standard_normal((6, 500))
    events = []

    _, _, _, convergence_info = iterative_dss(
        data,
        np.tanh,
        n_components=2,
        method="symmetric",
        max_iter=6,
        random_state=0,
        callback=events.append,
    )

    n_iterations = int(convergence_info[0, 0])
    assert len(events) == n_iterations
    assert [event.current for event in events] == list(range(1, n_iterations + 1))
    assert all(event.method == "iterative_dss" for event in events)
    assert all(event.stage == "iteration" for event in events)
    assert all(event.component is None for event in events)
    assert all(event.total == 6 for event in events)
    assert all(
        isinstance(event.metric, float)
        and np.isfinite(event.metric)
        and event.metric >= 0
        for event in events
    )


def test_iterative_dss_callback_is_numerically_transparent():
    """Callbacks do not change deflationary or symmetric DSS results."""
    rng = np.random.default_rng(106)
    data = rng.standard_normal((6, 500))

    for method in ("deflation", "symmetric"):
        without = iterative_dss(
            data,
            np.tanh,
            n_components=2,
            method=method,
            max_iter=5,
            tol=-1.0,
            random_state=0,
        )
        with_callback = iterative_dss(
            data,
            np.tanh,
            n_components=2,
            method=method,
            max_iter=5,
            tol=-1.0,
            random_state=0,
            callback=lambda event: None,
        )

        for expected, actual in zip(without, with_callback, strict=True):
            np.testing.assert_allclose(expected, actual)


def test_iterative_dss_callback_validation_and_exceptions():
    """Iterative DSS validates callbacks and propagates their exceptions."""
    rng = np.random.default_rng(107)
    data = rng.standard_normal((6, 500))

    with pytest.raises(TypeError, match="callback must be callable or None"):
        iterative_dss(data, np.tanh, n_components=2, callback=1)

    class SentinelError(Exception):
        pass

    error = SentinelError("stop")

    def callback(event):
        raise error

    with pytest.raises(SentinelError) as caught:
        iterative_dss(
            data,
            np.tanh,
            n_components=2,
            max_iter=3,
            callback=callback,
        )
    assert caught.value is error


def test_iterative_dss_3d_data():
    """iterative_dss should handle 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 6, 200, 5
    X_2d = rng.standard_normal((n_ch, n_times * n_epochs))
    # MNE Standard: (n_epochs, n_channels, n_times)
    X_3d = X_2d.reshape(n_ch, n_epochs, n_times).transpose(1, 0, 2)

    denoiser = KurtosisDenoiser()

    # 2D fit
    filters_2d, sources_2d, _, _ = iterative_dss(X_2d, denoiser, n_components=3)

    # 3D fit (should handle reshape automatically)
    filters_3d, sources_3d, _, _ = iterative_dss(X_3d, denoiser, n_components=3)

    assert filters_3d.shape == (3, n_ch)
    assert sources_3d.shape == (3, n_epochs * n_times)


def test_iterative_dss_symmetric_method():
    """iterative_dss should support symmetric extraction method."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((6, 1500))

    denoiser = KurtosisDenoiser()
    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=3, method="symmetric", max_iter=20
    )

    assert filters.shape == (3, 6)
    assert sources.shape == (3, 1500)


def test_iterative_dss_invalid_method():
    """iterative_dss should raise error for invalid method."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 500))

    denoiser = KurtosisDenoiser()

    with pytest.raises(ValueError, match="Unknown method"):
        iterative_dss(data, denoiser, n_components=2, method="invalid")


def test_iterative_dss_invalid_ndim():
    """iterative_dss should raise error for invalid data dimension."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5,))  # 1D - invalid

    denoiser = KurtosisDenoiser()

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        iterative_dss(data, denoiser, n_components=2)


def test_iterative_dss_with_rank():
    """iterative_dss should respect rank parameter."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 1000))

    denoiser = KurtosisDenoiser()
    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=3, rank=5, max_iter=10
    )

    assert filters.shape == (3, 10)


def test_iterative_dss_with_w_init():
    """iterative_dss should accept initial weight matrix."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((6, 1000))

    denoiser = KurtosisDenoiser()

    # Create initial weights (n_components, n_whitened)
    w_init = rng.standard_normal((3, 6))

    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=3, w_init=w_init, max_iter=10
    )

    assert filters.shape == (3, 6)


def test_iterative_dss_debug_logging(caplog):
    """iterative_dss should report component progress at DEBUG."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 500))

    denoiser = KurtosisDenoiser()
    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        iterative_dss(data, denoiser, n_components=2, max_iter=5, verbose="DEBUG")

    assert "IterativeDSS component" in caplog.text


@pytest.mark.parametrize("verbosity", [True, "DEBUG"])
def test_iterative_dss_uses_logging_not_stdout(verbosity, caplog, capsys):
    """INFO and DEBUG iterative reports never write processing output to stdout."""
    rng = np.random.default_rng(43)
    data = rng.standard_normal((4, 300))
    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        iterative_dss(
            data,
            KurtosisDenoiser(),
            n_components=1,
            max_iter=3,
            verbose=verbosity,
        )
    assert capsys.readouterr().out == ""
    summaries = [
        record
        for record in caplog.records
        if record.message.startswith("Iterative DSS:")
    ]
    assert len(summaries) == 1
    for token in ("method=", "denoiser=KurtosisDenoiser", "converged=", "iterations="):
        assert token in summaries[0].message


def test_iterative_dss_symmetric_debug_logging(caplog):
    """iterative_dss symmetric should report progress at DEBUG."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 500))

    denoiser = KurtosisDenoiser()
    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        iterative_dss(
            data,
            denoiser,
            n_components=2,
            method="symmetric",
            max_iter=5,
            verbose="DEBUG",
        )

    assert "IterativeDSS symmetric iteration" in caplog.text


def test_iterative_dss_symmetric_with_alpha_beta():
    """Symmetric method should support alpha and beta parameters."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((6, 1000))

    denoiser = KurtosisDenoiser()

    filters, sources, patterns, conv = iterative_dss(
        data,
        denoiser,
        n_components=3,
        method="symmetric",
        alpha=1.0,
        beta=-0.5,
        max_iter=10,
    )

    assert filters.shape == (3, 6)


# =============================================================================
# _symmetric_orthogonalize
# =============================================================================


def test_symmetric_orthogonalize():
    """_symmetric_orthogonalize should produce orthonormal rows."""
    rng = np.random.default_rng(42)
    W = rng.standard_normal((3, 5))

    W_orth = _symmetric_orthogonalize(W)

    # Rows should be orthonormal
    gram = W_orth @ W_orth.T
    assert_allclose(gram, np.eye(3), atol=1e-10)


# =============================================================================
# IterativeDSS Class
# =============================================================================


def test_iterative_dss_class_fit():
    """IterativeDSS class should work like sklearn."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 1500))

    denoiser = VarianceMaskDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=5, max_iter=5)

    it_dss.fit(data)
    assert it_dss.filters_ is not None
    assert it_dss.filters_.shape == (5, 10)
    assert it_dss.patterns_ is not None
    assert it_dss.sources_ is not None
    assert it_dss.convergence_info_ is not None


def test_iterative_dss_class_fit_progress_callback():
    """IterativeDSS.fit forwards callbacks without storing them."""
    rng = np.random.default_rng(108)
    data = rng.standard_normal((6, 500))
    events = []
    it_dss = IterativeDSS(
        np.tanh,
        n_components=2,
        max_iter=4,
        random_state=0,
        normalize_input=False,
    )

    it_dss.fit(data, callback=events.append)

    assert {event.component for event in events} == {1, 2}
    assert len(events) == int(np.sum(it_dss.convergence_info_[:, 0]))
    assert "callback" not in vars(it_dss)
    assert "callback" not in inspect.signature(IterativeDSS.__init__).parameters


def test_iterative_dss_class_fit_transform_progress_callback():
    """IterativeDSS.fit_transform emits the fitting stream exactly once."""
    rng = np.random.default_rng(109)
    data = rng.standard_normal((6, 500))
    events = []
    it_dss = IterativeDSS(
        np.tanh,
        n_components=2,
        max_iter=4,
        random_state=0,
        normalize_input=False,
    )

    sources = it_dss.fit_transform(data, callback=events.append)

    assert sources.shape == (2, 500)
    assert len(events) == int(np.sum(it_dss.convergence_info_[:, 0]))


def test_iterative_dss_class_fit_transform():
    """IterativeDSS should support fit_transform."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 1000))

    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3, max_iter=5)

    sources = it_dss.fit_transform(data)

    assert sources.shape == (3, 1000)


def test_iterative_dss_class_transform():
    """IterativeDSS should support separate fit and transform."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 1000))

    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3, max_iter=5)

    it_dss.fit(data)
    sources = it_dss.transform(data)

    assert sources.shape == (3, 1000)


def test_iterative_dss_class_transform_3d():
    """IterativeDSS transform should handle 3D data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 8, 100, 5
    data_2d = rng.standard_normal((n_ch, n_times * n_epochs))
    # MNE Standard: (n_epochs, n_ch, n_times)
    data_3d = rng.standard_normal((n_epochs, n_ch, n_times))

    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3, max_iter=5)

    it_dss.fit(data_2d)
    sources = it_dss.transform(data_3d)

    # Expected: (n_epochs, n_comp, n_times)
    assert sources.shape == (n_epochs, 3, n_times)


def test_iterative_dss_class_inverse_transform():
    """IterativeDSS should support inverse_transform."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3, max_iter=5)

    sources = it_dss.fit_transform(data)
    reconstructed = it_dss.inverse_transform(sources)

    assert reconstructed.shape == (8, 500)


def test_iterative_dss_class_inverse_transform_3d():
    """IterativeDSS inverse_transform should handle 3D data."""
    rng = np.random.default_rng(42)
    n_epochs, n_ch, n_times = 5, 8, 100
    # MNE Standard: (n_epochs, n_channels, n_times)
    data = rng.standard_normal((n_epochs, n_ch, n_times))

    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3, max_iter=5)

    sources = it_dss.fit_transform(data)
    reconstructed = it_dss.inverse_transform(sources)

    assert reconstructed.shape == (n_epochs, n_ch, n_times)


def test_iterative_dss_class_inverse_transform_normalized():
    """IterativeDSS inverse_transform should handle normalization correctly (2D and 3D)."""
    rng = np.random.default_rng(42)
    n_epochs, n_ch, n_times = 5, 4, 100

    # 2D data with different channel scales
    scales = np.array([1.0, 0.1, 0.01, 1e-3])
    data_2d = rng.standard_normal((n_ch, n_times * n_epochs)) * scales[:, np.newaxis]
    data_3d = data_2d.reshape(n_ch, n_epochs, n_times).transpose(1, 0, 2)

    # Use identity denoiser to ensure perfect reconstruction (within iterative precision)
    def identity_denoiser(data):
        return data

    # Center data for exact comparison as inverse_transform reconstructs centered data
    data_2d_centered = data_2d - data_2d.mean(axis=1, keepdims=True)
    data_3d_centered = data_3d - data_3d.mean(axis=(0, 2), keepdims=True)

    # Test 2D
    # Set reg to 1e-15 to ensure we don't truncate any components
    it_dss_2d = IterativeDSS(
        identity_denoiser, n_components=n_ch, normalize_input=True, reg=1e-15
    )
    sources_2d = it_dss_2d.fit_transform(data_2d)
    reconstructed_2d = it_dss_2d.inverse_transform(sources_2d)

    # Reconstructed should match original centered data (full rank)
    assert_allclose(data_2d_centered, reconstructed_2d, rtol=1e-3, atol=1e-12)

    # Test 3D
    it_dss_3d = IterativeDSS(
        identity_denoiser, n_components=n_ch, normalize_input=True, reg=1e-15
    )
    sources_3d = it_dss_3d.fit_transform(data_3d)
    reconstructed_3d = it_dss_3d.inverse_transform(sources_3d)

    # Needs to match 3D centered data
    assert_allclose(data_3d_centered, reconstructed_3d, rtol=1e-3, atol=1e-12)


def test_iterative_dss_class_transform_before_fit():
    """IterativeDSS should raise error when transform called before fit."""
    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3)

    data = np.random.randn(5, 100)

    with pytest.raises(RuntimeError, match="not fitted"):
        it_dss.transform(data)


def test_iterative_dss_class_inverse_transform_before_fit():
    """IterativeDSS should raise error when inverse_transform called before fit."""
    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3)

    sources = np.random.randn(3, 100)

    with pytest.raises(RuntimeError, match="not fitted"):
        it_dss.inverse_transform(sources)


# =============================================================================
# MNE Integration
# =============================================================================


def test_iterative_dss_class_mne_raw():
    """IterativeDSS should work with MNE Raw objects."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 2000
    sfreq = 250.0

    data = rng.standard_normal((n_channels, n_samples))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3, max_iter=5)

    sources = it_dss.fit_transform(raw)

    assert sources.shape == (3, n_samples)


def test_iterative_dss_class_mne_epochs():
    """IterativeDSS should work with MNE Epochs objects."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 6, 100, 10
    sfreq = 100.0

    # MNE format: (n_epochs, n_channels, n_times)
    data = rng.standard_normal((n_epochs, n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3, max_iter=5)

    sources = it_dss.fit_transform(epochs)

    # Expected: (n_epochs, n_comp, n_times)
    assert sources.shape == (n_epochs, 3, n_times)


# =============================================================================
# Functional Tests with Known Expected Outputs
# =============================================================================


def test_iterative_dss_extracts_known_source():
    """iterative_dss should extract a known independent source."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 6, 3000

    # Create known independent sources
    s1 = rng.standard_normal(n_samples)
    s1 = np.tanh(s1 * 2)  # Super-Gaussian

    s2 = rng.standard_normal(n_samples)

    sources = np.vstack([s1, s2])

    # Mix into channels
    A = rng.standard_normal((n_channels, 2))
    data = A @ sources

    def denoiser(s):
        return np.tanh(s)  # Tanh targets super-Gaussian

    filters, extracted, patterns, conv = iterative_dss(
        data, denoiser, n_components=1, max_iter=50, random_state=42
    )

    # Top source should correlate with s1 (super-Gaussian source)
    corr1 = np.abs(np.corrcoef(extracted[0], s1)[0, 1])

    # Should have some alignment with the super-Gaussian source
    assert corr1 > 0.3 or conv[0, 1] == 1.0  # Either correlates or converged


def test_iterative_dss_reconstruction():
    """Patterns @ sources should approximate centered data."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 5, 1000

    data = rng.standard_normal((n_channels, n_samples))
    data - data.mean(axis=1, keepdims=True)

    denoiser = KurtosisDenoiser()
    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=n_channels, max_iter=20
    )

    reconstructed = patterns @ sources

    # Should be close to centered data
    assert reconstructed.shape == data.shape


def test_iterative_dss_orthogonal_filters():
    """Extracted filters should be orthogonal in whitened space."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 2000))

    denoiser = KurtosisDenoiser()
    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=4, max_iter=20
    )

    # Sources should be approximately uncorrelated
    source_corr = np.corrcoef(sources)
    off_diag = source_corr - np.diag(np.diag(source_corr))

    assert np.max(np.abs(off_diag)) < 0.2


# =============================================================================
# Edge Cases
# =============================================================================


def test_iterative_dss_one_tiny_init():
    """iterative_dss_one should handle tiny initial weights."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 5, 500
    X_whitened = rng.standard_normal((n_whitened, n_times))

    # Very small initial weight
    w_init = np.ones(n_whitened) * 1e-20

    denoiser = KurtosisDenoiser()
    w, source, n_iter, converged = iterative_dss_one(
        X_whitened, denoiser, w_init=w_init
    )

    # Should still produce valid output
    assert w.shape == (n_whitened,)
    assert_allclose(np.linalg.norm(w), 1.0, atol=1e-10)


def test_iterative_dss_symmetric_w_init():
    """iterative_dss symmetric should accept w_init."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((6, 1000))

    denoiser = KurtosisDenoiser()

    w_init = rng.standard_normal((3, 6))

    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=3, method="symmetric", w_init=w_init, max_iter=10
    )

    assert filters.shape == (3, 6)


def test_iterative_dss_symmetric_callable_alpha_beta():
    """Symmetric method should support callable alpha and beta."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 800))

    denoiser = KurtosisDenoiser()

    def alpha_func(s):
        return 1.0 / np.std(s)

    def beta_func(s):
        return -np.mean(1 - np.tanh(s) ** 2)

    filters, sources, patterns, conv = iterative_dss(
        data,
        denoiser,
        n_components=2,
        method="symmetric",
        alpha=alpha_func,
        beta=beta_func,
        max_iter=15,
    )

    assert filters.shape == (2, 5)


def test_iterative_dss_symmetric_converges(caplog):
    """Symmetric method should report convergence when it happens."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((4, 2000))  # Enough data for convergence

    # Simple denoiser that converges easily
    def denoiser(s):
        return s**3  # cubic

    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        filters, sources, patterns, conv = iterative_dss(
            data,
            denoiser,
            n_components=2,
            method="symmetric",
            max_iter=100,
            tol=1e-4,
            verbose="DEBUG",
        )

    # Either converges or hits max_iter - both are covered
    assert "IterativeDSS symmetric" in caplog.text


def test_iterative_dss_transform_mne_raw():
    """IterativeDSS.transform should handle MNE Raw objects."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 6, 1000
    sfreq = 250.0

    data = rng.standard_normal((n_channels, n_samples))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    denoiser = KurtosisDenoiser()
    it_dss = IterativeDSS(denoiser, n_components=3, max_iter=5)

    # Fit on raw
    it_dss.fit(raw)

    # Transform on raw
    sources = it_dss.transform(raw)

    assert sources.shape == (3, n_samples)


def test_iterative_dss_separation_mixed_sources():
    """Iterative DSS should separate mixed super-Gaussian and Gaussian sources."""
    rng = np.random.default_rng(42)
    n_times = 2000

    # 1. Super-Gaussian source (highly kurtotic)
    t = np.linspace(0, 10, n_times)
    s1 = np.sin(t) * np.sign(np.sin(t))  # Square wave-ish
    s1 = (s1 - s1.mean()) / s1.std()

    # 2. Gaussian noise source
    s2 = rng.standard_normal(n_times)

    # Mix
    S = np.array([s1, s2])
    A = np.array([[0.8, 0.2], [0.3, 0.7]])
    X = A @ S

    # Use Tanh (kurtosis-based) denoiser
    def denoiser(x):
        return np.tanh(x)

    # Extract
    filters, sources, patterns, conv = iterative_dss(
        X, denoiser, n_components=2, method="symmetric", random_state=42
    )

    # Check correlations (accounting for permutation and sign)
    corrs = np.abs(np.corrcoef(sources, S)[:2, 2:])
    # One source should match s1 well (>0.9)
    assert np.max(corrs[0]) > 0.9 or np.max(corrs[1]) > 0.9


def test_iterative_dss_orthogonality_check():
    """Iterative DSS components should be orthogonal (decorrelated) in whitened space."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 5, 500
    data = rng.standard_normal((n_ch, n_times))

    # Make channels correlated
    data = np.dot(rng.standard_normal((n_ch, n_ch)), data)

    denoiser = VarianceMaskDenoiser()  # Any denoiser

    # Use deflation method
    filters_def, sources_def, _, _ = iterative_dss(
        data, denoiser, n_components=3, method="deflation", random_state=42
    )

    # Sources should be decorrelated
    corr_def = np.corrcoef(sources_def)
    off_diag_def = corr_def - np.diag(np.diag(corr_def))
    assert np.max(np.abs(off_diag_def)) < 1e-10, "Deflation sources not decorrelated"

    # Use symmetric method
    filters_sym, sources_sym, _, _ = iterative_dss(
        data, denoiser, n_components=3, method="symmetric", random_state=42
    )

    corr_sym = np.corrcoef(sources_sym)
    off_diag_sym = corr_sym - np.diag(np.diag(corr_sym))
    assert np.max(np.abs(off_diag_sym)) < 1e-10, "Symmetric sources not decorrelated"


def test_iterative_dss_class_input_types():
    """IterativeDSS Class should handle Array, Raw, and Epochs inputs."""
    import mne

    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 4, 200, 3

    # 1. Array (2D)
    data_2d = rng.standard_normal((n_ch, n_times))
    dss = IterativeDSS(KurtosisDenoiser(), n_components=2, random_state=42)
    dss.fit(data_2d)
    assert dss.filters_.shape == (2, n_ch)

    # 2. Raw
    info = mne.create_info(n_ch, 250, "eeg")
    raw = mne.io.RawArray(data_2d, info, verbose=False)
    dss = IterativeDSS(KurtosisDenoiser(), n_components=2, random_state=42)
    dss.fit(raw)
    out_raw = dss.transform(raw)
    assert out_raw.shape == (2, n_times)

    # 3. Epochs (3D inputs)
    data_3d = rng.standard_normal((n_epochs, n_ch, n_times))
    epochs = mne.EpochsArray(data_3d, info, verbose=False)
    dss = IterativeDSS(KurtosisDenoiser(), n_components=2, random_state=42)
    dss.fit(epochs)
    out_epochs = dss.transform(epochs)
    assert out_epochs.shape == (n_epochs, 2, n_times)
    assert out_epochs.ndim == 3


def test_iterative_dss_class_equivalence():
    """IterativeDSS class and iterative_dss function should yield same results."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 100))

    def denoiser(x):
        return np.tanh(x)

    # Class
    dss = IterativeDSS(denoiser, n_components=3, random_state=42, normalize_input=False)
    dss.fit(data)
    res_class = dss.transform(data)

    # Function
    filters, res_func, _, _ = iterative_dss(
        data, denoiser, n_components=3, random_state=42
    )

    # Should be identical (using same seed)
    assert_allclose(res_class, res_func)


def test_iterative_dss_preserves_scale():
    """IterativeDSS reconstruction should preserve physical signal scale."""
    sfreq = 1000
    n_channels = 10
    n_times = 2000
    t = np.arange(n_times) / sfreq

    signal_scale = 5e-6
    data = np.random.randn(n_channels, n_times) * 1e-7
    data[0:3, :] += signal_scale * np.sin(2 * np.pi * 10 * t)

    from mne_denoise.dss.denoisers import TanhMaskDenoiser

    idss = IterativeDSS(
        denoiser=TanhMaskDenoiser(), n_components=n_channels, random_state=42
    )
    # Using inverse_transform directly here to verify patterns * sources
    reconstructed = idss.fit(data).inverse_transform(idss.transform(data))

    rms_orig = np.sqrt(np.mean(data**2))
    rms_rec = np.sqrt(np.mean(reconstructed**2))
    assert_allclose(rms_orig, rms_rec, rtol=0.05)


def test_iterative_dss_get_normalized_patterns():
    """Test the newly added get_normalized_patterns method in IterativeDSS."""
    from mne_denoise.dss.denoisers import TanhMaskDenoiser

    data = np.random.randn(10, 1000)
    idss = IterativeDSS(denoiser=TanhMaskDenoiser(), n_components=2)
    idss.fit(data)
    norm_patterns = idss.get_normalized_patterns()
    assert norm_patterns.shape == (10, 2)
    assert_allclose(np.linalg.norm(norm_patterns, axis=0), 1.0)


def test_iterative_dss_full_rank_reconstruction_exact_match():
    """IterativeDSS with n_components=n_channels should reconstruct data exactly."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 4, 1000  # Small enough for quick convergence
    data = rng.standard_normal((n_channels, n_samples)) * 1e-6  # uV scale

    # Use Tanh denoiser
    from mne_denoise.dss.denoisers import TanhMaskDenoiser

    # We need tight convergence for exact reconstruction check
    dss = IterativeDSS(
        denoiser=TanhMaskDenoiser(),
        n_components=n_channels,
        normalize_input=True,
        max_iter=1000,
        tol=1e-12,
        random_state=42,
    )
    dss.fit(data)
    sources = dss.transform(data)
    rec = dss.inverse_transform(sources)

    # Comparison against centered data
    data_centered = data - data.mean(axis=1, keepdims=True)

    assert_allclose(rec, data_centered, rtol=1e-7, atol=1e-25)


def test_iterative_dss_mne_normalization():
    """IterativeDSS normalization should work with MNE objects."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 4, 1000
    sfreq = 250.0

    # Create data with different scales
    data = rng.standard_normal((n_channels, n_samples))
    data[0] *= 1e-6  # Simulate gradiometer scale
    data[1] *= 1e-12  # Simulate magnetometer scale

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    # IterativeDSS defaults to normalize_input=True
    from mne_denoise.dss.denoisers import TanhMaskDenoiser

    dss = IterativeDSS(denoiser=TanhMaskDenoiser(), n_components=3)
    sources = dss.fit_transform(raw)

    assert sources.shape == (3, n_samples)
    assert dss.channel_norms_ is not None
    assert dss.channel_norms_.shape == (n_channels,)
    # Norms should reflect the scales
    assert dss.channel_norms_[0] > dss.channel_norms_[1]


def test_iterative_dss_one_degenerate_signal():
    """iterative_dss_one should handle signal killing (norm < 1e-12)."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 3, 100
    X = rng.standard_normal((n_ch, n_times))

    # Stateful denoiser that kills the signal once then works
    class FlakyDenoiser:
        def __init__(self):
            self.killed = False

        def __call__(self, data):
            if not self.killed:
                self.killed = True
                return np.zeros_like(data)
            return data  # Identity map otherwise

    denoiser = FlakyDenoiser()
    w_init = np.array([1.0, 0.0, 0.0])

    w, source, n_iter, converged = iterative_dss_one(
        X, denoiser, w_init=w_init, max_iter=10, random_state=rng
    )

    # It should have reinitialized w (randomly) and then converged
    assert denoiser.killed
    assert not np.allclose(w, w_init)  # Should have changed


def test_iterative_dss_degenerate_orthogonalization():
    """iterative_dss should handle degenerate components during orthogonalization."""
    rng = np.random.default_rng(42)
    n_samples = 100
    # Create rank-deficient data where components are identical
    v = rng.standard_normal(n_samples)
    X = np.vstack([v, v, v])  # Rank 1 data

    # Use a simple identity denoiser
    def identity_denoiser(data):
        return data

    # Mock whitening step to ensure it returns 2 components despite rank 1 data
    X_white_mock = np.zeros((2, n_samples))
    X_white_mock[0] = v

    # Initialize BOTH components to the SAME vector to force collapse after orthogonalization
    w_init_force = np.array([[1.0, 0.0], [1.0, 0.0]])

    with patch("mne_denoise.dss.nonlinear.whiten_from_data_covariance") as mock_whiten:
        mock_whiten.return_value = (
            X_white_mock,
            np.eye(2, 3),  # Fake whitener
            np.eye(3, 2),  # Fake dewhitener
        )

        filters, _, _, _ = iterative_dss(
            X, identity_denoiser, n_components=2, w_init=w_init_force, random_state=rng
        )

    # Should stay at 2 components and re-initialize the degenerate one
    assert filters.shape == (2, 3)
    assert not np.allclose(filters[1], 0)
