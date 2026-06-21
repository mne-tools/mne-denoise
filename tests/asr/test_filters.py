import numpy as np
import pytest

from mne_denoise.asr._filters import (
    _append_streaming_tail,
    _apply_statistics_filter,
    _apply_statistics_filter_streaming,
    _design_aasr_filter,
    _design_statistics_filter,
    _prepend_streaming_carry,
)


def test_design_statistics_filter() -> None:
    """Test that statistics filter design returns correct shapes."""
    # Test "none"
    b, a = _design_statistics_filter(250.0, "none")
    assert np.allclose(b, [1.0])
    assert np.allclose(a, [1.0])

    # Test "highpass" normal case
    b, a = _design_statistics_filter(250.0, "highpass")
    assert b.ndim == 1
    assert a.ndim == 1

    # Test cutoff bounding
    b, a = _design_statistics_filter(1.0, "highpass")  # cutoff = 0.1, nyq = 0.5
    assert b.ndim == 1

    # Test invalid kind
    with pytest.raises(ValueError, match="must be 'none', 'asr', or 'highpass'"):
        _design_statistics_filter(250.0, "invalid")


def test_apply_statistics_filter() -> None:
    """Test that offline filter application works."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))

    # Trivial case
    b, a = np.array([1.0]), np.array([1.0])
    out = _apply_statistics_filter(X, b, a)
    assert np.allclose(X, out)

    # Actual filter
    b, a = _design_statistics_filter(250.0, "highpass")
    out = _apply_statistics_filter(X, b, a)
    assert out.shape == X.shape
    # Ensure it's not identically zero or identical to X
    assert not np.allclose(X, out)
    assert not np.allclose(out, 0.0)


def test_append_streaming_tail() -> None:
    """Test lookahead tail reflection."""
    X = np.array([[1.0, 2.0, 3.0, 4.0]])

    # Trivial case
    out = _append_streaming_tail(X, 0)
    assert np.allclose(out, X)

    # Regular case
    out = _append_streaming_tail(X, 2)
    # Reflecting 4.0: [1.0, 2.0, 3.0, 4.0, 2*4.0 - 3.0, 2*4.0 - 2.0]
    # tail = [5.0, 6.0]
    assert np.allclose(out, [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    # Error case
    with pytest.raises(ValueError, match="more samples than the lookahead tail"):
        _append_streaming_tail(X, 5)


def test_prepend_streaming_carry() -> None:
    """Test initial carry reflection."""
    X = np.array([[1.0, 2.0, 3.0, 4.0]])

    # Trivial case
    out = _prepend_streaming_carry(X, 0)
    assert np.allclose(out, X)

    # Regular case
    out = _prepend_streaming_carry(X, 2)
    # Reflecting around 1.0: [2*1.0 - 3.0, 2*1.0 - 2.0, 1.0, 2.0, 3.0, 4.0]
    # carry = [-1.0, 0.0]
    assert np.allclose(out, [[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]])

    # Error case
    with pytest.raises(ValueError, match="more samples than the lookahead carry"):
        _prepend_streaming_carry(X, 5)


def test_apply_statistics_filter_streaming() -> None:
    """Test causal filtering."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))

    # Trivial case
    b, a = np.array([1.0]), np.array([1.0])
    out = _apply_statistics_filter_streaming(X, b, a)
    assert np.allclose(out, X)

    # Actual causal filtering
    b, a = _design_statistics_filter(250.0, "highpass")
    out = _apply_statistics_filter_streaming(X, b, a)
    assert out.shape == X.shape
    assert not np.allclose(X, out)


def test__design_aasr_filter() -> None:
    """Test Yule-Walker filter design for AASR."""
    b, a = _design_aasr_filter(250.0)
    # The order is 8, so coefficients should be length 9
    assert len(b) == 9
    assert len(a) == 9
    assert np.all(np.isfinite(b))
    assert np.all(np.isfinite(a))


def test_lfilter_channels() -> None:
    """Test stateful causal filtering across channels."""
    from mne_denoise.asr._filters import _lfilter_channels

    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    b = np.array([1.0, -0.5])
    a = np.array([1.0, 0.5])

    # Test without initial conditions
    out, zf = _lfilter_channels(X, b, a)
    assert out.shape == X.shape
    assert zf.shape == (3, 1)  # order is max(len(a), len(b)) - 1 = 1

    # Test with initial conditions
    zi = np.ones((3, 1))
    out2, zf2 = _lfilter_channels(X, b, a, zi=zi)
    assert out2.shape == X.shape
    assert zf2.shape == (3, 1)
    assert not np.allclose(out, out2)

    # Test invalid shape
    with pytest.raises(ValueError, match="Filter state shape does not match"):
        _lfilter_channels(X, b, a, zi=np.zeros((3, 2)))


def test_design_statistics_filter_cutoff_at_nyquist() -> None:
    """When cutoff >= nyquist, the filter should be a passthrough."""
    # sfreq=0.0: cutoff = min(0.5, 0.0)=0.0, nyq=0.0. 0.0 >= 0.0 -> passthrough
    b, a = _design_statistics_filter(0.0, "highpass")
    assert np.allclose(b, [1.0])
    assert np.allclose(a, [1.0])


def test_apply_statistics_filter_short_signal() -> None:
    """Short signals that can't support filtfilt should fallback to lfilter."""
    b, a = _design_statistics_filter(250.0, "highpass")
    # Create a signal shorter than the padding length (3 * max(len(a), len(b)))
    X_short = np.random.default_rng(42).standard_normal((2, 5))
    out = _apply_statistics_filter(X_short, b, a)
    assert out.shape == X_short.shape
    assert not np.allclose(out, X_short)


def test_lfilter_channels_zero_order() -> None:
    """Zero-order filter (scalar b and a) should be a passthrough."""
    from mne_denoise.asr._filters import _lfilter_channels

    X = np.random.default_rng(42).standard_normal((3, 50))
    b = np.array([2.0])
    a = np.array([1.0])
    out, zf = _lfilter_channels(X, b, a)
    # order = max(1,1)-1 = 0, so passthrough copy with empty zf
    assert np.allclose(out, X)
    assert zf.shape == (3, 0)


def test_polystab_complex() -> None:
    """Test _polystab with a complex polynomial."""
    import pytest

    from mne_denoise.asr._filters import _polystab

    a = np.array([1.0, -0.5j, 0.2])
    with pytest.warns(np.exceptions.ComplexWarning):
        b = _polystab(a)
    assert b.shape == a.shape


def test_yulewalk_duplicate_freqs() -> None:
    """Test yulewalk with duplicate frequencies."""
    from mne_denoise.asr._filters import _yulewalk

    F = np.array([0.0, 0.5, 0.5, 1.0])
    M = np.array([1.0, 1.0, 0.0, 0.0])
    b, a = _yulewalk(4, F, M)
    assert len(b) == 5
    assert len(a) == 5
