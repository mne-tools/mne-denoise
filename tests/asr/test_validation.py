import numpy as np
import pytest

from mne_denoise.asr._validation import (
    _check_enough_samples,
    _check_transform_channels,
    _resolve_max_dims,
    _resolve_max_dims_padded,
    _round_half_up,
    _validate_adaptive_params,
    _validate_array_2d,
    _validate_backend_params,
    _validate_common_params,
)


def test_validate_common_params():
    """Test param guard rails."""
    valid_params = {
        "sfreq": 100.0,
        "cutoff": 5.0,
        "window_length": 0.5,
        "window_overlap": 0.66,
        "max_dropout_fraction": 0.1,
        "min_clean_fraction": 0.25,
        "regularization": 0.01,
    }
    _validate_common_params(**valid_params)

    with pytest.raises(ValueError, match="sfreq must be positive"):
        _validate_common_params(**{**valid_params, "sfreq": 0})
    with pytest.raises(ValueError, match="cutoff must be positive"):
        _validate_common_params(**{**valid_params, "cutoff": -1})
    with pytest.raises(ValueError, match="window_length must be positive"):
        _validate_common_params(**{**valid_params, "window_length": 0})
    with pytest.raises(ValueError, match="window_overlap must be in"):
        _validate_common_params(**{**valid_params, "window_overlap": 1.0})
    with pytest.raises(ValueError, match="max_dropout_fraction must be in"):
        _validate_common_params(**{**valid_params, "max_dropout_fraction": -0.1})
    with pytest.raises(ValueError, match="min_clean_fraction must be in"):
        _validate_common_params(**{**valid_params, "min_clean_fraction": 0})
    with pytest.raises(ValueError, match="must be less than 1"):
        _validate_common_params(
            **{**valid_params, "max_dropout_fraction": 0.5, "min_clean_fraction": 0.5}
        )
    with pytest.raises(ValueError, match="regularization must be positive"):
        _validate_common_params(**{**valid_params, "regularization": -0.01})


def test_validate_array_2d():
    """Test 2D array validation and zero-variance/nan guards."""
    X = np.random.randn(4, 1000)
    X_out = _validate_array_2d(X)
    assert X_out.shape == X.shape

    with pytest.raises(ValueError, match="expects a 2D array"):
        _validate_array_2d(np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="at least two channels"):
        _validate_array_2d(np.zeros((1, 10)))

    X_nan = X.copy()
    X_nan[0, :5] = np.nan
    X_out = _validate_array_2d(X_nan)
    assert not np.isnan(X_out).any()

    X_too_many_nans = X.copy()
    X_too_many_nans[0, :50] = np.nan
    with pytest.raises(ValueError, match="too many non-finite samples"):
        _validate_array_2d(X_too_many_nans)

    X_zero_var = X.copy()
    X_zero_var[0, :] = 5.0
    with pytest.raises(ValueError, match="zero or near-zero variance: \\[0\\]"):
        _validate_array_2d(X_zero_var)

    with pytest.raises(
        ValueError, match="All channels have zero or near-zero variance"
    ):
        _validate_array_2d(np.zeros((4, 1000)))


def test_check_enough_samples():
    """Test window length sample checks."""
    _check_enough_samples(100, 100.0, 0.5)

    with pytest.raises(ValueError, match="too short"):
        _check_enough_samples(100, 100.0, 0.01)

    with pytest.raises(ValueError, match="exceeds data length"):
        _check_enough_samples(10, 100.0, 0.5)


def test_round_half_up():
    assert _round_half_up(2.4) == 2
    assert _round_half_up(2.5) == 3
    assert _round_half_up(2.6) == 3
    assert _round_half_up(3.5) == 4  # numpy would round to 4
    assert _round_half_up(4.5) == 5  # numpy would round to 4


def test_resolve_max_dims_padded():
    """Test padded max_dims resolver."""
    assert _resolve_max_dims_padded(0.5, 10) == 5
    assert _resolve_max_dims_padded(1.5, 10) == 1  # truncated to int(1.5)
    assert _resolve_max_dims_padded(3, 10) == 3
    assert _resolve_max_dims_padded(12, 10) == 10

    with pytest.raises(ValueError, match="non-negative"):
        _resolve_max_dims_padded(-0.1, 10)
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_max_dims_padded(-1, 10)


def test_resolve_max_dims_standard():
    """Test standard max_dims resolver."""
    assert _resolve_max_dims(0.5, 10) == 5
    assert _resolve_max_dims(3, 10) == 3

    with pytest.raises(ValueError, match="float max_dims"):
        _resolve_max_dims(1.5, 10)
    with pytest.raises(ValueError, match="integer max_dims"):
        _resolve_max_dims(20, 10)


def test_validate_backend_params():
    """Test validation of backend configuration parameters."""
    _validate_backend_params(
        method="standard",
        experimental=False,
        lookahead=0.5,
        stepsize=32,
        window_criterion=0.5,
    )

    _validate_backend_params(
        method="riemannian_windowed",
        experimental=False,
        lookahead=None,
        stepsize=None,
        window_criterion=None,
    )

    _validate_backend_params(
        method="riemannian",
        experimental=True,
        lookahead=0.5,
        stepsize=32,
        window_criterion=0.5,
    )

    with pytest.raises(NotImplementedError, match="Supported methods are"):
        _validate_backend_params(
            method="unknown",
            experimental=False,
            lookahead=None,
            stepsize=None,
            window_criterion=None,
        )

    with pytest.raises(ValueError, match="'riemannian' is experimental"):
        _validate_backend_params(
            method="riemannian",
            experimental=False,
            lookahead=None,
            stepsize=None,
            window_criterion=None,
        )

    with pytest.raises(ValueError, match="lookahead must be non-negative"):
        _validate_backend_params(
            method="standard",
            experimental=False,
            lookahead=-0.1,
            stepsize=None,
            window_criterion=None,
        )

    with pytest.raises(ValueError, match="stepsize must be at least 1 sample"):
        _validate_backend_params(
            method="standard",
            experimental=False,
            lookahead=None,
            stepsize=0,
            window_criterion=None,
        )

    with pytest.raises(ValueError, match="window_criterion must be numeric or None"):
        _validate_backend_params(
            method="standard",
            experimental=False,
            lookahead=None,
            stepsize=None,
            window_criterion="invalid",  # type: ignore
        )


def test_check_transform_channels():
    """Test channel count and name validation against fitted state."""
    _check_transform_channels(3, ["A", "B", "C"], 3, ["A", "B", "C"])
    _check_transform_channels(3, None, 3, None)

    with pytest.raises(ValueError, match="Input channel count does not match"):
        _check_transform_channels(3, ["A", "B", "C"], 2, ["A", "B"])

    with pytest.raises(ValueError, match="Input channel names/order do not match"):
        _check_transform_channels(3, ["A", "B", "C"], 3, ["C", "B", "A"])


def test_validate_adaptive_params():
    """Test validation of adaptive ASR parameters."""
    # Valid parameters
    _validate_adaptive_params(
        variant="psp",
        update_window_length=1.0,
        calibration_window_length=1.0,
        calibration_window_overlap=0.5,
        ref_max_bad_channels=0.2,
        learning_rate=0.01,
        tau=10.0,
        mw_window_length=1.0,
        mw_mode="final_state",
    )

    with pytest.raises(ValueError, match="variant must be 'psp', 'psw', or 'mw'"):
        _validate_adaptive_params(
            "invalid", 1.0, 1.0, 0.5, 0.2, 0.01, 10.0, 1.0, "final_state"
        )

    with pytest.raises(ValueError, match="update_window_length must be positive"):
        _validate_adaptive_params(
            "psp", -1.0, 1.0, 0.5, 0.2, 0.01, 10.0, 1.0, "final_state"
        )

    with pytest.raises(ValueError, match="clean_window_length must be positive"):
        _validate_adaptive_params(
            "psp", 1.0, -1.0, 0.5, 0.2, 0.01, 10.0, 1.0, "final_state"
        )

    with pytest.raises(ValueError, match="clean_window_overlap must be in"):
        _validate_adaptive_params(
            "psp", 1.0, 1.0, 1.5, 0.2, 0.01, 10.0, 1.0, "final_state"
        )

    with pytest.raises(ValueError, match="clean_max_bad_channels must be non-negative"):
        _validate_adaptive_params(
            "psp", 1.0, 1.0, 0.5, -0.2, 0.01, 10.0, 1.0, "final_state"
        )

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        _validate_adaptive_params(
            "psp", 1.0, 1.0, 0.5, 0.2, 0.0, 10.0, 1.0, "final_state"
        )

    with pytest.raises(ValueError, match="tau must be positive"):
        _validate_adaptive_params(
            "psp", 1.0, 1.0, 0.5, 0.2, 0.01, -10.0, 1.0, "final_state"
        )

    with pytest.raises(
        ValueError, match="mw_window_length must be positive for variant='mw'"
    ):
        _validate_adaptive_params(
            "mw", 1.0, 1.0, 0.5, 0.2, 0.01, 10.0, 0.0, "final_state"
        )

    with pytest.raises(ValueError, match="mw_mode must be 'final_state' or 'sliding'"):
        _validate_adaptive_params("mw", 1.0, 1.0, 0.5, 0.2, 0.01, 10.0, 1.0, "invalid")


def test_validate_juggler_params():
    import pytest

    from mne_denoise.asr._validation import _validate_juggler_params

    # Valid parameters should pass silently
    _validate_juggler_params(
        strategy="dbscan",
        dbscan_top_k=5,
        gev_grid_size=2048,
        min_reference_fraction=0.05,
    )
    _validate_juggler_params(
        strategy="gev",
        dbscan_top_k=1,
        gev_grid_size=32,
        min_reference_fraction=0.99,
    )

    # Invalid strategy
    with pytest.raises(ValueError, match="strategy must be 'dbscan' or 'gev'"):
        _validate_juggler_params(
            strategy="invalid",
            dbscan_top_k=5,
            gev_grid_size=2048,
            min_reference_fraction=0.05,
        )

    # Invalid dbscan_top_k
    with pytest.raises(ValueError, match="dbscan_top_k must be at least 1"):
        _validate_juggler_params(
            strategy="dbscan",
            dbscan_top_k=0,
            gev_grid_size=2048,
            min_reference_fraction=0.05,
        )

    # Invalid gev_grid_size
    with pytest.raises(ValueError, match="gev_grid_size must be at least 32"):
        _validate_juggler_params(
            strategy="dbscan",
            dbscan_top_k=5,
            gev_grid_size=31,
            min_reference_fraction=0.05,
        )

    # Invalid min_reference_fraction
    with pytest.raises(
        ValueError, match="min_reference_fraction must be in \\(0, 1\\)"
    ):
        _validate_juggler_params(
            strategy="dbscan",
            dbscan_top_k=5,
            gev_grid_size=2048,
            min_reference_fraction=1.0,
        )


def test_asr_validation_errors(synthetic_burst_data):
    import pytest

    from mne_denoise.asr import ASR, AdaptiveASR, JugglerASR

    data, _, _, sfreq = synthetic_burst_data
    with pytest.raises(ValueError, match="experimental=True"):
        ASR(sfreq=sfreq, method="riemannian").fit(data)
    with pytest.raises(ValueError, match="strategy"):
        JugglerASR(sfreq=sfreq, strategy="unknown").fit(data)
    with pytest.raises(ValueError, match="variant"):
        AdaptiveASR(sfreq=sfreq, variant="unknown").fit(data)
    with pytest.raises(NotImplementedError, match="Supported methods"):
        ASR(sfreq=sfreq, method="unknown").fit(data)
    with pytest.raises(ValueError, match="sfreq"):
        ASR().fit(data)
    with pytest.raises(ValueError, match="at least two channels"):
        ASR(sfreq=sfreq).fit(data[:1])
    with pytest.raises(RuntimeError, match="not fitted"):
        ASR(sfreq=sfreq).transform(data)


def test_asr_unknown_method_raises():
    import numpy as np
    import pytest

    from mne_denoise.asr import ASR

    rng = np.random.default_rng(42)
    with pytest.raises(NotImplementedError, match="Supported methods"):
        ASR(sfreq=250.0, method="bogus", verbose=False).fit(
            rng.standard_normal((8, 2000))
        )


def test_asr_riemannian_requires_experimental():
    import numpy as np
    import pytest

    from mne_denoise.asr import ASR

    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="experimental"):
        ASR(sfreq=250.0, method="riemannian", verbose=False).fit(
            rng.standard_normal((8, 2000))
        )


def test_get_rejection_mask_without_window_criterion_raises():
    import numpy as np
    import pytest

    from mne_denoise.asr import ASR

    rng = np.random.default_rng(42)
    asr = ASR(sfreq=250.0, cutoff=20.0, verbose=False)
    asr.fit_transform(rng.standard_normal((8, 2000)))
    with pytest.raises(RuntimeError, match="rejection mask"):
        asr.get_rejection_mask()


def test_to_annotations_bad_kind_raises():
    import numpy as np
    import pytest

    from mne_denoise.asr import ASR

    rng = np.random.default_rng(42)
    asr = ASR(sfreq=250.0, cutoff=20.0, verbose=False)
    asr.fit_transform(rng.standard_normal((8, 2000)))
    with pytest.raises(ValueError, match="kind must be"):
        asr.to_annotations("bogus")


def test_to_annotations_calibration_on_window_backend_raises():
    import numpy as np
    import pytest

    from mne_denoise.asr import ASR

    rng = np.random.default_rng(42)
    asr = ASR(sfreq=250.0, cutoff=20.0, verbose=False)
    asr.fit_transform(rng.standard_normal((8, 2000)))
    with pytest.raises(RuntimeError, match="sample-based"):
        asr.to_annotations("calibration")
