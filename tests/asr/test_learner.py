import numpy as np

from mne_denoise.asr._learner import (
    _AdaptiveSimilarityMatcher,
    _build_adaptive_learner,
)


def test_build_adaptive_learner():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    V = np.eye(3)

    learner = _build_adaptive_learner(
        X_filtered=X,
        V=V,
        variant="psw",
        learning_rate=0.01,
        tau=10.0,
        regularization=1e-5,
    )

    assert isinstance(learner, _AdaptiveSimilarityMatcher)
    assert learner.W.shape == (3, 3)
    assert learner.M.shape == (3, 3)
    assert learner.Minv.shape == (3, 3)

    # Check that M and Minv are inverse pairs
    assert np.allclose(learner.M @ learner.Minv, np.eye(3))


def test_adaptive_learner_fit_next():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    V = np.eye(3)

    learner = _build_adaptive_learner(
        X_filtered=X,
        V=V,
        variant="psw",
        learning_rate=0.01,
        tau=10.0,
        regularization=1e-5,
    )

    # Process a new chunk
    X_new = rng.standard_normal((3, 50))
    y = learner.fit_next(X_new)

    assert y.shape == (3, 50)

    # State should have evolved
    components = learner.get_components()
    assert components.shape == (3, 3)

    # The components should be orthonormal basis vectors
    assert np.allclose(components.T @ components, np.eye(3))

    # Check "non-psw" variant
    learner2 = _build_adaptive_learner(
        X_filtered=X,
        V=V,
        variant="other",
        learning_rate=0.01,
        tau=10.0,
        regularization=1e-5,
    )
    learner2.fit_next(X_new)
    assert learner2.M.shape == (3, 3)

    # Force LinAlgError inside fit_next using mock
    import unittest.mock

    learner3 = _build_adaptive_learner(
        X_filtered=X,
        V=V,
        variant="other",
        learning_rate=0.01,
        tau=10.0,
        regularization=1e-5,
    )
    original_inv = np.linalg.inv
    side_effects = [np.linalg.LinAlgError("singular matrix"), original_inv]
    with unittest.mock.patch(
        "mne_denoise.asr._learner.np.linalg.inv", side_effect=side_effects
    ):
        y3 = learner3.fit_next(X_new)
    assert y3.shape == (3, 50)
