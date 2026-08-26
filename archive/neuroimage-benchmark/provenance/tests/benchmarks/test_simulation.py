"""Semi-synthetic simulation: same-A replicate, oracle, train-fitted source matching."""

import numpy as np
import pytest

from mne_denoise.benchmarks import simulation as sim


def test_same_A_within_replicate_X_equals_AS():
    rep = sim.simulate_replicate(regime="generic", n_channels=6, n_brain=3,
                                 n_artifact=3, n_train=1000, n_test=800, seed=3)
    np.testing.assert_allclose(rep.X_train, rep.A @ rep.S_train, atol=1e-9)
    np.testing.assert_allclose(rep.X_test, rep.A @ rep.S_test, atol=1e-9)
    assert rep.is_square  # 6 channels == 6 sources → Amari-valid


def test_distinct_A_across_replicates_same_within_seed():
    a1 = sim.simulate_replicate(seed=1).A
    a1b = sim.simulate_replicate(seed=1).A
    a2 = sim.simulate_replicate(seed=2).A
    np.testing.assert_allclose(a1, a1b)
    assert not np.allclose(a1, a2)


def test_oracle_is_brain_only_reconstruction():
    rep = sim.simulate_replicate(seed=5)
    orc = sim.oracle_reconstruction(rep.A, rep.S_train, rep.brain_idx)
    expected = rep.A[:, rep.brain_idx] @ rep.S_train[rep.brain_idx]
    np.testing.assert_allclose(orc, expected)
    assert orc.shape == rep.X_train.shape


def test_forward_regime_runs_and_is_smooth():
    rep = sim.simulate_replicate(regime="forward", n_channels=12, n_brain=6,
                                 n_artifact=6, n_train=500, n_test=500, seed=0)
    assert rep.A.shape == (12, 12)
    assert np.all(np.isfinite(rep.A))


def test_source_matching_fit_on_train_apply_to_test():
    rep = sim.simulate_replicate(regime="generic", n_channels=5, n_brain=3,
                                 n_artifact=2, n_train=3000, n_test=3000, seed=7)
    # "recovered" = a known permutation + sign + scale of the true train sources
    perm = np.array([2, 0, 4, 1, 3])
    signs = np.array([1, -1, 1, -1, 1.0])
    scales = np.array([2.0, 0.5, 1.5, 1.0, 3.0])
    recov_train = (rep.S_train[perm] * signs[:, None]) * scales[:, None]
    m = sim.fit_source_matching(recov_train, rep.S_train, min_corr=0.5)
    # every recovered row should be confidently matched
    assert m.unmatched == []
    assert np.all(m.matched_corr > 0.99)
    # applying the train-fitted sign+scale to a held-out segment removes them
    recov_test = (rep.S_test[perm] * signs[:, None]) * scales[:, None]
    aligned = sim.apply_source_matching(recov_test, m)
    # aligned[k] should match the true source rep.S_test[perm[k]] up to ~unit scale
    for k in range(5):
        a = aligned[k] - aligned[k].mean()
        t = rep.S_test[perm[k]] - rep.S_test[perm[k]].mean()
        corr = a @ t / (np.linalg.norm(a) * np.linalg.norm(t) + 1e-12)
        assert corr > 0.99


def test_simulate_rejects_too_few_channels():
    with pytest.raises(ValueError):
        sim.simulate_replicate(n_channels=3, n_brain=3, n_artifact=3)
