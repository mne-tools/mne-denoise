"""Comparator contract + leakage-barrier tests."""

import numpy as np
import pytest

from mne_denoise.benchmarks import comparators as C


def _data(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((8, 500))


def test_registry_has_trivial_comparators():
    assert "none" in C.available()
    assert "pca_reconstruct" in C.available()


def test_none_is_identity_roundtrip():
    x = _data()
    comp = C.get("none")
    state = comp.fit(x)
    res = comp.transform(x, state)
    assert res.status == "success"
    np.testing.assert_allclose(np.asarray(res.cleaned), x)
    assert res.runtime_seconds is not None and res.cpu_seconds is not None


def test_pca_fit_then_transform_reduces_rank():
    train, evl = _data(1), _data(2)
    comp = C.get("pca_reconstruct", n_components=3)
    state = comp.fit(train)
    res = comp.transform(evl, state)
    assert res.status == "success"
    assert res.rank_after == 3
    assert np.asarray(res.cleaned).shape == evl.shape


def test_transform_without_fit_state_raises():
    comp = C.get("pca_reconstruct", n_components=2)
    with pytest.raises(RuntimeError):
        comp.transform(_data(), state=None)  # leakage barrier: must fit first


def test_fit_signature_excludes_eval():
    # the contract: fit(train, ctx) — there is no `eval` parameter to peek at.
    import inspect

    params = list(inspect.signature(C.Comparator.fit).parameters)
    assert "evaluation" not in params and "eval" not in params


def test_comparator_result_rejects_bad_status():
    with pytest.raises(ValueError):
        C.ComparatorResult(status="totally_made_up")


def test_comparator_result_summary_is_scalar_only():
    res = C.ComparatorResult(cleaned=np.zeros((2, 2)), rank_after=2)
    s = res.summary()
    assert "cleaned" not in s and s["rank_after"] == 2


def test_meta_reference_aware_sets_tier():
    m = C.ComparatorMeta("x", reference_aware=True)
    assert m.information_tier == "reference_aware"


def test_meta_bad_fit_scope_raises():
    with pytest.raises(ValueError):
        C.ComparatorMeta("x", fit_scope="whenever")
