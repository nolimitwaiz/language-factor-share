"""RMFS assembly: soft-min limits, frozen T language aggregation, modes."""

import math

import numpy as np
import pytest

from rmfs.components.transfer import EPS_B
from rmfs.metrics.rmfs import (
    TAU,
    assemble,
    pool_languages,
    softmin,
    t_language,
)


# ---------------------------------------------------------------- softmin
def test_softmin_equal_components_is_identity():
    for j in (3, 4):
        assert softmin([0.42] * j) == pytest.approx(0.42, abs=1e-12)


def test_softmin_bounds():
    rng = np.random.default_rng(13)
    for _ in range(200):
        j = rng.choice([3, 4])
        q = rng.uniform(0, 1, size=j)
        v = softmin(q)
        assert q.min() <= v + 1e-12
        assert v <= q.min() + TAU * math.log(j) + 1e-12


def test_softmin_approaches_min_as_tau_shrinks():
    q = [0.9, 0.8, 0.05]
    assert softmin(q, tau=1e-4) == pytest.approx(0.05, abs=1e-3)


def test_softmin_dominated_by_weakest_gauge():
    # one collapsed gauge drags the scalar to it despite three healthy ones
    healthy = softmin([0.8, 0.8, 0.8, 0.8])
    crashed = softmin([0.8, 0.8, 0.8, 0.05])
    assert crashed < 0.05 + TAU * math.log(4) + 1e-12
    assert healthy - crashed > 0.5


def test_softmin_nan_propagates():
    assert math.isnan(softmin([0.5, math.nan, 0.7]))


def test_softmin_rejects_bad_shapes():
    with pytest.raises(ValueError):
        softmin([])
    with pytest.raises(ValueError):
        softmin(np.zeros((2, 2)))


# ------------------------------------------------------------- t_language
def _mk(none, nat, mis, inj, injw, n=6):
    one = np.ones(n)
    return dict(nll_none=none * one, nll_native=nat * one,
                nll_mismatch=mis * one, nll_inj=inj * one,
                nll_injw=injw * one)


def test_t_language_exact_algebra():
    # denom = mis − nat = 1.0; numer = injw − inj = 0.4 → T = 0.4
    r = t_language(**_mk(none=5.0, nat=3.0, mis=4.0, inj=4.2, injw=4.6))
    assert r.t_raw == pytest.approx(0.4, abs=1e-12)
    assert r.q_t == pytest.approx(0.4, abs=1e-12)
    assert not r.undefined and not r.saturated
    assert r.b_native == pytest.approx(2.0)
    assert r.b_mismatch == pytest.approx(1.0)


def test_t_language_aggregates_pairs_before_ratio():
    # pairwise ratios explode on one pair; language-level B's are tame.
    none = np.array([5.0, 5.0])
    nat = np.array([3.0, 4.999])       # pair 2 has near-zero pair denom
    mis = np.array([4.0, 5.0])
    inj = np.array([4.0, 4.0])
    injw = np.array([4.5, 4.5])
    r = t_language(none, nat, mis, inj, injw)
    denom = mis.mean() - nat.mean()
    assert r.denom == pytest.approx(denom, abs=1e-12)
    assert r.t_raw == pytest.approx(0.5 / denom, abs=1e-12)


def test_t_language_undefined_below_eps_b():
    # denom = 0.05 < ε_B = 0.072 → undefined, NaN, never zero
    r = t_language(**_mk(none=5.0, nat=3.95, mis=4.0, inj=4.0, injw=4.5))
    assert r.undefined
    assert math.isnan(r.t_raw) and math.isnan(r.q_t)
    assert EPS_B == pytest.approx(0.072)


def test_t_language_saturated_flag_and_clip():
    # inj WORSE than injw beyond ε_B → saturated; raw negative, q_t clips to 0
    r = t_language(**_mk(none=5.0, nat=3.0, mis=4.0, inj=4.8, injw=4.5))
    assert r.saturated
    assert r.t_raw == pytest.approx(-0.3, abs=1e-12)
    assert r.q_t == 0.0


def test_t_language_rejects_ragged():
    with pytest.raises(ValueError):
        t_language(np.ones(3), np.ones(3), np.ones(2), np.ones(3), np.ones(3))


# --------------------------------------------------------------- assemble
def test_assemble_modes():
    obs = assemble(0.4, 0.3, 0.6)
    assert obs.mode == "observational" and math.isnan(obs.q_c)
    assert obs.rmfs == pytest.approx(softmin([0.4, 0.3, 0.6]))
    itv = assemble(0.4, 0.3, 0.6, q_c=0.05)
    assert itv.mode == "intervention"
    assert itv.rmfs == pytest.approx(softmin([0.4, 0.3, 0.6, 0.05]))
    assert itv.rmfs < obs.rmfs          # the collapsed gauge dominates


def test_assemble_undefined_component_makes_cell_undefined():
    cell = assemble(math.nan, 0.3, 0.6)
    assert cell.undefined and math.isnan(cell.rmfs)


# ---------------------------------------------------------- pool_languages
def test_pool_languages_median_primary_and_undefined_counted():
    p = pool_languages([0.2, 0.4, 0.6, math.nan])
    assert p["median"] == pytest.approx(0.4)
    assert p["mean"] == pytest.approx(0.4)
    assert p["n_defined"] == 3 and p["n_undefined"] == 1


def test_pool_languages_all_undefined():
    p = pool_languages([math.nan, math.nan])
    assert math.isnan(p["median"]) and p["n_defined"] == 0
