"""Battery for the tournament baselines (Phase 1.6): exact legacy parity for
the shared-prep ports, invariance/blindness properties for CKA, and
MDL/selectivity behavior for the probes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rmfs.metrics.baselines import (
    aar_shared,
    cka_linear,
    language_probe,
    mexa_shared,
    official_position_weights,
)
from rmfs.utils.seeding import rng

D = 48


def _legacy_metrics():
    root = Path(__file__).resolve().parents[1] / "legacy"
    spec = importlib.util.spec_from_file_location(
        "legacy_study_metrics", root / "study_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- exact parity
def test_mexa_and_aar_exact_parity_with_legacy():
    leg = _legacy_metrics()
    g = rng(13)
    E_en = g.normal(size=(120, D))
    E_l = E_en + 0.6 * g.normal(size=(120, D))     # partial alignment
    ours_m = mexa_shared(E_en, E_l)
    ours_a = aar_shared(E_en, E_l)
    assert ours_m.value == pytest.approx(leg.mexa_score(E_en, E_l), abs=1e-12)
    t, m = leg.aar(E_en, E_l)
    assert ours_a.tail_mean == pytest.approx(t, abs=1e-12)
    assert ours_a.mean == pytest.approx(m, abs=1e-12)


# ------------------------------------------------------- semantic anchors
def test_mexa_perfect_and_permuted():
    g = rng(42)
    E = g.normal(size=(100, D))
    assert mexa_shared(E, E).value == 1.0
    perm = g.permutation(100)
    assert mexa_shared(E, E[perm]).value < 0.05


def test_aar_tail_below_mean_and_negative_on_shuffle():
    g = rng(71)
    E_en = g.normal(size=(200, D))
    E_l = E_en + 0.4 * g.normal(size=(200, D))
    r = aar_shared(E_en, E_l)
    assert r.tail_mean <= r.mean
    perm = g.permutation(200)
    assert aar_shared(E_en, E_l[perm]).tail_mean < 0


# --------------------------------------------------------------------- CKA
def test_cka_invariances_are_the_documented_blindness():
    g = rng(13)
    X = g.normal(size=(150, D))
    Z = X + 0.3 * g.normal(size=(150, D))
    base = cka_linear(X, Z).value
    Q = np.linalg.qr(g.normal(size=(D, D)))[0]
    assert cka_linear(X @ Q, Z).value == pytest.approx(base, abs=1e-9)
    assert cka_linear(5.0 * X, Z).value == pytest.approx(base, abs=1e-9)
    # and it DOES respond to destroyed correspondence
    assert cka_linear(X[g.permutation(150)], Z).value < base


# ------------------------------------------------------------------ bridge
def test_official_position_weights():
    w = official_position_weights(5)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(np.diff(w) > 0)           # later tokens weigh more
    assert w[-1] / w[0] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        official_position_weights(0)


# ------------------------------------------------------------------ probes
def _langs_data(sep: float, n=120, n_lang=4, seed=42):
    g = rng(seed)
    base = g.normal(size=(n, D))
    return {f"l{i}": base + sep * g.normal(size=(1, D))
            for i in range(n_lang)}


def test_probe_selectivity_positive_on_real_structure():
    E = _langs_data(sep=2.0)
    idx = np.arange(120)
    res = language_probe(E, idx[:80], idx[80:])
    assert res.accuracy > 0.9
    assert res.selectivity > 0            # real labels cost fewer bits
    assert res.mdl_bits < res.mdl_bits_control


def test_probe_selectivity_near_zero_without_structure():
    # identical distributions across "languages": the probe can memorize
    # nothing transferable; selectivity collapses toward zero even though
    # a flexible probe might fit noise — the MDL control's whole point
    g = rng(13)
    E = {f"l{i}": g.normal(size=(120, D)) for i in range(4)}
    idx = np.arange(120)
    res = language_probe(E, idx[:80], idx[80:])
    assert res.accuracy < 0.45
    assert abs(res.selectivity) < 0.25 * res.mdl_bits_control
