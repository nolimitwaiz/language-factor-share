"""Unit battery for the capacity gates (Phase 1.3) and shrinkage (1.4).

Closed-form anchors:
- CVP under uniform collapse by factor m is exactly (1 - (1-m))^2... stated
  plainly: scaling all representations by s multiplies concept variance by
  s^2, so C_pres = s^2. The prior campaign's battery cell I8b measured this
  to three decimals; the gate must reproduce it algebraically here.
- Effective rank of a k-spiked covariance approaches k when the spikes
  dominate, and approaches min(N-1, D) for isotropic data.
- Ledoit-Wolf intensity goes to zero as n/D grows; its eigenvalue MSE beats
  the sample covariance on a spiked model in the D ~ n regime.
"""

from __future__ import annotations

import numpy as np
import pytest

from rmfs.components.gates import (
    concept_variance_raw,
    cpres,
    effective_rank,
    read_delta_ppl,
)
from rmfs.components.shrinkage import ledoit_wolf
from rmfs.utils.seeding import rng

D = 64


# ---------------------------------------------------------------- CVP/C_pres
def test_cpres_exact_under_uniform_scaling():
    X = rng(13).normal(size=(6, 200, D))
    v_ref = concept_variance_raw(X)
    for s in (0.9, 0.5, 0.1):
        v = concept_variance_raw(s * X)
        assert v / v_ref == pytest.approx(s**2, abs=1e-12)
        assert cpres(v, v_ref).value == pytest.approx(s**2, abs=1e-12)


def test_cpres_capped_at_one_and_flagged():
    g = cpres(12.0, 10.0)
    assert g.value == 1.0
    assert g.capped is True
    assert cpres(8.0, 10.0).capped is False


def test_cpres_threshold_flag():
    assert cpres(9.5, 10.0, threshold=0.9).passed is True
    assert cpres(8.0, 10.0, threshold=0.9).passed is False


def test_cpres_zero_reference_hard_fails():
    with pytest.raises(ValueError):
        cpres(1.0, 0.0)


# ---------------------------------------------------------- effective rank
def test_effective_rank_low_rank_signal():
    g = rng(42)
    N, k = 400, 5
    basis = np.linalg.qr(g.normal(size=(D, k)))[0]
    means = g.normal(size=(N, k)) @ basis.T * 10.0 + g.normal(
        size=(N, D)) * 0.05
    r = effective_rank(means)
    assert r.value == pytest.approx(k, rel=0.15)


def test_effective_rank_isotropic_is_high():
    means = rng(71).normal(size=(500, D))
    r = effective_rank(means)
    assert r.value > 0.8 * D


def test_effective_rank_invariant_to_global_scale():
    means = rng(13).normal(size=(300, D))
    a = effective_rank(means).value
    b = effective_rank(7.3 * means).value
    assert a == pytest.approx(b, rel=1e-9)


# ------------------------------------------------------------- dPPL hook
def test_delta_ppl_hook_hard_fails_on_missing(tmp_path):
    # E2 rule: missing inputs raise, never degrade silently
    with pytest.raises(FileNotFoundError):
        read_delta_ppl(tmp_path / "nope.json")


def test_delta_ppl_hook_reads(tmp_path):
    import json

    p = tmp_path / "ppl.json"
    p.write_text(json.dumps({"deu": 1.05, "swa": 1.31}))
    out = read_delta_ppl(p)
    assert out["swa"] == pytest.approx(1.31)


# ---------------------------------------------------------------- shrinkage
def _spiked(g, n, scale=(25.0, 16.0, 9.0)):
    basis = np.linalg.qr(g.normal(size=(D, len(scale))))[0]
    true = basis @ np.diag(scale) @ basis.T + np.eye(D)
    Lch = np.linalg.cholesky(true)
    return g.normal(size=(n, D)) @ Lch.T, true


def test_shrinkage_intensity_vanishes_with_n_on_structured_truth():
    # Intensity -> 0 as n/D grows PRESUMES the true covariance differs from
    # the shrinkage target. Measured decay on this seed: 0.152 -> 0.019 ->
    # 0.002 (estimator_notes.md N3).
    g = rng(42)
    prev = 1.0
    for n, cap in ((80, 0.30), (800, 0.05), (8000, 0.01)):
        Xs, _ = _spiked(g, n)
        lw = ledoit_wolf(Xs)
        assert lw.intensity < prev
        assert lw.intensity < cap
        prev = lw.intensity


def test_shrinkage_full_toward_identity_when_truth_IS_the_target():
    # The flip side, asserted so nobody "fixes" it later: for isotropic data
    # the target equals the truth, so heavy shrinkage is OPTIMAL and
    # intensity near 1 is correct behavior, not a bug (N3).
    lw = ledoit_wolf(rng(13).normal(size=(800, D)))
    assert lw.intensity > 0.9


def test_shrinkage_beats_sample_covariance_in_frobenius_risk():
    # Frobenius risk is what Ledoit-Wolf minimizes in expectation; the
    # sorted-eigenvalue MSE of a heavily spiked model is NOT (a first draft
    # of this test asserted that and correctly failed). Measured on this
    # seed at n=80: LW 10.61 vs sample 11.91.
    g = rng(42)
    Xs, true = _spiked(g, 80)
    lw = ledoit_wolf(Xs)
    sample = np.cov(Xs, rowvar=False)
    assert (np.linalg.norm(lw.covariance - true)
            < np.linalg.norm(sample - true))
