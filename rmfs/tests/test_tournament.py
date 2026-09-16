"""Tournament stats: residual Spearman, increment, Romano-Wolf, breadth."""

import numpy as np
import pandas as pd
import pytest

from rmfs.validity.tournament import (
    effective_breadth,
    orthogonalized_increment,
    residual_spearman,
    romano_wolf,
)


def _frame(seed=13, n_langs=30, n_models=5, signal=0.8, confound=0.0):
    """Target driven by candidate (signal) and/or log_tokens (confound)."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_models):
        for lg in range(n_langs):
            tok = rng.normal()
            cand = rng.normal() + confound * tok
            targ = signal * cand + confound * tok + 0.3 * rng.normal()
            rows.append(dict(model=f"m{m}", flores_code=f"L{lg}",
                             log_tokens=tok, fertility=rng.normal(),
                             macro_family_g=f"f{lg % 3}",
                             script_g=f"s{lg % 2}",
                             cand=cand, targ=targ,
                             null=rng.normal()))
    return pd.DataFrame(rows)


def test_residual_spearman_finds_true_signal():
    r = residual_spearman(_frame(signal=0.8), "cand", "targ", n_boot=200)
    assert r.rho > 0.5 and r.ci_lo > 0.3
    assert r.n_langs == 30


def test_residual_spearman_null_calibrated():
    # a single seed can legitimately exclude zero ~5% of the time; the
    # testable property is the exclusion RATE (D4: null first)
    exc = 0
    for s in range(12):
        r = residual_spearman(_frame(seed=100 + s), "null", "targ",
                              n_boot=120, seed=7)
        exc += not (r.ci_lo < 0 < r.ci_hi)
    assert exc <= 3          # ~5% nominal; 4+/12 signals miscalibration


def test_residual_spearman_removes_confound():
    # target driven ONLY by tokens; candidate = tokens + noise. Raw corr
    # is strong; residualized corr must collapse toward zero.
    df = _frame(signal=0.0, confound=1.5)
    from scipy import stats
    raw = stats.spearmanr(df.cand, df.targ).statistic
    r = residual_spearman(df, "cand", "targ", n_boot=200)
    assert raw > 0.5
    assert abs(r.rho) < raw / 2
    assert r.ci_lo < 0 < r.ci_hi


def test_increment_dies_when_candidate_duplicates_incumbent():
    df = _frame(signal=0.8)
    df["cand2"] = df.cand + 0.01 * np.random.default_rng(1).normal(
        size=len(df))
    r = orthogonalized_increment(df, "cand2", "cand", "targ", n_boot=200)
    assert r.ci_lo < 0 < r.ci_hi          # no increment beyond incumbent


def test_increment_survives_for_independent_signal():
    df = _frame(signal=0.6)
    rng = np.random.default_rng(7)
    extra = rng.normal(size=len(df))
    df["cand2"] = extra
    df["targ"] = df.targ + 0.8 * extra
    r = orthogonalized_increment(df, "cand2", "cand", "targ", n_boot=200)
    assert r.rho > 0.3 and r.ci_lo > 0


def test_romano_wolf_orders_bounds_and_fwer():
    # per-seed invariants: monotone ordering, valid bounds, true signal
    # detected; across seeds: null family-wise rejections near nominal
    null_rej = 0
    for s in range(8):
        df = _frame(seed=200 + s, signal=0.8)
        out = romano_wolf(df, ["cand", "null"], "targ", n_boot=120,
                          seed=7)
        p_cand = float(out[out.candidate == "cand"].p_rw.iloc[0])
        p_null = float(out[out.candidate == "null"].p_rw.iloc[0])
        assert p_cand <= p_null
        assert (out.p_rw <= 1).all() and (out.p_rw > 0).all()
        assert p_cand <= 0.05          # strong true signal always found
        null_rej += p_null <= 0.05
    assert null_rej <= 2               # FWER calibration (~5% nominal)


def test_effective_breadth_bounds():
    df = _frame()
    ne = effective_breadth(df, "cand", "targ")
    assert 1.0 <= ne <= 30.0


def test_effective_breadth_collapses_under_duplication():
    # identical per-language values across languages -> rho_bar ~ 1 -> ~1
    rng = np.random.default_rng(3)
    rows = []
    base = rng.normal(size=6)
    for m in range(6):
        for lg in range(10):
            rows.append(dict(model=f"m{m}", flores_code=f"L{lg}",
                             cand=base[m], targ=rng.normal()))
    ne = effective_breadth(pd.DataFrame(rows), "cand", "targ")
    assert ne == pytest.approx(1.0, abs=0.2)
