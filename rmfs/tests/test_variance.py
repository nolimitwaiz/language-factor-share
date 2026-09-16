"""Unit battery for the variance components (Phase 1.1/1.2, tests-first).

Every test here has a closed-form or Monte-Carlo-bounded expected answer,
written BEFORE the implementation. The pure-noise contrast (REML components
near zero while the legacy sum-of-squares ratio sits at its known floor
(L-1)/((L-1)+(N-1))) is a paper figure; its harness lives here.

Data convention: X has shape (L, N, D) — languages x concepts x dimensions,
matching the legacy grid layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from rmfs.components.variance import (
    lde,
    legacy_lfs_ratio,
    variance_components,
)
from rmfs.utils.seeding import rng

# (L, N) grid from the kickoff, with per-configuration Monte-Carlo
# tolerances: the sigma_L^2 estimator pools (L-1) x D degrees of freedom,
# so small-L configurations are intrinsically noisy and get wider bands.
GRID = [(2, 50), (12, 300), (128, 300), (12, 1500)]
D = 32


def synth(L, N, sd_L, sd_C, sd_E, seed):
    g = rng(seed)
    b = g.normal(0.0, sd_L, size=(L, 1, D))
    a = g.normal(0.0, sd_C, size=(1, N, D))
    e = g.normal(0.0, sd_E, size=(L, N, D))
    return b + a + e


# ---------------------------------------------------------------- pure noise
def _trunc_bound(df_x: int, df_e: int, mult: int) -> float:
    """Expected truncated-positive mass of a nonnegative variance estimate
    under pure noise, per dimension, times a 2.5x MC headroom factor.

    The ANOVA component (MS_X - MS_E)/mult has per-dim SD
    sqrt(2/df_x + 2/df_e)/mult under sigma2=1; truncation at zero keeps
    ~0.4*SD of positive mass in expectation. "Near zero" for nonnegative
    REML therefore MUST scale with degrees of freedom — a flat threshold is
    wrong at small df (estimator_notes.md N1)."""
    sd = np.sqrt(2.0 / df_x + 2.0 / df_e) / mult
    return 2.5 * 0.4 * D * sd


@pytest.mark.parametrize("L,N", GRID)
def test_pure_noise_reml_near_zero(L, N):
    X = rng(13).normal(size=(L, N, D))
    vc = variance_components(X)
    df_e = (L - 1) * (N - 1)
    # REML systematic components sit within the truncation mass of zero
    assert vc.sigma2_L_reml < _trunc_bound(L - 1, df_e, N)
    assert vc.sigma2_C_reml < _trunc_bound(N - 1, df_e, L)
    # error variance itself is recovered (true 1.0 per dim, summed over D).
    # rel=0.10 not 0.05: REML boundary pooling conditions on ms_L < ms_E,
    # which selects low ss_L into the pooled error and biases sigma2_E
    # slightly DOWN at tiny L (half of all dims sit on the boundary under
    # pure noise at L=2). Documented in paper/appendix/estimator_notes.md.
    assert vc.sigma2_E == pytest.approx(D, rel=0.10)


@pytest.mark.parametrize("L,N", GRID)
def test_pure_noise_legacy_ratio_sits_at_floor(L, N):
    X = rng(42).normal(size=(L, N, D))
    floor = (L - 1) / ((L - 1) + (N - 1))
    ratio = legacy_lfs_ratio(X)
    # The legacy estimator's pure-noise value is the DF ratio exactly in
    # expectation; MC tolerance scales with the pooled degrees of freedom.
    tol = 6.0 * floor / np.sqrt((L - 1) * D)
    assert ratio == pytest.approx(floor, abs=max(tol, 0.01))


# ---------------------------------------------------------------- recovery
@pytest.mark.parametrize(
    "L,N,rel",
    [(2, 50, 0.9), (12, 300, 0.30), (128, 300, 0.12), (12, 1500, 0.30)],
)
def test_component_recovery(L, N, rel):
    sd_L, sd_C, sd_E = 1.5, 2.0, 1.0
    X = synth(L, N, sd_L, sd_C, sd_E, seed=71)
    vc = variance_components(X)
    assert vc.sigma2_L_reml == pytest.approx(D * sd_L**2, rel=rel)
    assert vc.sigma2_C_reml == pytest.approx(D * sd_C**2, rel=rel)
    # rel=0.10: boundary-pooling bias, see estimator_notes N1
    assert vc.sigma2_E == pytest.approx(D * sd_E**2, rel=0.10)
    true_share = sd_L**2 / (sd_L**2 + sd_C**2)
    assert vc.lfs_vc == pytest.approx(true_share, abs=0.5 * rel)


def test_reml_equals_anova_when_interior():
    X = synth(12, 300, 1.5, 2.0, 1.0, seed=13)
    vc = variance_components(X)
    if vc.n_neg_L == 0:
        assert vc.sigma2_L_reml == pytest.approx(vc.sigma2_L_anova, rel=1e-9)
    if vc.n_neg_C == 0:
        assert vc.sigma2_C_reml == pytest.approx(vc.sigma2_C_anova, rel=1e-9)


# ------------------------------------------------------- negative estimates
def test_negatives_flagged_never_silently_clipped():
    # sigma_L = 0 with few languages: roughly half the per-dimension ANOVA
    # language components come out negative. The ANOVA aggregate must KEEP
    # that negative mass (flagged), while REML is truncated nonnegative.
    X = synth(3, 40, 0.0, 1.0, 1.0, seed=42)
    vc = variance_components(X)
    assert vc.n_neg_L > 0
    assert vc.sigma2_L_reml >= 0.0
    assert vc.sigma2_L_anova < vc.sigma2_L_reml + 1e-12
    # and the REML truncation leaves the zero-signal component near zero
    assert vc.sigma2_L_reml < 0.05 * vc.sigma2_E


# ---------------------------------------------------------------------- LDE
def test_lde_recovers_offset_energy():
    L, N = 12, 300
    g = rng(71)
    offsets = g.normal(0.0, 1.0, size=(L, 1, D))
    X = offsets + g.normal(0.0, 1.0, size=(L, N, D))
    res = lde(X)
    true = ((offsets[:, 0, :] - offsets[:, 0, :].mean(0)) ** 2).sum(1)
    est = np.array([res.corrected[i] for i in range(L)])
    # per-language agreement within MC error, and tight in aggregate
    assert np.corrcoef(est, true)[0, 1] > 0.98
    assert est.sum() == pytest.approx(true.sum(), rel=0.10)


def test_lde_pure_noise_centers_on_zero():
    X = rng(13).normal(size=(12, 300, D))
    res = lde(X)
    vals = np.array(list(res.corrected.values()))
    # noise-corrected LDE is unbiased around zero: small relative to the
    # UNcorrected magnitude, and individual values may legitimately be
    # negative (that is the point of not clipping)
    raw = np.array(list(res.raw.values()))
    assert abs(vals.mean()) < 0.2 * raw.mean()


# ---------------------------------------------------------------- interface
def test_dataclass_diagnostics_present():
    X = synth(4, 60, 1.0, 1.0, 1.0, seed=13)
    vc = variance_components(X)
    for field in ("L", "N", "D", "mean_norm", "lfs_vc", "lfs_vc_sd",
                  "lfs_vc_anova", "n_neg_L", "n_neg_C"):
        assert hasattr(vc, field)
    assert vc.L == 4 and vc.N == 60 and vc.D == D
    assert 0.0 <= vc.lfs_vc <= 1.0
    assert 0.0 <= vc.lfs_vc_sd <= 1.0
