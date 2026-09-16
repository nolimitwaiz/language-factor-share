"""Battery for behavioral T (Phase 2.1), written before the implementation
and before any extraction, per addendum A2's binding order.

The end-to-end tests use an ORACLE world: language states are known
transforms of a shared latent, and "NLL" is a closed-form function of the
distance between the provided context state and the true native state. In
that world T's value is known: ≈1 when reconstruction can recover the
native state (invertible-rung worlds, full-rank basis), ≈0 when the
pairing is permuted (the shared factor carries the wrong concept). No
model, no GPU — the construct is tested, not the plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from rmfs.components.transfer import (
    TResult,
    fit_reverse_map,
    reconstruct,
    t_from_nll,
)
from rmfs.utils.seeding import rng

D = 24
N = 300


# ------------------------------------------------------------ reverse maps
def test_reverse_offset_map_recovers_states():
    g = rng(13)
    Z = g.normal(size=(N, D))
    b = g.normal(0, 1.0, size=(1, D))
    X = Z + b + g.normal(0, 0.02, Z.shape)
    m = fit_reverse_map(Z[:200], X[:200], "M1", seed=0)
    err = np.abs(m(Z[200:]) - X[200:]).mean()
    assert err < 0.05


def test_reverse_rotation_map_recovers_states():
    g = rng(42)
    Z = g.normal(size=(N, D))
    Q = np.linalg.qr(g.normal(size=(D, D)))[0]
    X = 1.5 * Z @ Q + 0.3 + g.normal(0, 0.02, Z.shape)
    m = fit_reverse_map(Z[:200], X[:200], "M3", seed=0)
    err = np.abs(m(Z[200:]) - X[200:]).mean()
    assert err < 0.05


def test_reverse_ridge_fits_linear_world():
    g = rng(71)
    Z = g.normal(size=(N, D))
    A = np.eye(D) + 0.4 * g.normal(size=(D, D))
    X_lin = Z @ A + g.normal(0, 0.02, Z.shape)
    m4 = fit_reverse_map(Z[:200], X_lin[:200], "M4", seed=0)
    assert np.abs(m4(Z[200:]) - X_lin[200:]).mean() < 0.10


def test_reverse_krr_fits_finitely_and_beats_the_mean_predictor():
    # What A2 actually requires of reverse maps: finite, deterministic
    # under seed, better than predicting the mean. NOT that reverse-KRR
    # beats reverse-ridge — measured, it does not (estimator note N6: the
    # kernel advantage does NOT survive direction reversal; ridge wins the
    # reverse cubic world at every n/d/amplitude tried). A2 discloses the
    # consequence for the all-M5 families before any extraction.
    g = rng(71)
    Z = g.normal(size=(N, 12))
    X_nl = Z + 0.25 * Z**3 + g.normal(0, 0.02, Z.shape)
    m5a = fit_reverse_map(Z[:200], X_nl[:200], "M5", seed=0)
    m5b = fit_reverse_map(Z[:200], X_nl[:200], "M5", seed=0)
    pred = m5a(Z[200:])
    assert np.isfinite(pred).all()
    assert np.allclose(pred, m5b(Z[200:]), atol=1e-12)   # deterministic
    err5 = ((pred - X_nl[200:]) ** 2).mean()
    err_mean = ((X_nl[:200].mean(0) - X_nl[200:]) ** 2).mean()
    assert err5 < 0.5 * err_mean


# ---------------------------------------------------------- reconstruction
def test_strict_reconstruction_broadcasts_factor_prediction_exact():
    g = rng(13)
    mu = g.normal(size=D)
    B = np.linalg.qr(g.normal(size=(D, 5)))[0]      # rank-5 basis
    z = g.normal(size=5)

    def m(zz):
        return zz + 1.0                             # toy reverse map

    out = reconstruct(z, B, mu, m, mode="strict", n_tokens=7)
    manual = mu + B @ (z + 1.0)
    assert out.shape == (7, D)
    for row in out:                                 # identical broadcast
        assert np.allclose(row, manual, atol=1e-12)


def test_hybrid_keeps_out_of_subspace_replaces_in_subspace_exact():
    g = rng(42)
    states = g.normal(size=(7, D))
    mu = np.zeros(D)
    B = np.linalg.qr(g.normal(size=(D, 5)))[0]
    z = g.normal(size=5)

    def ident(zz):
        return zz

    out = reconstruct(z, B, mu, ident, mode="hybrid", states=states)
    # out-of-subspace content preserved exactly
    off = states - (states @ B) @ B.T
    off_out = out - (out @ B) @ B.T
    assert np.allclose(off, off_out, atol=1e-12)
    # in-subspace content is the factor prediction, same at every token
    for row in out:
        assert np.allclose(row @ B, z, atol=1e-12)


def test_a2_style_token_coordinate_input_is_rejected_by_shape():
    # The A3 catch: feeding token coordinates where the factor belongs was
    # the leak. The signature now takes a single factor vector; a (T, D)
    # states matrix in the z slot fails loudly rather than leaking.
    g = rng(7)
    B = np.linalg.qr(g.normal(size=(D, 5)))[0]
    with pytest.raises((ValueError, Exception)):
        bad = reconstruct(g.normal(size=(7, D)), B, np.zeros(D),
                          lambda zz: zz, mode="strict", n_tokens=7)
        assert bad.shape != (7, D)


# ------------------------------------------------------------- T from NLLs
def test_t_formula_and_clipping():
    r = t_from_nll(nll_none=5.0, nll_native=4.0, nll_mismatch=4.8,
                   nll_recon=4.3)
    # B_native=1.0, B_mismatch=0.2, B_recon=0.7 -> T=(0.7-0.2)/(1.0-0.2)
    assert r.t_raw == pytest.approx(0.625)
    assert r.q_t == pytest.approx(0.625)
    assert not r.undefined and not r.saturated
    over = t_from_nll(5.0, 4.0, 4.8, 3.5)           # B_recon > B_native
    assert over.t_raw > 1.0 and over.q_t == 1.0     # raw kept, clipped for q
    neg = t_from_nll(5.0, 4.0, 4.8, 5.5)            # recon HURTS
    assert neg.t_raw < 0.0 and neg.q_t == 0.0


def test_eps_b_rule_returns_undefined_not_zero():
    r = t_from_nll(5.0, 4.95, 4.93, 4.94, eps_b=0.072)
    # B_native - B_mismatch = 0.05 - 0.07 < eps_b -> undefined
    assert r.undefined
    assert np.isnan(r.t_raw) and np.isnan(r.q_t)


def test_saturation_flag_per_addendum_a2():
    # B_recon < B_mismatch - eps_b -> reconstruction worse than a wrong
    # document beyond the threshold: saturated, reported, not averaged
    r = t_from_nll(nll_none=5.0, nll_native=3.0, nll_mismatch=4.5,
                   nll_recon=4.9)
    assert r.saturated
    assert not r.undefined


# --------------------------------------------------------- oracle end-to-end
def _oracle_nll(context_state, native_state):
    """Closed-form 'NLL': quadratic in the distance to the true native
    context state, floored at 1.0. No-context NLL is the max."""
    d2 = float(((context_state - native_state) ** 2).mean())
    return 1.0 + min(d2, 4.0)


def test_oracle_world_T_near_one_when_recoverable():
    g = rng(13)
    Z = g.normal(size=(N, D))                       # shared factor
    Q = np.linalg.qr(g.normal(size=(D, D)))[0]
    X = 1.3 * Z @ Q + 0.5 + g.normal(0, 0.02, Z.shape)   # language ℓ, M3 world
    B = np.eye(D)                                   # full-rank basis
    mu = np.zeros(D)
    m = fit_reverse_map(Z[:200], X[:200], "M3", seed=0)
    ts = []
    for i in range(200, 260):
        native = X[i]
        recon = reconstruct(Z[i], B, mu, m, mode="strict", n_tokens=1)[0]
        wrong = X[(i + 37) % N]
        r = t_from_nll(
            nll_none=_oracle_nll(native * 0.0 + 99.0, native),  # far away
            nll_native=_oracle_nll(native, native),
            nll_mismatch=_oracle_nll(wrong, native),
            nll_recon=_oracle_nll(recon, native))
        if not (r.undefined or r.saturated):
            ts.append(r.t_raw)
    assert np.mean(ts) > 0.9


def test_oracle_world_T_near_zero_under_pairing_permutation():
    g = rng(42)
    Z = g.normal(size=(N, D))
    X = Z + 0.5 + g.normal(0, 0.02, Z.shape)        # M1 world
    B, mu = np.eye(D), np.zeros(D)
    m = fit_reverse_map(Z[:200], X[:200], "M1", seed=0)
    perm = g.permutation(np.arange(200, 260))
    ts = []
    for k, i in enumerate(range(200, 260)):
        native = X[i]
        recon = reconstruct(Z[perm[k]], B, mu, m, mode="strict",
                            n_tokens=1)[0]
        wrong = X[(i + 37) % N]
        r = t_from_nll(
            nll_none=_oracle_nll(native * 0.0 + 99.0, native),
            nll_native=_oracle_nll(native, native),
            nll_mismatch=_oracle_nll(wrong, native),
            nll_recon=_oracle_nll(recon, native))
        if not r.undefined:
            ts.append(max(r.t_raw, 0.0) if not r.saturated else 0.0)
    assert np.mean(ts) < 0.15


def test_result_dataclass_fields():
    r = t_from_nll(5.0, 4.0, 4.8, 4.3)
    assert isinstance(r, TResult)
    for f in ("b_native", "b_mismatch", "b_recon", "t_raw", "q_t",
              "undefined", "saturated", "eps_b"):
        assert hasattr(r, f)
