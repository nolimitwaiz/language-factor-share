"""Ladder additivity on synthetic data with known components: each share
lands in its rung within tolerance, cross-fitted. Plus gauge invariance of
rungs >= M3 and permutation-null sanity."""
import numpy as np

from src.geometry.ladder import (fit_ladder, ladder_with_splits,
                                 permutation_null)


def _rand_orth(r, rng):
    Q, R = np.linalg.qr(rng.standard_normal((r, r)))
    return Q * np.sign(np.diag(R))


def _split(n, rng, frac=0.5):
    p = rng.permutation(n)
    k = int(n * frac)
    return p[:k], p[k:]


def _make(n=400, r=6, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, r))
    return Z, rng


def test_offset_only_lands_in_M1():
    Z, rng = _make()
    X = Z - 3.0 * np.ones(Z.shape[1])
    tr, te = _split(len(Z), rng)
    res = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    assert res['shares']['share_M1'] > 0.95
    for k in ('M2', 'M3', 'M4', 'M5'):
        assert abs(res['shares'][f'share_{k}']) < 0.03


def test_scale_only_lands_in_M2():
    Z, rng = _make(seed=1)
    X = Z / 3.0
    tr, te = _split(len(Z), rng)
    res = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    assert res['shares']['share_M2'] > 0.9
    for k in ('M3', 'M4', 'M5'):
        assert abs(res['shares'][f'share_{k}']) < 0.03


def test_rotation_only_lands_in_M3():
    """Pure rotation: the M3 rung COMPLETES the fit (E3 ~ 0) and dominates.
    NOTE (signature-matrix lesson): under the canonical nested order, an
    isotropic shrink s < 1 mechanically reduces error for ANY rotated copy
    (optimal s = mean diagonal cosine), so M1/M2 legitimately absorb part of
    a pure rotation's share. The frozen prediction for rotation is
    'M3 largest AND E3 ~ 0', not 'M3 = 1'."""
    Z, rng = _make(seed=2)
    R0 = _rand_orth(Z.shape[1], rng)
    X = Z @ R0.T
    tr, te = _split(len(Z), rng)
    res = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    s = res['shares']
    assert res['errors']['M3'] / res['errors']['M0'] < 0.02
    assert s['share_M3'] > 0.4
    assert s['share_M3'] > s['share_M1'] and s['share_M3'] > s['share_M2']
    for k in ('M4', 'M5'):
        assert abs(s[f'share_{k}']) < 0.03


def test_general_linear_lands_in_M4():
    """A distinctly non-similarity linear map (anisotropic stretch between
    two rotations) cannot be absorbed by M3; M4 must complete the fit."""
    Z, rng = _make(seed=3)
    r = Z.shape[1]
    A = (_rand_orth(r, rng)
         @ np.diag([2.0, 0.5, 1.6, 0.6, 1.3, 0.75][:r])
         @ _rand_orth(r, rng))
    X = Z @ np.linalg.inv(A)
    tr, te = _split(len(Z), rng)
    res = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    # NOTE (signature-matrix lesson): similarity Procrustes explains most of
    # a stretch-between-rotations map (rotation does the heavy lifting), so
    # the frozen prediction for a general linear injection is 'M4 COMPLETES
    # the fit (E4 ~ 0, E4 << E3) and clears the noise level' — not
    # 'M4 owns the majority share'.
    assert res['errors']['M4'] / res['errors']['M0'] < 0.02
    assert res['errors']['M4'] < 0.25 * res['errors']['M3']
    assert res['shares']['share_M4'] > 0.05
    assert abs(res['shares']['share_M5']) < 0.03


def test_nonlinear_lands_in_M5():
    rng = np.random.default_rng(4)
    n, r = 400, 4
    X = rng.standard_normal((n, r))
    Z = X + 0.6 * np.sin(2.0 * X)                          # monotone nonlinear
    tr, te = _split(n, rng)
    res = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    assert res['shares']['share_M5'] > 0.05
    assert res['errors']['M5'] < res['errors']['M4']


def test_mixed_components_additive():
    """offset + scale + rotation combined: shares sum to ~1 and M4/M5 stay
    near zero."""
    Z, rng = _make(seed=5)
    r = Z.shape[1]
    R0 = _rand_orth(r, rng)
    X = (Z @ R0.T) / 2.0 - 1.5
    tr, te = _split(len(Z), rng)
    res = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    s = res['shares']
    total = sum(s.values())
    assert total > 0.97
    assert abs(s['share_M4']) < 0.03 and abs(s['share_M5']) < 0.03
    assert res['errors']['M3'] / res['errors']['M0'] < 0.02


def test_gauge_invariance_of_rotation_and_above():
    """Under a global re-orientation Z -> Z Q, the errors of rungs >= M3 are
    exactly invariant (their model class absorbs Q). M0-M2 are pinned by the
    GPA gauge anchor instead — documented in gpa.py."""
    Z, rng = _make(seed=6)
    X = Z @ _rand_orth(Z.shape[1], rng).T + 0.1 * rng.standard_normal(Z.shape)
    Q = _rand_orth(Z.shape[1], rng)
    tr, te = _split(len(Z), rng)
    r1 = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    r2 = fit_ladder(X[tr], Z[tr] @ Q, X[te], Z[te] @ Q)
    for k in ('M3', 'M4'):
        np.testing.assert_allclose(r1['errors'][k], r2['errors'][k],
                                   rtol=1e-6)
    # M5 kernel ridge: RBF kernel on inputs is unchanged; targets rotate ->
    # predictions rotate with them; SSE invariant
    np.testing.assert_allclose(r1['errors']['M5'], r2['errors']['M5'],
                               rtol=1e-5)


def test_permutation_null_centers_on_zero():
    """Shuffled data yields shares centered on ~0 (M3/M4)."""
    Z, rng = _make(n=300, seed=7)
    X = rng.standard_normal(Z.shape)                       # no correspondence
    tr, te = _split(len(Z), rng)
    null = permutation_null(X[tr], Z[tr], X[te], Z[te], B=20,
                            rungs=('M3', 'M4'), seed=0)
    for r_ in ('M3', 'M4'):
        assert abs(null[r_].mean()) < 0.05


def test_ladder_with_splits_shape():
    Z, rng = _make(n=200, seed=8)
    X = Z - 1.0
    out = ladder_with_splits(X, Z, n_splits=3, seed=0)
    assert len(out) == 3
    assert all('shares' in o and 'errors' in o for o in out)
