"""Marchenko-Pastur edge + pooled PCA sanity."""
import numpy as np

from src.geometry.subspace import mp_edge, mp_rank_ceiling, pooled_pca


def test_mp_edge_matches_theory_on_iid_gaussian():
    rng = np.random.default_rng(0)
    n, d = 4000, 200
    X = rng.standard_normal((n, d))
    eig = np.linalg.eigvalsh(np.cov(X.T))
    edge = mp_edge(n, d, 1.0)
    # bulk edge: no eigenvalue materially exceeds it, top eigenvalue near it
    assert eig.max() < edge * 1.05
    assert eig.max() > edge * 0.90
    k, _, s2 = mp_rank_ceiling(eig, n, d)          # sigma2 estimated from bulk
    assert k <= 2                                   # ~0 spikes on pure noise
    assert abs(s2 - 1.0) < 0.1


def test_mp_detects_planted_spikes():
    rng = np.random.default_rng(1)
    n, d, k_true = 4000, 200, 3
    X = rng.standard_normal((n, d))
    for i in range(k_true):                         # strong rank-1 spikes
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        X += np.sqrt(10.0) * rng.standard_normal((n, 1)) * u
    eig = np.linalg.eigvalsh(np.cov(X.T))
    k, _, _ = mp_rank_ceiling(eig, n, d)
    assert k == k_true


def test_pooled_pca_recovers_basis_and_lw_bounded():
    rng = np.random.default_rng(2)
    n, d, r = 2000, 50, 3
    U, _ = np.linalg.qr(rng.standard_normal((d, r)))
    X = rng.standard_normal((n, r)) * 5.0 @ U.T + 0.1 * rng.standard_normal((n, d))
    res = pooled_pca(X, r)
    # recovered top-r basis spans the true subspace
    proj = res['basis'].T @ U
    sv = np.linalg.svd(proj, compute_uv=False)
    assert sv.min() > 0.99
    assert 0.0 <= res['lw_shrinkage'] <= 1.0
    assert res['eig_lw'].shape == res['eig'].shape
