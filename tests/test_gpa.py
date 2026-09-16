"""Procrustes recovery, GPA convergence, gauge invariance of relative
rotations."""
import numpy as np

from src.geometry.gpa import (consensus_target, gpa, orthogonal_procrustes,
                              relative_rotation_angles,
                              similarity_procrustes)


def _rand_orth(r, rng):
    Q, R = np.linalg.qr(rng.standard_normal((r, r)))
    return Q * np.sign(np.diag(R))


def test_procrustes_exact_recovery():
    """Synthetic rotation at n >> r, zero noise -> exact recovery."""
    rng = np.random.default_rng(0)
    n, r = 500, 8
    X = rng.standard_normal((n, r))
    R0 = _rand_orth(r, rng)
    R = orthogonal_procrustes(X, X @ R0)
    np.testing.assert_allclose(R, R0, atol=1e-10)


def test_similarity_procrustes_recovers_s_R_b():
    rng = np.random.default_rng(1)
    n, r = 500, 6
    X = rng.standard_normal((n, r))
    R0, s0, b0 = _rand_orth(r, rng), 2.5, rng.standard_normal(r)
    Z = s0 * X @ R0 + b0
    s, R, b = similarity_procrustes(X, Z)
    assert abs(s - s0) < 1e-9
    np.testing.assert_allclose(R, R0, atol=1e-9)
    np.testing.assert_allclose(b, b0, atol=1e-9)
    np.testing.assert_allclose(s * X @ R + b, Z, atol=1e-9)


def test_gpa_aligns_rotated_copies():
    """Languages = similarity transforms of one configuration; GPA must
    align them to near-zero spread and converge."""
    rng = np.random.default_rng(2)
    n, r = 300, 6
    base = rng.standard_normal((n, r))
    X = {}
    for i in range(5):
        X[f'l{i}'] = ((0.5 + i * 0.3) * base @ _rand_orth(r, rng)
                      + rng.standard_normal(r))
    g = gpa(X)
    aligned = [s * X[l] @ R + b for l, (s, R, b) in
               ((l, g['fits'][l]) for l in g['langs'])]
    spread = np.mean([np.linalg.norm(a - g['Z_train']) for a in aligned])
    assert spread / np.linalg.norm(g['Z_train']) < 1e-6
    assert g['history'][-1] < 1e-6


def test_relative_rotation_gauge_invariance():
    """R_a R_b^T angle spectrum is exactly invariant under a global
    re-orientation of the consensus (R_l -> R_l Q)."""
    rng = np.random.default_rng(3)
    r = 8
    Ra, Rb, Q = _rand_orth(r, rng), _rand_orth(r, rng), _rand_orth(r, rng)
    ang1, g1 = relative_rotation_angles(Ra, Rb)
    ang2, g2 = relative_rotation_angles(Ra @ Q, Rb @ Q)
    np.testing.assert_allclose(np.sort(ang1), np.sort(ang2), atol=1e-9)
    assert abs(g1 - g2) < 1e-9


def test_consensus_target_and_loo():
    rng = np.random.default_rng(4)
    n, r = 200, 5
    base = rng.standard_normal((n, r))
    X = {f'l{i}': base @ _rand_orth(r, rng) for i in range(4)}
    g = gpa(X)
    Z_all = consensus_target(g, X)
    Z_loo = consensus_target(g, X, exclude='l0')
    assert Z_all.shape == base.shape
    # perfect-copy case: LOO and full targets agree up to numerics
    np.testing.assert_allclose(Z_all, Z_loo, atol=1e-6)
