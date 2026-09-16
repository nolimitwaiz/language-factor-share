"""Injections do exactly what they claim; rotation calibration recovers the
injected angle."""
import numpy as np

from src.battery import inject
from src.geometry.gpa import orthogonal_procrustes, relative_rotation_angles
from src.geometry.subspace import pooled_pca


def _fake_E(n_lang=4, N=200, D=32, seed=0):
    rng = np.random.default_rng(seed)
    concepts = rng.standard_normal((N, D)) * 2.0
    E = {}
    for i, l in enumerate(['eng', 'aaa', 'bbb', 'ccc'][:n_lang]):
        E[l] = (concepts + 0.3 * rng.standard_normal((N, D))
                + 0.5 * rng.standard_normal(D)).astype(np.float32)
    return E, rng


def _subspace(E, r=8):
    flat = np.concatenate(list(E.values()), 0)
    res = pooled_pca(flat, r)
    return res['basis'], res['mean']


def test_i1_offset_moves_means_only():
    E, rng = _fake_E()
    out, meta = inject.i1_offset(E, 0.5, rng)
    np.testing.assert_array_equal(out['eng'], E['eng'])       # pivot untouched
    for l in ('aaa', 'bbb', 'ccc'):
        d = out[l] - E[l]
        # constant shift: identical across sentences
        assert np.abs(d - d[0]).max() < 1e-6
        assert np.linalg.norm(d[0]) > 0


def test_i2_scale_multiplies():
    E, rng = _fake_E()
    out, meta = inject.i2_scale(E, 0.5, rng)
    for l in ('aaa', 'bbb', 'ccc'):
        s = meta['scales'][l]
        np.testing.assert_allclose(out[l], E[l] * s, rtol=1e-6)


def test_i3_global_rotation_preserves_geometry():
    E, rng = _fake_E()
    out, _ = inject.i3_global_rotation(E, rng)
    for l in E:
        # norms and pairwise inner products preserved (orthogonal map)
        np.testing.assert_allclose(
            np.linalg.norm(out[l], axis=1), np.linalg.norm(E[l], axis=1),
            rtol=1e-4)
    G0 = E['aaa'] @ E['bbb'].T
    G1 = out['aaa'] @ out['bbb'].T
    np.testing.assert_allclose(G0, G1, rtol=1e-3, atol=1e-2)


def test_i4_rotation_angle_recovery():
    """Calibration property: recover the injected angle from the subspace
    coordinates via Procrustes."""
    E, rng = _fake_E(N=400, D=32)
    r, theta = 8, 0.3
    basis, mean = _subspace(E, r)
    out, meta = inject.i4_lang_rotation(E, theta, r, basis, mean, rng)
    for l in ('aaa', 'bbb'):
        Zc = (E[l].astype(np.float64) - mean) @ basis
        Zi = (out[l].astype(np.float64) - mean) @ basis
        R = orthogonal_procrustes(Zc, Zi)
        ang, _ = relative_rotation_angles(R, np.eye(r))
        planes = ang[ang > 0.05]
        assert len(planes) == r // 2
        np.testing.assert_allclose(planes, theta, atol=0.05)
    np.testing.assert_array_equal(out['eng'], E['eng'])


def test_i8_collapse_shrinks_concept_variance():
    E, rng = _fake_E()
    m = 0.6
    out, _ = inject.i8_collapse(E, m)
    for l in ('aaa', 'bbb'):
        v0 = E[l].var(0).sum()
        v1 = out[l].var(0).sum()
        np.testing.assert_allclose(v1 / v0, (1 - m) ** 2, rtol=1e-3)


def test_i9_permutes_rows():
    E, rng = _fake_E()
    out, meta = inject.i9_permute(E, rng)
    for l in ('aaa', 'bbb', 'ccc'):
        assert not np.array_equal(out[l], E[l])
        np.testing.assert_array_equal(np.sort(out[l], 0), np.sort(E[l], 0))
    np.testing.assert_array_equal(out['eng'], E['eng'])


def test_i10_identity():
    E, _ = _fake_E()
    out, _ = inject.i10_identity(E)
    for l in E:
        np.testing.assert_array_equal(out[l], E[l])


def test_i4_lands_in_rotation_rung_of_ladder():
    """End-to-end: a language-specific subspace rotation shows up in the
    ladder's M3 rung, not M1/M2."""
    from src.geometry.ladder import fit_ladder
    E, rng = _fake_E(N=400, D=32)
    r = 8
    basis, mean = _subspace(E, r)
    out, _ = inject.i4_lang_rotation(E, 0.8, r, basis, mean, rng)
    # ladder target: clean subspace coords; input: injected coords
    l = 'aaa'
    Z = (E[l].astype(np.float64) - mean) @ basis
    X = (out[l].astype(np.float64) - mean) @ basis
    p = np.random.default_rng(0).permutation(len(Z))
    tr, te = p[:200], p[200:]
    res = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    assert res['shares']['share_M3'] > 0.5
    # NOTE (goes in the I4 signature): rotation about the POOLED mean also
    # displaces each language's centroid, so a small mechanical offset/scale
    # share is EXPECTED under I4 — the prediction is "M3 dominant", not
    # "M1 exactly zero".
    assert res['shares']['share_M1'] < 0.15
    assert res['shares']['share_M4'] < 0.05 and res['shares']['share_M5'] < 0.05


def test_i4b_preserves_centroids_exactly():
    """A1: centroid-preserving rotation leaves every language mean unchanged
    and recovers the injected angle under centered Procrustes."""
    from src.geometry.gpa import orthogonal_procrustes, relative_rotation_angles
    E, rng = _fake_E(N=400, D=32)
    r, theta = 8, 0.4
    basis, mean = _subspace(E, r)
    out, _ = inject.i4b_lang_rotation_centered(E, theta, r, basis, rng)
    for l in ('aaa', 'bbb', 'ccc'):
        np.testing.assert_allclose(out[l].mean(0), E[l].mean(0),
                                   atol=1e-3)
        Z0 = (E[l].astype(np.float64)) @ basis
        Z1 = (out[l].astype(np.float64)) @ basis
        Z0, Z1 = Z0 - Z0.mean(0), Z1 - Z1.mean(0)
        R = orthogonal_procrustes(Z0, Z1)
        ang, _ = relative_rotation_angles(R, np.eye(r))
        planes = ang[ang > 0.05]
        np.testing.assert_allclose(planes, theta, atol=0.05)
    np.testing.assert_array_equal(out['eng'], E['eng'])


def test_i4b_no_offset_leak_in_ladder():
    """A1's power-restoration claim, mechanically: even with a HUGE language
    offset (OLMo-like geometry), i4b's rotation lands in M3, not M1."""
    from src.geometry.ladder import fit_ladder
    E, rng = _fake_E(N=400, D=32)
    E['aaa'] = E['aaa'] + 25.0 * np.ones(32, dtype=np.float32)  # big offset
    r = 8
    basis, mean = _subspace(E, r)
    out, _ = inject.i4b_lang_rotation_centered(E, 0.8, r, basis, rng)
    Z = (E['aaa'].astype(np.float64) - mean) @ basis
    X = (out['aaa'].astype(np.float64) - mean) @ basis
    p = np.random.default_rng(0).permutation(len(Z))
    tr, te = p[:200], p[200:]
    res = fit_ladder(X[tr], Z[tr], X[te], Z[te])
    assert res['shares']['share_M3'] > 0.5
    assert abs(res['shares']['share_M1']) < 0.05    # NO centroid leak


def test_i8b_collapses_toward_global_centroid():
    E, _ = _fake_E()
    m = 0.6
    out, _ = inject.i8b_global_collapse(E, m)
    mu = np.mean([X.mean(0) for X in E.values()], axis=0)
    for l in ('aaa', 'bbb', 'ccc'):
        # centroids move toward global mean by factor (1-m)
        np.testing.assert_allclose(out[l].mean(0) - mu,
                                   (1 - m) * (E[l].mean(0) - mu), atol=1e-3)
        # per-language concept variance shrinks by (1-m)^2
        np.testing.assert_allclose(out[l].var(0).sum() / E[l].var(0).sum(),
                                   (1 - m) ** 2, rtol=1e-3)
    np.testing.assert_array_equal(out['eng'], E['eng'])
