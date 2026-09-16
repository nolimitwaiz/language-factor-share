"""Synthetic injection framework (battery I1-I10).

Injection point pinned by spec: pooled, PRE-z-score fp32 embeddings, one
(model, layer) at a time: E = {lang: [N, D]}. After injection the FULL
standard pipeline runs untouched (z-scoring included) — the battery tests the
pipeline end to end, not the metric in isolation.

Every injector returns (E_new, meta) where meta records the exact injected
quantities (per-language angles, scales, ...) for calibration curves.
Magnitude grids and language subsets are [DECIDE] fields in the prereg.
"""
import numpy as np

PIVOT = 'eng'


def _targets(E, include_pivot=False):
    return [l for l in E if include_pivot or l != PIVOT]


def _rms(E):
    return float(np.sqrt(np.mean([np.mean(v.astype(np.float64) ** 2)
                                  for v in E.values()])))


def i1_offset(E, magnitude, rng, include_pivot=False):
    """Additive per-language offset: x += m * rms * u_l (u_l random unit)."""
    scale = magnitude * _rms(E)
    out, meta = {}, {'offsets': {}}
    for l, X in E.items():
        if l in _targets(E, include_pivot):
            u = rng.standard_normal(X.shape[1])
            u /= np.linalg.norm(u)
            out[l] = X + (scale * u).astype(X.dtype)
            meta['offsets'][l] = scale
        else:
            out[l] = X.copy()
    return out, meta


def i2_scale(E, magnitude, rng, include_pivot=False):
    """Isotropic per-language scale: x *= s_l, s_l log-uniform around 1."""
    out, meta = {}, {'scales': {}}
    for l, X in E.items():
        if l in _targets(E, include_pivot):
            s = float(np.exp(rng.uniform(-np.log1p(magnitude),
                                         np.log1p(magnitude))))
            out[l] = (X * s).astype(X.dtype)
            meta['scales'][l] = s
        else:
            out[l] = X.copy()
    return out, meta


def i3_global_rotation(E, rng):
    """Shared GLOBAL rotation (gauge/pipeline control): the SAME full-D
    orthogonal Q applied to every language including the pivot.
    Pre-normalization quantities are exactly invariant; the post-z-scoring
    pipeline is only APPROXIMATELY invariant (per-dimension z-scoring is not
    rotation-invariant) — the measured deviation is the anisotropy guard's
    gauge sensitivity."""
    D = next(iter(E.values())).shape[1]
    Q = _random_orthogonal(D, rng)
    return {l: (X @ Q).astype(X.dtype) for l, X in E.items()}, {'D': D}


def i4_lang_rotation(E, theta, r, basis, mean, rng, include_pivot=False):
    """Language-specific rotation by angle theta in the top-r subspace:
    rotate the subspace component about the pooled mean through r//2 disjoint
    random 2-planes, each by exactly theta (calibration target)."""
    out, meta = {}, {'theta': float(theta), 'r': r, 'planes': {}}
    for l, X in E.items():
        if l not in _targets(E, include_pivot):
            out[l] = X.copy()
            continue
        pairs = _random_plane_pairs(r, rng)
        R = _plane_rotation(r, pairs, theta)
        Zc = (X - mean) @ basis                       # [N, r] subspace coords
        delta = Zc @ (R - np.eye(r)) @ basis.T
        out[l] = (X + delta).astype(X.dtype)
        meta['planes'][l] = pairs
    return out, meta


def i4b_lang_rotation_centered(E, theta, r, basis, rng, include_pivot=False):
    """ADDENDUM A1: centroid-preserving language rotation. Rotates each
    injected language's cloud about its OWN centroid in the top-r subspace:
    x' = c_l + R(x - c_l). Unlike I4 (pooled-mean rotation), the language
    centroid is unchanged EXACTLY, so no injected energy leaks into the
    budget's offset rung — pure within-cloud rotation."""
    out, meta = {}, {'theta': float(theta), 'r': r, 'planes': {}}
    for l, X in E.items():
        if l not in _targets(E, include_pivot):
            out[l] = X.copy()
            continue
        pairs = _random_plane_pairs(r, rng)
        R = _plane_rotation(r, pairs, theta)
        c = X.mean(0, keepdims=True)
        Zc = (X - c) @ basis                          # centered subspace coords
        delta = Zc @ (R - np.eye(r)) @ basis.T
        out[l] = (X + delta).astype(X.dtype)
        meta['planes'][l] = pairs
    return out, meta


def i8b_global_collapse(E, magnitude, include_pivot=False):
    """ADDENDUM A1: collapse toward the GLOBAL centroid (the Aim-2 study's
    gaming mode): x' = mu_global + (1-m)(x - mu_global) for non-pivot
    languages. Shrinks language offsets AND concept variance together —
    predicted to LOWER LFS (the Goodhart direction) while CVP collapses."""
    mu = np.mean([X.mean(0) for X in E.values()], axis=0, keepdims=True)
    out, meta = {}, {'magnitude': magnitude}
    for l, X in E.items():
        if l not in _targets(E, include_pivot):
            out[l] = X.copy()
            continue
        out[l] = (mu + (1.0 - magnitude) * (X - mu)).astype(X.dtype)
    return out, meta


def i5_shear(E, magnitude, r, basis, mean, rng, include_pivot=False):
    """Anisotropic per-language scale (shear) in the top-r subspace:
    diag(1 + m*u_i), u ~ U(-1, 1)."""
    out, meta = {}, {'diags': {}}
    for l, X in E.items():
        if l not in _targets(E, include_pivot):
            out[l] = X.copy()
            continue
        d = 1.0 + magnitude * rng.uniform(-1, 1, size=r)
        Zc = (X - mean) @ basis
        delta = Zc @ (np.diag(d) - np.eye(r)) @ basis.T
        out[l] = (X + delta).astype(X.dtype)
        meta['diags'][l] = d.tolist()
    return out, meta


def i6_interaction(E, magnitude, k, r, basis, mean, rng, include_pivot=False):
    """Concept-dependent low-rank interaction: per-language rank-k linear map
    in the subspace, unit spectral norm, applied at strength m:
    z' = z + m * z B_l."""
    out, meta = {}, {'k': k, 'magnitude': magnitude}
    for l, X in E.items():
        if l not in _targets(E, include_pivot):
            out[l] = X.copy()
            continue
        P = rng.standard_normal((r, k))
        Q = rng.standard_normal((k, r))
        B = P @ Q
        B /= np.linalg.norm(B, 2)                     # unit spectral norm
        Zc = (X - mean) @ basis
        delta = magnitude * Zc @ B @ basis.T
        out[l] = (X + delta).astype(X.dtype)
    return out, meta


def i7_warp(E, magnitude, r, basis, mean, rng, include_pivot=False):
    """Monotone nonlinear warp, language-specific: mix subspace coords with a
    per-language random rotation, tanh-squash (a*tanh(z/a), compressive),
    unmix. magnitude m sets a = std(z)/m — larger m = stronger squash."""
    out, meta = {}, {'magnitude': magnitude}
    for l, X in E.items():
        if l not in _targets(E, include_pivot):
            out[l] = X.copy()
            continue
        M = _random_orthogonal(r, rng)                # per-language mixing
        Zc = (X - mean) @ basis @ M
        a = Zc.std(0, keepdims=True) / max(magnitude, 1e-9) + 1e-9
        Zw = a * np.tanh(Zc / a)
        delta = (Zw - Zc) @ M.T @ basis.T
        out[l] = (X + delta).astype(X.dtype)
    return out, meta


def i8_collapse(E, magnitude, include_pivot=False):
    """Partial collapse: shrink concept variance toward each language's own
    centroid: x' = xbar_l + (1 - m)(x - xbar_l). m=1 -> full collapse."""
    out, meta = {}, {'magnitude': magnitude}
    for l, X in E.items():
        if l not in _targets(E, include_pivot):
            out[l] = X.copy()
            continue
        c = X.mean(0, keepdims=True)
        out[l] = (c + (1.0 - magnitude) * (X - c)).astype(X.dtype)
    return out, meta


def i9_permute(E, rng, include_pivot=False):
    """Pairing permutation null: shuffle concept correspondence within each
    non-pivot language."""
    out, meta = {}, {'perms': {}}
    for l, X in E.items():
        if l not in _targets(E, include_pivot):
            out[l] = X.copy()
            continue
        p = rng.permutation(len(X))
        out[l] = X[p].copy()
        meta['perms'][l] = p.tolist()
    return out, meta


def i10_identity(E):
    """No-op. Run TWICE on independently resampled sentence halves: rows are
    the noise floor AND the split-half reliability samples."""
    return {l: X.copy() for l, X in E.items()}, {}


def _random_orthogonal(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def _random_plane_pairs(r, rng):
    idx = rng.permutation(r)
    return [(int(idx[2 * i]), int(idx[2 * i + 1])) for i in range(r // 2)]


def _plane_rotation(r, pairs, theta):
    R = np.eye(r)
    c, s = np.cos(theta), np.sin(theta)
    for (i, j) in pairs:
        G = np.eye(r)
        G[i, i] = G[j, j] = c
        G[i, j], G[j, i] = -s, s
        R = R @ G
    return R
