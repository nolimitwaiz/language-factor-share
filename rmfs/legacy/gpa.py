"""Generalized Procrustes consensus + relative rotations.

Conventions pinned by spec (do not change silently):
  * Map each language INTO the consensus frame: s_l * X_l @ R_l + b_l ~= Z.
  * Consensus transforms are fit on TRAIN concepts only. For held-out test
    concepts the consensus target is built by applying the FROZEN train
    transforms to test rows and averaging across languages (self-inclusion
    bias ~1/L — disclosed, not corrected; leave-one-out target available as
    a sensitivity alternative via `consensus_target(..., exclude=lang)`).
  * GAUGE: a global re-orientation Z -> Z Q sends R_l -> R_l Q, so any
    rotation summary against the consensus is reference-dependent. Only
    pairwise RELATIVE rotations R_l R_m^T are exactly invariant; summarize
    those via the principal-angle spectrum. NEVER headline ||log R_l||
    against the consensus.
"""
import numpy as np


def orthogonal_procrustes(X, Z):
    """R = argmin_{R orthogonal} ||X R - Z||_F. Returns R [r, r]."""
    U, _, Vt = np.linalg.svd(X.T @ Z)
    return U @ Vt


def similarity_procrustes(X, Z):
    """Full similarity fit: s * X @ R + b ~= Z (centered Procrustes with
    isotropic scale). Returns (s, R, b)."""
    xm, zm = X.mean(0), Z.mean(0)
    Xc, Zc = X - xm, Z - zm
    U, S, Vt = np.linalg.svd(Xc.T @ Zc)
    R = U @ Vt
    denom = (Xc ** 2).sum()
    s = float(S.sum() / denom) if denom > 0 else 1.0
    b = zm - s * xm @ R
    return s, R, b


def gpa(X_by_lang, n_iter=50, tol=1e-9):
    """Generalized Procrustes over languages. X_by_lang: {lang: [N, r]} TRAIN
    rows in the shared subspace. Returns dict with per-language (s, R, b),
    consensus Z_train, and convergence history."""
    langs = sorted(X_by_lang)
    Z = np.mean([X_by_lang[l] for l in langs], axis=0)
    Z = Z - Z.mean(0)
    scale0 = np.linalg.norm(Z)
    hist = []
    fits = {}
    for _ in range(n_iter):
        aligned = []
        for l in langs:
            s, R, b = similarity_procrustes(X_by_lang[l], Z)
            fits[l] = (s, R, b)
            aligned.append(s * X_by_lang[l] @ R + b)
        Z_new = np.mean(aligned, axis=0)
        Z_new = Z_new - Z_new.mean(0)
        nrm = np.linalg.norm(Z_new)
        if nrm > 0:                                # freeze scale: no shrink drift
            Z_new = Z_new * (scale0 / nrm)
        delta = float(np.linalg.norm(Z_new - Z) / (np.linalg.norm(Z) + 1e-12))
        hist.append(delta)
        Z = Z_new
        if delta < tol:
            break
    # GAUGE ANCHOR (spec refinement, disclosed): the consensus orientation is
    # arbitrary up to a global rotation, and the M0-M2 budget entries are NOT
    # invariant to it (only rungs >= M3 absorb a re-orientation). We therefore
    # pin the gauge deterministically: Procrustes-align the converged Z back
    # to the raw cross-language mean configuration. All budget entries are
    # then well-defined; rungs >= M3 remain exactly gauge-invariant anyway.
    Z0 = np.mean([X_by_lang[l] for l in langs], axis=0)
    Z0 = Z0 - Z0.mean(0)
    Ra = orthogonal_procrustes(Z, Z0)
    Z = Z @ Ra
    fits = {l: (s, R @ Ra, b @ Ra) for l, (s, R, b) in fits.items()}
    return {'langs': langs, 'fits': fits, 'Z_train': Z, 'history': hist}


def consensus_target(g, X_by_lang_rows, exclude=None):
    """Consensus for arbitrary rows (e.g. TEST concepts): apply the FROZEN
    train transforms to each language's rows and average. `exclude` drops one
    language from the average (leave-one-out sensitivity target)."""
    acc, cnt = None, 0
    for l in g['langs']:
        if l == exclude or l not in X_by_lang_rows:
            continue
        s, R, b = g['fits'][l]
        a = s * X_by_lang_rows[l] @ R + b
        acc = a if acc is None else acc + a
        cnt += 1
    return acc / cnt


def relative_rotation_angles(R_a, R_b):
    """Principal-angle spectrum of the relative rotation R_a R_b^T, from the
    complex eigenvalues e^{+-i theta} of the orthogonal matrix. Returns sorted
    angles (radians, >=0) and the geodesic norm sqrt(sum theta^2).
    Exactly invariant to a global re-orientation of the consensus."""
    M = R_a @ R_b.T
    ev = np.linalg.eigvals(M)
    ang = np.abs(np.angle(ev))
    ang = np.sort(ang)[::-1]
    # each rotation plane contributes a conjugate pair; keep one per pair
    ang = ang[::2] if len(ang) > 1 else ang
    return ang, float(np.sqrt((ang ** 2).sum()))
