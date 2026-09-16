"""Nested mapping ladder M0-M5: the misalignment budget.

Convention (pinned): map language into consensus frame, X_l -> Z.
    M0: X ~= Z                      (shared space only, no fit)
    M1: + offset b
    M2: + isotropic scale s         (fertility/length channel, kept explicit)
    M3: + orthogonal R              (full similarity Procrustes)
    M4: + general linear A          (ridge, lambda by inner CV on train)
    M5: + nonlinear                 (RBF kernel ridge, bandwidth+lambda by
                                     inner CV; cheap detector = kernel-vs-
                                     linear CKA gap)

Budget entries are held-out error reductions (E_{k-1} - E_k) / E_0 per rung,
computed on TEST concepts only; ALL fits on TRAIN. Negative held-out
contributions are REPORTED, not clipped (overfit signal). Canonical order is
frozen (offset -> scale -> rotation -> linear -> nonlinear): rotation is
defined on centered data and scale-before-rotation is the similarity-
Procrustes convention.

Gauge: budget entries are gauge-invariant by construction — reconstruction
error does not depend on the orientation of Z.

Permutation null: shuffled concept pairing (train fit AND test eval both on
shuffled pairing), B draws; a rung's share is only claimed nonzero if it
clears its own null. Code emits numbers; judgment happens against the frozen
prereg.
"""
import numpy as np

from .gpa import similarity_procrustes

RIDGE_LAMBDAS = np.logspace(-2, 3, 8)
KRR_LAMBDAS = np.logspace(-3, 1, 5)
KRR_GAMMA_SCALES = (0.25, 1.0, 4.0)
RUNGS = ('M0', 'M1', 'M2', 'M3', 'M4', 'M5')


def _sse(pred, Z):
    return float(((pred - Z) ** 2).sum())


def _fit_offset(X, Z):
    b = (Z - X).mean(0)
    return lambda Xn: Xn + b


def _fit_scale(X, Z):
    xm, zm = X.mean(0), Z.mean(0)
    Xc, Zc = X - xm, Z - zm
    denom = (Xc ** 2).sum()
    s = float((Xc * Zc).sum() / denom) if denom > 0 else 1.0
    b = zm - s * xm
    return lambda Xn: s * Xn + b


def _fit_similarity(X, Z):
    s, R, b = similarity_procrustes(X, Z)
    return lambda Xn: s * Xn @ R + b


def _fit_ridge(X, Z, lambdas=RIDGE_LAMBDAS, inner_folds=3, seed=0):
    xm, zm = X.mean(0), Z.mean(0)
    Xc, Zc = X - xm, Z - zm
    lam = _select_ridge_lambda(Xc, Zc, lambdas, inner_folds, seed)
    A = _ridge_solve(Xc, Zc, lam)
    return (lambda Xn: (Xn - xm) @ A + zm), lam


def _ridge_solve(Xc, Zc, lam):
    r = Xc.shape[1]
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(r), Xc.T @ Zc)


def _select_ridge_lambda(Xc, Zc, lambdas, folds, seed):
    idx = _fold_indices(len(Xc), folds, seed)
    errs = np.zeros(len(lambdas))
    for f in range(folds):
        te = idx == f
        tr = ~te
        for i, lam in enumerate(lambdas):
            A = _ridge_solve(Xc[tr], Zc[tr], lam)
            errs[i] += ((Xc[te] @ A - Zc[te]) ** 2).sum()
    return float(lambdas[int(np.argmin(errs))])


def _fit_krr(X, Z, lambdas=KRR_LAMBDAS, gamma_scales=KRR_GAMMA_SCALES,
             inner_folds=3, seed=0):
    """Multivariate RBF kernel ridge. gamma grid = median heuristic x scales."""
    d2 = _sq_dists(X, X)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    gammas = [s / med for s in gamma_scales]
    idx = _fold_indices(len(X), inner_folds, seed)
    best, best_err = None, np.inf
    for g in gammas:
        for lam in lambdas:
            err = 0.0
            for f in range(inner_folds):
                te = idx == f
                tr = ~te
                a = _krr_solve(X[tr], Z[tr], g, lam)
                err += ((_rbf(X[te], X[tr], g) @ a - Z[te]) ** 2).sum()
            if err < best_err:
                best_err, best = err, (g, lam)
    g, lam = best
    alpha = _krr_solve(X, Z, g, lam)
    Xtr = X.copy()
    return (lambda Xn: _rbf(Xn, Xtr, g) @ alpha), {'gamma': g, 'lambda': lam}


def _krr_solve(X, Z, gamma, lam):
    K = _rbf(X, X, gamma)
    return np.linalg.solve(K + lam * len(X) * np.eye(len(X)), Z)


def _rbf(A, B, gamma):
    return np.exp(-gamma * _sq_dists(A, B))


def _sq_dists(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def _fold_indices(n, folds, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n) % folds
    rng.shuffle(idx)
    return idx


def fit_ladder(X_tr, Z_tr, X_te, Z_te, seed=0, fit_m5=True):
    """Fit all rungs on train, evaluate SSE on test. Returns errors, shares
    (held-out error reduction per rung / E0), and selected hyperparams.
    fit_m5=False skips the kernel rung (used by M3/M4 permutation nulls,
    where B x languages x splits kernel refits would dominate compute)."""
    maps = {'M0': lambda Xn: Xn}
    maps['M1'] = _fit_offset(X_tr, Z_tr)
    maps['M2'] = _fit_scale(X_tr, Z_tr)
    maps['M3'] = _fit_similarity(X_tr, Z_tr)
    maps['M4'], lam4 = _fit_ridge(X_tr, Z_tr, seed=seed)
    hyper = {'ridge_lambda': lam4}
    rungs = RUNGS if fit_m5 else RUNGS[:-1]
    if fit_m5:
        maps['M5'], hp5 = _fit_krr(X_tr, Z_tr, seed=seed)
        hyper.update(hp5)
    E = {k: _sse(maps[k](X_te), Z_te) for k in rungs}
    E0 = E['M0'] if E['M0'] > 0 else 1.0
    shares = {f'share_{rungs[i]}': (E[rungs[i - 1]] - E[rungs[i]]) / E0
              for i in range(1, len(rungs))}
    return {'errors': E, 'shares': shares, 'hyper': hyper}


def ladder_with_splits(X, Z, n_splits=20, train_frac=0.5, seed=0):
    """Cross-fitted budget: repeat over S random concept splits; report
    per-split results (mean +- SD is computed downstream)."""
    n = len(X)
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_splits):
        perm = rng.permutation(n)
        k = int(n * train_frac)
        tr, te = perm[:k], perm[k:]
        out.append(fit_ladder(X[tr], Z[tr], X[te], Z[te], seed=seed + s))
    return out


def permutation_null(X_tr, Z_tr, X_te, Z_te, B=200, rungs=('M3', 'M4', 'M5'),
                     seed=0):
    """Null distribution of rung shares under shuffled concept pairing.
    Pairing is shuffled consistently: train fit on shuffled train pairs, test
    eval on (independently) shuffled test pairs — no exploitable
    correspondence remains beyond distributional match."""
    rng = np.random.default_rng(seed)
    draws = {r: [] for r in rungs}
    need_m5 = 'M5' in rungs
    for _ in range(B):
        pt = rng.permutation(len(Z_tr))
        pe = rng.permutation(len(Z_te))
        res = fit_ladder(X_tr, Z_tr[pt], X_te, Z_te[pe], seed=seed,
                         fit_m5=need_m5)
        for r in rungs:
            draws[r].append(res['shares'][f'share_{r}'])
    return {r: np.asarray(v) for r, v in draws.items()}


def cka_linear(X, Z):
    """Linear CKA (Kornblith et al. 2019), centered."""
    Xc, Zc = X - X.mean(0), Z - Z.mean(0)
    num = np.linalg.norm(Xc.T @ Zc, 'fro') ** 2
    den = (np.linalg.norm(Xc.T @ Xc, 'fro') *
           np.linalg.norm(Zc.T @ Zc, 'fro'))
    return float(num / den) if den > 0 else 0.0


def cka_rbf(X, Z, gamma_scale=1.0):
    """RBF-kernel CKA with median-heuristic bandwidth x gamma_scale."""
    def _gram(A):
        d2 = _sq_dists(A, A)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        return np.exp(-gamma_scale * d2 / med)

    def _center(K):
        n = len(K)
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    Kx, Kz = _center(_gram(X)), _center(_gram(Z))
    num = (Kx * Kz).sum()
    den = np.sqrt((Kx * Kx).sum() * (Kz * Kz).sum())
    return float(num / den) if den > 0 else 0.0
