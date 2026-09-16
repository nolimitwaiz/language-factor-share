"""Shared subspace estimation: pooled PCA + Ledoit-Wolf eigenvalue shrinkage
+ Marchenko-Pastur edge as a sanity ceiling on retainable rank.

Notes pinned by spec:
  * Ledoit-Wolf shrinkage toward a scaled identity does NOT change the
    eigenbasis of the empirical covariance — only the eigenvalue spectrum.
    We therefore take eigenvectors from the empirical covariance and report
    the shrunk spectrum alongside for MP comparisons.
  * The MP edge is a CEILING on how many components are distinguishable from
    noise; final rank selection happens in the ladder via held-out
    reconstruction error of the most flexible linear model (one-SE rule),
    never on downstream numbers.
"""
import numpy as np


def pooled_pca(X, r):
    """X: [n_samples, D] fp32 pooled pre-z embeddings (all langs stacked).
    Returns dict with mean, top-r basis U [D, r], full eigenvalue spectrum
    (empirical + LW-shrunk), and LW shrinkage intensity."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    # economical eigendecomposition via SVD of the data matrix
    U_svd, S, _ = np.linalg.svd(Xc, full_matrices=False)
    eig = (S ** 2) / (n - 1)                      # empirical covariance spectrum
    # Ledoit-Wolf intensity (Ledoit & Wolf 2004, target = mu_LW * I)
    shrink, mu_lw = _lw_intensity(Xc, eig)
    eig_lw = (1 - shrink) * eig + shrink * mu_lw
    basis = (Xc.T @ U_svd[:, :r]) / S[:r]         # top-r right singular vectors
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    return {'mean': mu, 'basis': basis.astype(np.float64), 'eig': eig,
            'eig_lw': eig_lw, 'lw_shrinkage': float(shrink), 'n': n, 'd': d}


def _lw_intensity(Xc, eig):
    """Ledoit-Wolf optimal shrinkage intensity toward mu*I (2004 estimator)."""
    n, d = Xc.shape
    mu = eig.sum() / d                             # = trace(S)/d
    # alpha2 = ||S - mu I||_F^2 / d ; beta2 = E||xxT - S||^2 (LW b2), capped
    alpha2 = (eig ** 2).sum() / d - mu ** 2 + mu ** 2 * max(d - len(eig), 0) / d
    sq_norms = (Xc ** 2).sum(1)
    # E ||x xT||_F^2 term computed without forming DxD matrices:
    # ||S||_F^2 = sum(eig^2); sum_i ||x_i x_iT||_F^2 = sum_i ||x_i||^4
    b2 = ((sq_norms ** 2).sum() / n - (eig ** 2).sum()) / n
    b2 = max(min(b2, alpha2), 0.0)
    shrink = 0.0 if alpha2 <= 0 else b2 / alpha2
    return float(np.clip(shrink, 0.0, 1.0)), float(mu)


def mp_edge(n, d, sigma2=1.0):
    """Marchenko-Pastur upper bulk edge for sample covariance of n iid samples
    in d dims with true covariance sigma2*I: sigma2 * (1 + sqrt(d/n))^2."""
    gamma = d / n
    return sigma2 * (1.0 + np.sqrt(gamma)) ** 2


def mp_rank_ceiling(eig, n, d, sigma2=None):
    """Number of eigenvalues above the MP edge. If sigma2 is None, estimate it
    from the bulk by matching the median eigenvalue to the MP median."""
    eig = np.sort(np.asarray(eig))[::-1]
    if sigma2 is None:
        sigma2 = _sigma2_from_bulk(eig, n, d)
    edge = mp_edge(n, d, sigma2)
    return int((eig > edge).sum()), float(edge), float(sigma2)


def _sigma2_from_bulk(eig, n, d):
    """sigma2 estimate: median(eig) / median of the standard MP law (sigma2=1).
    Robust to a small number of spikes."""
    med_emp = np.median(eig[eig > 0])
    med_mp = _mp_median(d / n)
    return float(med_emp / med_mp)


def _mp_median(gamma, grid=20001):
    """Median of the Marchenko-Pastur distribution with ratio gamma, sigma2=1,
    computed numerically from the density on [(1-sqrt g)^2, (1+sqrt g)^2]."""
    g = min(gamma, 1.0)                            # d>n: density on same support
    lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2
    x = np.linspace(lo + 1e-12, hi - 1e-12, grid)
    dens = np.sqrt((hi - x) * (x - lo)) / (2 * np.pi * g * x)
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    return float(x[np.searchsorted(cdf, 0.5)])
