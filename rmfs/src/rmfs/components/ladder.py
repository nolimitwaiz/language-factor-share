"""M0-M5 mapping ladder with purged cross-fitting and one-SE selection
(Phase 1.5; prereg §3.3).

Ported from the legacy pipeline (legacy/ladder.py, legacy/gpa.py) with the
rung mathematics kept IDENTICAL — the unit battery asserts exact numerical
parity of `fit_ladder` against the legacy implementation on shared inputs,
so the port cannot drift silently. Two changes, both mandated by the
kickoff and the Phase 0 audit:

1. Cross-fitting is DOCUMENT-PURGED and concept-disjoint
   (`rmfs.data.ntrex.purged_concept_splits`). The legacy length-stratified
   sentence splits leaked 99.9% of held-out sentences into training
   documents (ERROR_LEDGER E1); the purged re-estimate showed the M5 rung's
   overfitting had been understated by ~6pp.

2. Rung selection by the ONE-SE RULE on held-out reconstruction error:
   the smallest rung whose mean held-out error is within one standard
   error of the best rung's mean. K = selected_rung / 5, q_K = 1 - K.
   Selection is on ERRORS, never on shares (shares can be legitimately
   negative — that is the overfit signal, reported not clipped).

Rungs (canonical order frozen; see legacy docstring for the gauge
conventions, which are preserved verbatim):
    M0 identity | M1 +offset | M2 +isotropic scale | M3 +orthogonal
    (similarity Procrustes) | M4 +general linear (ridge, inner CV)
    | M5 +nonlinear (RBF kernel ridge, median-heuristic bandwidth grid)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RungSelection", "fit_ladder", "gpa", "consensus_target",
           "ladder_purged", "select_rung", "similarity_procrustes",
           "orthogonal_procrustes"]

RIDGE_LAMBDAS = np.logspace(-2, 3, 8)
KRR_LAMBDAS = np.logspace(-3, 1, 5)
KRR_GAMMA_SCALES = (0.25, 1.0, 4.0)
RUNGS = ("M0", "M1", "M2", "M3", "M4", "M5")


# --------------------------------------------------------------- procrustes
def orthogonal_procrustes(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(X.T @ Z)
    return U @ Vt


def similarity_procrustes(X: np.ndarray, Z: np.ndarray):
    """s * X @ R + b ~= Z (centered Procrustes with isotropic scale)."""
    xm, zm = X.mean(0), Z.mean(0)
    Xc, Zc = X - xm, Z - zm
    U, S, Vt = np.linalg.svd(Xc.T @ Zc)
    R = U @ Vt
    denom = (Xc**2).sum()
    s = float(S.sum() / denom) if denom > 0 else 1.0
    b = zm - s * xm @ R
    return s, R, b


def gpa(X_by_lang: dict, n_iter: int = 50, tol: float = 1e-9) -> dict:
    """Generalized Procrustes consensus over languages (legacy-verbatim,
    including the frozen-scale iteration and the deterministic gauge anchor
    back to the raw cross-language mean configuration)."""
    langs = sorted(X_by_lang)
    Z = np.mean([X_by_lang[lang] for lang in langs], axis=0)
    Z = Z - Z.mean(0)
    scale0 = np.linalg.norm(Z)
    hist, fits = [], {}
    for _ in range(n_iter):
        aligned = []
        for lang in langs:
            s, R, b = similarity_procrustes(X_by_lang[lang], Z)
            fits[lang] = (s, R, b)
            aligned.append(s * X_by_lang[lang] @ R + b)
        Z_new = np.mean(aligned, axis=0)
        Z_new = Z_new - Z_new.mean(0)
        nrm = np.linalg.norm(Z_new)
        if nrm > 0:
            Z_new = Z_new * (scale0 / nrm)
        delta = float(np.linalg.norm(Z_new - Z) / (np.linalg.norm(Z) + 1e-12))
        hist.append(delta)
        Z = Z_new
        if delta < tol:
            break
    Z0 = np.mean([X_by_lang[lang] for lang in langs], axis=0)
    Z0 = Z0 - Z0.mean(0)
    Ra = orthogonal_procrustes(Z, Z0)
    Z = Z @ Ra
    fits = {lang: (s, R @ Ra, b @ Ra) for lang, (s, R, b) in fits.items()}
    return {"langs": langs, "fits": fits, "Z_train": Z, "history": hist}


def consensus_target(g: dict, X_by_lang_rows: dict,
                     exclude: str | None = None) -> np.ndarray:
    """Consensus for arbitrary rows: apply the FROZEN train transforms and
    average across languages (optionally excluding one, the leave-one-out
    sensitivity variant; self-inclusion bias ~1/L is disclosed, not
    corrected — legacy convention preserved)."""
    rows = []
    for lang, X in X_by_lang_rows.items():
        if lang == exclude:
            continue
        s, R, b = g["fits"][lang]
        rows.append(s * X @ R + b)
    if not rows:
        raise ValueError("consensus_target: no languages left after exclude")
    return np.mean(rows, axis=0)


# -------------------------------------------------------------- rung fits
def _sse(pred, Z):
    return float(((pred - Z) ** 2).sum())


def _fit_offset(X, Z):
    b = (Z - X).mean(0)
    return lambda Xn: Xn + b


def _fit_scale(X, Z):
    xm, zm = X.mean(0), Z.mean(0)
    Xc, Zc = X - xm, Z - zm
    denom = (Xc**2).sum()
    s = float((Xc * Zc).sum() / denom) if denom > 0 else 1.0
    b = zm - s * xm
    return lambda Xn: s * Xn + b


def _fit_similarity(X, Z):
    s, R, b = similarity_procrustes(X, Z)
    return lambda Xn: s * Xn @ R + b


def _ridge_solve(Xc, Zc, lam):
    r = Xc.shape[1]
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(r), Xc.T @ Zc)


def _fold_indices(n, folds, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n) % folds
    rng.shuffle(idx)
    return idx


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


def _fit_ridge(X, Z, lambdas=RIDGE_LAMBDAS, inner_folds=3, seed=0):
    xm, zm = X.mean(0), Z.mean(0)
    Xc, Zc = X - xm, Z - zm
    lam = _select_ridge_lambda(Xc, Zc, lambdas, inner_folds, seed)
    A = _ridge_solve(Xc, Zc, lam)
    return (lambda Xn: (Xn - xm) @ A + zm), lam


def _sq_dists(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def _rbf(A, B, gamma):
    return np.exp(-gamma * _sq_dists(A, B))


def _krr_solve(X, Z, gamma, lam):
    K = _rbf(X, X, gamma)
    return np.linalg.solve(K + lam * len(X) * np.eye(len(X)), Z)


def _fit_krr(X, Z, lambdas=KRR_LAMBDAS, gamma_scales=KRR_GAMMA_SCALES,
             inner_folds=3, seed=0):
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
    return (lambda Xn: _rbf(Xn, Xtr, g) @ alpha), {"gamma": g, "lambda": lam}


def fit_ladder(X_tr, Z_tr, X_te, Z_te, seed=0, fit_m5=True) -> dict:
    """Fit all rungs on train, evaluate SSE on held-out. Legacy-identical:
    the unit battery asserts exact parity. Negative shares are reported."""
    maps = {"M0": lambda Xn: Xn}
    maps["M1"] = _fit_offset(X_tr, Z_tr)
    maps["M2"] = _fit_scale(X_tr, Z_tr)
    maps["M3"] = _fit_similarity(X_tr, Z_tr)
    maps["M4"], lam4 = _fit_ridge(X_tr, Z_tr, seed=seed)
    hyper = {"ridge_lambda": lam4}
    rungs = RUNGS if fit_m5 else RUNGS[:-1]
    if fit_m5:
        maps["M5"], hp5 = _fit_krr(X_tr, Z_tr, seed=seed)
        hyper.update(hp5)
    E = {k: _sse(maps[k](X_te), Z_te) for k in rungs}
    E0 = E["M0"] if E["M0"] > 0 else 1.0
    shares = {f"share_{rungs[i]}": (E[rungs[i - 1]] - E[rungs[i]]) / E0
              for i in range(1, len(rungs))}
    return {"errors": E, "shares": shares, "hyper": hyper}


# ---------------------------------------------------- purged cross-fitting
@dataclass(frozen=True)
class RungSelection:
    """One-SE rung selection with its diagnostics."""

    rung: str                  # e.g. "M1"
    K: float                   # rung index / 5
    q_K: float                 # 1 - K
    mean_errors: dict          # rung -> mean held-out SSE over splits
    se_best: float             # SE of the best rung's error
    n_splits: int
    per_split: tuple           # raw fit_ladder outputs, kept for shares


def ladder_purged(X, Z, doc_ids, n_splits: int, seed: int,
                  strata=None, fit_m5: bool = True) -> list[dict]:
    """Cross-fitted ladder over document-purged concept splits."""
    from rmfs.data.ntrex import purged_concept_splits

    splits = purged_concept_splits(doc_ids, n_splits, seed, strata=strata)
    return [fit_ladder(X[tr], Z[tr], X[te], Z[te], seed=seed + s,
                       fit_m5=fit_m5)
            for s, (tr, te) in enumerate(splits)]


def select_rung(per_split: list[dict]) -> RungSelection:
    """One-SE rule on held-out reconstruction error.

    The best rung is the one with the lowest mean error; the SELECTED rung
    is the smallest whose mean error is within one standard error of that
    minimum. Conservative by construction: complexity must earn its keep
    beyond split noise — the Phase 0 audit showed leaky splits made the M5
    rung look ~6pp better than it is, which is exactly the mistake this
    rule guards against."""
    rungs = [r for r in RUNGS if r in per_split[0]["errors"]]
    S = len(per_split)
    errs = {r: np.array([d["errors"][r] for d in per_split]) for r in rungs}
    means = {r: float(errs[r].mean()) for r in rungs}
    best = min(rungs, key=lambda r: means[r])
    se = (float(errs[best].std(ddof=1) / np.sqrt(S)) if S > 1 else 0.0)
    selected = next(r for r in rungs if means[r] <= means[best] + se)
    idx = RUNGS.index(selected)
    return RungSelection(rung=selected, K=idx / 5.0, q_K=1.0 - idx / 5.0,
                         mean_errors=means, se_best=se, n_splits=S,
                         per_split=tuple(per_split))
