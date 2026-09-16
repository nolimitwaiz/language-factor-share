"""Tournament core (kickoff 3.3): S5/S6/S7 statistics.

All statistics use the language-cluster bootstrap (languages are the
exchangeable units; rows within a language share a cluster) with the
frozen 2,000 reps and seed 13 unless stated.

- residual_spearman: both sides residualized on the deflation covariates
  (log_tokens + macro-family + script groups + fertility) by OLS within
  the pooled frame, then Spearman. S5's statistic.
- orthogonalized_increment: candidate residualized on the shared-prep
  incumbent (per prereg S7), residual correlated with the target under
  the same deflation controls.
- romano_wolf: stepdown maxT over a pre-registered candidate family,
  cluster-bootstrap null. Controls FWER for the family comparison table.
- effective_breadth: Grinold N_eff from the mean pairwise cluster
  correlation of per-language signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

COVARIATES = ["log_tokens", "fertility"]
CATEGORICAL = ["macro_family_g", "script_g"]


def _design(df: pd.DataFrame) -> np.ndarray:
    cols = [np.ones(len(df))]
    for c in COVARIATES:
        cols.append(df[c].astype(float).values)
    for c in CATEGORICAL:
        d = pd.get_dummies(df[c].astype(str), drop_first=True)
        for k in d.columns:
            cols.append(d[k].astype(float).values)
    return np.column_stack(cols)


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


@dataclass(frozen=True)
class ResidCorr:
    rho: float
    ci_lo: float
    ci_hi: float
    p_boot: float          # two-sided bootstrap p for rho = 0
    n_rows: int
    n_langs: int


def _cluster_boot(df: pd.DataFrame, stat_fn, n_boot: int = 2000,
                  seed: int = 13) -> tuple[float, float, float, float]:
    langs = df.flores_code.unique()
    by = {lg: df[df.flores_code == lg] for lg in langs}
    point = stat_fn(df)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(langs, size=len(langs), replace=True)
        try:
            vals.append(stat_fn(pd.concat([by[lg] for lg in pick])))
        except Exception:
            continue
    v = np.array([x for x in vals if np.isfinite(x)])
    lo, hi = np.percentile(v, [2.5, 97.5])
    # two-sided percentile p: fraction of bootstrap mass beyond 0
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return float(point), float(lo), float(hi), float(min(max(p, 1 / len(v)),
                                                         1.0))


def residual_spearman(df: pd.DataFrame, cand: str, target: str,
                      n_boot: int = 2000, seed: int = 13) -> ResidCorr:
    d = df.dropna(subset=[cand, target, *COVARIATES,
                          *CATEGORICAL]).copy()

    def stat(sample: pd.DataFrame) -> float:
        X = _design(sample)
        rc = _residualize(sample[cand].astype(float).values, X)
        rt = _residualize(sample[target].astype(float).values, X)
        return stats.spearmanr(rc, rt).statistic

    rho, lo, hi, p = _cluster_boot(d, stat, n_boot, seed)
    return ResidCorr(rho=rho, ci_lo=lo, ci_hi=hi, p_boot=p,
                     n_rows=len(d), n_langs=d.flores_code.nunique())


def orthogonalized_increment(df: pd.DataFrame, cand: str, incumbent: str,
                             target: str, n_boot: int = 2000,
                             seed: int = 13) -> ResidCorr:
    d = df.dropna(subset=[cand, incumbent, target, *COVARIATES,
                          *CATEGORICAL]).copy()

    def stat(sample: pd.DataFrame) -> float:
        X = _design(sample)
        Xi = np.column_stack(
            [X, sample[incumbent].astype(float).values])
        rc = _residualize(sample[cand].astype(float).values, Xi)
        rt = _residualize(sample[target].astype(float).values, X)
        return stats.spearmanr(rc, rt).statistic

    rho, lo, hi, p = _cluster_boot(d, stat, n_boot, seed)
    return ResidCorr(rho=rho, ci_lo=lo, ci_hi=hi, p_boot=p,
                     n_rows=len(d), n_langs=d.flores_code.nunique())


def romano_wolf(df: pd.DataFrame, candidates: list[str], target: str,
                n_boot: int = 2000, seed: int = 13) -> pd.DataFrame:
    """Stepdown maxT over |deflated Spearman|, language-cluster bootstrap
    null (centered at the point estimates). Returns per-candidate
    adjusted p-values."""
    d0 = df.copy()
    langs = d0.flores_code.unique()
    by = {lg: d0[d0.flores_code == lg] for lg in langs}

    def stat_all(sample: pd.DataFrame) -> np.ndarray:
        out = []
        for c in candidates:
            s = sample.dropna(subset=[c, target, *COVARIATES,
                                      *CATEGORICAL])
            if len(s) < 10:
                out.append(np.nan)
                continue
            X = _design(s)
            rc = _residualize(s[c].astype(float).values, X)
            rt = _residualize(s[target].astype(float).values, X)
            out.append(stats.spearmanr(rc, rt).statistic)
        return np.array(out, dtype=float)

    point = stat_all(d0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(langs, size=len(langs), replace=True)
        boots.append(stat_all(pd.concat([by[lg] for lg in pick])))
    B = np.array(boots)                       # (n_boot, k)
    centered = np.abs(B - np.nanmean(B, axis=0))

    order = np.argsort(-np.abs(point))        # most significant first
    adj = np.full(len(candidates), np.nan)
    prev = 0.0
    for rank, idx in enumerate(order):
        rest = order[rank:]
        maxdist = np.nanmax(centered[:, rest], axis=1)
        p = float(np.mean(maxdist >= abs(point[idx])))
        p = max(p, prev, 1 / n_boot)          # monotone stepdown
        adj[idx] = min(p, 1.0)
        prev = adj[idx]
    return pd.DataFrame({"candidate": candidates, "rho_deflated": point,
                         "p_rw": adj}).sort_values("p_rw")


def effective_breadth(df: pd.DataFrame, cand: str, target: str) -> float:
    """Grinold N_eff = N / (1 + (N-1) * rho_bar) with rho_bar the mean
    pairwise correlation of per-language (candidate, target) products
    across models — a conservative panel-breadth summary."""
    d = df.dropna(subset=[cand, target]).copy()
    piv = d.pivot_table(index="model", columns="flores_code",
                        values=cand, aggfunc="mean")
    piv = piv.dropna(axis=1)
    n = piv.shape[1]
    if n < 2 or piv.shape[0] < 3:
        return float(n)
    corr = np.corrcoef(piv.values.T)
    rho_bar = float((corr.sum() - n) / (n * (n - 1)))
    rho_bar = max(rho_bar, 0.0)
    return float(n / (1 + (n - 1) * rho_bar))
