"""Deflated validity: the attribution regression + estimator bracket
(Phase 1.7; prereg §2 covariates and bracket).

Port of `legacy/attribution.py` with the regression spec preserved exactly —
the Phase-1 acceptance requires reproducing the published MEXA/AaR
correlations within ±0.02 on this stack before any new candidate is scored
through it. What the port adds:

* Arbitrary candidate columns (mexa, aar10, lde, q_K, T, RMFS, ...), not a
  hardcoded pair.
* The full estimator bracket per candidate: raw-residual (primary),
  family-demeaned, and median-s² disattenuation — three answers per
  candidate, reported together, never averaged.
* Language-cluster bootstrap CIs (resample LANGUAGES with replacement, the
  unit of dependence; 2,000 reps per prereg).
* Typed results, hard-fail loading (E2 rule), and the tokens-beta sanity
  gate promoted from a printout to a machine-readable flag: if the proxy
  token coefficient is weak or wrong-signed the deflation is invalid and
  every downstream verdict carries that flag.

The regression, verbatim from the advisor-reviewed spec:
    y = logit((acc − 0.25) / 0.75)          # 4-choice floor
    y ~ C(model) + log10_tokens + C(macro_family_g) + C(script_g) + fertility
    cluster-robust SEs by language; alpha = residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["DeflationFit", "BracketRow", "fit_attribution",
           "shrink_alphas", "candidate_bracket"]


@dataclass(frozen=True)
class DeflationFit:
    alphas: pd.DataFrame           # input rows + y, alpha (+ alpha_shrunk)
    tokens_beta: float
    tokens_beta_p: float
    tokens_gate_passed: bool       # positive and significant, else INVALID
    r2: float
    n_rows: int
    n_dropped: int


@dataclass(frozen=True)
class BracketRow:
    candidate: str
    target: str                    # "raw_acc" | "alpha"
    scheme: str                    # "raw_residual" | "family_demeaned"
    #                                | "disattenuated"
    spearman: float
    ci_low: float
    ci_high: float
    n: int
    n_boot: int
    tokens_gate_passed: bool
    flags: tuple = field(default_factory=tuple)


def _logit_excess(acc: pd.Series) -> pd.Series:
    a = acc.clip(0.26, 0.99)
    p = (a - 0.25) / 0.75
    return np.log(p / (1 - p))


def fit_attribution(df: pd.DataFrame,
                    tokens_alpha: float = 0.05) -> DeflationFit:
    """OLS with model fixed effects and cluster-robust SEs by language.

    Required columns: model, flores_code, acc, log_tokens, macro_family_g,
    script_g, fertility. Rows with missing covariates are dropped and
    counted, never imputed here (the CC-100 backstop happens upstream,
    flagged)."""
    import statsmodels.formula.api as smf

    need = ["model", "flores_code", "acc", "log_tokens",
            "macro_family_g", "script_g", "fertility"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"fit_attribution: missing columns {missing}")
    d = df.dropna(subset=["acc", "log_tokens", "fertility"]).copy()
    n_dropped = len(df) - len(d)
    d["y"] = _logit_excess(d.acc)
    m = smf.ols(
        "y ~ C(model) + log_tokens + C(macro_family_g) + C(script_g)"
        " + fertility", data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d.flores_code})
    beta = float(m.params["log_tokens"])
    p = float(m.pvalues["log_tokens"])
    d["alpha"] = m.resid
    return DeflationFit(
        alphas=d, tokens_beta=beta, tokens_beta_p=p,
        tokens_gate_passed=bool(beta > 0 and p < tokens_alpha),
        r2=float(m.rsquared), n_rows=len(d), n_dropped=n_dropped,
    )


def shrink_alphas(d: pd.DataFrame, n_questions: int = 300) -> pd.DataFrame:
    """Empirical-Bayes shrinkage toward macro-family means (legacy-verbatim:
    w = tau2/(tau2 + s_i2), s_i2 the binomial sampling variance propagated
    through the logit). Reported as a diagnostic column; the PRIMARY target
    stays the raw alpha — the prior campaign measured EB collapsing to
    family means on this data and inflating correlations."""
    acc = d.acc.clip(0.26, 0.99)
    p = (acc - 0.25) / 0.75
    dyda = 1.0 / (0.75 * p * (1 - p))
    d = d.assign(s2=((acc * (1 - acc) / float(n_questions)) * dyda**2).values)
    out = []
    for _, g in d.groupby("macro_family_g"):
        tau2 = max(g.alpha.var(ddof=1) - g.s2.mean(), 1e-4)
        w = tau2 / (tau2 + g.s2)
        mu = g.alpha.mean()
        out.append(g.assign(alpha_shrunk=mu + w * (g.alpha - mu)))
    return pd.concat(out)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats

    return float(stats.spearmanr(x, y).statistic)


def _family_demean(sub: pd.DataFrame, cols) -> pd.DataFrame:
    out = sub.copy()
    for c in cols:
        out[c] = sub[c] - sub.groupby("macro_family_g")[c].transform("mean")
    return out


def candidate_bracket(fit: DeflationFit, candidate: str,
                      n_boot: int = 2000, seed: int = 13,
                      n_questions: int = 300) -> list[BracketRow]:
    """The estimator bracket for one candidate column, with language-cluster
    bootstrap CIs. Returns rows for (raw_acc, alpha) x schemes."""
    d = fit.alphas.dropna(subset=[candidate]).copy()
    rows: list[BracketRow] = []
    g = np.random.default_rng(seed)
    langs = d.flores_code.unique()
    by_lang = {lg: d[d.flores_code == lg] for lg in langs}

    def boot_ci(stat_fn) -> tuple[float, float]:
        vals = []
        for _ in range(n_boot):
            pick = g.choice(langs, size=len(langs), replace=True)
            sample = pd.concat([by_lang[lg] for lg in pick])
            try:
                vals.append(stat_fn(sample))
            except Exception:
                continue
        v = np.array([x for x in vals if np.isfinite(x)])
        return (float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)))

    specs = [
        ("raw_acc", "raw_residual",
         lambda s: _spearman(s[candidate], s.acc)),
        ("alpha", "raw_residual",
         lambda s: _spearman(s[candidate], s.alpha)),
        ("alpha", "family_demeaned",
         lambda s: _spearman(*_family_demean(s, [candidate, "alpha"])
                             [[candidate, "alpha"]].values.T)),
    ]
    for target, scheme, fn in specs:
        est = fn(d)
        lo, hi = boot_ci(fn)
        rows.append(BracketRow(
            candidate=candidate, target=target, scheme=scheme,
            spearman=est, ci_low=lo, ci_high=hi, n=len(d), n_boot=n_boot,
            tokens_gate_passed=fit.tokens_gate_passed,
            flags=() if fit.tokens_gate_passed
            else ("TOKENS_GATE_FAILED_DEFLATION_INVALID",),
        ))

    # median-s² disattenuation: correct the alpha correlation for sampling
    # noise in alpha itself; r_true ≈ r / sqrt(reliability), reliability =
    # 1 − med(s2)/var(alpha). Reported only when reliability is meaningful.
    acc = d.acc.clip(0.26, 0.99)
    p_ = (acc - 0.25) / 0.75
    dyda = 1.0 / (0.75 * p_ * (1 - p_))
    s2 = (acc * (1 - acc) / float(n_questions)) * dyda**2
    var_a = float(d.alpha.var(ddof=1))
    rel = 1.0 - float(np.median(s2)) / var_a if var_a > 0 else np.nan
    raw_r = _spearman(d[candidate], d.alpha)
    if np.isfinite(rel) and rel > 0.1:
        dis = raw_r / np.sqrt(rel)
        lo, hi = boot_ci(lambda s: _spearman(s[candidate], s.alpha)
                         / np.sqrt(rel))
        flags: tuple = () if fit.tokens_gate_passed else (
            "TOKENS_GATE_FAILED_DEFLATION_INVALID",)
    else:
        dis, lo, hi = float("nan"), float("nan"), float("nan")
        flags = ("RELIABILITY_TOO_LOW_FOR_DISATTENUATION",)
    rows.append(BracketRow(
        candidate=candidate, target="alpha", scheme="disattenuated",
        spearman=float(dis), ci_low=lo, ci_high=hi, n=len(d),
        n_boot=n_boot, tokens_gate_passed=fit.tokens_gate_passed,
        flags=flags,
    ))
    return rows
