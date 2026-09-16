"""Variance components for the two-way crossed random model (prereg §3.2).

Per dimension d of the representation, the model is

    h[c, l, d] = mu_d + a[c, d] + b[l, d] + e[c, l, d]

with concepts a ~ N(0, sigma2_C), languages b ~ N(0, sigma2_L), noise
e ~ N(0, sigma2_E), one observation per (concept, language) cell. All
estimation is per-dimension with closed forms, then aggregated by summing
over dimensions (the LFS-VC ratio is invariant to sum-vs-mean).

Two estimators, per the kickoff:

* Bias-corrected ANOVA (method of moments):
      sigma2_L = (MS_L - MS_E) / N,  sigma2_C = (MS_C - MS_E) / L,
      sigma2_E = MS_E,
  with E[MS_L] = sigma2_E + N sigma2_L and E[MS_C] = sigma2_E + L sigma2_C.
  Negative per-dimension estimates are KEPT in the ANOVA aggregate and
  counted in n_neg_* — never silently clipped (RULES.md coding standard;
  the prior campaign's floors came from exactly this kind of hidden bias).

* Nonnegative REML (primary). For a balanced design the interior REML
  solution coincides with the ANOVA estimator; when an ANOVA component is
  negative the REML solution sits on the boundary: the component is zero
  and the error variance re-pools that component's sum of squares into its
  degrees of freedom. Implemented as the closed-form boundary lattice
  (interior / L-boundary / C-boundary / both), vectorized over dimensions.

Why this replaces the legacy ratio: the legacy estimator is a raw
sum-of-squares share whose pure-noise value is the degrees-of-freedom ratio
(L-1)/((L-1)+(N-1)) rather than zero — noise masquerades as signal at small
N. `legacy_lfs_ratio` reproduces it here solely for the contrast figure and
parity tests; it is not a component of RMFS.

LDE (prereg 1.2): per-language deviation energy
    D_l = || mean_c h[:, l, :] - grand mean ||^2
noise-corrected by subtracting its null expectation D * sigma2_E_dim / N.
Raw and corrected are both stored; corrected values may legitimately be
negative (unbiasedness over clipping). Residualization against covariates
happens in validity/deflation.py, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["VarianceComponents", "LDEResult", "variance_components", "lde",
           "legacy_lfs_ratio"]


@dataclass(frozen=True)
class VarianceComponents:
    """Aggregated (summed over dimensions) variance components + diagnostics."""

    sigma2_L_reml: float
    sigma2_C_reml: float
    sigma2_E: float
    sigma2_L_anova: float
    sigma2_C_anova: float
    lfs_vc: float          # REML ratio sigma2_L / (sigma2_L + sigma2_C)
    lfs_vc_sd: float       # SD-scale variant sigma_L / (sigma_L + sigma_C)
    lfs_vc_anova: float    # ANOVA-ratio variant (negatives kept upstream)
    n_neg_L: int           # dims where the ANOVA language component < 0
    n_neg_C: int
    L: int
    N: int
    D: int
    mean_norm: float       # mean over cells of ||h||_2, capacity diagnostic
    degenerate: bool       # both REML systematic components ~ 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class LDEResult:
    """Per-language deviation energy, raw and noise-corrected."""

    raw: dict = field(default_factory=dict)
    corrected: dict = field(default_factory=dict)
    correction: float = 0.0
    L: int = 0
    N: int = 0
    D: int = 0


def _mean_squares(X: np.ndarray):
    """Per-dimension mean squares for the balanced two-way layout.

    X: (L, N, D). Returns per-dimension arrays (MS_L, MS_C, MS_E) plus the
    per-dimension sums of squares and their degrees of freedom.
    """
    L, N, _ = X.shape
    grand = X.mean(axis=(0, 1))                    # (D,)
    lang_means = X.mean(axis=1)                    # (L, D)
    conc_means = X.mean(axis=0)                    # (N, D)
    ss_L = N * ((lang_means - grand) ** 2).sum(axis=0)          # (D,)
    ss_C = L * ((conc_means - grand) ** 2).sum(axis=0)          # (D,)
    resid = X - lang_means[:, None, :] - conc_means[None, :, :] + grand
    ss_E = (resid ** 2).sum(axis=(0, 1))                        # (D,)
    df_L, df_C, df_E = L - 1, N - 1, (L - 1) * (N - 1)
    return ss_L, ss_C, ss_E, df_L, df_C, df_E


def variance_components(X: np.ndarray) -> VarianceComponents:
    """ANOVA + nonnegative-REML variance components on X of shape (L, N, D)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"X must be (L, N, D); got shape {X.shape}")
    L, N, D = X.shape
    if L < 2 or N < 2:
        raise ValueError(f"need L >= 2 and N >= 2; got L={L}, N={N}")

    ss_L, ss_C, ss_E, df_L, df_C, df_E = _mean_squares(X)
    ms_L, ms_C, ms_E = ss_L / df_L, ss_C / df_C, ss_E / df_E

    # --- bias-corrected ANOVA, negatives kept -----------------------------
    s2_L_d = (ms_L - ms_E) / N
    s2_C_d = (ms_C - ms_E) / L
    n_neg_L = int((s2_L_d < 0).sum())
    n_neg_C = int((s2_C_d < 0).sum())

    # --- nonnegative REML: closed-form boundary lattice -------------------
    # interior: ANOVA values. L-boundary: sigma2_L = 0, error re-pools ss_L.
    # C-boundary symmetric. Both: everything pools into the error.
    neg_L, neg_C = s2_L_d < 0, s2_C_d < 0
    both = neg_L & neg_C
    only_L = neg_L & ~neg_C
    only_C = neg_C & ~neg_L

    r_L = np.where(neg_L, 0.0, s2_L_d)
    r_C = np.where(neg_C, 0.0, s2_C_d)
    r_E = ms_E.copy()
    # re-pooled error variances on each boundary
    e_pool_L = (ss_L + ss_E) / (df_L + df_E)
    e_pool_C = (ss_C + ss_E) / (df_C + df_E)
    e_pool_B = (ss_L + ss_C + ss_E) / (df_L + df_C + df_E)
    r_E = np.where(only_L, e_pool_L, r_E)
    r_E = np.where(only_C, e_pool_C, r_E)
    r_E = np.where(both, e_pool_B, r_E)
    # the surviving component re-estimates against the pooled error
    r_C = np.where(only_L, np.maximum((ms_C - e_pool_L) / L, 0.0), r_C)
    r_L = np.where(only_C, np.maximum((ms_L - e_pool_C) / N, 0.0), r_L)

    S_L, S_C = float(r_L.sum()), float(r_C.sum())
    S_E = float(r_E.sum())
    A_L, A_C = float(s2_L_d.sum()), float(s2_C_d.sum())

    denom = S_L + S_C
    degenerate = denom < 1e-12 * max(S_E, 1e-300)
    lfs_vc = float("nan") if degenerate else S_L / denom
    sd_denom = np.sqrt(max(S_L, 0.0)) + np.sqrt(max(S_C, 0.0))
    lfs_vc_sd = (float("nan") if sd_denom == 0.0
                 else np.sqrt(max(S_L, 0.0)) / sd_denom)
    a_denom = A_L + A_C
    lfs_vc_anova = float("nan") if a_denom == 0.0 else A_L / a_denom

    return VarianceComponents(
        sigma2_L_reml=S_L, sigma2_C_reml=S_C, sigma2_E=S_E,
        sigma2_L_anova=A_L, sigma2_C_anova=A_C,
        lfs_vc=lfs_vc, lfs_vc_sd=lfs_vc_sd, lfs_vc_anova=lfs_vc_anova,
        n_neg_L=n_neg_L, n_neg_C=n_neg_C, L=L, N=N, D=D,
        mean_norm=float(np.linalg.norm(X, axis=2).mean()),
        degenerate=bool(degenerate),
    )


def lde(X: np.ndarray) -> LDEResult:
    """Per-language deviation energy with its noise correction (prereg 1.2)."""
    X = np.asarray(X, dtype=np.float64)
    L, N, D = X.shape
    grand = X.mean(axis=(0, 1))
    lang_means = X.mean(axis=1)
    raw = ((lang_means - grand) ** 2).sum(axis=1)              # (L,)
    # null expectation of raw under no language effect: each of the D
    # dimensions contributes sigma2_E_dim / N (up to the (L-1)/L centering
    # factor, absorbed into the estimator's own centering)
    _, _, ss_E, _, _, df_E = _mean_squares(X)
    sigma2_E_dim = float((ss_E / df_E).mean())
    correction = D * sigma2_E_dim / N * (L - 1) / L
    return LDEResult(
        raw={i: float(raw[i]) for i in range(L)},
        corrected={i: float(raw[i] - correction) for i in range(L)},
        correction=float(correction), L=L, N=N, D=D,
    )


def legacy_lfs_ratio(X: np.ndarray) -> float:
    """The prior campaign's raw sum-of-squares share, kept ONLY for the
    contrast figure and parity tests. Its pure-noise value is
    (L-1)/((L-1)+(N-1)), not zero — the reason it was replaced."""
    X = np.asarray(X, dtype=np.float64)
    L, N, _ = X.shape
    grand = X.mean(axis=(0, 1))
    # The N and L multipliers are load-bearing: they are what make the
    # pure-noise value the DF ratio (L-1)/((L-1)+(N-1)). Omitting them (as a
    # first draft here did — caught by the floor test) gives a different,
    # much smaller floor. Matches the campaign's batch_lfs convention.
    v_lang = N * ((X.mean(axis=1) - grand) ** 2).sum()
    v_conc = L * ((X.mean(axis=0) - grand) ** 2).sum()
    return float(v_lang / (v_lang + v_conc))
