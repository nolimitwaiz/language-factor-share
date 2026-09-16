"""RMFS v1 assembly: frozen language-level T aggregation and soft-min.

Prereg §3.1 (T, ε_B, undefined rule), §3.5 (aggregation, τ = 0.1, J = 4
intervention / 3 observational), addendum A4 (injection numerator,
saturation analog). The component vector is always published beside the
scalar; the scalar is a ranking statistic only.

Language pooling for arm-level comparisons (S1): the prereg freezes the
per-language score but not the cross-language pool. Declared HERE, before
any arm is assembled: median over defined languages is primary, mean is
reported beside it as sensitivity. Undefined languages are excluded from
the pool and their count is published.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rmfs.components.transfer import EPS_B

TAU = 0.1


@dataclass(frozen=True)
class TLanguage:
    """Frozen language-level T: B's aggregate over pairs FIRST, then one
    T per language (the ε_B provenance cells were built this way)."""

    t_raw: float           # unclipped, NaN when undefined
    q_t: float             # clip(T, 0, 1), NaN when undefined
    b_native: float        # mean over pairs of (NLL_none − NLL_native)
    b_mismatch: float      # mean over pairs of (NLL_none − NLL_mismatch)
    denom: float           # B_native − B_mismatch
    numer: float           # A4: mean(NLL_injw) − mean(NLL_inj)
    undefined: bool        # denom < ε_B → reported undefined, never zero
    saturated: bool        # A4 analog: mean NLL_inj > mean NLL_injw + ε_B
    n_pairs: int


def t_language(nll_none: np.ndarray, nll_native: np.ndarray,
               nll_mismatch: np.ndarray, nll_inj: np.ndarray,
               nll_injw: np.ndarray, eps_b: float = EPS_B) -> TLanguage:
    arrs = [np.asarray(a, dtype=np.float64) for a in
            (nll_none, nll_native, nll_mismatch, nll_inj, nll_injw)]
    n = arrs[0].shape[0]
    if any(a.shape != (n,) for a in arrs) or n == 0:
        raise ValueError("all five NLL arrays must be equal-length, nonempty")
    none_m, nat_m, mis_m, inj_m, injw_m = (float(a.mean()) for a in arrs)
    b_native = none_m - nat_m
    b_mismatch = none_m - mis_m
    denom = b_native - b_mismatch          # = mean(mismatch) − mean(native)
    numer = injw_m - inj_m
    undefined = denom < eps_b
    saturated = inj_m > injw_m + eps_b
    t_raw = math.nan if undefined else numer / denom
    q_t = math.nan if undefined else float(np.clip(t_raw, 0.0, 1.0))
    return TLanguage(t_raw=t_raw, q_t=q_t, b_native=b_native,
                     b_mismatch=b_mismatch, denom=denom, numer=numer,
                     undefined=undefined, saturated=saturated, n_pairs=n)


def softmin(q, tau: float = TAU) -> float:
    """RMFS = −τ · log((1/J) Σ_j exp(−q_j/τ)). NaN components propagate:
    an undefined gauge makes the scalar undefined, never silently ignored.
    Bounds: min(q) ≤ RMFS ≤ min(q) + τ·log J."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q must be a nonempty 1-D component vector")
    if np.isnan(q).any():
        return math.nan
    m = float(q.min())
    return m - tau * math.log(float(np.mean(np.exp(-(q - m) / tau))))


@dataclass(frozen=True)
class RMFSCell:
    """One (model-or-arm, language) score with its full component vector."""

    mode: str              # "observational" (J=3) | "intervention" (J=4)
    q_t: float
    q_l: float
    q_k: float
    q_c: float             # NaN in observational mode (not a component)
    rmfs: float
    undefined: bool


def assemble(q_t: float, q_l: float, q_k: float,
             q_c: float | None = None, tau: float = TAU) -> RMFSCell:
    if q_c is None:
        mode, vec = "observational", [q_t, q_l, q_k]
    else:
        mode, vec = "intervention", [q_t, q_l, q_k, q_c]
    val = softmin(vec, tau=tau)
    return RMFSCell(mode=mode, q_t=float(q_t), q_l=float(q_l),
                    q_k=float(q_k),
                    q_c=math.nan if q_c is None else float(q_c),
                    rmfs=val, undefined=math.isnan(val))


def pool_languages(scores) -> dict:
    """Arm/model-level pool of per-language RMFS: median primary, mean as
    sensitivity, undefined excluded and counted (declared pre-assembly)."""
    s = np.asarray(list(scores), dtype=np.float64)
    defined = s[~np.isnan(s)]
    return {
        "median": float(np.median(defined)) if defined.size else math.nan,
        "mean": float(defined.mean()) if defined.size else math.nan,
        "n_defined": int(defined.size),
        "n_undefined": int(np.isnan(s).sum()),
    }
