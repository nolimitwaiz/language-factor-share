"""Behavioral T (prereg §3.1; addendum A2), Phase 2.1.

T_ℓ = (B_recon − B_mismatch) / (B_native − B_mismatch), where each B is the
target-sentence NLL reduction relative to no context. B_recon uses context
token states reconstructed from the shared factor.

This module is the CPU core: reverse-direction rung maps, the two
reconstruction modes A2 froze, and the T arithmetic with its ε_B and
saturation rules. The GPU side (extracting the four NLL conditions from a
model, with layer-k states hook-substituted for the reconstruction
condition) lives in scripts/t_extract.py and consumes these functions.

Design decisions are FROZEN in prereg/addenda/A2_reconstruction_spec.md,
written before any extraction:
- Reconstruction uses the language's SELECTED rung (Phase-1 K selection,
  not refit) with a map of that rung's family fit consensus → ℓ — reverse
  direction, same hyperparameter procedure. M0–M3 coincide with the exact
  inverse up to estimation noise; M4/M5 are proper reverse regressions.
- Strict mode (primary): x̂ = μ + B·m(Bᵀ(x−μ)) — factor-predicted content
  plus the pooled mean, nothing token-specific outside the subspace.
- Hybrid mode (disclosed sensitivity): keeps the out-of-subspace component
  of the original state; reported, never selected on.
- Saturation: B_recon < B_mismatch − ε_B ⇒ the cell is reported saturated,
  excluded from means, never silently folded in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rmfs.components.ladder import (
    _fit_krr,
    _fit_offset,
    _fit_ridge,
    _fit_scale,
    _fit_similarity,
)

__all__ = ["TResult", "fit_reverse_map", "reconstruct", "t_from_nll",
           "EPS_B"]

EPS_B = 0.072   # prereg §3.1: pooled 5th percentile of the legacy 3.1.3
#                 B_native − B_mismatch distribution (160 cells)


@dataclass(frozen=True)
class TResult:
    b_native: float
    b_mismatch: float
    b_recon: float
    t_raw: float           # unclipped; may be negative or exceed 1
    q_t: float             # clip(t_raw, 0, 1); NaN when undefined
    undefined: bool        # denominator below eps_b (prereg rule)
    saturated: bool        # A2: recon worse than wrong-document beyond eps_b
    eps_b: float


def fit_reverse_map(Z_train: np.ndarray, X_train: np.ndarray, rung: str,
                    seed: int = 0):
    """Map of `rung`'s family fit consensus → ℓ (A2 fork-1 resolution).

    Reuses the ladder's fit machinery with the arguments in the reverse
    order — deliberately the same code paths the forward fits and the
    parity tests exercise, so the reverse maps inherit their validation.
    Returns a callable rows → rows."""
    if rung == "M0":
        return lambda z: z
    if rung == "M1":
        return _fit_offset(Z_train, X_train)
    if rung == "M2":
        return _fit_scale(Z_train, X_train)
    if rung == "M3":
        return _fit_similarity(Z_train, X_train)
    if rung == "M4":
        fn, _ = _fit_ridge(Z_train, X_train, seed=seed)
        return fn
    if rung == "M5":
        fn, _ = _fit_krr(Z_train, X_train, seed=seed)
        return fn
    raise ValueError(f"unknown rung {rung!r}")


def reconstruct(z: np.ndarray, basis: np.ndarray, mu: np.ndarray,
                rev_map, mode: str = "strict",
                states: np.ndarray | None = None,
                n_tokens: int | None = None) -> np.ndarray:
    """Factor-input reconstruction (A3, superseding A2's fork-2 formula).

    z: (r,) the sentence-level shared-factor coordinates for this concept
    (LEAVE-ONE-OUT consensus, exclude=ℓ — A3 §2). The reverse map is
    applied to the FACTOR, never to the token's own coordinates: A2's
    per-token-coordinate formula was a frame mismatch that would have
    leaked ℓ's content and inflated T (A3 §1), caught before any run.

    strict (primary): x̂ = μ + B·m(z), broadcast to n_tokens positions —
      only factor-predicted content plus the pooled mean survives.
    hybrid (disclosed sensitivity): per-token; keeps each token's
      out-of-subspace content, replaces in-subspace content with the
      factor prediction. Requires `states` (T, D)."""
    mu = np.asarray(mu, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(1, -1)
    xhat_in = np.asarray(rev_map(z)) @ B.T          # (1, D) in-subspace pred
    if mode == "strict":
        if n_tokens is None:
            raise ValueError("strict mode needs n_tokens for the broadcast")
        return np.repeat(mu[None, :] + xhat_in, n_tokens, axis=0)
    if mode == "hybrid":
        if states is None:
            raise ValueError("hybrid mode needs the original token states")
        states = np.asarray(states, dtype=np.float64)
        coords = (states - mu) @ B
        return states - coords @ B.T + xhat_in
    raise ValueError(f"unknown mode {mode!r}")


def t_from_nll(nll_none: float, nll_native: float, nll_mismatch: float,
               nll_recon: float, eps_b: float = EPS_B) -> TResult:
    """The T arithmetic with the prereg's ε_B rule and A2's saturation flag.

    B_x = nll_none − nll_x (positive = that context helps)."""
    b_native = float(nll_none - nll_native)
    b_mismatch = float(nll_none - nll_mismatch)
    b_recon = float(nll_none - nll_recon)
    denom = b_native - b_mismatch
    undefined = bool(denom < eps_b)
    saturated = bool((not undefined) and (b_recon < b_mismatch - eps_b))
    if undefined:
        t_raw = float("nan")
        q_t = float("nan")
    else:
        t_raw = (b_recon - b_mismatch) / denom
        q_t = float(np.clip(t_raw, 0.0, 1.0))
    return TResult(b_native=b_native, b_mismatch=b_mismatch,
                   b_recon=b_recon, t_raw=t_raw, q_t=q_t,
                   undefined=undefined, saturated=saturated, eps_b=eps_b)
