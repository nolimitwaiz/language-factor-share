"""Capacity gates (prereg §3.4, Phase 1.3): the collapse detectors.

The prior campaign's founding result is that scale-invariant metrics (the
legacy ratio, retrieval, tails) ranked an 89%-collapsed model best in the
study. These gates are the quantities that caught it, promoted from ad-hoc
tripwires to typed components:

- C_pres: concept-variance preservation against a frozen reference
  (intervention mode). Under uniform scaling by s it equals s^2 exactly —
  calibrated in the prior battery (cell I8b) to three decimals, asserted
  algebraically in the unit battery here.
- effective rank: exp(spectral entropy) of the concept-mean covariance
  eigenvalues. Scale-invariant by construction (it reads the SHAPE of the
  spectrum), so it complements C_pres rather than duplicating it: uniform
  collapse moves C_pres and not r_eff; rank collapse moves r_eff.
- mean representation norm: the plain magnitude diagnostic.
- read_delta_ppl: hook that READS held-out perplexity outputs produced by
  cluster LM evaluations. It does not run models. It HARD-FAILS on a
  missing file — the E2 rule: a loader that silently degrades on missing
  input survived a full campaign; that pattern is banned.

Every gate returns a dataclass carrying the value and its pass/fail flag
against an explicit threshold. Thresholds come from the prereg (or addendum
A1), never from defaults buried here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["GateResult", "concept_variance_raw", "cpres", "effective_rank",
           "mean_norm", "read_delta_ppl"]


@dataclass(frozen=True)
class GateResult:
    """A gate value with its diagnostics; never a bare float."""

    name: str
    value: float
    threshold: float | None = None
    passed: bool | None = None      # None when no threshold was supplied
    capped: bool = False            # C_pres only: raw ratio exceeded 1
    n: int = 0                      # sample count behind the estimate


def concept_variance_raw(X: np.ndarray) -> float:
    """Unstandardized concept variance on X of shape (L, N, D).

    Mean over concepts of the squared distance of the concept mean (over
    languages) from the grand mean, summed over dimensions — the exact
    quantity whose 20x separation between collapsed and sound models the
    scalar ratio divided away. Raw by design: standardizing here would
    reintroduce the scale blindness this gate exists to remove.
    """
    X = np.asarray(X, dtype=np.float64)
    grand = X.mean(axis=(0, 1))
    concept_means = X.mean(axis=0)                 # (N, D)
    return float(((concept_means - grand) ** 2).sum(axis=1).mean())


def cpres(v_concept: float, v_concept_ref: float,
          threshold: float | None = None) -> GateResult:
    """C_pres = min(V_concept / V_concept_ref, 1), prereg §3.4.

    The reference is the frozen base model of the intervention. A zero or
    negative reference is a protocol error, not a value to coerce."""
    if v_concept_ref <= 0.0:
        raise ValueError(
            f"reference concept variance must be positive; got "
            f"{v_concept_ref!r} — a zero reference means the reference "
            f"model was not measured, which is a protocol error")
    ratio = v_concept / v_concept_ref
    value = min(ratio, 1.0)
    return GateResult(
        name="C_pres", value=float(value), threshold=threshold,
        passed=None if threshold is None else bool(value >= threshold),
        capped=bool(ratio > 1.0),
    )


def effective_rank(means: np.ndarray,
                   threshold: float | None = None) -> GateResult:
    """exp(spectral entropy) of the covariance eigenvalues of `means`.

    means: (N, D) concept means. Scale-invariant: eigenvalues are
    normalized to a distribution before the entropy."""
    means = np.asarray(means, dtype=np.float64)
    centered = means - means.mean(axis=0)
    # eigenvalues of the covariance via SVD of the centered matrix — never
    # form the DxD covariance when D is large
    s = np.linalg.svd(centered, compute_uv=False)
    lam = s**2
    lam = lam[lam > 1e-12 * lam.max()] if lam.max() > 0 else lam
    p = lam / lam.sum()
    value = float(np.exp(-(p * np.log(p)).sum()))
    return GateResult(
        name="effective_rank", value=value, threshold=threshold,
        passed=None if threshold is None else bool(value >= threshold),
        n=means.shape[0],
    )


def mean_norm(X: np.ndarray) -> GateResult:
    """Mean L2 norm over all (language, concept) cells of X (L, N, D)."""
    X = np.asarray(X, dtype=np.float64)
    return GateResult(name="mean_norm",
                      value=float(np.linalg.norm(X, axis=2).mean()),
                      n=int(X.shape[0] * X.shape[1]))


def read_delta_ppl(path: str | Path) -> dict[str, float]:
    """Read a held-out perplexity table produced by a cluster LM evaluation.

    HARD-FAILS on a missing file. The E2 lesson is one line: `if
    os.path.exists(...)` guards that silently change behavior survived a
    full campaign; this loader raises instead."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"delta-PPL table not found: {p} — refusing to continue "
            f"without it (E2 rule: no silent degradation on missing input)")
    with open(p) as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}
