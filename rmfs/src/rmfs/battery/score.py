"""Signature-cell scoring for the S2 battery (conventions frozen in
prereg §4 / signatures CSV; estimator details declared here BEFORE any
scoring, same pre-declaration pattern as the assembly pool rule).

Declared aggregation per (config, component) scalar:
- q_L    : pooled 1 − LFS-VC (REML, z-scored pipeline), all languages.
- q_K    : median selected rung INDEX over affected languages (rung cells
           compare this median to the named rung).
- C_pres : REML σ²_C ratio vs the base world (uniform/global configs);
           per-language own-centroid deviation-energy ratio for the
           affected language in collapse_per_language.
- T      : median T_ℓ (frozen language-level form) over affected
           languages with defined denominators.
- RMFS   : median over affected languages of the intervention-mode
           soft-min [q_T, q_L, q_K, q_C].

Floors: per-component sample SD over the identity replicates (seeds
13/42/71, independent pipeline randomness), floored at 1e-9. Conventions:
  band     |Δ| ≤ k·floor
  down     Δ < −k·floor          up      Δ > +k·floor
  down_mag down AND monotone nonincreasing across the family magnitudes
  ceiling  Δ ≤ +k·floor (any fall allowed)
  exact    |value − expected| ≤ abs_tol (0.01, C_pres closed form)
  rung_le / rung_ge   median rung index vs named rung
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RUNGS = ["M0", "M1", "M2", "M3", "M4", "M5"]
EXACT_ABS_TOL = 0.01
FLOOR_MIN = 1e-9


def floors_from_identity(reps: list[dict]) -> dict:
    """Per-component noise floor: SD over identity replicates."""
    if len(reps) < 2:
        raise ValueError("need >= 2 identity replicates for floors")
    out = {}
    for comp in reps[0]:
        vals = np.array([r[comp] for r in reps], dtype=np.float64)
        out[comp] = max(float(np.nanstd(vals, ddof=1)), FLOOR_MIN)
    return out


@dataclass(frozen=True)
class CellVerdict:
    injection: str
    magnitude: str
    component: str
    expectation: str
    value_base: float
    value_inj: float
    delta: float
    threshold: float       # k·floor, or abs_tol for exact, or rung index
    passed: bool
    note: str = ""


def _rung_idx(v) -> int:
    return RUNGS.index(v) if isinstance(v, str) else int(round(float(v)))


def score_cell(injection: str, magnitude: str, component: str,
               expectation: str, expected_value, k: float,
               value_base: float, value_inj: float, floor: float,
               family_values: list[float] | None = None) -> CellVerdict:
    """Score one signature cell. family_values: the component's injected
    values across the family's magnitude ladder IN INCREASING-severity
    order (needed only for down_mag)."""
    numeric = not isinstance(value_inj, str)
    delta = (value_inj - value_base) if numeric else float("nan")
    thr = k * floor
    note = ""
    if expectation == "band":
        passed = abs(delta) <= thr
    elif expectation == "down":
        passed = delta < -thr
    elif expectation == "up":
        passed = delta > thr
    elif expectation == "ceiling":
        passed = delta <= thr
    elif expectation == "down_mag":
        if family_values is None or len(family_values) < 2:
            raise ValueError("down_mag needs the family magnitude ladder")
        mono = all(family_values[i] >= family_values[i + 1] - thr
                   for i in range(len(family_values) - 1))
        passed = (delta < -thr) and mono
        note = f"monotone={mono}"
    elif expectation == "exact":
        thr = EXACT_ABS_TOL
        passed = abs(value_inj - float(expected_value)) <= thr
        note = f"expected={float(expected_value):.4f}"
    elif expectation in ("rung_le", "rung_ge"):
        want = _rung_idx(expected_value)
        got = _rung_idx(value_inj)
        thr = float(want)
        passed = got <= want if expectation == "rung_le" else got >= want
        note = f"median_rung={RUNGS[got]} vs {RUNGS[want]}"
    else:
        raise ValueError(f"unknown expectation: {expectation}")
    vb = float(value_base) if not isinstance(value_base, str) \
        else float(_rung_idx(value_base))
    vi = float(value_inj) if numeric else float(_rung_idx(value_inj))
    return CellVerdict(injection=injection, magnitude=magnitude,
                       component=component, expectation=expectation,
                       value_base=vb, value_inj=vi, delta=float(delta),
                       threshold=float(thr), passed=bool(passed), note=note)
