"""Battery injection worlds: thin adapter over legacy/inject.py (read-only,
rule 7) mapping the frozen signature-CSV injection names onto the legacy
calibrated forms (I8b exactness was established on those implementations).

DECLARED BEFORE SCORING (2026-08-04, ledgered with the battery run):
- Free "calibrated" magnitudes: offset 0.5 (rms units), scale 0.5
  (log-uniform band), shear 0.5, low-rank interaction 0.5 with k = 8,
  warp 2.0 (a = std/m tanh squash). Frozen-in-CSV magnitudes are used as
  frozen (rotation θ ∈ {0.1, 0.4, 0.8}, collapse m ∈ {0.9, 0.5, 0.1}).
- uniform_collapse -> legacy i8b GLOBAL-centroid collapse with
  include_pivot=True: a training-arm collapse affects every language
  including English (WA-C did), and with all languages scaled every REML
  sum of squares scales by (1−m)² so C_pres is exact to the signature's
  3-decimal calibration.
- collapse_per_language -> legacy i8 OWN-centroid collapse restricted to
  the declared affected language "deu".
- Injection point per legacy spec: pooled PRE-z-score fp32 embeddings;
  the full pipeline (z-scoring included) runs after injection.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "legacy", "inject.py")


def _load_legacy():
    spec = importlib.util.spec_from_file_location(
        "legacy_inject", os.path.abspath(_LEGACY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LEGACY = _load_legacy()

AFFECTED_LANG = "deu"          # collapse_per_language target, declared
MAG = {"offset_per_language": 0.5, "scale_isotropic_per_language": 0.5,
       "shear_general_linear": 0.5, "lowrank_CxL_interaction": 0.5,
       "monotone_warp": 2.0, "lowrank_k": 8}

# (csv_injection_name, csv_magnitude_label) in signature order; identity
# runs once per floor seed, all others once at seed 13.
CONFIGS = [
    ("identity", "-"),
    ("offset_per_language", "calibrated"),
    ("scale_isotropic_per_language", "calibrated"),
    ("rotation_global_shared", "-"),
    ("rotation_per_language", "0.1"),
    ("rotation_per_language", "0.4"),
    ("rotation_per_language", "0.8"),
    ("shear_general_linear", "calibrated"),
    ("lowrank_CxL_interaction", "calibrated"),
    ("monotone_warp", "calibrated"),
    ("uniform_collapse", "0.9"),
    ("uniform_collapse", "0.5"),
    ("uniform_collapse", "0.1"),
    ("collapse_per_language", "0.5"),
    ("pairing_permutation", "-"),
]


def affected_languages(name: str, langs) -> list:
    """Languages whose cells the config is scored on (non-pivot targets;
    single language for the per-language collapse)."""
    if name == "collapse_per_language":
        return [AFFECTED_LANG]
    return [lg for lg in langs if lg != LEGACY.PIVOT]


def apply_config(E: dict, name: str, mag: str, seed: int,
                 basis: np.ndarray, mean: np.ndarray) -> tuple[dict, dict]:
    """Apply one signature-CSV config to E = {lang: (N, D)}. basis/mean are
    the BASE-world rank-r subspace (fit pre-injection, as in the legacy
    battery). Returns (E_new, meta)."""
    rng = np.random.default_rng(seed)
    r = basis.shape[1]
    if name == "identity":
        return LEGACY.i10_identity(E)
    if name == "offset_per_language":
        return LEGACY.i1_offset(E, MAG[name], rng)
    if name == "scale_isotropic_per_language":
        return LEGACY.i2_scale(E, MAG[name], rng)
    if name == "rotation_global_shared":
        return LEGACY.i3_global_rotation(E, rng)
    if name == "rotation_per_language":
        return LEGACY.i4_lang_rotation(E, float(mag), r, basis, mean, rng)
    if name == "shear_general_linear":
        return LEGACY.i5_shear(E, MAG[name], r, basis, mean, rng)
    if name == "lowrank_CxL_interaction":
        return LEGACY.i6_interaction(E, MAG[name], MAG["lowrank_k"], r,
                                     basis, mean, rng)
    if name == "monotone_warp":
        return LEGACY.i7_warp(E, MAG[name], r, basis, mean, rng)
    if name == "uniform_collapse":
        return LEGACY.i8b_global_collapse(E, float(mag), include_pivot=True)
    if name == "collapse_per_language":
        sub = {AFFECTED_LANG: E[AFFECTED_LANG], LEGACY.PIVOT: E[LEGACY.PIVOT]}
        out, meta = LEGACY.i8_collapse(sub, float(mag))
        merged = {lg: (out[lg] if lg == AFFECTED_LANG else X.copy())
                  for lg, X in E.items()}
        return merged, meta
    if name == "pairing_permutation":
        return LEGACY.i9_permute(E, rng)
    raise ValueError(f"unknown battery config: {name}")
