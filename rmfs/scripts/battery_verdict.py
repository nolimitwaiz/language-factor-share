#!/usr/bin/env python
"""S2 battery verdict: combine battery_geo + battery_t, score all 75
frozen signature cells (local CPU; reads pulled artifacts only).

Aggregates and floors exactly as declared in worlds.py / battery_geo.py /
battery_t.py BEFORE the runs. Emits results/tables/battery_verdict.csv
(one row per cell) and prints the scorecard. Cells whose inputs are
missing are reported UNSCORED, never guessed (rule 3).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd

RMFS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS_DIR, "src"))

from rmfs.battery.score import (  # noqa: E402
    FLOOR_MIN,
    RUNGS,
    score_cell,
)
from rmfs.battery.worlds import AFFECTED_LANG  # noqa: E402
from rmfs.metrics.rmfs import softmin, t_language  # noqa: E402

RUNS = os.path.join(RMFS_DIR, "results", "runs")
SIG = os.path.join(RMFS_DIR, "prereg", "signatures", "rmfs_signatures.csv")
PIVOT = "eng"

# CSV family order for down_mag ladders: increasing severity
FAMILY_ORDER = {"rotation_per_language": ["0.1", "0.4", "0.8"],
                "uniform_collapse": ["0.9", "0.5", "0.1"]}
# uniform_collapse severity INCREASES with m (m is the collapse fraction);
# rotation severity increases with theta. Order above lists CSV magnitudes
# from least to most severe.
FAMILY_ORDER["uniform_collapse"] = ["0.1", "0.5", "0.9"]


def med_rung_idx(rungs: dict, langs) -> float:
    idx = [RUNGS.index(rungs[lg]["rung"]) for lg in langs if lg in rungs]
    return float(np.median(idx))


def t_aggregate(tdf: pd.DataFrame, base: pd.DataFrame, affected) -> float:
    """Median over affected languages of frozen language-level T, using
    config inject NLLs joined with base text-condition NLLs."""
    m = tdf[["lang", "pair", "nll_inject", "nll_inject_wrong"]].merge(
        base[["lang", "pair", "nll_none", "nll_native", "nll_mismatch"]],
        on=["lang", "pair"], how="inner")
    vals = []
    for lg, g in m.groupby("lang"):
        if lg not in affected:
            continue
        tl = t_language(g.nll_none.values, g.nll_native.values,
                        g.nll_mismatch.values, g.nll_inject.values,
                        g.nll_inject_wrong.values)
        if not tl.undefined:
            vals.append(tl.t_raw)
    return float(np.median(vals)) if vals else math.nan


def main() -> None:
    geo = json.load(open(os.path.join(RUNS, "battery_geo", "result.json")))
    trows = pd.read_parquet(os.path.join(RUNS, "battery_t",
                                         "t_rows.parquet"))
    tbase = pd.read_parquet(os.path.join(RUNS, "battery_t",
                                         "base_rows.parquet"))
    langs = sorted(geo["base"]["v_lang"].keys())
    nonpivot = [lg for lg in langs if lg != PIVOT]
    t_langs = sorted(set(trows.lang))

    # ---------------- component values per world -------------------------
    def geo_vals(w: dict, affected, base_sigma) -> dict:
        return {"q_L": w["q_L"],
                "q_K": med_rung_idx(w["rungs"], affected),
                "C_pres": w["sigma2_C_raw"] / base_sigma}

    base_sigma = geo["base"]["sigma2_C_raw"]
    base_geo = geo_vals(geo["base"], nonpivot, base_sigma)

    # T values: base from the Phase-2 run rows; configs from battery_t
    t_base_val = t_aggregate(tbase, tbase,
                             [lg for lg in t_langs if lg != PIVOT])
    t_cfg = {}
    for key, g in trows.groupby("config"):
        name = key.split("|")[0]
        aff = ([AFFECTED_LANG] if name == "collapse_per_language"
               else [lg for lg in t_langs if lg != PIVOT])
        t_cfg[key] = t_aggregate(g, tbase, aff)

    # ---------------- floors ---------------------------------------------
    fw = geo["floor_worlds"]
    floors = {
        "q_L": max(float(np.std([w["q_L"] for w in [geo["base"], *fw]],
                                ddof=1)), FLOOR_MIN),
        "q_K": max(float(np.std(
            [med_rung_idx(w["rungs"], nonpivot)
             for w in [geo["base"], *fw]], ddof=1)), FLOOR_MIN),
        "C_pres": max(float(np.std(
            [w["sigma2_C_raw"] / base_sigma for w in [geo["base"], *fw]],
            ddof=1)), FLOOR_MIN),
        "q_L_raw": max(float(np.std(
            [w.get("q_L_raw", math.nan) for w in [geo["base"], *fw]],
            ddof=1)), FLOOR_MIN) if "q_L_raw" in geo["base"] else FLOOR_MIN,
    }
    id_reps = [t_cfg[k] for k in t_cfg if k.startswith("identity|")]
    id_s13 = t_cfg.get("identity|-|s13", math.nan)
    floors["T"] = max(
        float(np.nanstd(id_reps, ddof=1)) if len(id_reps) > 1 else 0.0,
        abs(id_s13 - t_base_val) if not math.isnan(id_s13) else 0.0,
        FLOOR_MIN)

    # RMFS per world: soft-min of [q_T, q_L, q_K_norm, q_C] medians —
    # battery RMFS cells use the aggregate component vector (declared).
    def rmfs_of(tval, g):
        q_t = min(max(tval, 0.0), 1.0) if not math.isnan(tval) else math.nan
        q_k = 1.0 - g["q_K"] / 5.0
        q_c = min(g["C_pres"], 1.0)
        return softmin([q_t, g["q_L"], q_k, q_c])

    base_rmfs = rmfs_of(t_base_val, base_geo)
    _seeds = [13, 42, 71]
    floors["RMFS"] = max(
        float(np.nanstd([rmfs_of(t_cfg.get(f"identity|-|s{_seeds[i]}",
                                           t_base_val),
                                 geo_vals(fw[i], nonpivot, base_sigma))
                         for i in range(min(len(_seeds), len(fw)))],
                        ddof=1)),
        FLOOR_MIN)

    # ---------------- score every CSV cell -------------------------------
    verdicts, unscored = [], []
    with open(SIG) as f:
        cells = list(csv.DictReader(f))

    # family value ladders for down_mag (per component)
    def cfg_component(name, mag, comp):
        key = f"{name}|{mag}"
        aff = ([AFFECTED_LANG] if name == "collapse_per_language"
               else nonpivot)
        if comp == "T":
            return t_cfg.get(f"{key}|s13", math.nan)
        w = geo["configs"].get(key)
        if w is None:
            return math.nan
        if comp == "q_L":
            # the rotation_global cell's frozen note pins q_L to the
            # INVARIANT track (raw REML, no z-scoring); all other q_L
            # cells score the metric's native z-track
            if name == "rotation_global_shared":
                return w.get("q_L_raw", math.nan)
            return w["q_L"]
        if comp == "q_K":
            # numeric q_K cells speak in the q_K VALUE (1 - rung/5);
            # only rung_le/rung_ge cells speak in rungs (handled below)
            return med_rung_idx(w["rungs"], aff)
        if comp == "C_pres":
            if name == "collapse_per_language":
                return (w["v_lang"][AFFECTED_LANG]
                        / geo["base"]["v_lang"][AFFECTED_LANG])
            return w["sigma2_C_raw"] / base_sigma
        if comp == "RMFS":
            tv = t_cfg.get(f"{key}|s13", math.nan)
            return rmfs_of(tv, geo_vals(w, aff, base_sigma))
        raise ValueError(comp)

    base_val = {"T": t_base_val, "q_L": base_geo["q_L"],
                "q_K": base_geo["q_K"], "C_pres": 1.0, "RMFS": base_rmfs}

    for c in cells:
        name, mag, comp = c["injection"], c["magnitude"], c["component"]
        exp, val, k = c["expectation"], c["value"], float(c["k_noise"])
        if name == "identity":
            vi = ({"T": id_s13, "q_L": fw[0]["q_L"],
                   "q_K": med_rung_idx(fw[0]["rungs"], nonpivot),
                   "C_pres": fw[0]["sigma2_C_raw"] / base_sigma,
                   "RMFS": rmfs_of(id_s13,
                                   geo_vals(fw[0], nonpivot, base_sigma))}
                  [comp])
        else:
            vi = cfg_component(name, mag, comp)
        if isinstance(vi, float) and math.isnan(vi):
            unscored.append((name, mag, comp, "input missing/undefined"))
            continue
        fam = None
        if exp == "down_mag":
            fam = [cfg_component(name, m2, comp)
                   for m2 in FAMILY_ORDER[name]]
        vb = base_val[comp]
        if exp in ("rung_le", "rung_ge"):
            vi_arg = RUNGS[int(round(vi))]
            v = score_cell(name, mag, comp, exp, val, k, vb, vi_arg,
                           floors.get(comp, FLOOR_MIN))
        else:
            floor_c = floors.get(comp, FLOOR_MIN)
            if name == "rotation_global_shared" and comp == "q_L":
                vb = geo["base"].get("q_L_raw", math.nan)
                floor_c = floors["q_L_raw"]
                if math.isnan(vb) or math.isnan(vi):
                    unscored.append((name, mag, comp,
                                     "invariant track absent from geo run"))
                    continue
            if comp == "q_K":
                # numeric q_K cells speak in the VALUE 1 - rung/5; the
                # med_rung_idx plumbing speaks in indices. Linear map:
                # sign flips (down on value = up on index), floor scales.
                vi, vb = 1.0 - vi / 5.0, 1.0 - vb / 5.0
                floor_c = max(floor_c / 5.0, FLOOR_MIN)
                if fam is not None:
                    fam = [1.0 - x / 5.0 for x in fam]
            expected = float(val) if exp == "exact" else None
            v = score_cell(name, mag, comp, exp, expected, k, vb, vi,
                           floor_c, family_values=fam)
        verdicts.append(v)

    out = pd.DataFrame([vars(v) for v in verdicts])
    tables = os.path.join(RMFS_DIR, "results", "tables")
    os.makedirs(tables, exist_ok=True)
    out.to_csv(os.path.join(tables, "battery_verdict.csv"), index=False,
               float_format="%.6f")

    n_pass = int(out.passed.sum())
    print(f"\n=== S2 BATTERY SCORECARD: {n_pass}/{len(out)} cells pass "
          f"({len(unscored)} unscored) ===")
    print("floors: " + ", ".join(f"{k}={v:.5f}"
                                  for k, v in floors.items()))
    for comp in ["T", "q_L", "q_K", "C_pres", "RMFS"]:
        sub = out[out.component == comp]
        print(f"  {comp:7s} {int(sub.passed.sum())}/{len(sub)}")
    fails = out[~out.passed]
    if len(fails):
        print("\nFAILED cells:")
        for _, r in fails.iterrows():
            print(f"  {r.injection}|{r.magnitude} {r.component} "
                  f"[{r.expectation}] base={r.value_base:.4f} "
                  f"inj={r.value_inj:.4f} thr={r.threshold:.5f} {r.note}")
    for u in unscored:
        print(f"  UNSCORED: {u}")


if __name__ == "__main__":
    main()
