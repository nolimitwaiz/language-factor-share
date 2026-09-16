#!/usr/bin/env python
"""RMFS v1 assembly (Phase 3). Local CPU — reads pulled run artifacts only.

Base models  -> observational mode: [q_T, q_L, q_K], J = 3.
Arm ckpts    -> intervention mode:  [q_T, q_L, q_K, q_C], J = 4, with
                q_L from the arm's own REML (arm_components_*), q_C =
                C_pres = min(sigma2_C_arm / sigma2_C_base_ref, 1) against
                the SAME-protocol base reference (arm_components task 75),
                q_K inherited from the base rung selection (A3 section 3).

Arms whose arm_components result is not yet pulled are SKIPPED and counted
loudly — rule 3: a blank cell beats an invented one. Re-run after the
components array drains to fill them in.

Outputs: results/tables/rmfs_v1.parquet (long: model, language, layer,
seed, component, value, run_id), results/tables/rmfs_v1_summary.csv (one
row per model/arm: pooled RMFS + component medians + flags), manifest
under results/runs/rmfs_assembly_v1/, and an S1 verdict block when all
three arms x 3 seeds are present.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import socket
import subprocess
import sys

import numpy as np
import pandas as pd

RMFS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS_DIR, "src"))

from rmfs.metrics.rmfs import (  # noqa: E402
    assemble,
    pool_languages,
    t_language,
)

RUNS = os.path.join(RMFS_DIR, "results", "runs")
TABLES = os.path.join(RMFS_DIR, "results", "tables")
BASE_TAG = "Qwen3-0.6B-Base"
BASE_MODELS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base",
               "Qwen3-8B-Base", "OLMo-2-0425-1B", "OLMo-2-1124-7B",
               "bloom-1b7", "bloom-7b1", "EuroLLM-1.7B", "SmolLM2-1.7B",
               "Falcon3-7B-Base", "Mistral-7B-v0.3", "salamandra-2b",
               "salamandra-7b"]
ANALYSIS_SEED = 13
RUN_ID = "rmfs_assembly_v1"

# S1 (prereg section 5): WA-C below WA-G AND below the LM-only control,
# >= 2 of 3 seeds per comparison, driven by q_T and/or q_C.
S1_COLLAPSED, S1_SOUND, S1_CONTROL = "armWA-C", "armWA-G", "armB"


def grid(tag: str) -> dict:
    p = os.path.join(RUNS, f"grid_components_{tag}", "result.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing Phase-1 grid result: {p}")
    return json.load(open(p))


def t_by_lang(run_dir: str) -> dict:
    p = os.path.join(run_dir, "t_rows.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing T rows: {p}")
    df = pd.read_parquet(p)
    out = {}
    for lg, g in df.groupby("lang"):
        out[lg] = t_language(g.nll_none.values, g.nll_native.values,
                             g.nll_mismatch.values, g.nll_inject.values,
                             g.nll_inject_wrong.values)
    return out


def arm_vc(name: str) -> dict | None:
    p = os.path.join(RUNS, f"arm_components_{name}", "result.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main() -> None:
    os.makedirs(TABLES, exist_ok=True)
    base_grid = grid(BASE_TAG)
    base_qk = {lg: v["q_K"] for lg, v in base_grid["K"].items()}

    long_rows, summary = [], []

    def emit(model: str, mode: str, dip: int, tset: dict, q_l: float,
             qk_map: dict, q_c: float | None) -> None:
        cells = {}
        for lg, tl in tset.items():
            if lg not in qk_map:
                continue                      # lang outside grid ladder set
            cell = assemble(tl.q_t, q_l, qk_map[lg], q_c)
            cells[lg] = (tl, cell)
            for comp, val in [("q_T", tl.q_t), ("T_raw", tl.t_raw),
                              ("q_L", cell.q_l), ("q_K", cell.q_k),
                              ("q_C", cell.q_c), ("RMFS", cell.rmfs)]:
                long_rows.append(dict(model=model, language=lg, layer=dip,
                                      seed=ANALYSIS_SEED, component=comp,
                                      value=val, run_id=RUN_ID, mode=mode))
        pool = pool_languages([c.rmfs for _, c in cells.values()])
        n_sat = sum(tl.saturated for tl, _ in cells.values())
        summary.append(dict(
            model=model, mode=mode, layer=dip,
            rmfs_median=pool["median"], rmfs_mean=pool["mean"],
            n_defined=pool["n_defined"], n_undefined=pool["n_undefined"],
            n_saturated=n_sat,
            q_T_median=float(np.nanmedian([tl.q_t for tl, _ in
                                           cells.values()])),
            q_L=q_l, q_C=(math.nan if q_c is None else q_c),
            q_K_median=float(np.median([c.q_k for _, c in cells.values()])),
        ))

    # ---- base models, observational --------------------------------------
    for tag in BASE_MODELS:
        g = grid(tag)
        emit(tag, "observational", g["dip_layer"],
             t_by_lang(os.path.join(RUNS, f"t_extract_{tag}")),
             1.0 - g["lfs_vc"], {lg: v["q_K"] for lg, v in g["K"].items()},
             None)

    # ---- arms, intervention ----------------------------------------------
    ref = arm_vc(BASE_TAG)
    arm_dirs = sorted(d for d in os.listdir(RUNS)
                      if d.startswith("t_extract_ckpt_"))
    missing = []
    for d in arm_dirs:
        name = d.removeprefix("t_extract_")      # ckpt_<arm>_seed<k>
        vc = arm_vc(name)
        if vc is None or ref is None:
            missing.append(name)
            continue
        s2c = vc["variance_components"]["sigma2_C_reml"]
        s2c_ref = ref["variance_components"]["sigma2_C_reml"]
        emit(name, "intervention", vc["protocol"]["dip"],
             t_by_lang(os.path.join(RUNS, d)),
             1.0 - vc["variance_components"]["lfs_vc"], base_qk,
             min(s2c / s2c_ref, 1.0))
    if ref is None:
        print(f"[skip] base reference arm_components_{BASE_TAG} not pulled "
              f"-> ALL {len(arm_dirs)} arms skipped", flush=True)
    elif missing:
        print(f"[skip] {len(missing)}/{len(arm_dirs)} arms lack "
              f"arm_components (array still draining)", flush=True)

    long_df = pd.DataFrame(long_rows)
    sum_df = pd.DataFrame(summary)
    long_df.to_parquet(os.path.join(TABLES, "rmfs_v1.parquet"), index=False)
    sum_df.to_csv(os.path.join(TABLES, "rmfs_v1_summary.csv"), index=False,
                  float_format="%.4f")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sum_df.to_string(index=False))

    # ---- S1 verdict when the crash-test trio is fully assembled ----------
    s1 = {}
    for arm in (S1_COLLAPSED, S1_SOUND, S1_CONTROL):
        for s in (0, 1, 2):
            row = sum_df[sum_df.model == f"ckpt_{arm}_seed{s}"]
            if len(row) == 1:
                s1[(arm, s)] = row.iloc[0]
    if len(s1) == 9:
        print("\n=== S1 VERDICT (intervention-mode RMFS, median pool; "
              "mean in parens) ===")
        wins_g = wins_b = 0
        for s in (0, 1, 2):
            c, g_, b = (s1[(a, s)] for a in
                        (S1_COLLAPSED, S1_SOUND, S1_CONTROL))
            vg, vb = (c.rmfs_median < g_.rmfs_median,
                      c.rmfs_median < b.rmfs_median)
            wins_g += vg
            wins_b += vb
            print(f" seed {s}: WA-C {c.rmfs_median:.4f} ({c.rmfs_mean:.4f})"
                  f" | WA-G {g_.rmfs_median:.4f} ({g_.rmfs_mean:.4f})"
                  f" | B {b.rmfs_median:.4f} ({b.rmfs_mean:.4f})"
                  f" | C<G {vg}  C<B {vb}")
        verdict = "PASS" if (wins_g >= 2 and wins_b >= 2) else "FAIL"
        print(f" S1: {verdict}  (C<G {wins_g}/3, C<B {wins_b}/3; "
              f"bar >= 2/3 each)")
    else:
        print(f"\n[S1] deferred: {len(s1)}/9 crash-test cells assembled")

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS_DIR,
                         capture_output=True, text=True).stdout.strip()
    run_dir = os.path.join(RUNS, RUN_ID)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({
            "run_id": RUN_ID, "script": "scripts/assemble_rmfs.py",
            "git": git, "host": socket.gethostname(),
            "when": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tau": 0.1, "analysis_seed": ANALYSIS_SEED,
            "pool_rule": "median primary, mean sensitivity, undefined "
                         "excluded+counted (declared pre-assembly)",
            "n_models": len(sum_df),
            "n_arms_missing_components": len(missing) if ref else
                len(arm_dirs),
        }, f, indent=1)
    print(f"\n[done] {len(sum_df)} models/arms -> results/tables/", flush=True)


if __name__ == "__main__":
    main()
