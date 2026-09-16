#!/usr/bin/env python
"""A5 verdict: score the frozen predictions P1-P4 verbatim and produce
the arm-panel validity table (local CPU; pulled artifacts only).

Panel statistic (declared in A5 BEFORE any score existed): per-language
soft-min over [q_L, q_K, q_C] (tau = 0.1), median over languages. T and
the T-carrying intervention RMFS are reported BESIDE, in no primary
claim. Validity: Spearman across the 75 checkpoints vs panel_acc with
arm-family cluster bootstrap (25 families, 2000 reps, seed 13).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

RMFS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS_DIR, "src"))

from rmfs.metrics.rmfs import softmin  # noqa: E402

RUNS = os.path.join(RMFS_DIR, "results", "runs")
BEL = os.path.join(RMFS_DIR, "results", "belebele_arms")
TABLES = os.path.join(RMFS_DIR, "results", "tables")
BASE_TAG = "Qwen3-0.6B-Base"
N_BOOT, SEED = 2000, 13


def load_panel() -> pd.DataFrame:
    grid = json.load(open(os.path.join(
        RUNS, f"grid_components_{BASE_TAG}", "result.json")))
    qk = {lg: v["q_K"] for lg, v in grid["K"].items()}
    ref = json.load(open(os.path.join(
        RUNS, f"arm_components_{BASE_TAG}", "result.json")))
    ref_sigma = ref["variance_components"]["sigma2_C_reml"]

    summ = pd.read_csv(os.path.join(TABLES, "rmfs_v1_summary.csv"))

    rows = []
    for d in sorted(os.listdir(BEL)):
        files = glob.glob(os.path.join(BEL, d, "**", "results_*.json"),
                          recursive=True)
        if not files:
            raise FileNotFoundError(f"no lm-eval results for {d}")  # E2
        res = json.load(open(sorted(files)[-1]))["results"]
        accs = {k: v["acc,none"] for k, v in res.items()
                if k.startswith("belebele_")}
        non_eng = [v for k, v in accs.items() if k != "belebele_eng_Latn"]
        if len(non_eng) != 40:
            raise ValueError(f"{d}: {len(non_eng)} non-eng tasks != 40")

        vc = json.load(open(os.path.join(
            RUNS, f"arm_components_{d}", "result.json")))
        bl = json.load(open(os.path.join(
            RUNS, f"arm_baselines_{d}", "result.json")))["baselines"]
        q_l = 1.0 - vc["variance_components"]["lfs_vc"]
        c_pres = min(vc["variance_components"]["sigma2_C_reml"] / ref_sigma,
                     1.0)
        panel_langs = [lg for lg in bl if lg in qk]
        panel_stat = float(np.median(
            [softmin([q_l, qk[lg], c_pres]) for lg in panel_langs]))

        srow = summ[summ.model == d]
        rows.append(dict(
            ckpt=d, family=d.replace("ckpt_", "").rsplit("_seed", 1)[0],
            seed=int(d.rsplit("seed", 1)[1]),
            panel_acc=float(np.mean(non_eng)),
            eng_acc=accs["belebele_eng_Latn"],
            mexa=float(np.mean([v["mexa"] for v in bl.values()])),
            aar=float(np.mean([v["aar_tail"] for v in bl.values()])),
            q_L=q_l, C_pres=c_pres,
            q_K_med=float(np.median([qk[lg] for lg in panel_langs])),
            panel_stat=panel_stat,
            rmfs_int=(float(srow.rmfs_median.iloc[0]) if len(srow)
                      else np.nan),
            q_T=(float(srow.q_T_median.iloc[0]) if len(srow) else np.nan),
        ))
    return pd.DataFrame(rows)


def cluster_spearman(df: pd.DataFrame, col: str) -> tuple:
    fams = df.family.unique()
    by = {f: df[df.family == f] for f in fams}
    point = stats.spearmanr(df[col], df.panel_acc).statistic
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(fams, size=len(fams), replace=True)
        s = pd.concat([by[f] for f in pick])
        if s[col].nunique() > 2:
            vals.append(stats.spearmanr(s[col], s.panel_acc).statistic)
    v = np.array([x for x in vals if np.isfinite(x)])
    if v.size == 0:                     # constant column (e.g. inherited
        return float(point), np.nan, np.nan   # q_K): no rank signal
    return float(point), float(np.percentile(v, 2.5)), \
        float(np.percentile(v, 97.5))


def main() -> None:
    df = load_panel()
    df.to_csv(os.path.join(TABLES, "a5_arm_panel.csv"), index=False,
              float_format="%.4f")
    print(f"[panel] {len(df)} checkpoints, {df.family.nunique()} families")

    def acc(fam, s):
        r = df[(df.family == fam) & (df.seed == s)]
        return float(r.panel_acc.iloc[0]) if len(r) else np.nan

    # ---- P1: WA-C ability degraded >= 5pp vs B, below WA-G, >=2/3 seeds
    p1_hits = 0
    print("\n=== P1: WA-C ability (panel_acc, seed-matched) ===")
    for s in (0, 1, 2):
        c, b, g = acc("armWA-C", s), acc("armB", s), acc("armWA-G", s)
        hit = (c <= b - 0.05) and (c < g)
        p1_hits += hit
        print(f"  seed {s}: WA-C {c:.4f} | B {b:.4f} | WA-G {g:.4f} "
              f"| degraded>=5pp & <WA-G: {hit}")
    p1 = p1_hits >= 2
    print(f"  P1: {'PASS' if p1 else 'FAIL'} ({p1_hits}/3)")
    if not p1:
        print("  -> C_pres FALSE-ALARM finding recorded; P2/P3 void per A5")

    # ---- validity table --------------------------------------------------
    print("\n=== arm-panel validity: Spearman vs panel_acc "
          "(family-cluster bootstrap) ===")
    table = []
    for col in ["panel_stat", "mexa", "aar", "q_L", "C_pres", "q_K_med",
                "rmfs_int", "q_T"]:
        rho, lo, hi = cluster_spearman(df.dropna(subset=[col]), col)
        beside = "  (beside; not primary)" if col in ("rmfs_int", "q_T") \
            else ""
        table.append(dict(metric=col, rho=rho, ci_lo=lo, ci_hi=hi))
        print(f"  {col:10s} rho={rho:+.3f}  CI[{lo:+.3f},{hi:+.3f}]{beside}")
    tab = pd.DataFrame(table)
    tab.to_csv(os.path.join(TABLES, "a5_validity.csv"), index=False,
               float_format="%.4f")

    rho_panel = float(tab[tab.metric == "panel_stat"].rho.iloc[0])
    lo_panel = float(tab[tab.metric == "panel_stat"].ci_lo.iloc[0])
    rho_mexa = float(tab[tab.metric == "mexa"].rho.iloc[0])

    if p1:
        # ---- P2: MEXA mis-ranks WA-C high + lower validity than panel
        b_med = float(df[df.family == "armB"].mexa.median())
        mis = [float(df[(df.family == "armWA-C") & (df.seed == s)]
                     .mexa.iloc[0]) > b_med for s in (0, 1, 2)]
        p2 = all(mis) and (rho_mexa < rho_panel)
        print(f"\n  P2: WA-C mexa > armB median in {sum(mis)}/3 seeds; "
              f"mexa rho {rho_mexa:+.3f} < panel rho {rho_panel:+.3f}: "
              f"{rho_mexa < rho_panel}  -> {'PASS' if p2 else 'FAIL'}")
        # ---- P3: panel validity positive, CI excludes 0
        p3 = rho_panel > 0 and lo_panel > 0
        print(f"  P3: panel rho {rho_panel:+.3f} CI_lo {lo_panel:+.3f} "
              f"-> {'PASS' if p3 else 'FAIL'}")

    # ---- P4: alignment gives no downstream gain
    gd = (np.mean([acc("armWA-G", s) for s in (0, 1, 2)])
          - np.mean([acc("armB", s) for s in (0, 1, 2)]))
    p4 = gd <= 0.02
    print(f"  P4: WA-G - B = {gd:+.4f} (bar <= +0.02) "
          f"-> {'PASS' if p4 else 'FAIL'}")

    print("\n[done] tables -> a5_arm_panel.csv, a5_validity.csv")


if __name__ == "__main__":
    main()
