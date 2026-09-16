#!/usr/bin/env python3
"""Score the W0 offline offset pre-test against the frozen predictions in
prereg/addenda/AIM2_W0_OFFSET_OFFLINE_PROTOCOL_2026-09-07.md.

Reads results/aim2/offset_offline/<model>/readings.json and results/budget/<model>/budget.json.
Writes results/aim2/offset_offline/summary.json and summary.md.
"""
import glob, json, os, statistics
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "aim2", "offset_offline")


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def main():
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*", "readings.json"))):
        r = json.load(open(f)); m = r["model"]; dip = str(r["dip_layer"]); L = r["per_layer"][dip]["conditions"]
        I0, I1, I2 = L["I0"], L["I1"], L["I2"]
        I3 = [L[k] for k in L if k.startswith("I3_")]
        rows[m] = {
            "dip": int(dip),
            "lfs_I0": I0["lfs"], "lfs_I1": I1["lfs"], "lfs_drop": I0["lfs"] - I1["lfs"],
            "ss_lang_drop_frac": 1 - I1["ss_lang"] / I0["ss_lang"],
            "rawcv_ratio_I1": I1["raw_concept_var"] / I0["raw_concept_var"],
            "mexa_I0": I0["mexa_mean"], "mexa_I1": I1["mexa_mean"], "mexa_change_I1": I1["mexa_mean"] - I0["mexa_mean"],
            "mexa_I2_minus_I1": I2["mexa_mean"] - I1["mexa_mean"], "aar10_I2_minus_I1": I2["aar10_mean"] - I1["aar10_mean"],
            "ss_res_share_I1": I1["ss_res_share"], "ss_res_share_I2": I2["ss_res_share"],
            "mexa_change_I3_mean": float(np.mean([c["mexa_mean"] - I0["mexa_mean"] for c in I3])),
            "lfs_drop_calib300_diff": abs((I0["lfs"] - r["per_layer"][dip]["I1_calib300_lfs"]) - (I0["lfs"] - I1["lfs"])),
            "parity_dip": r["per_layer"][dip].get("parity_grid_dip"),
            "boot": r["per_layer"][dip].get("dip_lfs_drop_bootstrap"),
        }
        bpath = os.path.join(ROOT, "results", "budget", m, "budget.json")
        if os.path.exists(bpath):
            b = json.load(open(bpath)); pl = b["per_language"]
            rows[m]["budget_M2_median"] = float(np.median([pl[l]["share_M2"]["mean"] for l in pl]))
    n = len(rows)
    P = {}
    P["P1 LFS drop >= 0.10 and SS_lang drop >= 50% in all; raw concept var ratio 1.000"] = (
        sum(v["lfs_drop"] >= 0.10 and v["ss_lang_drop_frac"] >= 0.50 for v in rows.values()), n,
        all(abs(v["rawcv_ratio_I1"] - 1) < 5e-4 for v in rows.values()))
    resp = sum(abs(v["mexa_change_I1"]) >= 0.010 for v in rows.values())
    P["P2 |MEXA change| >= 0.010 (respond) count"] = (resp, n, "respond" if resp >= 10 else ("near-invariant" if n - resp >= 10 else "mixed"))
    P["P3a MEXA and AaR10 identical under I2 vs I1 (3 decimals)"] = (sum(abs(v["mexa_I2_minus_I1"]) < 5e-4 and abs(v["aar10_I2_minus_I1"]) < 5e-4 for v in rows.values()), n)
    with_b = [m for m in rows if "budget_M2_median" in rows[m]]
    if len(with_b) >= 5:
        drop = [rows[m]["ss_res_share_I1"] - rows[m]["ss_res_share_I2"] for m in with_b]
        P["P3b Spearman(residual-share drop I1->I2, budget M2 median) >= 0.5"] = (spearman(drop, [rows[m]["budget_M2_median"] for m in with_b]), len(with_b))
    P["P4 |I3 change| < 0.5 * |I1 change| count"] = (sum(abs(v["mexa_change_I3_mean"]) < 0.5 * abs(v["mexa_change_I1"]) for v in rows.values()), n)
    P["P5 calib-300 vs calib-1200 LFS drop within 0.02 count"] = (sum(v["lfs_drop_calib300_diff"] <= 0.02 for v in rows.values()), n)
    summary = {"n_models": n, "predictions": {k: list(v) if isinstance(v, tuple) else v for k, v in P.items()}, "rows": rows}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    md = ["# W0 offline offset pre-test: scorecard", "", f"Models scored: {n}", ""]
    for k, v in P.items(): md.append(f"- {k}: {v}")
    md += ["", "| model | dip | LFS I0 | LFS I1 | SS_lang drop | rawCV ratio | MEXA I0 | MEXA I1 | dMEXA I1 | dMEXA I3 | I2-I1 MEXA | calib300 diff |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in sorted(rows.items(), key=lambda kv: -kv[1]["lfs_drop"]):
        md.append(f"| {m} | {v['dip']} | {v['lfs_I0']:.3f} | {v['lfs_I1']:.3f} | {v['ss_lang_drop_frac']*100:.0f}% | {v['rawcv_ratio_I1']:.4f} | {v['mexa_I0']:.3f} | {v['mexa_I1']:.3f} | {v['mexa_change_I1']:+.3f} | {v['mexa_change_I3_mean']:+.3f} | {v['mexa_I2_minus_I1']:+.4f} | {v['lfs_drop_calib300_diff']:.4f} |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
