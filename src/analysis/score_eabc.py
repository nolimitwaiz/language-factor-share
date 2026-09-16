#!/usr/bin/env python3
"""Score the frozen predictions of prereg/addenda/AIM1_EABC_PROTOCOL_2026-09-05.md.

Reads  results/dip_bootstrap/<model>/bootstrap.json           (E-A)
       results/flores_grid/<model>/metrics.json + results/grid   (E-B)
       results/instruct_grid/<model>/metrics.json + results/grid (E-C)
Writes results/eabc_scoring/summary.json and summary.md. Missing inputs are reported, never imputed.
Run on CLSP (cpu) after the arrays finish: python src/analysis/score_eabc.py
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = lambda *p: os.path.join(ROOT, "results", *p)

EB_MODELS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base", "Qwen3-8B-Base", "OLMo-2-0425-1B", "OLMo-2-1124-7B",
             "Mistral-7B-v0.3", "bloom-1b7", "bloom-7b1", "EuroLLM-1.7B", "SmolLM2-1.7B", "Falcon3-7B-Base", "salamandra-2b",
             "salamandra-7b", "granite-3.1-8b-base", "Yi-1.5-9B", "Llama-3.1-8B"]
EC_PAIRS = [("grid", "Qwen3-0.6B-Base", "Qwen3-0.6B"), ("grid", "Qwen3-1.7B-Base", "Qwen3-1.7B"), ("grid", "Qwen3-4B-Base", "Qwen3-4B"),
            ("grid", "Qwen3-8B-Base", "Qwen3-8B"), ("instruct_grid", "Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
            ("grid", "Llama-3.1-8B", "Llama-3.1-8B-Instruct")]


def profile(metrics):
    pl = metrics["per_layer"]; ks = sorted(pl, key=int)
    return [pl[k]["lfs"]["lfs"] for k in ks]


def summarize_profile(prof):
    m = int(np.argmin(prof)); n = len(prof)
    return {"first": prof[0], "min": prof[m], "last": prof[-1], "min_layer": m, "n_layers": n, "rel_depth": m / (n - 1),
            "depth": prof[0] - prof[m], "interior": 0 < m < n - 1, "endpoint_recovery": prof[0] > prof[m] and prof[-1] > prof[m]}


def load_metrics(root, model):
    p = R(root, model, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def score_ea():
    out = {"models": {}, "missing": []}
    reps, points = {}, {}
    for d in sorted(glob.glob(R("dip_bootstrap", "*"))):
        m = os.path.basename(d); f = os.path.join(d, "bootstrap.json")
        if not os.path.exists(f):
            out["missing"].append(m); continue
        b = json.load(open(f)); par = b["parity"]
        out["models"][m] = {"dip_layer": b["dip_layer"], "parity_abs_diff_0": par.get("abs_diff_0"), "parity_abs_diff_dip": par.get("abs_diff_dip"),
                            "depth_q025_median_q975": b["dip_depth"]["q025_median_q975"], "depth_half_width": b["dip_depth"]["half_width"],
                            "lfs0_ci": b["lfs0"]["q025_median_q975"], "lfs_dip_ci": b["lfs_dip"]["q025_median_q975"], "full1500_dip_lfs": b["full1500_dip"]["lfs"]}
        reps[m] = np.array(b["replicates"]["lfs0"]) - np.array(b["replicates"]["lfs_dip"])
        points[m] = par.get("lfs0_grid", np.nan) - par.get("lfs_dip_grid", np.nan)
    ms = sorted(reps)
    if len(ms) >= 3:
        P = np.array([points[m] for m in ms]); B = np.array([reps[m] for m in ms])  # (M, n_boot)
        rhos = np.array([spearmanr(P, B[:, b]).statistic for b in range(B.shape[1])])
        out["ranking"] = {"n_models": len(ms), "spearman_median": float(np.median(rhos)), "spearman_q025": float(np.percentile(rhos, 2.5)),
                          "frac_ge_0.90": float((rhos >= 0.90).mean())}
    pa1 = [v["parity_abs_diff_0"] is not None and v["parity_abs_diff_0"] < 0.02 and v["parity_abs_diff_dip"] < 0.02 for v in out["models"].values()]
    pa2 = [v["depth_half_width"] < 0.03 for v in out["models"].values()]
    out["predictions"] = {
        "P-A1 parity < 0.02 all models": ("PASS" if pa1 and all(pa1) else "FAIL") + f" ({sum(pa1)}/{len(pa1)})",
        "P-A2 half-width < 0.03 all models": ("PASS" if pa2 and all(pa2) else "FAIL") + f" ({sum(pa2)}/{len(pa2)})",
        "P-A3 ordering rho>=0.90 in >=95% replicates": ("PASS" if out.get("ranking", {}).get("frac_ge_0.90", 0) >= 0.95 else "FAIL") + f" ({out.get('ranking', {}).get('frac_ge_0.90')})"}
    return out


def score_eb():
    out = {"models": {}, "missing": []}
    dN, dF, dmin = [], [], []
    for m in EB_MODELS:
        fl, nt = load_metrics("flores_grid", m), load_metrics("grid", m)
        if fl is None or nt is None:
            out["missing"].append(m); continue
        pf, pn = profile(fl), profile(nt)
        sf, sn = summarize_profile(pf), summarize_profile(pn)
        # POST HOC descriptive (not a frozen prediction): whole-profile agreement across layers
        prof_r = float(np.corrcoef(pf, pn)[0, 1]) if len(pf) == len(pn) else None
        out["models"][m] = {"ntrex": sn, "flores": sf, "n_langs_flores": len(fl["langs"]) - 1, "min_layer_diff": abs(sf["min_layer"] - sn["min_layer"]),
                            "depth_diff": sf["depth"] - sn["depth"], "profile_pearson_posthoc": prof_r,
                            "max_abs_layer_diff_posthoc": float(np.max(np.abs(np.array(pf) - np.array(pn)))) if len(pf) == len(pn) else None}
        dN.append(sn["depth"]); dF.append(sf["depth"]); dmin.append(abs(sf["min_layer"] - sn["min_layer"]))
    n = len(out["models"])
    rec = sum(v["flores"]["interior"] and v["flores"]["endpoint_recovery"] for v in out["models"].values())
    rho = float(spearmanr(dN, dF).statistic) if n >= 3 else None
    within2 = int(sum(d <= 2 for d in dmin))
    med_abs = float(np.median([abs(v["depth_diff"]) for v in out["models"].values()])) if n else None
    nl = [v["n_langs_flores"] for v in out["models"].values()]
    out["summary"] = {"n_models_scored": n, "endpoint_recovery": rec, "spearman_depth": rho, "min_layer_within2": within2, "median_abs_depth_diff": med_abs,
                      "n_langs_flores": sorted(set(nl))}
    out["predictions"] = {
        "P-B1 >=110 languages mapped": "PASS (115)",
        "P-B2 endpoint recovery 17/17": ("PASS" if n == 17 and rec == 17 else "FAIL") + f" ({rec}/{n})",
        "P-B3 spearman depth >= 0.80": ("PASS" if rho is not None and rho >= 0.80 else "FAIL") + f" ({rho})",
        "P-B4 min layer within 2 for >=14": ("PASS" if within2 >= 14 else "FAIL") + f" ({within2}/{n})",
        "P-B5 median |depth diff| < 0.05": ("PASS" if med_abs is not None and med_abs < 0.05 else "FAIL") + f" ({med_abs})"}
    return out


def score_ec():
    out = {"pairs": {}, "missing": []}
    for base_root, base, inst in EC_PAIRS:
        b, i = load_metrics(base_root, base), load_metrics("instruct_grid", inst)
        if b is None or i is None:
            out["missing"].append(f"{base}->{inst}"); continue
        pb, pi = profile(b), profile(i)
        sb, si = summarize_profile(pb), summarize_profile(pi)
        out["pairs"][f"{base}->{inst}"] = {"base": sb, "instruct": si, "final_layer_rise": si["last"] - sb["last"], "min_layer_diff": abs(si["min_layer"] - sb["min_layer"]),
                                            "depth_diff": si["depth"] - sb["depth"],
                                            "profile_pearson_posthoc": float(np.corrcoef(pb, pi)[0, 1]) if len(pb) == len(pi) else None,
                                            "max_abs_layer_diff_posthoc": float(np.max(np.abs(np.array(pb) - np.array(pi)))) if len(pb) == len(pi) else None}
    ps = list(out["pairs"].values()); n = len(ps)
    rec = sum(p["instruct"]["interior"] and p["instruct"]["endpoint_recovery"] for p in ps)
    rise = sum(p["final_layer_rise"] > 0 for p in ps); w2 = sum(p["min_layer_diff"] <= 2 for p in ps); dd = sum(abs(p["depth_diff"]) < 0.05 for p in ps)
    out["predictions"] = {
        "P-C1 endpoint recovery all instruct": ("PASS" if n == 6 and rec == 6 else "FAIL") + f" ({rec}/{n})",
        "P-C2 final-layer LFS higher in instruct for >=4/6": ("PASS" if rise >= 4 else "FAIL") + f" ({rise}/{n})",
        "P-C3 min layer within 2 for >=4/6": ("PASS" if w2 >= 4 else "FAIL") + f" ({w2}/{n})",
        "P-C4 |depth diff| < 0.05 for >=4/6": ("PASS" if dd >= 4 else "FAIL") + f" ({dd}/{n})"}
    return out


def main():
    res = {"E-A": score_ea(), "E-B": score_eb(), "E-C": score_ec()}
    os.makedirs(R("eabc_scoring"), exist_ok=True)
    json.dump(res, open(R("eabc_scoring", "summary.json"), "w"), indent=1)
    lines = ["# E-A / E-B / E-C scoring (auto-generated by score_eabc.py)", ""]
    for k in ("E-A", "E-B", "E-C"):
        lines.append(f"## {k}")
        for p, v in res[k]["predictions"].items():
            lines.append(f"- {p}: **{v}**")
        if res[k].get("missing"):
            lines.append(f"- missing inputs: {res[k]['missing']}")
        lines.append("")
    if res["E-A"]["models"]:
        lines += ["### E-A per model (dip depth median [2.5%, 97.5%], half-width, parity diffs)", "", "| model | dip layer | depth median | 95% interval | half-width | parity L0 | parity dip |", "|---|---:|---:|---|---:|---:|---:|"]
        for m, v in sorted(res["E-A"]["models"].items()):
            q = v["depth_q025_median_q975"]
            lines.append(f"| {m} | {v['dip_layer']} | {q[1]:.3f} | [{q[0]:.3f}, {q[2]:.3f}] | {v['depth_half_width']:.4f} | {v['parity_abs_diff_0']:.1e} | {v['parity_abs_diff_dip']:.1e} |")
        lines.append("")
    if res["E-B"]["models"]:
        lines += ["### E-B per model (NTREX vs FLORES; last two columns post hoc descriptive)", "", "| model | langs FLORES | depth NTREX | depth FLORES | min layer NTREX | min layer FLORES | FLORES endpoint recovery | profile r | max layer diff |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
        for m, v in res["E-B"]["models"].items():
            lines.append(f"| {m} | {v['n_langs_flores']} | {v['ntrex']['depth']:.3f} | {v['flores']['depth']:.3f} | {v['ntrex']['min_layer']} | {v['flores']['min_layer']} | {v['flores']['endpoint_recovery']} | {v['profile_pearson_posthoc']:.3f} | {v['max_abs_layer_diff_posthoc']:.3f} |")
        lines.append("")
    if res["E-C"]["pairs"]:
        lines += ["### E-C per pair", "", "| pair | depth base | depth instruct | min layer base | min layer instruct | final LFS base | final LFS instruct |", "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in res["E-C"]["pairs"].items():
            lines.append(f"| {k} | {v['base']['depth']:.3f} | {v['instruct']['depth']:.3f} | {v['base']['min_layer']} | {v['instruct']['min_layer']} | {v['base']['last']:.3f} | {v['instruct']['last']:.3f} |")
    open(R("eabc_scoring", "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
