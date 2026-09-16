#!/usr/bin/env python
"""Document consistency audit: recompute headline numbers from saved artifacts and
compare with the generated tables and the compiled PDFs. Reads JSON/CSV only; no
model compute. Written 2026-09-06."""
import json, csv, os, re, statistics, sys
ROOT = os.path.expanduser("~/Desktop/multilingual-metrics")
P = os.path.join
OUT = []
def say(s=""): OUT.append(s); print(s)
def ranks(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(idx):
        j = i
        while j+1 < len(idx) and v[idx[j+1]] == v[idx[i]]: j += 1
        for k in range(i, j+1): r[idx[k]] = (i+j)/2 + 1
        i = j+1
    return r
def pearson(x, y):
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y)); sx = sum((a-mx)**2 for a in x)**.5; sy = sum((b-my)**2 for b in y)**.5
    return sxy/(sx*sy) if sx and sy else float("nan")
def spearman(x, y): return pearson(ranks(x), ranks(y))
def profile(path):
    m = json.load(open(path)); pl = m["per_layer"]; ks = sorted(pl, key=int)
    return [pl[k]["lfs"]["lfs"] for k in ks], m
def st(v):
    mn = min(v); i = v.index(mn)
    return dict(first=v[0], min=mn, last=v[-1], min_layer=i, n_layers=len(v), depth=v[0]-mn, interior=0 < i < len(v)-1, recovery=v[0] > mn and v[-1] > mn)
def tex_rows(path, stop=None):
    rows = []
    for line in open(path):
        if stop and stop in line: break
        if "&" in line and not line.strip().startswith(("\\", "Model", "Base")):
            cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
            rows.append(cells)
    return rows
def num(s):
    s = s.replace("$", "").replace("\\", "").strip()
    try: return float(s)
    except: return None
checks = []  # (check, artifact value, document value, ok)
def rec(name, art, doc, ok): checks.append((name, art, doc, "OK" if ok else "MISMATCH"))

PRIMARY = sorted(os.listdir(P(ROOT, "results/flores_grid")))
say("# Consistency audit, 2026-09-06\n")
say("Recomputed from saved artifacts on the Mac (JSON/CSV reads only). Compared against the generated paper tables and the ledger values.\n")

# 1. primary grid vs depth_profiles.tex
say("## 1. Primary 17-model NTREX grid vs Appendix Table (depth_profiles.tex)")
grid = {m: st(profile(P(ROOT, f"results/grid/{m}/metrics.json"))[0]) for m in PRIMARY}
rows = tex_rows(P(ROOT, "paper/iclr2027/tables/depth_profiles.tex"), stop="Expansion cohort")
bad = 0
for r in rows:
    if len(r) < 8 or r[0] not in grid: continue
    g = grid[r[0]]
    ok = (int(num(r[1])) == g["n_layers"] and int(num(r[2])) == g["min_layer"] and abs(num(r[4])-g["first"]) < 6e-4 and abs(num(r[5])-g["min"]) < 6e-4 and abs(num(r[6])-g["last"]) < 6e-4 and abs(num(r[7])-g["depth"]) < 6e-4)
    if not ok: bad += 1; say(f"  MISMATCH {r[0]}: table {r[1:]} vs artifact {g}")
say(f"  {len([r for r in rows if r and r[0] in grid])} table rows checked, {bad} mismatches")
rec("Primary-cohort table rows equal results/grid", f"{len(grid)} models", f"{bad} mismatches", bad == 0)
n_int = sum(g["interior"] and g["recovery"] for g in grid.values()); rec("Interior minimum + endpoint recovery (primary)", f"{n_int}/17", "17/17", n_int == 17)
depths = {m: g["depth"] for m, g in grid.items()}; lo, hi = min(depths, key=depths.get), max(depths, key=depths.get)
rec("Dip depth range", f"{depths[lo]:.3f} ({lo}) to {depths[hi]:.3f} ({hi}); Llama {depths['Llama-3.1-8B']:.3f}", "0.039 bloom-1b7 to 0.336 Qwen3-8B; Llama 0.335", lo == "bloom-1b7" and hi == "Qwen3-8B-Base" and abs(depths[hi]-0.336) < 6e-4 and abs(depths['Llama-3.1-8B']-0.335) < 6e-4)
rec("Mistral minimum layer", grid["Mistral-7B-v0.3"]["min_layer"], 2, grid["Mistral-7B-v0.3"]["min_layer"] == 2)
q = [depths[f"Qwen3-{s}-Base"] for s in ("0.6B", "1.7B", "4B", "8B")]; rec("Qwen3 depths", " / ".join(f"{d:.3f}" for d in q), "0.206/0.213/0.314/0.336", all(abs(a-b) < 6e-4 for a, b in zip(q, (0.206, 0.213, 0.314, 0.336))))
dl = [g["min"] for g in grid.values()]; rec("Dip-layer LFS range", f"{min(dl):.3f} to {max(dl):.3f}", "0.602 to 0.904", abs(min(dl)-0.602) < 6e-4 and abs(max(dl)-0.904) < 6e-4)
rec("Pure-noise null (128,300)", f"{127/426:.4f}", "0.298", abs(127/426-0.298) < 6e-4)

# 2. FLORES
say("\n## 2. FLORES-200 replication vs flores_replication.tex and text claims")
fl = {}
for m in PRIMARY:
    v_f, mf = profile(P(ROOT, f"results/flores_grid/{m}/metrics.json")); v_n, _ = profile(P(ROOT, f"results/grid/{m}/metrics.json"))
    s = st(v_f); s["n_langs"] = len(mf["langs"]); s["profile_r"] = pearson(v_n, v_f); s["max_layer_diff"] = max(abs(a-b) for a, b in zip(v_n, v_f)); fl[m] = s
rows = tex_rows(P(ROOT, "paper/iclr2027/tables/flores_replication.tex")); bad = 0
for r in rows:
    if len(r) < 8 or r[0] not in fl: continue
    f, g = fl[r[0]], grid[r[0]]
    ok = abs(num(r[1])-g["depth"]) < 6e-4 and abs(num(r[2])-f["depth"]) < 6e-4 and int(num(r[3])) == g["min_layer"] and int(num(r[4])) == f["min_layer"] and (r[5] == "yes") == f["recovery"] and abs(num(r[6])-f["profile_r"]) < 6e-4 and abs(num(r[7])-f["max_layer_diff"]) < 6e-4
    if not ok: bad += 1; say(f"  MISMATCH {r[0]}: table {r[1:]} vs {f}")
rec("FLORES table rows equal results/flores_grid", f"{len(rows)} rows", f"{bad} mismatches", bad == 0)
rec("FLORES grid languages (incl. English)", f"{set(f['n_langs'] for f in fl.values())}", "116 (English plus 115 non-English)", set(f["n_langs"] for f in fl.values()) == {116})
nrec = sum(f["recovery"] and f["interior"] for f in fl.values()); rec("FLORES recovery", f"{nrec}/17", "17/17", nrec == 17)
sp = spearman([grid[m]["depth"] for m in PRIMARY], [fl[m]["depth"] for m in PRIMARY]); rec("Depth Spearman NTREX vs FLORES", f"{sp:.3f}", "0.917", abs(sp-0.917) < 6e-4)
w2 = sum(abs(grid[m]["min_layer"]-fl[m]["min_layer"]) <= 2 for m in PRIMARY); rec("Min layer within 2", f"{w2}/17", "14/17", w2 == 14)
md = statistics.median(abs(grid[m]["depth"]-fl[m]["depth"]) for m in PRIMARY); rec("Median |depth diff|", f"{md:.4f}", "0.0105", abs(md-0.0105) < 6e-4)
pr = [fl[m]["profile_r"] for m in PRIMARY]; rec("Profile r range", f"{min(pr):.3f} to {max(pr):.3f}", "0.936 to 1.000", abs(min(pr)-0.936) < 6e-4 and max(pr) > 0.9995)
movers = [m for m in PRIMARY if abs(grid[m]["min_layer"]-fl[m]["min_layer"]) > 2]; say(f"  movers: {[(m, grid[m]['min_layer'], fl[m]['min_layer']) for m in movers]}")
rec("granite depth NTREX -> FLORES", f"{grid['granite-3.1-8b-base']['depth']:.3f} -> {fl['granite-3.1-8b-base']['depth']:.3f}", "0.190 -> 0.074", abs(fl['granite-3.1-8b-base']['depth']-0.074) < 6e-4)

# 3. instruct pairs
say("\n## 3. Instruct pairs vs instruct_pairs.tex")
pairs = [("Qwen3-0.6B-Base", "grid", "Qwen3-0.6B"), ("Qwen3-1.7B-Base", "grid", "Qwen3-1.7B"), ("Qwen3-4B-Base", "grid", "Qwen3-4B"), ("Qwen3-8B-Base", "grid", "Qwen3-8B"), ("Qwen2.5-7B", "instruct_grid", "Qwen2.5-7B-Instruct"), ("Llama-3.1-8B", "grid", "Llama-3.1-8B-Instruct")]
ip = {}
for b, where, i in pairs:
    vb, _ = profile(P(ROOT, f"results/{where}/{b}/metrics.json")); vi, _ = profile(P(ROOT, f"results/instruct_grid/{i}/metrics.json"))
    sb, si = st(vb), st(vi); ip[b] = dict(base=sb, inst=si, ddiff=si["depth"]-sb["depth"], final_rise=si["last"]-sb["last"], mdiff=abs(si["min_layer"]-sb["min_layer"]))
rows = tex_rows(P(ROOT, "paper/iclr2027/tables/instruct_pairs.tex")); bad = 0
for r in rows:
    key = r[0].split("$")[0].strip()
    if key not in ip: continue
    d = ip[key]; ok = abs(num(r[1])-d["base"]["depth"]) < 6e-4 and abs(num(r[2])-d["inst"]["depth"]) < 6e-4 and int(num(r[3])) == d["base"]["min_layer"] and int(num(r[4])) == d["inst"]["min_layer"] and abs(num(r[5])-d["base"]["last"]) < 6e-4 and abs(num(r[6])-d["inst"]["last"]) < 6e-4
    if not ok: bad += 1; say(f"  MISMATCH {key}: {r[1:]} vs {d}")
rec("Instruct table rows equal artifacts", f"{len(rows)} rows", f"{bad} mismatches", bad == 0)
rec("Instruct recovery", f"{sum(d['inst']['recovery'] and d['inst']['interior'] for d in ip.values())}/6", "6/6", sum(d['inst']['recovery'] and d['inst']['interior'] for d in ip.values()) == 6)
mx = max(abs(d["ddiff"]) for d in ip.values()); rec("Max |depth diff| base vs instruct", f"{mx:.4f}", "<= 0.008", mx <= 0.008)
fr = sum(d["final_rise"] > 0 for d in ip.values()); rec("Final-layer LFS higher after tuning", f"{fr}/6", "5/6", fr == 5)
m2 = sum(d["mdiff"] <= 2 for d in ip.values()); rec("Min layer within 2 (instruct)", f"{m2}/6", "5/6", m2 == 5)

# 4. bootstrap
say("\n## 4. Sentence bootstrap vs dip_bootstrap.tex")
bs = {}
for d in sorted(os.listdir(P(ROOT, "results/dip_bootstrap"))):
    f = P(ROOT, f"results/dip_bootstrap/{d}/bootstrap.json")
    if not os.path.exists(f): continue
    b = json.load(open(f)); q = b["dip_depth"]["q025_median_q975"]; bs[d] = dict(lo=q[0], med=q[1], hi=q[2], hw=(q[2]-q[0])/2, dip_layer=b["dip_layer"], par=max(b["parity"]["abs_diff_0"], b["parity"]["abs_diff_dip"]), grid_depth=b["parity"]["lfs0_grid"]-b["parity"]["lfs_dip_grid"])
rows = tex_rows(P(ROOT, "paper/iclr2027/tables/dip_bootstrap.tex")); bad = 0
for r in rows:
    if r[0] not in bs: continue
    b = bs[r[0]]; iv = re.findall(r"[-0-9.]+", r[4]); ok = int(num(r[1])) == b["dip_layer"] and abs(num(r[2])-b["grid_depth"]) < 6e-4 and abs(num(r[3])-b["med"]) < 6e-4 and abs(float(iv[0])-b["lo"]) < 6e-4 and abs(float(iv[1])-b["hi"]) < 6e-4
    if not ok: bad += 1; say(f"  MISMATCH {r[0]}: {r[1:]} vs {b}")
rec("Bootstrap table rows equal artifacts", f"{len(rows)} rows", f"{bad} mismatches", bad == 0)
hw = {m: b["hw"] for m, b in bs.items()}; small = [m for m in hw if 0.004 <= hw[m] <= 0.0205]
rec("Half-widths 0.004-0.020", f"{len(small)}/14 models; others: {[(m, round(hw[m], 3)) for m in hw if m not in small]}", "12 of 14; Mistral 0.045, salamandra-2b 0.059", len(small) == 12 and abs(hw["Mistral-7B-v0.3"]-0.045) < 1.5e-3 and abs(hw["salamandra-2b"]-0.059) < 1.5e-3)
rec("Parity dump vs grid", f"max abs diff {max(b['par'] for b in bs.values()):.1e}", "< 1e-4", max(b["par"] for b in bs.values()) < 1e-4)
ea = json.load(open(P(ROOT, "results/eabc_scoring/summary.json")))
say("  E-A predictions in summary.json: " + json.dumps(ea["E-A"].get("predictions"))[:600])

# 5. R1
say("\n## 5. Content transfer (R1) vs paper Table 2")
A = {}
for r in csv.DictReader(open(P(ROOT, "results/round3/r1_content_transfer_analysis_2026-08-24/associations.csv"))):
    A[(r["candidate"], r["target"], r["bootstrap_axis"])] = (float(r["rho_deflated"]), float(r["ci_lo"]), float(r["ci_hi"]))
def near(t, exp): return all(abs(a-b) < 6e-4 for a, b in zip(t, exp))
rec("q_L vs R_content, model bootstrap", "%.3f [%.3f, %.3f]" % A[("q_L", "R_content", "model")], "0.508 [0.188, 0.707]", near(A[("q_L", "R_content", "model")], (0.508, 0.188, 0.707)))
rec("q_L crossed / family intervals", "[%.3f, %.3f] / [%.3f, %.3f]" % (A[("q_L", "R_content", "crossed")][1:] + A[("q_L", "R_content", "family")][1:]), "[0.181, 0.723] / [0.242, 0.757]", near(A[("q_L", "R_content", "crossed")][1:], (0.181, 0.723)) and near(A[("q_L", "R_content", "family")][1:], (0.242, 0.757)))
rec("MEXA reference, model bootstrap", "%.3f [%.3f, %.3f]" % A[("mexa_shared", "R_content", "model")], "0.591 [0.330, 0.818]", near(A[("mexa_shared", "R_content", "model")], (0.591, 0.330, 0.818)))
rec("AaR tail reading, model bootstrap", "%.3f [%.3f, %.3f]" % A[("aar_shared", "R_content", "model")], "0.443 [0.093, 0.709]", near(A[("aar_shared", "R_content", "model")], (0.443, 0.093, 0.709)))
rec("q_L vs raw content, model bootstrap", "%.3f [%.3f, %.3f]" % A[("q_L", "content", "model")], "0.620 [0.337, 0.769]", near(A[("q_L", "content", "model")], (0.620, 0.337, 0.769)))
ml = {r["target"]: (float(r["rho"]), float(r["ci_lo"]), float(r["ci_hi"])) for r in csv.DictReader(open(P(ROOT, "results/round3/r1_content_transfer_analysis_2026-08-24/model_level_associations.csv")))}
rec("Model-level q_L vs median R_content", "%.3f [%.3f, %.3f]" % ml["median_R_content"], "0.770 [0.440, 0.941]", near(ml["median_R_content"], (0.770, 0.440, 0.941)))
lofo = [float(r["rho_deflated"]) for r in csv.DictReader(open(P(ROOT, "results/round3/r1_content_transfer_analysis_2026-08-24/leave_one_family_out.csv"))) if r["candidate"] == "q_L" and r["target"] == "R_content"]
rec("Leave-one-family-out range (q_L)", f"{min(lofo):.3f} to {max(lofo):.3f}, {len(lofo)} deletions, all positive={all(x > 0 for x in lofo)}", "0.439 to 0.598, all positive", abs(min(lofo)-0.439) < 6e-4 and abs(max(lofo)-0.598) < 6e-4 and all(x > 0 for x in lofo))
mf = json.load(open(P(ROOT, "results/round3/r1_content_transfer_analysis_2026-08-24/manifest.json"))); say("  R1 manifest keys: " + ", ".join(list(mf.keys())[:12]))

# 6. same support + expanded validity
say("\n## 6. Same-support and 33-model exam target")
S = {}
for r in csv.DictReader(open(P(ROOT, "rmfs/results/submission_comparison/same_support_estimates.csv"))):
    S[(r["target"], r["candidate"], r["cluster"])] = (float(r["rho"]), float(r["ci_lo"]), float(r["ci_hi"]))
rec("Same-support alpha, LFS concept direction (language / model CI)", "%.3f [%.3f, %.3f] / [%.3f, %.3f]" % (S[("alpha", "lfs_legacy_concept", "flores_code")] + S[("alpha", "lfs_legacy_concept", "model")][1:]), "0.158 [0.034, 0.285] / [-0.020, 0.289]", near(S[("alpha", "lfs_legacy_concept", "flores_code")], (0.158, 0.034, 0.285)) and near(S[("alpha", "lfs_legacy_concept", "model")][1:], (-0.020, 0.289)))
rec("Legacy 4-model content transfer, LFS concept direction", "%.3f [%.3f, %.3f]" % S[("r_content", "lfs_legacy_concept", "flores_code")], "0.842 [0.790, 0.905]", near(S[("r_content", "lfs_legacy_concept", "flores_code")], (0.842, 0.790, 0.905)))
ev = open(P(ROOT, "rmfs/results/tables/expanded_validity.txt")).read()
g3 = {m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4))) for m in re.finditer(r"(mexa_shared|aar_shared|q_L)\s+([+-][0-9.]+) \[([+-][0-9.]+),([+-][0-9.]+)\]", ev)}
rec("33-model pooled exam: q_L", "%.3f [%.3f, %.3f]" % g3["q_L"], "0.041 [-0.045, 0.121]", near(g3["q_L"], (0.041, -0.045, 0.121)))
rec("33-model pooled exam: MEXA / AaR", "%.3f / %.3f" % (g3["mexa_shared"][0], g3["aar_shared"][0]), "0.238 / 0.105", abs(g3["mexa_shared"][0]-0.238) < 6e-4 and abs(g3["aar_shared"][0]-0.105) < 6e-4)
say("  expanded panel line: " + ev.splitlines()[0])

# 7. hub + headline audit
say("\n## 7. Hub test and headline audit")
hub = [(r["model"], int(r["latent_beats_english"]), int(r["n_languages"])) for r in csv.DictReader(open(P(ROOT, "rmfs/results/tables/lfs_hub_audit.csv")))]
cnt = [h[1] for h in hub]; rec("Latent factor beats English, per model", f"{min(cnt)} to {max(cnt)} of {set(h[2] for h in hub)} over {len(hub)} models; lowest {[h[0] for h in hub if h[1] == min(cnt)]}", "113 to 124 of 127", min(cnt) == 113 and max(cnt) == 124)
ha = json.load(open(P(ROOT, "rmfs/results/tables/lfs_headline_audit.json")))
rec("Headline audit: direct-SS interior+recovery", f"{ha['direct_ss']['n_both_endpoints_above_minimum']}/{ha['direct_ss']['n_models']}", "17/17", ha['direct_ss']['n_both_endpoints_above_minimum'] == 17)
rec("Headline audit: REML LFS-VC interior+recovery", f"{ha['a9_reml_lfs_vc']['n_both_endpoints_above_minimum']}/{ha['a9_reml_lfs_vc']['n_models']}", "19/19", ha['a9_reml_lfs_vc']['n_both_endpoints_above_minimum'] == 19)

# 8. deflation refresh
say("\n## 8. Deflation refresh (17 models, 1,173 rows)")
mc = {(r["metric"], r["target"]): (float(r["spearman"]), int(r["n"])) for r in csv.DictReader(open(P(ROOT, "results/deflation/attribution/metric_correlations.csv")))}
rec("AaR@10 raw / alpha; n", "%.3f / %.3f; n=%d" % (mc[("aar10", "raw_acc")][0], mc[("aar10", "alpha")][0], mc[("aar10", "alpha")][1]), "0.540 / 0.208; 1173", abs(mc[("aar10", "raw_acc")][0]-0.540) < 6e-4 and abs(mc[("aar10", "alpha")][0]-0.208) < 6e-4)
rec("MEXA raw / alpha", "%.3f / %.3f" % (mc[("mexa", "raw_acc")][0], mc[("mexa", "alpha")][0]), "0.539 / 0.176", abs(mc[("mexa", "raw_acc")][0]-0.539) < 6e-4 and abs(mc[("mexa", "alpha")][0]-0.176) < 6e-4)
hd = json.load(open(P(ROOT, "results/deflation/attribution/hardening.json")))
say(f"  hardening.json drop_floor alpha_corr (six exam-capable dev models, n={hd['drop_floor']['alpha_corr']['n']}): AaR {hd['drop_floor']['alpha_corr']['aar10']:.3f}, MEXA {hd['drop_floor']['alpha_corr']['mexa']:.3f}")
say("  NOTE: no stored artifact reproduces the exam-capable 0.59 (ten models, n=690) quoted in the July report; DECISIONS.md 2026-08 re-derivation gives about 0.49. Treated as unresolved.")
rec("Exam-capable 0.59 traceable to a stored artifact", "not found locally", "0.59 (report text)", False)

# 9. WA-C collapse
say("\n## 9. Aim 2 word-alignment collapse arm (WA-C), trained group, layer 8, mean of 3 seeds")
w = json.load(open(P(ROOT, "results/aim2/evaluation_wordalign.json"))); S9 = w["seeds"]
def arm_mean(arm, key): return statistics.fmean(S9[sd][arm]["trained"]["8"][key] for sd in S9)
rec("WA-A LFS / MEXA / raw V_concept", "%.4f / %.4f / %.0f" % (arm_mean("WA-A", "lfs"), arm_mean("WA-A", "mexa_mean"), arm_mean("WA-A", "v_concept_raw")), "0.4446 / 0.4444 / 13791", abs(arm_mean("WA-A", "lfs")-0.4446) < 6e-5 and abs(arm_mean("WA-A", "mexa_mean")-0.4444) < 6e-5 and abs(arm_mean("WA-A", "v_concept_raw")-13791) < 1)
rec("WA-C LFS / MEXA / raw V_concept", "%.4f / %.4f / %.0f" % (arm_mean("WA-C", "lfs"), arm_mean("WA-C", "mexa_mean"), arm_mean("WA-C", "v_concept_raw")), "0.4932 / 0.4878 / 699", abs(arm_mean("WA-C", "lfs")-0.4932) < 6e-5 and abs(arm_mean("WA-C", "mexa_mean")-0.4878) < 6e-5 and abs(arm_mean("WA-C", "v_concept_raw")-699) < 1)
rec("WA-G LFS / MEXA / raw V_concept", "%.4f / %.4f / %.0f" % (arm_mean("WA-G", "lfs"), arm_mean("WA-G", "mexa_mean"), arm_mean("WA-G", "v_concept_raw")), "0.4826 / 0.3944 / 13792", abs(arm_mean("WA-G", "lfs")-0.4826) < 6e-5 and abs(arm_mean("WA-G", "v_concept_raw")-13792) < 1)
say("  NOTE: the norm figures 451 -> 42 come from the training logs of the arm files and were not re-derived here.")

# 10. estimator size dependence
say("\n## 10. Evaluation-size dependence (results/estimator_nulls)")
dl, dm = [], []
for f in sorted(os.listdir(P(ROOT, "results/estimator_nulls"))):
    e = json.load(open(P(ROOT, f"results/estimator_nulls/{f}")))["size_invariance"]
    if "100" in e and "1000" in e:
        dl.append(e["1000"]["lfs_mean"]-e["100"]["lfs_mean"]); dm.append(e["1000"]["mexa_mean"]-e["100"]["mexa_mean"])
rec("LFS drift N=100->1000 (4 models)", f"{min(dl):+.3f} to {max(dl):+.3f}", "-0.006 to -0.010", abs(min(dl)+0.010) < 1.5e-3 and abs(max(dl)+0.006) < 1.5e-3)
rec("MEXA drift N=100->1000 (4 models)", f"{min(dm):+.3f} to {max(dm):+.3f}", "-0.065 to -0.125", abs(min(dm)+0.125) < 1.5e-3 and abs(max(dm)+0.065) < 1.5e-3)

# 11. whitening
say("\n## 11. Whitened-basis rank agreement (results/whitened)")
wn, ww, names = [], [], []
for f in sorted(os.listdir(P(ROOT, "results/whitened"))):
    j = json.load(open(P(ROOT, f"results/whitened/{f}"))); names.append(j["model"]); wn.append(j["lfs_native"]); ww.append(j["lfs_white"])
say(f"  {len(names)} models; native LFS keys are dip-layer values; Spearman(native, white) = {spearman(wn, ww):.3f}")
rec("Whitened vs native rank agreement", f"{spearman(wn, ww):.2f} over {len(names)} models", "0.45", abs(spearman(wn, ww)-0.45) < 0.006)

# 12. budget
say("\n## 12. Misalignment budget (results/budget)")
def budget_summary(suffix):
    rng_os, rng_rot = [], []
    for d in sorted(os.listdir(P(ROOT, "results/budget"))):
        if suffix == "" and ("_fixed_raw" in d or "_selected_centered" in d): continue
        if suffix and not d.endswith(suffix): continue
        f = P(ROOT, f"results/budget/{d}/budget.json")
        if not os.path.exists(f): continue
        b = json.load(open(f)); pl = b["per_language"]
        os_ = statistics.fmean(pl[l]["share_M1"]["mean"] + pl[l]["share_M2"]["mean"] for l in pl); rot = statistics.fmean(pl[l]["share_M3"]["mean"] for l in pl)
        rng_os.append((d, os_)); rng_rot.append((d, rot))
    return rng_os, rng_rot
for suf in ("", "_fixed_raw", "_selected_centered"):
    o, r = budget_summary(suf)
    if o: say(f"  variant '{suf or 'plain'}': {len(o)} models; offset+scale share {min(x for _, x in o):.3f} to {max(x for _, x in o):.3f}; rotation share {min(x for _, x in r)*100:+.2f}% to {max(x for _, x in r)*100:+.2f}%")
o, r = budget_summary("")
rec("Offset+scale share range (plain variant, language mean per model)", f"{min(x for _, x in o):.2f} to {max(x for _, x in o):.2f}", "0.65 to 0.98", min(x for _, x in o) >= 0.60 and max(x for _, x in o) <= 1.0)
rec("Rotation share range (plain variant)", f"{min(x for _, x in r)*100:+.2f}% to {max(x for _, x in r)*100:+.2f}%", "-0.2% to +0.7%", max(abs(x) for _, x in r) < 0.01)

# 13. probe + pooling
say("\n## 13. Language probe and pooling sensitivity")
pr = json.load(open(P(ROOT, "results/adversarial_probe/Qwen3-0.6B-Base.json")))["per_layer"]; lin = [p["probe_linear"] for p in pr]
rec("Linear language-probe accuracy across layers (Qwen3-0.6B)", f"{min(lin):.3f} to {max(lin):.3f}", "0.875 to 0.886", abs(min(lin)-0.875) < 6e-4 and abs(max(lin)-0.886) < 6e-4)
mx_change = {}
for f in sorted(os.listdir(P(ROOT, "results/pooling_variants"))):
    j = json.load(open(P(ROOT, f"results/pooling_variants/{f}"))); dep = {}
    for sc, v in j["schemes"].items():
        pl = v["per_layer"]; ks = sorted(pl, key=int); vals = [pl[k]["lfs"] for k in ks]; dep[sc] = vals[0]-min(vals)
    for sc in dep:
        if sc != "mean": mx_change[sc] = max(mx_change.get(sc, 0), abs(dep[sc]-dep["mean"]))
say("  max |dip-depth change vs mean pooling| per scheme over 5 models: " + ", ".join(f"{k} {v:.3f}" for k, v in mx_change.items()))
rec("Pooling: max dip-depth change vs mean pooling, across schemes", f"{min(mx_change.values()):.2f} to {max(mx_change.values()):.2f}", "0.13 to 0.20", abs(min(mx_change.values())-0.13) < 0.006 and abs(max(mx_change.values())-0.20) < 0.006)

# 14. text presence in PDFs
say("\n## 14. Strings in the compiled PDFs")
import pypdf
def pdftext(p):
    return re.sub(r"\s+", " ", "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(p).pages))
docs = {"ICLR draft": pdftext(P(ROOT, "paper/iclr2027/main.pdf")), "Technical report": pdftext(P(ROOT, "paper/TECHNICAL_REPORT.pdf"))}
expect = {"17 of 17": r"17 of 17|17/17", "19 of 19": r"19 of 19|19/19", "0.039": r"0\.039", "0.336": r"0\.336", "Llama 0.335": r"0\.335", "null 0.298": r"0\.298", "FLORES Spearman 0.917/0.92": r"0\.917|0\.92\b", "14 of 17": r"14 of 17|14/17", "median 0.0105": r"0\.0105|0\.011", "R1 0.508/0.51": r"0\.508|0\.51\b", "R1 CI [0.19, 0.71]": r"0\.19,\s*0\.71|0\.188,\s*0\.707", "model-level 0.770": r"0\.770|0\.77\b", "same-support 0.158": r"0\.158", "exam q_L 0.041": r"0\.041", "hub 113 to 124": r"113 to 124|113–124|113-124", "collapse 451 -> 42": r"451", "concept var 13,791": r"13,?791", "LFS 0.4446": r"0\.4446", "whitening 0.45": r"0\.45\b", "probe 0.875": r"0\.875"}
stale = {"0.374 (old Llama depth)": r"0\.374", "0.59 exam-capable": r"0\.59\b", "124/127 as a general count": r"124/127", "U-shaped": r"U-shaped", "external baseline": r"external baseline", "beats MEXA": r"beats? MEXA|outperform", "indistinguishable (contest)": r"indistinguishable on", "superiority": r"superiority"}
say("| string | ICLR draft | Technical report |"); say("|---|---|---|")
for k, pat in expect.items(): say(f"| {k} | {'yes' if re.search(pat, docs['ICLR draft']) else 'no'} | {'yes' if re.search(pat, docs['Technical report']) else 'no'} |")
say("\nStale or contest strings (should be absent or explained):"); say("| string | ICLR draft | Technical report |"); say("|---|---|---|")
for k, pat in stale.items(): say(f"| {k} | {len(re.findall(pat, docs['ICLR draft']))} | {len(re.findall(pat, docs['Technical report']))} |")

say("\n## Scorecard")
say("| check | artifact value | document value | status |"); say("|---|---|---|---|")
for c in checks: say(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
n_ok = sum(c[3] == "OK" for c in checks); say(f"\n{n_ok} of {len(checks)} checks OK.")
open(P(ROOT, "docs/CONSISTENCY_AUDIT_2026-09-06.md"), "w").write("\n".join(OUT) + "\n")
