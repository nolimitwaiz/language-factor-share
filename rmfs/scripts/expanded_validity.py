#!/usr/bin/env python
"""A9 scoring: build the expanded panel (33 models) and score G2, G3, G4.

Local CPU. Reads pulled artifacts only. Writes
results/tables/expanded_panel.csv and expanded_validity.txt.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

RMFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS, "src"))
sys.path.insert(0, os.path.join(RMFS, "scripts"))

from rmfs.validity.deflation import fit_attribution  # noqa: E402
from rmfs.validity.tournament import _design, _residualize  # noqa: E402

RUNS = os.path.join(RMFS, "results", "runs")
TAB = os.path.join(RMFS, "results", "tables")
N_BOOT = 2000


def _fert():
    spec = importlib.util.spec_from_file_location(
        "fert", os.path.join(RMFS, "legacy", "fertility.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["fert"] = m
    spec.loader.exec_module(m)
    return m


def boot_spearman(d, cand, target, cluster, n_boot=N_BOOT, seed=13):
    d = d.dropna(subset=[cand, target, "log_tokens", "fertility",
                         "macro_family_g", "script_g"])
    # Canonical label order makes seeded cluster bootstrap intervals portable
    # across dataframe/parquet environments.
    units = np.sort(d[cluster].dropna().unique())
    by = {u: d[d[cluster] == u] for u in units}

    def stat(s):
        X = _design(s)
        return stats.spearmanr(
            _residualize(s[cand].astype(float).values, X),
            _residualize(s[target].astype(float).values, X)).statistic

    pt = stat(d)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        try:
            v.append(stat(pd.concat([by[u] for u in pick])))
        except Exception:
            pass
    v = np.array([x for x in v if np.isfinite(x)])
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(pt), float(lo), float(hi), len(d), d[cluster].nunique()


def main() -> None:
    from tournament import load_frame

    fm = _fert()
    base = load_frame()
    n2f = {}
    for fc in base.flores_code.unique():
        n2f.setdefault(fm.flores_to_ntrex(fc), fc)

    rows = []
    for d in sorted(os.listdir(RUNS)):
        if not d.startswith("new_readings_"):
            continue
        r = json.load(open(os.path.join(RUNS, d, "result.json")))
        for ntrex, mx in r["mexa"].items():
            fc = n2f.get(ntrex)
            if fc:
                rows.append(dict(model=r["tag"], flores_code=fc,
                                 mexa_shared=mx,
                                 aar_shared=r["aar"].get(ntrex),
                                 q_L=1 - r["lfs_vc"]))
    newr = pd.DataFrame(rows)

    incl = json.load(open(os.path.join(TAB, "include_native_acc.json")))
    beln = json.load(open(os.path.join(TAB, "belebele_new_acc.json")))
    tgt = []
    for m, per in incl.items():
        for lg, a in per.items():
            fc = n2f.get(lg)
            if fc:
                tgt.append(dict(model=m, flores_code=fc, acc_native=a))
    for m, per in beln.items():
        for lg, a in per.items():
            tgt.append(dict(model=m, flores_code=lg, acc_trans=a))
    tg = (pd.DataFrame(tgt).groupby(["model", "flores_code"])
          .first().reset_index())

    cov = base.drop_duplicates("flores_code")[
        ["flores_code", "log_tokens", "macro_family_g", "script_g"]]
    fmean = base.groupby("flores_code").fertility.mean()
    new = newr.merge(cov, on="flores_code", how="left").merge(
        tg, on=["model", "flores_code"], how="inner")
    new["fertility"] = new.flores_code.map(fmean)

    old = base.rename(columns={"acc": "acc_trans"}).copy()
    old["acc_native"] = np.nan
    for m, per in incl.items():
        for lg, a in per.items():
            fc = n2f.get(lg)
            if fc:
                old.loc[(old.model == m) & (old.flores_code == fc),
                        "acc_native"] = a
    keep = ["model", "flores_code", "mexa_shared", "aar_shared", "q_L",
            "r_content", "log_tokens", "macro_family_g", "script_g",
            "fertility", "acc_native", "acc_trans"]
    old = old[[c for c in keep if c in old.columns]]
    new = new[[c for c in keep if c in new.columns]]
    allm = pd.concat([old, new], ignore_index=True)
    allm["acc"] = allm[["acc_native", "acc_trans"]].mean(axis=1)
    allm = allm.dropna(subset=["acc", "log_tokens", "fertility"])
    allm.to_csv(os.path.join(TAB, "expanded_panel.csv"), index=False)

    lines = [f"EXPANDED PANEL: {len(allm)} rows, "
             f"{allm.model.nunique()} models, "
             f"{allm.flores_code.nunique()} languages"]
    fit = fit_attribution(allm)
    dd = fit.alphas
    lines.append(f"tokens gate "
                 f"{'PASS' if fit.tokens_gate_passed else 'FAIL'}")

    lines.append("")
    lines.append("=== G3: deflated validity vs pooled exam target ===")
    g3 = None
    for c in ["mexa_shared", "aar_shared", "q_L"]:
        if c not in dd.columns or dd[c].notna().sum() < 40:
            continue
        pt, lo, hi, n, k = boot_spearman(dd, c, "alpha", "flores_code")
        if c == "mexa_shared":
            g3 = lo > 0
        lines.append(f"  {'*' if lo > 0 else ' '} {c:12s} {pt:+.3f} "
                     f"[{lo:+.3f},{hi:+.3f}]  n={n}/{k}L")
    lines.append(f"G3 -> {'PASS' if g3 else 'FAIL'}")

    lines.append("")
    lines.append("=== G2: content transfer, models as the unit ===")
    rc = allm.dropna(subset=["r_content"]) if "r_content" in allm else \
        pd.DataFrame()
    if len(rc) > 40:
        pt, lo, hi, n, k = boot_spearman(rc, "q_L", "r_content", "model")
        width = hi - lo
        old_width = 0.837 - 0.124
        lines.append(f"  q_L vs content: {pt:+.3f} [{lo:+.3f},{hi:+.3f}] "
                     f"width {width:.3f}  ({k} models)")
        lines.append(f"  previous width (14 models): {old_width:.3f}")
        shrink = 1 - width / old_width
        lines.append(f"  narrowing: {shrink*100:.1f}%  (bar 25%) -> "
                     f"{'PASS' if shrink >= 0.25 else 'FAIL'}")
    else:
        lines.append("  content transfer target not available on the "
                     "expanded panel (4 model subset only) -> UNSCORED")

    txt = "\n".join(lines)
    open(os.path.join(TAB, "expanded_validity.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
