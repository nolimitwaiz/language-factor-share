#!/usr/bin/env python
"""Tournament core (Phase 3.3): S5/S6/S7 + Romano-Wolf + breadth.
Local CPU — reads pulled artifacts and legacy result files only.

Candidates (pre-registered family, kickoff 3.3), per (model, language)
where they exist:
  rmfs_obs   frozen 3-gauge scalar, 4 T models      (scored though failed)
  texcl      pre-registered T-excluded aggregate [q_L, q_K], 14 models
  q_T        behavioral T alone, 4 models
  q_L        1 - LFS-VC broadcast per model, 14 models
  q_K        ladder simplicity per language, 14 models
  lde        per-language deviation energy (corrected), 14 models
  legacy_lfs old ratio broadcast, 14 models
  mexa_shared / aar_shared   OUR-stack baselines at the dip layer
  mexa_leg / aar10_leg       legacy grid best-layer values (bridge)
CKA and probe-MDL were never computed on the grid: reported as
NOT COMPUTED (rule 3), queued as an optional cluster pass.

Targets: alpha (Belebele logit-excess residual after deflation controls,
via the verified legacy assemble + new-stack fit) and R_content
(3.1.3 held-out content transfer, 4-model set) for S5.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import socket
import subprocess
import sys

import pandas as pd

RMFS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIOR = os.environ.get(
    "MULTILINGUAL_METRICS_ROOT",
    os.path.expanduser("~/Desktop/multilingual-metrics"),
)
sys.path.insert(0, os.path.join(RMFS_DIR, "src"))

from rmfs.metrics.rmfs import softmin  # noqa: E402
from rmfs.validity.deflation import fit_attribution  # noqa: E402
from rmfs.validity.tournament import (  # noqa: E402
    effective_breadth,
    orthogonalized_increment,
    residual_spearman,
    romano_wolf,
)

RUNS = os.path.join(RMFS_DIR, "results", "runs")
TABLES = os.path.join(RMFS_DIR, "results", "tables")
T_MODELS = ["Qwen3-0.6B-Base", "Qwen3-4B-Base", "OLMo-2-0425-1B",
            "bloom-1b7"]
N_BOOT, SEED = 2000, 13


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_frame() -> pd.DataFrame:
    leg = _load_module("legacy_attribution",
                       os.path.join(RMFS_DIR, "legacy", "attribution.py"))
    leg.ROOT = PRIOR
    # Full CLSP mirrors keep the external control tables in rmfs/data/external,
    # while the laptop campaign historically placed them under the original
    # project's results/deflation directory.  Require an explicit override on
    # CLSP rather than copying or silently omitting a control.
    leg.DEF = os.environ.get(
        "RMFS_DEFLATION_ROOT",
        os.path.join(PRIOR, "results", "deflation"),
    )
    leg.OUT = os.path.join(RMFS_DIR, "results", "deflation_repro")
    os.makedirs(leg.OUT, exist_ok=True)
    df = leg.assemble().rename(columns={"mexa": "mexa_leg",
                                        "aar10": "aar10_leg"})
    fert = _load_module("legacy_fertility",
                        os.path.join(RMFS_DIR, "legacy", "fertility.py"))
    n2f = {}
    for fc in df.flores_code.unique():
        n2f.setdefault(fert.flores_to_ntrex(fc), fc)

    # ---- new-stack candidates from grid_components runs ------------------
    rows = []
    for d in sorted(os.listdir(RUNS)):
        if not d.startswith("grid_components_"):
            continue
        tag = d.removeprefix("grid_components_")
        r = json.load(open(os.path.join(RUNS, d, "result.json")))
        if not r.get("K"):
            continue
        q_l = 1.0 - r["lfs_vc"]
        for ntrex, kv in r["K"].items():
            fc = n2f.get(ntrex)
            if fc is None:
                continue
            rows.append(dict(
                model=tag, flores_code=fc, q_K=kv["q_K"], q_L=q_l,
                texcl=softmin([q_l, kv["q_K"]]),
                lde=r["lde_corrected"].get(ntrex),
                legacy_lfs=r["legacy_lfs"],
                mexa_shared=r["mexa"].get(ntrex),
                aar_shared=(r["aar"].get(ntrex) or {}).get("tail")))
    cand = pd.DataFrame(rows)

    # ---- RMFS-obs + q_T from the assembly table --------------------------
    rv = pd.read_parquet(os.path.join(TABLES, "rmfs_v1.parquet"))
    rv = rv[rv["mode"] == "observational"]
    piv = rv.pivot_table(index=["model", "language"], columns="component",
                         values="value", aggfunc="mean").reset_index()
    piv["flores_code"] = [n2f.get(lang) for lang in piv.language]
    piv = piv.dropna(subset=["flores_code"])
    piv = piv.rename(columns={"RMFS": "rmfs_obs", "q_T": "q_T"})
    cand = cand.merge(piv[["model", "flores_code", "rmfs_obs", "q_T"]],
                      on=["model", "flores_code"], how="left")

    # ---- R_content target (S5, 3.1.3 set) --------------------------------
    rc_rows = []
    for tag in ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base",
                "OLMo-2-0425-1B"]:
        p = os.path.join(PRIOR, "results", "transfer313", f"{tag}_v2.json")
        if not os.path.exists(p):
            raise FileNotFoundError(p)          # E2: hard fail, no skip
        res = json.load(open(p))["results"]
        for ntrex, v in res.items():
            if ntrex == "eng":
                continue
            fc = n2f.get(ntrex)
            if fc:
                rc_rows.append(dict(model=tag, flores_code=fc,
                                    r_content=v["R_content"]))
    rc = pd.DataFrame(rc_rows)

    out = (df.merge(cand, on=["model", "flores_code"], how="left")
             .merge(rc, on=["model", "flores_code"], how="left"))
    return out


def fmt(name: str, r) -> str:
    return (f"{name:12s} rho={r.rho:+.3f}  CI[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]"
            f"  p={r.p_boot:.4f}  n={r.n_rows}/{r.n_langs}L")


def main() -> None:
    df = load_frame()
    fit = fit_attribution(df)
    d = fit.alphas
    print(f"[frame] {len(d)} rows, {d.model.nunique()} models, "
          f"{d.flores_code.nunique()} languages | tokens gate "
          f"{'PASS' if fit.tokens_gate_passed else 'FAIL'}", flush=True)

    fam = ["rmfs_obs", "texcl", "q_T", "q_L", "q_K", "lde", "legacy_lfs",
           "mexa_shared", "aar_shared", "mexa_leg", "aar10_leg"]
    fam = [c for c in fam if c in d.columns and d[c].notna().sum() >= 40]

    results = {}
    print("\n=== deflated validity vs alpha (raw-residual, cluster boot) ===")
    for c in fam:
        r = residual_spearman(d, c, "alpha", N_BOOT, SEED)
        results[c] = r
        print("  " + fmt(c, r), flush=True)

    print("\n=== Romano-Wolf stepdown (family FWER) ===")
    rw = romano_wolf(d, fam, "alpha", N_BOOT, SEED)
    for _, row in rw.iterrows():
        print(f"  {row.candidate:12s} rho={row.rho_deflated:+.3f} "
              f"p_RW={row.p_rw:.4f}")

    print("\n=== S5: residual Spearman vs R_content (3.1.3 set) ===")
    s5 = {}
    for c in ["rmfs_obs", "texcl", "q_T", "q_L", "q_K", "mexa_shared"]:
        if c not in d.columns or d.dropna(
                subset=[c, "r_content"]).empty:
            continue
        r = residual_spearman(d, c, "r_content", N_BOOT, SEED)
        s5[c] = r
        print("  " + fmt(c, r), flush=True)
    s5_head = s5.get("rmfs_obs")
    s5_free = s5.get("texcl")
    if s5_head:
        ok = s5_head.rho >= 0.30 and s5_head.p_boot < 0.05
        print(f"  S5 headline (rmfs_obs >= 0.30, p<0.05): "
              f"{'PASS' if ok else 'FAIL'}")
    if s5_free:
        ok = s5_free.rho >= 0.30 and s5_free.p_boot < 0.05
        print(f"  S5 circularity-free line (texcl):        "
              f"{'PASS' if ok else 'FAIL'}")

    print("\n=== S6: rmfs_obs vs alpha, lower CI > 0? ===")
    s6 = results.get("rmfs_obs")
    if s6:
        print(f"  {fmt('rmfs_obs', s6)}  ->  "
              f"{'PASS' if s6.ci_lo > 0 else 'FAIL'}")

    print("\n=== S7: orthogonalized increment over shared-prep MEXA ===")
    s7 = {}
    for c in ["rmfs_obs", "texcl", "q_K", "lde"]:
        if c not in d.columns:
            continue
        r = orthogonalized_increment(d, c, "mexa_shared", "alpha",
                                     N_BOOT, SEED)
        s7[c] = r
        print("  " + fmt(c, r) + ("  -> increment PASS" if r.ci_lo > 0
                                  else "  -> no increment"), flush=True)

    print("\n=== Grinold effective breadth (language panel) ===")
    for c in ["texcl", "mexa_shared"]:
        if c in d.columns:
            print(f"  {c:12s} N_eff = "
                  f"{effective_breadth(d, c, 'alpha'):.1f}")

    print("\n  NOT COMPUTED: cka, probe_mdl (no grid pass exists; queued)")

    # ---- persist ---------------------------------------------------------
    os.makedirs(TABLES, exist_ok=True)
    tab = pd.DataFrame(
        [{"candidate": c, "target": "alpha", "rho": r.rho,
          "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "p_boot": r.p_boot,
          "n_rows": r.n_rows, "n_langs": r.n_langs} for c, r in
         results.items()] +
        [{"candidate": c, "target": "r_content", "rho": r.rho,
          "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "p_boot": r.p_boot,
          "n_rows": r.n_rows, "n_langs": r.n_langs} for c, r in
         s5.items()] +
        [{"candidate": f"{c}|orth_mexa", "target": "alpha", "rho": r.rho,
          "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "p_boot": r.p_boot,
          "n_rows": r.n_rows, "n_langs": r.n_langs} for c, r in
         s7.items()])
    tab.to_csv(os.path.join(TABLES, "tournament_core.csv"), index=False,
               float_format="%.4f")
    rw.to_csv(os.path.join(TABLES, "tournament_rw.csv"), index=False,
              float_format="%.4f")

    run_dir = os.path.join(RUNS, "tournament_core")
    os.makedirs(run_dir, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS_DIR,
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": "tournament_core", "git": git,
                   "host": socket.gethostname(),
                   # timezone.utc (not UTC alias): local runner is py3.9
                   "when": datetime.datetime.now(
                       datetime.timezone.utc).isoformat(),  # noqa: UP017
                   "n_boot": N_BOOT, "seed": SEED,
                   "family": fam,
                   "not_computed": ["cka", "probe_mdl"],
                   "tokens_gate": fit.tokens_gate_passed}, f, indent=1)
    print("\n[done] -> results/tables/tournament_core.csv, "
          "tournament_rw.csv", flush=True)


if __name__ == "__main__":
    main()
