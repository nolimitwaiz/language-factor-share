#!/usr/bin/env python
"""Phase 1.7 finale: deflated-validity brackets for LDE and q_K, with
manifests. Inputs: the grid_quick_* and grid_components_* results plus the
prior repo's committed covariate/Belebele tables (assembled through the
read-only legacy loaders, roots re-pointed, exactly as the acceptance
reproduction did).

Local CPU only. Emits results/runs/component_brackets/{result.json,
manifest.json} and appends nothing to any ledger by itself — the ledger
entry is authored from the printed table.
"""

from __future__ import annotations

import datetime
import glob
import importlib.util
import json
import os
import socket
import subprocess
import sys

import pandas as pd

RMFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIOR = os.path.expanduser("~/Desktop/multilingual-metrics")
sys.path.insert(0, os.path.join(RMFS, "src"))

from rmfs.validity.deflation import candidate_bracket, fit_attribution  # noqa: E402


def _load(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    leg = _load("leg_attr", os.path.join(RMFS, "legacy", "attribution.py"))
    leg.ROOT = PRIOR
    leg.DEF = os.path.join(PRIOR, "results", "deflation")
    leg.OUT = os.path.join(RMFS, "results", "deflation_repro")
    df = leg.assemble()

    fert = _load("leg_fert", os.path.join(RMFS, "legacy", "fertility.py"))
    meta = pd.read_csv(os.path.join(leg.DEF, "language_meta.csv"))
    n2f: dict = {}
    for fc in meta.flores_code:
        n2f.setdefault(fert.flores_to_ntrex(fc), fc)

    def merge_component(pattern: str, extract) -> pd.DataFrame:
        rows = []
        for f in glob.glob(os.path.join(RMFS, "results", "runs", pattern)):
            d = json.load(open(f))
            for ntrex, val in extract(d).items():
                fc = n2f.get(ntrex)
                if fc is not None:
                    rows.append({"model": d["model"], "flores_code": fc,
                                 **val})
        return pd.DataFrame(rows)

    lde_df = merge_component(
        "grid_quick_*/result.json",
        lambda d: {k: {"lde": v} for k, v in d["lde_corrected"].items()})
    qk_df = merge_component(
        "grid_components_*/result.json",
        lambda d: {k: {"q_K": v["q_K"], "K": v["K"]}
                   for k, v in d["K"].items()})
    df = (df.merge(lde_df, on=["model", "flores_code"], how="left")
            .merge(qk_df, on=["model", "flores_code"], how="left"))
    print(f"[merge] lde rows {df.lde.notna().sum()}, "
          f"q_K rows {df.q_K.notna().sum()}")

    fit = fit_attribution(df)
    print(f"[fit] tokens gate "
          f"{'PASS' if fit.tokens_gate_passed else 'FAIL'} "
          f"(beta {fit.tokens_beta:+.4f}, p {fit.tokens_beta_p:.2e}), "
          f"R2 {fit.r2:.3f}, rows {fit.n_rows}")

    out = {"fit": {"tokens_beta": fit.tokens_beta,
                   "tokens_beta_p": fit.tokens_beta_p,
                   "gate": fit.tokens_gate_passed, "r2": fit.r2,
                   "n_rows": fit.n_rows},
           "brackets": {}}
    print(f"\n{'cand':>6} {'target':>8} {'scheme':>16} {'rho':>8} "
          f"{'95% CI':>18} {'n':>5}")
    for cand in ("lde", "q_K"):
        rows = candidate_bracket(fit, cand, n_boot=2000, seed=13)
        out["brackets"][cand] = [r.__dict__ for r in rows]
        for r in rows:
            print(f"{cand:>6} {r.target:>8} {r.scheme:>16} "
                  f"{r.spearman:>+8.4f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
                  f" {r.n:>5}")

    run_id = "component_brackets"
    out_dir = os.path.join(RMFS, "results", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    git = subprocess.run(["git", "-C", RMFS, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    manifest = {"run_id": run_id, "git": git,
                "host": socket.gethostname(),
                "when": datetime.datetime.now().isoformat(timespec="seconds"),
                "inputs": {"quick": sorted(glob.glob(os.path.join(
                    RMFS, "results", "runs", "grid_quick_*"))),
                    "k": sorted(glob.glob(os.path.join(
                        RMFS, "results", "runs", "grid_components_*"))),
                    "covariates": leg.DEF},
                "config": {"n_boot": 2000, "seed": 13,
                           "candidates": ["lde", "q_K"]}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
