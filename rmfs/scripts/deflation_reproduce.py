#!/usr/bin/env python
"""Phase 1.7 acceptance: reproduce the published MEXA/AaR deflation numbers
on the NEW stack, within ±0.02.

Data assembly runs through the LEGACY loaders (imported read-only, roots
re-pointed at the prior repo) so the input table is constructed exactly as
the published run constructed it; the regression and correlations then run
through `rmfs.validity.deflation`. Any mismatch is therefore attributable
to the ported fit, not to input drift.

Published reference: the prior repo's committed
results/deflation/attribution/metric_correlations.csv.

Local CPU only; no cluster, no model.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

RMFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIOR = os.path.expanduser("~/Desktop/multilingual-metrics")
sys.path.insert(0, os.path.join(RMFS, "src"))

from rmfs.validity.deflation import candidate_bracket, fit_attribution  # noqa: E402

TOL = 0.02


def load_legacy_attribution():
    spec = importlib.util.spec_from_file_location(
        "legacy_attribution", os.path.join(RMFS, "legacy", "attribution.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["legacy_attribution"] = mod
    spec.loader.exec_module(mod)
    # re-point every root at the prior repo (the module computed them from
    # its own location inside rmfs/)
    mod.ROOT = PRIOR
    mod.DEF = os.path.join(PRIOR, "results", "deflation")
    mod.OUT = os.path.join(RMFS, "results", "deflation_repro")
    os.makedirs(mod.OUT, exist_ok=True)
    return mod


def main() -> None:
    leg = load_legacy_attribution()
    df = leg.assemble()
    print(f"[assemble] {len(df)} rows via legacy loaders "
          f"(prior repo inputs)")

    fit = fit_attribution(df)
    print(f"[fit] rows {fit.n_rows} (dropped {fit.n_dropped}) | "
          f"tokens beta {fit.tokens_beta:+.4f} p={fit.tokens_beta_p:.2e} "
          f"gate={'PASS' if fit.tokens_gate_passed else 'FAIL'} | "
          f"R2 {fit.r2:.3f}")

    pub = pd.read_csv(os.path.join(
        PRIOR, "results", "deflation", "attribution",
        "metric_correlations.csv"))
    pub_map = {(r.metric, r.target): float(r.spearman)
               for r in pub.itertuples()}

    print(f"\n{'candidate':>10} {'target':>8} {'published':>10} "
          f"{'new stack':>10} {'delta':>8} {'CI (cluster boot)':>22} verdict")
    all_ok = True
    for cand in ("mexa", "aar10"):
        rows = candidate_bracket(fit, cand, n_boot=2000, seed=13)
        for r in rows:
            if r.scheme != "raw_residual":
                continue
            key = (cand, "raw_acc" if r.target == "raw_acc" else "alpha")
            ref = pub_map.get(key)
            if ref is None:
                continue
            delta = r.spearman - ref
            ok = abs(delta) <= TOL
            all_ok &= ok
            print(f"{cand:>10} {r.target:>8} {ref:>10.4f} "
                  f"{r.spearman:>10.4f} {delta:>+8.4f} "
                  f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
                  f"{'':>3}{'PASS' if ok else 'FAIL'}")
    print(f"\nACCEPTANCE (±{TOL}): "
          f"{'ALL PASS' if all_ok else 'MISMATCH — ledger before any fix'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
