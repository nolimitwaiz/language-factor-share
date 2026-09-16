#!/usr/bin/env python
"""Layerwise Aim-1 profiles (CPU, cluster: reads existing dumps only).

The proposal's research objective is to understand representation
"across Transformer layers"; every number so far lives at one frozen
layer per model. This pass computes, at EVERY dumped layer (5 per
model, n = 1500, 41 languages):

  - LFS-VC (REML, z-track) -> the language/concept split BY DEPTH
  - effective rank + mean norm (capacity gates)
  - shared-prep MEXA and AaR (mean over languages)
  - CKA (mean linear CKA vs English)
  - language-identity probe (accuracy, prequential MDL, selectivity)
    on a document-purged split — ALSO the input A1 has been waiting
    for (capacity-gate thresholds = 5th percentiles of the grid
    distribution at the frozen layer rule).

One model per invocation; results/runs/layer_profiles_<tag>/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

RMFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS, "src"))
PRIOR = os.path.expanduser("~/multilingual-metrics")
DUMPS = os.path.join(PRIOR, "results", "dumps")
DOC_IDS = os.path.join(PRIOR, "data", "NTREX", "DOCUMENT_IDS.tsv")

from rmfs.components.gates import effective_rank, mean_norm  # noqa: E402
from rmfs.components.variance import variance_components  # noqa: E402
from rmfs.data.ntrex import purged_concept_splits  # noqa: E402
from rmfs.metrics.baselines import (  # noqa: E402
    aar_shared,
    cka_linear,
    language_probe,
    mexa_shared,
)

PIVOT = "eng"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    args = ap.parse_args()
    tag = args.model_tag
    t0 = time.time()

    meta = json.load(open(os.path.join(DUMPS, tag, "meta.json")))
    n_sents = meta["n_sents"]
    dip = meta["dip_layer"]
    if not os.path.exists(DOC_IDS):
        raise FileNotFoundError(DOC_IDS)                 # E2 rule
    with open(DOC_IDS) as f:
        doc_ids = [line.strip().split("\t")[0] for line in f][:n_sents]
    tr, te = purged_concept_splits(doc_ids, n_splits=1, seed=13)[0]

    layers = sorted(int(f[5:8]) for f in os.listdir(os.path.join(DUMPS, tag))
                    if f.startswith("layer") and f.endswith(".npz"))
    out = {"tag": tag, "n_sents": n_sents, "layers": {}}
    for ly in layers:
        z = np.load(os.path.join(DUMPS, tag, f"layer{ly:03d}.npz"),
                    allow_pickle=False)
        X = z["X"].astype(np.float32)                    # (L, N, D)
        langs = [str(x) for x in z["langs"]]
        order = sorted(range(len(langs)), key=lambda i: langs[i])
        langs = [langs[i] for i in order]
        X = X[order]
        L, N, D = X.shape

        mu = X.reshape(-1, D).mean(0)
        sd = X.reshape(-1, D).std(0) + 1e-9
        Xz = (X - mu) / sd
        vc = variance_components(Xz)

        E = {lg: X[i] for i, lg in enumerate(langs)}
        Ez = {lg: Xz[i] for i, lg in enumerate(langs)}
        mexa = float(np.mean([mexa_shared(Ez[PIVOT], Ez[lg]).value
                              for lg in langs if lg != PIVOT]))
        aar = float(np.mean([aar_shared(Ez[PIVOT], Ez[lg]).tail_mean
                             for lg in langs if lg != PIVOT]))
        cka = float(np.mean([cka_linear(E[PIVOT], E[lg]).value
                             for lg in langs if lg != PIVOT]))
        r_eff = effective_rank(X.mean(0)).value
        norm = mean_norm(X).value
        # prequential MDL at n=1500 x 41 classes took ~110 min/layer
        # (1709803 TIMEOUT); probe runs at the DIP layer only, n=300
        # dev protocol — which is exactly the distribution A1 needs.
        probe = None
        if ly == dip:
            keep = np.arange(300)
            Ez300 = {lg: Ez[lg][keep] for lg in Ez}
            tr3 = np.asarray([i for i in tr if i < 300])
            te3 = np.asarray([i for i in te if i < 300])
            probe = language_probe(Ez300, tr3, te3, seed=13)

        out["layers"][str(ly)] = {
            "L": L, "N": N, "D": D,
            "lfs_vc": vc.lfs_vc, "sigma2_C": vc.sigma2_C_reml,
            "sigma2_L": vc.sigma2_L_reml, "degenerate": vc.degenerate,
            "effective_rank": r_eff, "mean_norm": norm,
            "mexa": mexa, "aar_tail": aar, "cka": cka,
            "probe_acc": probe.accuracy if probe else None,
            "probe_mdl_bits": probe.mdl_bits if probe else None,
            "probe_selectivity": probe.selectivity if probe else None,
        }
        pa = f"{probe.accuracy:.4f}" if probe else "-"
        print(f"[{tag} L{ly}] lfs_vc={vc.lfs_vc:.4f} r_eff={r_eff:.1f} "
              f"mexa={mexa:.4f} cka={cka:.4f} probe={pa} "
              f"({time.time() - t0:.0f}s)", flush=True)

    run_dir = os.path.join(RMFS, "results", "runs", f"layer_profiles_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS,
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": f"layer_profiles_{tag}",
                   "script": "scripts/layer_profiles.py", "git": git,
                   "host": socket.gethostname(),
                   "when": datetime.datetime.now(
                       datetime.timezone.utc).isoformat(),  # noqa: UP017
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"[done] {len(layers)} layers -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
