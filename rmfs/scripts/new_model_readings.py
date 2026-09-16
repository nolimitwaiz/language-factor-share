#!/usr/bin/env python
"""A9: readings for a model that is not yet in the reference grid.

Self contained. Sweeps every layer on the standard 300 sentences and 41
languages, selects the measurement layer by the rule already used for
the grid (the layer where the language share is lowest), then computes
the readings there. One model per invocation.

Writes results/runs/new_readings_<tag>/result.json with the full depth
profile as well as the selected layer, so the sweep is auditable.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

RMFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS, "src"))
sys.path.insert(0, os.path.join(RMFS, "scripts"))
PRIOR = os.path.expanduser("~/multilingual-metrics")
NTREX = os.path.join(PRIOR, "data", "NTREX", "NTREX-128")
DOC_IDS = os.path.join(PRIOR, "data", "NTREX", "DOCUMENT_IDS.tsv")

from rmfs.components.gates import effective_rank, mean_norm  # noqa: E402
from rmfs.components.variance import variance_components  # noqa: E402
from rmfs.metrics.baselines import aar_shared, mexa_shared  # noqa: E402
from t_extract import CTX_LANGS, N_SENTS  # noqa: E402

PIVOT = "eng"


def sweep_embed(model, tok, sentences, dev, batch=8, max_len=128):
    """Pooled embedding at EVERY layer in one pass: list over layers of
    (N, D) float32, fp32 pooling as everywhere else."""
    import torch

    out = None
    for i in range(0, len(sentences), batch):
        enc = tok(sentences[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = [((h.float() * m).sum(1) / m.sum(1)).cpu().numpy()
                  for h in hs]
        if out is None:
            out = [[] for _ in pooled]
        for k, p in enumerate(pooled):
            out[k].append(p)
        del hs, pooled
    return [np.concatenate(chunks, 0) for chunks in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    t0 = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (AutoModelForCausalLM.from_pretrained(
        args.hf_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .to("cuda").eval())

    from rmfs.data.ntrex import load_ntrex
    langs = list(CTX_LANGS)
    data = load_ntrex(NTREX, DOC_IDS, langs, N_SENTS)
    all_langs = [PIVOT, *langs]

    per_layer: dict = {}
    for lg in all_langs:
        layers = sweep_embed(model, tok, data.sentences[lg], "cuda")
        for k, E in enumerate(layers):
            per_layer.setdefault(k, {})[lg] = E.astype(np.float32)
        del layers
        gc.collect()
        print(f"  [embed] {lg} ({time.time() - t0:.0f}s)", flush=True)

    n_layers = len(per_layer)
    profile = {}
    for k in range(n_layers):
        X = np.stack([per_layer[k][lg] for lg in all_langs], 0)
        D = X.shape[-1]
        mu = X.reshape(-1, D).mean(0)
        sd = X.reshape(-1, D).std(0) + 1e-9
        vc = variance_components((X - mu) / sd)
        profile[k] = float(vc.lfs_vc)
    dip = min(profile, key=profile.get)
    print(f"[{args.tag}] {n_layers} layers, dip at {dip} "
          f"(language share {profile[dip]:.4f})", flush=True)

    X = np.stack([per_layer[dip][lg] for lg in all_langs], 0)
    D = X.shape[-1]
    mu = X.reshape(-1, D).mean(0)
    sd = X.reshape(-1, D).std(0) + 1e-9
    Xz = (X - mu) / sd
    vc = variance_components(Xz)
    Ez = {lg: Xz[i] for i, lg in enumerate(all_langs)}
    mexa = {lg: mexa_shared(Ez[PIVOT], Ez[lg]).value
            for lg in langs}
    aar = {lg: aar_shared(Ez[PIVOT], Ez[lg]).tail_mean for lg in langs}

    res = {
        "tag": args.tag, "hf_id": args.hf_id, "n_layers": n_layers,
        "dip_layer": dip, "depth_profile": profile,
        "lfs_vc": vc.lfs_vc, "sigma2_L": vc.sigma2_L_reml,
        "sigma2_C": vc.sigma2_C_reml,
        "effective_rank": effective_rank(X.mean(0)).value,
        "mean_norm": mean_norm(X).value,
        "mexa": mexa, "aar": aar, "n_sents": N_SENTS,
    }
    run_dir = os.path.join(RMFS, "results", "runs",
                           f"new_readings_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS,
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": f"new_readings_{args.tag}",
                   "script": "scripts/new_model_readings.py", "git": git,
                   "host": socket.gethostname(),
                   "when": datetime.datetime.now(
                       datetime.timezone.utc).isoformat(),  # noqa: UP017
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"[done] -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
