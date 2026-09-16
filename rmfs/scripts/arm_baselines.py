#!/usr/bin/env python
"""Per-arm shared-prep baselines (A5). GPU, cluster-only, one checkpoint
per invocation: embed n=300 at dip L8, z-score per dimension over all
cells (grid protocol), then mexa_shared and aar_shared per language.

Run IDs arm_baselines_* (rule 8: never overwrite arm_components_*).
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
sys.path.insert(0, os.path.join(RMFS, "scripts"))
PRIOR = os.path.expanduser("~/multilingual-metrics")
NTREX = os.path.join(PRIOR, "data", "NTREX", "NTREX-128")
DOC_IDS = os.path.join(PRIOR, "data", "NTREX", "DOCUMENT_IDS.tsv")
CKPTS = os.path.join(PRIOR, "results", "aim2")

from rmfs.metrics.baselines import aar_shared, mexa_shared  # noqa: E402
from t_extract import CTX_LANGS, N_SENTS, load_registry, pooled_embed  # noqa: E402

BASE_TAG = "Qwen3-0.6B-Base"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_ckpt", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    reg = load_registry()
    dip = reg[BASE_TAG]["dip_layer"]
    model_path = os.path.join(CKPTS, args.arm_ckpt)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"checkpoint dir missing: {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16)
             .to("cuda").eval())

    from rmfs.data.ntrex import load_ntrex
    langs = list(CTX_LANGS)
    data = load_ntrex(NTREX, DOC_IDS, langs, N_SENTS)
    all_langs = ["eng", *langs]
    E = {}
    for lg in all_langs:
        E[lg] = pooled_embed(model, tok, data.sentences[lg], "cuda", dip)
        print(f"  [embed] {lg}", flush=True)

    X = np.stack([E[lg] for lg in all_langs], 0).astype(np.float32)
    D = X.shape[-1]
    mu = X.reshape(-1, D).mean(0)
    sd = X.reshape(-1, D).std(0) + 1e-9
    Ez = {lg: (E[lg] - mu) / sd for lg in all_langs}

    out = {}
    for lg in langs:
        m = mexa_shared(Ez["eng"], Ez[lg])
        a = aar_shared(Ez["eng"], Ez[lg])
        out[lg] = {"mexa": m.value, "aar_tail": a.tail_mean,
                   "aar_mean": a.mean}
    print(f"[done metrics] mean mexa "
          f"{np.mean([v['mexa'] for v in out.values()]):.4f}", flush=True)

    run_dir = os.path.join(RMFS, "results", "runs",
                           f"arm_baselines_{args.arm_ckpt}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump({"run": args.arm_ckpt, "dip": dip,
                   "protocol": "n300-dip8-zscored-sharedprep",
                   "baselines": out}, f, indent=1)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS,
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": f"arm_baselines_{args.arm_ckpt}",
                   "script": "scripts/arm_baselines.py", "git": git,
                   "host": socket.gethostname(),
                   "when": datetime.datetime.now(
                       datetime.timezone.utc).isoformat(),  # noqa: UP017
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"[done] -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
