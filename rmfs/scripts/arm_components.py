#!/usr/bin/env python
"""Arm-level REML components (Phase 3 input). GPU, cluster-only, one model
or arm checkpoint per invocation.

T extraction (t_extract.py) holds pooled embeddings in memory and discards
them; assembling intervention-mode RMFS for the 75 arm checkpoints needs
two components T never produced:

  q_L   = 1 - LFS-VC  : REML concept share on the arm's own embeddings
  C_pres              : sigma2_C_reml(arm) / sigma2_C_reml(frozen base),
                        SAME protocol both sides (n=300, dip layer, fp32
                        pooling, identical language set)

Each task saves its own raw sigma2 components; C_pres is a ratio taken at
assembly time against the base task's output — no cross-task dependency
inside the array. LDE is included from the same X at zero extra GPU cost
(nulled for validity, kept for the record).

Usage:
  python scripts/arm_components.py --model_tag Qwen3-0.6B-Base   # reference
  python scripts/arm_components.py --arm_ckpt ckpt_armWA-C_seed0
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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

from rmfs.components.variance import lde, variance_components  # noqa: E402
from t_extract import CTX_LANGS, N_SENTS, load_registry, pooled_embed  # noqa: E402

BASE_TAG = "Qwen3-0.6B-Base"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag")
    ap.add_argument("--arm_ckpt")
    args = ap.parse_args()
    if bool(args.model_tag) == bool(args.arm_ckpt):
        raise SystemExit("exactly one of --model_tag / --arm_ckpt")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reg = load_registry()
    if args.arm_ckpt:
        run_name = args.arm_ckpt
        model_path = os.path.join(CKPTS, args.arm_ckpt)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"checkpoint dir missing: {model_path}")
        dip = reg[BASE_TAG]["dip_layer"]
    else:
        run_name = args.model_tag
        m = reg[args.model_tag]
        model_path = m["hf_id"]
        dip = m["dip_layer"]

    t0 = time.time()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16)
             .to(dev).eval())
    print(f"[{run_name}] dip L{dip}, dtype bf16", flush=True)

    from rmfs.data.ntrex import load_ntrex

    langs = list(CTX_LANGS)
    data = load_ntrex(NTREX, DOC_IDS, langs, N_SENTS)

    all_langs = ["eng", *langs]
    E = []
    for lg in all_langs:
        E.append(pooled_embed(model, tok, data.sentences[lg], dev, dip))
        print(f"  [embed] {lg}", flush=True)
    X = np.stack(E, 0)                      # (L, N, D), fp32-pooled

    vc = variance_components(X)
    ld = lde(X)
    print(f"[reml] lfs_vc={vc.lfs_vc:.4f}  sigma2_C={vc.sigma2_C_reml:.4f}  "
          f"sigma2_L={vc.sigma2_L_reml:.4f}  degenerate={vc.degenerate}",
          flush=True)

    run_dir = os.path.join(RMFS, "results", "runs", f"arm_components_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump({
            "run": run_name,
            "protocol": {"n_sents": N_SENTS, "dip": dip,
                         "langs": all_langs, "pooling": "fp32-mean"},
            "variance_components": vc.as_dict(),
            "lde_corrected": {lg: ld.corrected[lg] for lg in ld.corrected},
        }, f, indent=1)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS,
                         capture_output=True, text=True).stdout.strip()
    doc_hash = hashlib.sha256(open(DOC_IDS, "rb").read()).hexdigest()[:16]
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({
            "run_id": f"arm_components_{run_name}",
            "script": "scripts/arm_components.py",
            "git": git, "host": socket.gethostname(),
            "when": datetime.datetime.now(datetime.UTC).isoformat(),
            "precision": "bf16 forward / fp32 pooling / fp64 REML",
            "wall_s": round(time.time() - t0, 1),
            "inputs": {"model_path": model_path, "doc_ids_sha16": doc_hash},
        }, f, indent=1)
    print(f"[done] -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
