#!/usr/bin/env python
"""A7: are the readings driven by translationese? GPU, cluster only.

WMT19 test sets are partitioned by the original language of the
document, so for one language L we can compare, at equal N, same domain
and same year:

  native L      L-en set: L side natively authored, English translated
  translated L  en-L set: English original, L side translated

Three readings are computed on each side and compared: sentence
matching, the meaning against language split over the two languages,
and the weakest aligned tail. One model per invocation.

Frozen predictions T1-T4: prereg/addenda/A7_translationese.md.
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

from rmfs.components.variance import variance_components  # noqa: E402
from rmfs.metrics.baselines import aar_shared, mexa_shared  # noqa: E402
from t_extract import load_registry, pooled_embed  # noqa: E402

# cs has only the en-cs direction in WMT19; excluded (needs both)
LANGS = ["de", "ru", "zh", "fi", "lt", "gu", "kk"]
N_PAIRS = 300
PIVOT = "eng"


def load_pair(lang: str, native: bool):
    """Return (english_sentences, target_sentences) for one direction.

    native=True  -> L-en set: L is the original, English is translated.
    native=False -> en-L set: English is the original, L is translated.
    """
    from sacrebleu.utils import get_reference_files, get_source_file

    lp = f"{lang}-en" if native else f"en-{lang}"
    src = [ln.strip() for ln in open(get_source_file("wmt19", lp))]
    ref = [ln.strip() for ln in open(get_reference_files("wmt19", lp)[0])]
    if native:
        tgt, eng = src, ref          # source side is the native L text
    else:
        eng, tgt = src, ref          # source side is the native English
    keep = [i for i in range(min(len(eng), len(tgt)))
            if len(eng[i]) > 20 and len(tgt[i]) > 20][:N_PAIRS]
    if len(keep) < 100:
        raise RuntimeError(f"{lp}: only {len(keep)} usable pairs")
    return [eng[i] for i in keep], [tgt[i] for i in keep]


def readings(E_eng: np.ndarray, E_tgt: np.ndarray) -> dict:
    """The three bilingual readings, on the standard preprocessing."""
    X = np.stack([E_eng, E_tgt], 0).astype(np.float32)     # (2, N, D)
    D = X.shape[-1]
    mu = X.reshape(-1, D).mean(0)
    sd = X.reshape(-1, D).std(0) + 1e-9
    Xz = (X - mu) / sd
    vc = variance_components(Xz)
    m = mexa_shared(Xz[0], Xz[1])
    a = aar_shared(Xz[0], Xz[1])
    return {"language_share": vc.lfs_vc, "matching": m.value,
            "tail": a.tail_mean, "n": int(X.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    args = ap.parse_args()
    tag = args.model_tag
    t0 = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reg = load_registry()
    m = reg[tag]
    dip = m["dip_layer"]
    tok = AutoTokenizer.from_pretrained(m["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if "bloom" in tag.lower() else torch.bfloat16
    model = (AutoModelForCausalLM.from_pretrained(m["hf_id"],
                                                  torch_dtype=dtype)
             .to("cuda").eval())
    print(f"[{tag}] layer {dip}", flush=True)

    out = {"tag": tag, "dip": dip, "n_pairs": N_PAIRS, "languages": {}}
    for lang in LANGS:
        entry = {}
        for native in (True, False):
            key = "native" if native else "translated"
            try:
                eng, tgt = load_pair(lang, native)
            except Exception as e:
                print(f"  [{lang}/{key}] skipped: {e}", flush=True)
                continue
            E_eng = pooled_embed(model, tok, eng, "cuda", dip)
            E_tgt = pooled_embed(model, tok, tgt, "cuda", dip)
            entry[key] = readings(E_eng, E_tgt)
        if {"native", "translated"} <= set(entry):
            n, t = entry["native"], entry["translated"]
            print(f"  [{lang}] matching {n['matching']:.4f} native vs "
                  f"{t['matching']:.4f} translated | language share "
                  f"{n['language_share']:.4f} vs {t['language_share']:.4f}",
                  flush=True)
        out["languages"][lang] = entry

    run_dir = os.path.join(RMFS, "results", "runs", f"translationese_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS,
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": f"translationese_{tag}",
                   "script": "scripts/translationese.py", "git": git,
                   "host": socket.gethostname(),
                   "when": datetime.datetime.now(
                       datetime.timezone.utc).isoformat(),  # noqa: UP017
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"[done] -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
