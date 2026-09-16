#!/usr/bin/env python
"""S2 battery, behavioral-T cells (GPU, cluster). One task, all configs.

For every signature config: inject the config into the base pooled
embeddings, refit the FULL factor pipeline on the injected world (basis,
GPA, LOO consensus, reverse maps — rungs inherited frozen from Phase 1,
A3 pattern), then measure ONLY the two injection conditions per pair
(NLL_inj, NLL_injw). The text conditions (none/native/mismatch/scaffold)
are injection-invariant and are REUSED from the Phase-2 base run's
t_rows.parquet — same pairs, same targets, same scaffold lengths.

Floors (declared with worlds.py): T's stochastic channel is the
REVERSE-MAP fit seed — identity is re-run with rev seeds {13, 42, 71}
while the split and pair set stay pinned to seed 13 (the text-condition
NLLs being reused exist only for those pairs). Deterministic rungs
(M0-M4) make the seed spread degenerate, so the declared floor is
  floor_T = max( SD over the three replicates,
                 |T(identity s13) - T(base run)|,   # harness recompute
                 1e-9 )                             # noise (bf16 forward)
"""

from __future__ import annotations

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

from rmfs.battery.worlds import (  # noqa: E402
    CONFIGS,
    apply_config,
)
from rmfs.components.ladder import consensus_target, gpa  # noqa: E402
from rmfs.components.transfer import fit_reverse_map, reconstruct  # noqa: E402
from rmfs.data.ntrex import load_ntrex, purged_concept_splits  # noqa: E402
from t_extract import (  # noqa: E402
    CTX_LANGS,
    DOC_IDS,
    N_SENTS,
    NTREX,
    load_registry,
    make_pairs,
    nll_of_target,
    rungs_for,
    scaffold_ids,
)

TAG = "Qwen3-0.6B-Base"
RANK = 64
IDENTITY_SEEDS = [13, 42, 71]


def main() -> None:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    reg = load_registry()
    dip = reg[TAG]["dip_layer"]
    rung_by_lang = rungs_for(TAG)

    base_rows = pd.read_parquet(os.path.join(
        RMFS, "results", "runs", f"t_extract_{TAG}", "t_rows.parquet"))

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(reg[TAG]["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        reg[TAG]["hf_id"], torch_dtype=torch.bfloat16).to(dev).eval()

    langs = list(CTX_LANGS)
    data = load_ntrex(NTREX, DOC_IDS, langs, N_SENTS)
    doc_ids = list(data.doc_ids)
    eng = data.sentences["eng"]
    all_langs = ["eng", *langs]

    from t_extract import pooled_embed
    E0 = {}
    for lg in all_langs:
        E0[lg] = pooled_embed(model, tok, data.sentences[lg], dev, dip)
        print(f"  [embed] {lg}", flush=True)

    flat0 = np.concatenate([E0[lg] for lg in all_langs], 0)
    mu_b = flat0.mean(0, keepdims=True).astype(np.float64)
    fc = flat0 - mu_b.astype(np.float32)
    _, vecs = np.linalg.eigh((fc.T @ fc).astype(np.float64))
    basis_b = vecs[:, ::-1][:, :RANK]

    runs = [("identity", "-", s) for s in IDENTITY_SEEDS]
    runs += [(n, m, 13) for n, m in CONFIGS if n != "identity"]

    rows = []
    for name, mag, rev_seed in runs:
        key = f"{name}|{mag}|s{rev_seed}"
        E, _ = apply_config(E0, name, mag, rev_seed, basis_b, mu_b)

        flat = np.concatenate([E[lg] for lg in all_langs], 0).astype(
            np.float32)
        mu = flat.mean(0).astype(np.float64)
        fcc = flat - mu.astype(np.float32)
        _, ev = np.linalg.eigh((fcc.T @ fcc).astype(np.float64))
        basis = ev[:, ::-1][:, :RANK]
        Zc = {lg: ((E[lg] - mu) @ basis) for lg in all_langs}

        # split and pairs PINNED to seed 13: the reused text-condition
        # NLLs exist only for these pairs; replicate noise enters through
        # rev_seed alone.
        tr, te = purged_concept_splits(doc_ids, n_splits=1, seed=13)[0]
        g = gpa({lg: Zc[lg][tr] for lg in all_langs})
        pairs, mis = make_pairs(doc_ids, te, 60)
        te_pos = {int(c): k for k, c in enumerate(te)}

        for lg in langs:
            rung = rung_by_lang.get(lg)
            if rung is None:
                continue
            z_tr = consensus_target(g, {l2: Zc[l2][tr] for l2 in all_langs},
                                    exclude=lg)
            z_te = consensus_target(g, {l2: Zc[l2][te] for l2 in all_langs},
                                    exclude=lg)
            rev = fit_reverse_map(z_tr, Zc[lg][tr], rung, seed=rev_seed)
            sents = data.sentences[lg]
            for k in range(len(pairs)):
                i, j = pairs[k], mis[k]
                target = eng[i + 1]
                n_ctx = tok(sents[i] + "\n",
                            return_tensors="pt").input_ids.shape[1]
                scaf = scaffold_ids(tok, n_ctx)
                rec_i = reconstruct(z_te[te_pos[i]], basis, mu, rev,
                                    mode="strict", n_tokens=n_ctx)
                rec_j = reconstruct(z_te[te_pos[j]], basis, mu, rev,
                                    mode="strict", n_tokens=n_ctx)
                nll_inj = nll_of_target(model, tok, dev, dip, target,
                                        context_ids=scaf,
                                        replace_states=rec_i)
                nll_injw = nll_of_target(model, tok, dev, dip, target,
                                         context_ids=scaf,
                                         replace_states=rec_j)
                rows.append({"config": key, "lang": lg, "pair": i,
                             "nll_inject": nll_inj,
                             "nll_inject_wrong": nll_injw})
        print(f"[{key}] done ({time.time() - t0:.0f}s)", flush=True)

    run_dir = os.path.join(RMFS, "results", "runs", "battery_t")
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(run_dir, "t_rows.parquet"),
                                  index=False)
    base_rows.to_parquet(os.path.join(run_dir, "base_rows.parquet"),
                         index=False)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS,
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": "battery_t", "git": git,
                   "host": socket.gethostname(),
                   "when": datetime.datetime.now(
                       datetime.UTC).isoformat(),
                   "wall_s": round(time.time() - t0, 1),
                   "n_runs": len(runs),
                   "script": "scripts/battery_t.py"}, f, indent=1)
    print(f"[done] {len(rows)} rows, {len(runs)} runs -> {run_dir}",
          flush=True)


if __name__ == "__main__":
    main()
