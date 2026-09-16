#!/usr/bin/env python
"""S2 battery, geometric components (CPU, cluster: needs the Qwen3-0.6B
dump). Runs every signature config through the UNTOUCHED grid pipeline
(inject pre-z-score fp32 -> refit basis -> z-track REML -> raw-track
ladder), plus the noise-floor replicates.

DECLARED BEFORE SCORING (with worlds.py; ledgered):
- World: Qwen3-0.6B-Base dump, dip layer, rows 0:300 (dev protocol),
  rank 64, S = 20 document-purged splits, seed 13.
- Floors: identity is bit-exact, so recompute noise is zero by
  construction; the honest floor is SENTENCE-SAMPLING noise — the full
  pipeline re-run on dump rows 300:600, 600:900, 900:1200 (legacy i10
  doctrine: resampled halves are the floor). floor_c = SD over the four
  world replicates (base + 3).
- Two-track estimators: q_L (1 - LFS-VC) on the z-scored track (grid
  parity; the blindness cells assume the scale-invariant ratio); C_pres
  on the RAW REML sigma2_C ratio (assembly parity; z-scoring after a
  collapse would partially undo it and break the (1-m)^2 closed form).
  Per-language V_ratio (own-centroid deviation energy) is saved for the
  collapse_per_language cells and per-language RMFS assembly.
- Ladder rungs refit per config (the pipeline is end-to-end); T's rung
  map is inherited from Phase 1 as frozen (A3 pattern) in battery_t.py.
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
PRIOR = os.path.expanduser("~/multilingual-metrics")
DUMPS = os.path.join(PRIOR, "results", "dumps")
NTREX_SRC = os.path.join(PRIOR, "data", "NTREX", "NTREX-128",
                         "newstest2019-src.eng.txt")
DOC_IDS = os.path.join(PRIOR, "data", "NTREX", "DOCUMENT_IDS.tsv")

from rmfs.battery.worlds import (  # noqa: E402
    CONFIGS,
    affected_languages,
    apply_config,
)
from rmfs.components.ladder import (  # noqa: E402
    consensus_target,
    fit_ladder,
    gpa,
    select_rung,
)
from rmfs.components.variance import variance_components  # noqa: E402
from rmfs.data.ntrex import purged_concept_splits  # noqa: E402

TAG = "Qwen3-0.6B-Base"
N, RANK, S_SPLITS, SEED = 300, 64, 20, 13
FLOOR_OFFSETS = [300, 600, 900]

_POOL: dict = {}


def _pool_init(Zs, langs, z_pivot):
    os.environ["OMP_NUM_THREADS"] = "1"
    _POOL.update(Zs=Zs, langs=langs)


def _fit_split(task):
    s, tr, te, lang_subset = task
    Zs, langs = _POOL["Zs"], _POOL["langs"]
    g = gpa({lg: Zs[lg][tr] for lg in langs})
    z_te = consensus_target(g, {lg: Zs[lg][te] for lg in langs})
    return s, {lg: fit_ladder(Zs[lg][tr], g["Z_train"], Zs[lg][te], z_te,
                              seed=s)
               for lg in lang_subset}


def pipeline(E: dict, doc_ids, strata, langs, n_workers: int,
             affected: list) -> dict:
    """The untouched grid pipeline on one (possibly injected) world."""
    X = np.stack([E[lg] for lg in langs], 0).astype(np.float32)
    L, n, D = X.shape

    mu = X.reshape(-1, D).mean(0)
    sd = X.reshape(-1, D).std(0) + 1e-9
    vc_z = variance_components((X - mu) / sd)
    vc_raw = variance_components(X)

    v_lang = {lg: float(((E[lg] - E[lg].mean(0)) ** 2).mean())
              for lg in langs}

    flat = np.concatenate([E[lg] for lg in langs], 0)
    mu_raw = flat.mean(0).astype(np.float64)
    fc = flat - mu_raw.astype(np.float32)
    evals, evecs = np.linalg.eigh((fc.T @ fc).astype(np.float64))
    basis = evecs[:, ::-1][:, :RANK]
    Zs = {lg: ((E[lg] - mu_raw) @ basis).astype(np.float64) for lg in langs}

    import multiprocessing as mp

    splits = purged_concept_splits(doc_ids, S_SPLITS, SEED, strata=strata)
    tasks = [(s, tr, te, affected) for s, (tr, te) in enumerate(splits)]
    with mp.get_context("fork").Pool(n_workers, initializer=_pool_init,
                                     initargs=(Zs, langs, None)) as pool:
        by_s = dict(pool.map(_fit_split, tasks, chunksize=1))
    rungs = {}
    for lg in affected:
        sel = select_rung([by_s[s][lg] for s in range(S_SPLITS)])
        rungs[lg] = {"rung": sel.rung, "q_K": sel.q_K}

    return {"q_L": 1.0 - vc_z.lfs_vc, "lfs_vc_z": vc_z.lfs_vc,
            "q_L_raw": 1.0 - vc_raw.lfs_vc,   # invariant track (no
            "sigma2_C_raw": vc_raw.sigma2_C_reml,  # z-scoring): the
            "v_lang": v_lang,                 # rotation-cell's frozen note
            "rungs": rungs}                   # pins q_L to THIS track


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    meta = json.load(open(os.path.join(DUMPS, TAG, "meta.json")))
    dip = meta["dip_layer"]
    z = np.load(os.path.join(DUMPS, TAG, f"layer{dip:03d}.npz"),
                allow_pickle=False)
    Xfull = z["X"].astype(np.float32)
    langs = sorted(str(x) for x in z["langs"])
    order = {str(x): i for i, x in enumerate(z["langs"])}

    if not os.path.exists(DOC_IDS):
        raise FileNotFoundError(f"DOC_IDS missing: {DOC_IDS}")  # E2 rule
    with open(DOC_IDS) as f:
        all_docs = [line.strip().split("\t")[0] for line in f]
    with open(NTREX_SRC) as f:
        all_lens = np.array([len(line.strip()) for line in f])

    def world(row0: int):
        E = {lg: Xfull[order[lg]][row0:row0 + N].copy() for lg in langs}
        docs = all_docs[row0:row0 + N]
        lens = all_lens[row0:row0 + N]
        q = np.quantile(lens, [1 / 3, 2 / 3])
        return E, docs, np.digitize(lens, q)

    E0, docs0, strata0 = world(0)
    aff_all = affected_languages("offset_per_language", langs)

    flat = np.concatenate([E0[lg] for lg in langs], 0)
    mu_b = flat.mean(0, keepdims=True).astype(np.float64)
    fc = flat - mu_b.astype(np.float32)
    _, vecs = np.linalg.eigh((fc.T @ fc).astype(np.float64))
    basis_b = vecs[:, ::-1][:, :RANK]

    results = {"tag": TAG, "dip": dip, "protocol":
               {"n": N, "rank": RANK, "splits": S_SPLITS, "seed": SEED},
               "configs": {}, "floor_worlds": []}

    print(f"[base] world rows 0:{N}", flush=True)
    results["base"] = pipeline(E0, docs0, strata0, langs, args.workers,
                               aff_all)
    print(f"[base] q_L={results['base']['q_L']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    for off in FLOOR_OFFSETS:
        Ei, di, si = world(off)
        print(f"[floor] world rows {off}:{off + N}", flush=True)
        results["floor_worlds"].append(
            pipeline(Ei, di, si, langs, args.workers, aff_all))

    for name, mag in CONFIGS:
        if name == "identity":
            continue                    # floors replace identity replicates
        key = f"{name}|{mag}"
        aff = affected_languages(name, langs)
        Einj, _ = apply_config(E0, name, mag, SEED, basis_b, mu_b)
        print(f"[inject] {key} ({time.time() - t0:.0f}s)", flush=True)
        results["configs"][key] = pipeline(Einj, docs0, strata0, langs,
                                           args.workers, aff)

    run_dir = os.path.join(RMFS, "results", "runs", "battery_geo")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(results, f, indent=1)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RMFS,
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": "battery_geo", "git": git,
                   "host": socket.gethostname(),
                   "when": datetime.datetime.now(
                       datetime.UTC).isoformat(),
                   "wall_s": round(time.time() - t0, 1),
                   "script": "scripts/battery_geo.py"}, f, indent=1)
    print(f"[done] {len(results['configs'])} configs + base + "
          f"{len(FLOOR_OFFSETS)} floor worlds -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
