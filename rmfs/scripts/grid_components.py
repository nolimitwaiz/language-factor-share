#!/usr/bin/env python
"""Phase 1.7: first RMFS-stack numbers on the model grid.

Per model (one SLURM array task each), at the frozen dip layer, from the
saved n=1500 campaign dump:

  native track (per-dimension z-scored, prereg §2):
    LFS-VC (REML), the legacy ratio (contrast column), LDE (raw +
    noise-corrected)
  raw track (scale must stay visible — the gates' whole point):
    concept variance, mean norm, effective rank
  baselines, shared prep, same embeddings:
    MEXA and AaR per language against the eng pivot
  K: the M0-M5 ladder with GPA consensus refit per split, over 20
    DOCUMENT-PURGED length-stratified concept splits at the PUBLISHED
    selected rank (held fixed so K is attributable to the mapping, not to
    rank selection), one-SE rung selection per language.

Everything runs on saved embeddings; no model is loaded. Output: one
parquet (long format: model, language, layer, component, value, run_id)
plus result.json + manifest.json per rule 9.

The sbatch prologue runs the unit battery first: no grid number is produced
on a machine where the battery has not passed.

Usage (cluster):  python scripts/grid_components.py --model_tag Qwen3-1.7B-Base
Local self-test:  python scripts/grid_components.py --selftest
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys

import numpy as np

RMFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS, "src"))
PRIOR = os.path.expanduser("~/multilingual-metrics")

from rmfs.components.gates import (  # noqa: E402
    concept_variance_raw,
    effective_rank,
    mean_norm,
)
from rmfs.components.ladder import (  # noqa: E402
    consensus_target,
    fit_ladder,
    gpa,
    select_rung,
)
from rmfs.components.variance import (  # noqa: E402
    lde,
    legacy_lfs_ratio,
    variance_components,
)
from rmfs.data.ntrex import purged_concept_splits  # noqa: E402
from rmfs.metrics.baselines import aar_shared, mexa_shared  # noqa: E402

DUMPS = os.path.join(PRIOR, "results", "dumps")
BUDGET = os.path.join(PRIOR, "results", "budget")
NTREX_SRC = os.path.join(PRIOR, "data", "NTREX", "NTREX-128",
                         "newstest2019-src.eng.txt")
DOC_IDS = os.path.join(PRIOR, "data", "NTREX", "DOCUMENT_IDS.tsv")
S_SPLITS = 20
PIVOT = "eng"

_POOL: dict = {}


def _pool_init(Zs, langs):
    os.environ["OMP_NUM_THREADS"] = "1"
    _POOL["Zs"], _POOL["langs"] = Zs, langs


def _fit_split(task):
    s, tr, te = task
    Zs, langs = _POOL["Zs"], _POOL["langs"]
    g = gpa({lang: Zs[lang][tr] for lang in langs})
    z_te = consensus_target(g, {lang: Zs[lang][te] for lang in langs})
    out = {lang: fit_ladder(Zs[lang][tr], g["Z_train"], Zs[lang][te], z_te,
                            seed=s)
           for lang in langs}
    print(f"[ladder split {s + 1}/{S_SPLITS}] done", flush=True)
    return s, out


def length_strata(n_sents: int) -> np.ndarray:
    with open(NTREX_SRC) as f:
        lens = np.array([len(line.strip()) for line in f][:n_sents])
    q = np.quantile(lens, [1 / 3, 2 / 3])
    return np.digitize(lens, q)


def load_dump(tag: str):
    meta = json.load(open(os.path.join(DUMPS, tag, "meta.json")))
    dip, n_sents = meta["dip_layer"], meta["n_sents"]
    z = np.load(os.path.join(DUMPS, tag, f"layer{dip:03d}.npz"),
                allow_pickle=False)
    X = z["X"].astype(np.float32)
    langs = sorted(str(x) for x in z["langs"])
    order = {str(x): i for i, x in enumerate(z["langs"])}
    E = {lang: X[order[lang]] for lang in langs}
    return E, langs, dip, n_sents


def run_model(tag: str, n_workers: int, skip_ladder: bool = False) -> dict:
    E, langs, dip, n_sents = load_dump(tag)
    if PIVOT not in langs:
        raise RuntimeError(f"pivot {PIVOT} missing from dump for {tag}")
    X = np.stack([E[lang] for lang in langs], 0)          # (L, N, D)
    L, N, D = X.shape
    print(f"[{tag}] dip L{dip}, {L} languages, N={N}, D={D}", flush=True)

    # ---- native track: per-dimension z-scoring over all cells
    mu = X.reshape(-1, D).mean(0)
    sd = X.reshape(-1, D).std(0) + 1e-9
    Xz = (X - mu) / sd
    vc = variance_components(Xz)
    legacy_ratio = legacy_lfs_ratio(Xz)
    lde_res = lde(Xz)

    # ---- raw track: the gates
    v_conc_raw = concept_variance_raw(X)
    r_eff = effective_rank(X.mean(0))
    norm_gate = mean_norm(X)

    # ---- shared-prep baselines per language vs pivot
    Ez = {lang: Xz[i] for i, lang in enumerate(langs)}
    mexa = {lang: mexa_shared(Ez[PIVOT], Ez[lang]).value
            for lang in langs if lang != PIVOT}
    aar = {}
    for lang in langs:
        if lang == PIVOT:
            continue
        r = aar_shared(Ez[PIVOT], Ez[lang])
        aar[lang] = {"tail": r.tail_mean, "mean": r.mean}

    # ---- K: purged cross-fitted ladder at the published rank
    if skip_ladder:
        return {
            "model": tag, "dip_layer": dip, "n_sents": n_sents,
            "L": L, "N": N, "D": D, "selected_rank_reused": None,
            "lfs_vc": vc.lfs_vc, "lfs_vc_sd": vc.lfs_vc_sd,
            "lfs_vc_anova": vc.lfs_vc_anova,
            "sigma2_L": vc.sigma2_L_reml, "sigma2_C": vc.sigma2_C_reml,
            "sigma2_E": vc.sigma2_E, "n_neg_L": vc.n_neg_L,
            "n_neg_C": vc.n_neg_C, "legacy_lfs": legacy_ratio,
            "lde_raw": {langs[i]: lde_res.raw[i] for i in range(L)},
            "lde_corrected": {langs[i]: lde_res.corrected[i]
                              for i in range(L)},
            "lde_correction": lde_res.correction,
            "concept_variance_raw": v_conc_raw,
            "effective_rank": r_eff.value, "mean_norm": norm_gate.value,
            "mexa": mexa, "aar": aar, "K": {},
        }
    # OLMo-2-0425-1B's budget primary timed out twice in the prior campaign
    # and was never rerun (INVENTORY); its published rank lives in the
    # _fixed_raw variant. The fallback is EXPLICIT and recorded in the
    # result + manifest — announced, never silent (E2 rule).
    primary = os.path.join(BUDGET, tag, "budget.json")
    fallback = os.path.join(BUDGET, f"{tag}_fixed_raw", "budget.json")
    if os.path.exists(primary):
        published, rank_source = json.load(open(primary)), "primary"
    elif os.path.exists(fallback):
        published, rank_source = json.load(open(fallback)), "fixed_raw"
        print(f"[rank] NOTE: no budget primary for {tag}; using the "
              f"_fixed_raw published rank (campaign primary timed out "
              f"twice; INVENTORY)", flush=True)
    else:
        raise FileNotFoundError(
            f"no published rank for {tag}: neither {primary} nor "
            f"{fallback} exists — refusing to choose a rank ad hoc")
    sel_r = published["selected_rank"]
    flat = np.concatenate([E[lang] for lang in langs], 0)
    mu_raw = flat.mean(0).astype(np.float64)
    # Basis via eigendecomposition of the DxD covariance rather than a full
    # SVD of the 192k x D matrix: one GEMM + a small eigh, minutes even
    # single-threaded. Single-threaded matters: OMP_NUM_THREADS=1 must be
    # exported in the sbatch BEFORE python starts (fork children inherit
    # BLAS thread state; setting it in the pool initializer is too late and
    # 10 workers x 10 BLAS threads thrashed the first run to a third of its
    # throughput - RUNS.md E3).
    flatc32 = (flat - mu_raw.astype(np.float32))
    C = (flatc32.T @ flatc32).astype(np.float64)
    evals, evecs = np.linalg.eigh(C)
    basis = evecs[:, ::-1][:, :sel_r]
    Zs = {lang: ((E[lang] - mu_raw) @ basis).astype(np.float64)
          for lang in langs}
    strata = length_strata(n_sents)
    with open(DOC_IDS) as f:
        doc_ids = [line.strip().split("\t")[0] for line in f][:n_sents]
    splits = purged_concept_splits(doc_ids, S_SPLITS, seed=13, strata=strata)

    import multiprocessing as mp

    tasks = [(s, tr, te) for s, (tr, te) in enumerate(splits)]
    with mp.get_context("fork").Pool(n_workers, initializer=_pool_init,
                                     initargs=(Zs, langs)) as pool:
        split_results = dict_by_s = pool.map(_fit_split, tasks, chunksize=1)
    dict_by_s = dict(split_results)
    K = {}
    for lang in langs:
        per_split = [dict_by_s[s][lang] for s in range(S_SPLITS)]
        sel = select_rung(per_split)
        K[lang] = {"rung": sel.rung, "K": sel.K, "q_K": sel.q_K,
                   "mean_errors": sel.mean_errors}

    return {
        "model": tag, "dip_layer": dip, "n_sents": n_sents,
        "L": L, "N": N, "D": D, "selected_rank_reused": sel_r,
        "rank_source": rank_source,
        "lfs_vc": vc.lfs_vc, "lfs_vc_sd": vc.lfs_vc_sd,
        "lfs_vc_anova": vc.lfs_vc_anova,
        "sigma2_L": vc.sigma2_L_reml, "sigma2_C": vc.sigma2_C_reml,
        "sigma2_E": vc.sigma2_E, "n_neg_L": vc.n_neg_L,
        "n_neg_C": vc.n_neg_C, "legacy_lfs": legacy_ratio,
        "lde_raw": {langs[i]: lde_res.raw[i] for i in range(L)},
        "lde_corrected": {langs[i]: lde_res.corrected[i] for i in range(L)},
        "lde_correction": lde_res.correction,
        "concept_variance_raw": v_conc_raw,
        "effective_rank": r_eff.value, "mean_norm": norm_gate.value,
        "mexa": mexa, "aar": aar, "K": K,
    }


def to_parquet(res: dict, run_id: str, out_dir: str):
    import pandas as pd

    rows = []

    def add(lang, comp, val):
        rows.append(dict(model=res["model"], language=lang,
                         layer=res["dip_layer"], seed=13, component=comp,
                         value=float(val), run_id=run_id))

    for comp in ("lfs_vc", "lfs_vc_sd", "lfs_vc_anova", "legacy_lfs",
                 "sigma2_L", "sigma2_C", "sigma2_E",
                 "concept_variance_raw", "effective_rank", "mean_norm"):
        add("__grid__", comp, res[comp])
    for lang, v in res["lde_corrected"].items():
        add(lang, "lde", v)
    for lang, v in res["mexa"].items():
        add(lang, "mexa_shared", v)
    for lang, v in res["aar"].items():
        add(lang, "aar10_tail", v["tail"])
        add(lang, "aar_mean", v["mean"])
    for lang, v in res["K"].items():
        add(lang, "K", v["K"])
        add(lang, "q_K", v["q_K"])
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, f"{res['model']}.parquet"))
    return len(rows)


def selftest() -> None:
    """Plumbing check on synthetic data, CPU, no dumps needed."""
    g = np.random.default_rng(13)
    langs = ["eng", "aaa", "bbb"]
    N, D = 60, 24
    base = g.normal(size=(N, D))
    E = {lang: base + 0.3 * g.normal(size=(N, D)) + i
         for i, lang in enumerate(langs)}
    X = np.stack([E[lang] for lang in langs], 0)
    mu = X.reshape(-1, D).mean(0)
    sd = X.reshape(-1, D).std(0) + 1e-9
    Xz = (X - mu) / sd
    vc = variance_components(Xz)
    assert 0 <= vc.lfs_vc <= 1
    r = aar_shared(Xz[0], Xz[1])
    assert r.tail_mean <= r.mean
    print("selftest OK: components + baselines wire together")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip_ladder", action="store_true",
                    help="quick pass: everything except K (the ladder is "
                         "~99 percent of the compute); lets the array ride "
                         "short-walltime backfill while the full-K pass "
                         "waits for real capacity")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.model_tag:
        raise SystemExit("--model_tag required (or --selftest)")

    tag = args.model_tag
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    res = run_model(tag, n_workers, skip_ladder=args.skip_ladder)

    run_id = (f"grid_quick_{tag}" if args.skip_ladder
              else f"grid_components_{tag}")
    out_dir = os.path.join(RMFS, "results", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    n_rows = to_parquet(res, run_id, out_dir)
    git_hash = subprocess.run(["git", "-C", RMFS, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    manifest = {
        "run_id": run_id, "git": git_hash, "host": socket.gethostname(),
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {"model_tag": tag, "S": S_SPLITS, "seed": 13,
                   "rank": res["selected_rank_reused"],
                   "tracks": "native=z-scored for LFS-VC/LDE/baselines; "
                             "raw for gates"},
        "inputs": {"dump": f"{DUMPS}/{tag}",
                   "published_rank_from": f"{BUDGET}/{tag}/budget.json",
                   "doc_ids": DOC_IDS},
        "precision": "fp32 load, fp64 statistics",
        "n_parquet_rows": n_rows,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[done] {tag}: {n_rows} rows -> {out_dir}")


if __name__ == "__main__":
    main()
