#!/usr/bin/env python
"""Phase 0.3b: re-estimate the misalignment-budget shares on Qwen3-1.7B with
document-purged splits, side by side with the published (leaky) splits.

Runs cluster-side against the prior repo's n=1500 dump. Wraps the legacy
modules verbatim (gpa, consensus_target, fit_ladder, pooled_pca) rather than
porting them — porting is Phase 1.5; the audit must use the machinery that
produced the published table, changed in exactly one respect (the splits).

Three deliberate choices:
1. The published selected rank is REUSED (read from the committed
   budget.json), so rank selection is held fixed and any share movement is
   attributable to the splits alone.
2. The legacy arm replicates the published run generator-for-generator
   (np.random.default_rng(0) consumed only by the 20 stratified_split
   calls; fit_ladder(seed=s) per split). It should reproduce the committed
   per-language means to float tolerance — that is the parity check that
   the wrapper is faithful, before the purged numbers mean anything.
3. Nulls are skipped (they dominated the published 34h wall time and are
   irrelevant to a split-construction audit).

Decision rule (kickoff 0.3, fixed in advance): any rung whose grid-mean
share moves by more than 2 percentage points under purging is flagged
loudly — a published table changes.

Usage (cluster):
    python scripts/audit_purged_budget.py --model_tag Qwen3-1.7B-Base
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import socket
import subprocess
import sys
import types

import numpy as np

RMFS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRIOR = os.path.expanduser("~/multilingual-metrics")

# legacy modules use RELATIVE imports (ladder.py: `from .gpa import ...`) —
# they lived in a package in the prior repo. legacy/ is read-only, so
# instead of adding an __init__.py we synthesize a package at import time.
# INVENTORY's "imports standalone" column is corrected accordingly.
_pkg = types.ModuleType("legacypkg")
_pkg.__path__ = [os.path.join(RMFS, "legacy")]
sys.modules["legacypkg"] = _pkg
for _name in ("gpa", "subspace", "ladder"):
    _spec = importlib.util.spec_from_file_location(
        f"legacypkg.{_name}", os.path.join(RMFS, "legacy", f"{_name}.py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"legacypkg.{_name}"] = _mod
    _spec.loader.exec_module(_mod)

from legacypkg.gpa import consensus_target, gpa  # noqa: E402
from legacypkg.ladder import fit_ladder  # noqa: E402
from legacypkg.subspace import pooled_pca  # noqa: E402

DUMPS = os.path.join(PRIOR, "results", "dumps")
PUBLISHED = os.path.join(PRIOR, "results", "budget")
NTREX_SRC = os.path.join(PRIOR, "data", "NTREX", "NTREX-128",
                         "newstest2019-src.eng.txt")
DOC_IDS = os.path.join(PRIOR, "data", "NTREX", "DOCUMENT_IDS.tsv")
S_SPLITS = 20
RUNGS = ("share_M1", "share_M2", "share_M3", "share_M4", "share_M5")


def length_strata(n_sents: int) -> np.ndarray:
    with open(NTREX_SRC) as f:
        lens = np.array([len(line.strip()) for line in f][:n_sents])
    q = np.quantile(lens, [1 / 3, 2 / 3])
    return np.digitize(lens, q)


def stratified_split(strata, rng):
    """Verbatim legacy split (leaky)."""
    tr = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        idx = rng.permutation(idx)
        tr.extend(idx[: len(idx) // 2])
    tr = np.array(sorted(tr))
    te = np.setdiff1d(np.arange(len(strata)), tr)
    return tr, te


def purged_split(doc_arr, strata, rng):
    """Documents split 50/50 (stratified by per-document median stratum);
    sentences follow their document. No document straddles."""
    uniq = np.array(sorted(set(doc_arr)))
    doc_stratum = np.array(
        [int(np.median(strata[doc_arr == doc])) for doc in uniq])
    tr_docs = []
    for s in np.unique(doc_stratum):
        idx = np.where(doc_stratum == s)[0]
        idx = rng.permutation(idx)
        tr_docs.extend(uniq[idx[: len(idx) // 2]])
    tr_set = set(tr_docs)
    tr = np.array([i for i in range(len(doc_arr)) if doc_arr[i] in tr_set])
    te = np.setdiff1d(np.arange(len(doc_arr)), tr)
    return tr, te


# One split (GPA + 128 ladder fits, M5 kernel included) costs ~30 min
# single-threaded — measured on job 1703720, which would have needed ~20 h
# serial for both arms. Splits are embarrassingly parallel ONCE their
# indices exist, and rng fidelity lives entirely in index GENERATION, which
# stays sequential and verbatim. So: generate all indices first, then farm
# the fitting out over a process pool. fit_ladder(seed=s) is unchanged.
_POOL_GLOBALS: dict = {}


def _pool_init(Zs, langs):
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"   # one BLAS thread per worker
    _POOL_GLOBALS["Zs"] = Zs
    _POOL_GLOBALS["langs"] = langs


def _fit_one_split(task):
    label, s, tr_c, te_c = task
    Zs, langs = _POOL_GLOBALS["Zs"], _POOL_GLOBALS["langs"]
    g = gpa({lang: Zs[lang][tr_c] for lang in langs})
    z_te = consensus_target(g, {lang: Zs[lang][te_c] for lang in langs})
    out = {}
    for lang in langs:
        res = fit_ladder(Zs[lang][tr_c], g["Z_train"], Zs[lang][te_c],
                         z_te, seed=s)
        out[lang] = res["shares"]
    print(f"[{label} split {s + 1}/{S_SPLITS}] done", flush=True)
    return label, s, out


def run_budgets(Zs, langs, split_sets, n_workers):
    """split_sets: {label: [(tr, te), ...]} with indices pre-generated
    sequentially (rng consumption identical to the serial legacy loop)."""
    import multiprocessing as mp

    tasks = [(label, s, tr, te)
             for label, splits in split_sets.items()
             for s, (tr, te) in enumerate(splits)]
    with mp.get_context("fork").Pool(
            n_workers, initializer=_pool_init, initargs=(Zs, langs)) as pool:
        results = pool.map(_fit_one_split, tasks, chunksize=1)
    means = {}
    for label in split_sets:
        rows = [r for lb, _, r in results if lb == label]
        means[label] = {
            lang: {k: float(np.mean([d[lang][k] for d in rows]))
                   for k in rows[0][lang]}
            for lang in langs}
    return means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", default="Qwen3-1.7B-Base")
    args = ap.parse_args()
    tag = args.model_tag

    meta = json.load(open(os.path.join(DUMPS, tag, "meta.json")))
    dip, n_sents = meta["dip_layer"], meta["n_sents"]
    published = json.load(open(os.path.join(PUBLISHED, tag, "budget.json")))
    sel_r = published["selected_rank"]
    print(f"[audit] {tag}: dip L{dip}, n={n_sents}, published rank r={sel_r}",
          flush=True)

    z = np.load(os.path.join(DUMPS, tag, f"layer{dip:03d}.npz"),
                allow_pickle=False)
    X = z["X"].astype(np.float32)
    langs = sorted(str(x) for x in z["langs"])
    order = {str(x): i for i, x in enumerate(z["langs"])}
    E = {lang: X[order[lang]] for lang in langs}
    print(f"[data] {len(langs)} languages, per-language shape "
          f"{E[langs[0]].shape}", flush=True)

    flat = np.concatenate([E[lang] for lang in langs], 0)
    mu_raw = flat.mean(0)
    pca = pooled_pca(flat, sel_r)
    Zs = {lang: ((E[lang] - mu_raw) @ pca["basis"]).astype(np.float64)
          for lang in langs}

    strata = length_strata(n_sents)
    with open(DOC_IDS) as f:
        doc_arr = np.array([line.strip().split("\t")[0]
                            for line in f][:n_sents])

    # Split indices are generated SEQUENTIALLY, one generator per arm,
    # consumed exactly as the serial legacy loop consumed it (budget_real.py
    # line 158) — rng fidelity is preserved here; only the fitting is
    # parallel.
    rng_leg = np.random.default_rng(0)
    legacy_splits = [stratified_split(strata, rng_leg)
                     for _ in range(S_SPLITS)]
    rng_pur = np.random.default_rng(0)
    purged_splits = [purged_split(doc_arr, strata, rng_pur)
                     for _ in range(S_SPLITS)]
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    means = run_budgets(Zs, langs,
                        {"legacy": legacy_splits, "purged": purged_splits},
                        n_workers)
    legacy_means, purged_means = means["legacy"], means["purged"]

    pub_means = {
        lang: {k: published["per_language"][lang][k]["mean"] for k in RUNGS}
        for lang in langs if lang in published["per_language"]
    }

    def grid_mean(d, k):
        return float(np.mean([d[lang][k] for lang in d]))

    report = {"model": tag, "dip_layer": dip, "n_sents": n_sents,
              "selected_rank_reused": sel_r, "S": S_SPLITS,
              "rows": [], "parity": {}, "per_language": {}}
    print("\n=== grid-mean shares (percent) ===")
    print(f"{'rung':>10} {'published':>10} {'replicated':>11} "
          f"{'purged':>8} {'purged-repl':>12} {'flag':>6}")
    for k in RUNGS:
        pub = grid_mean(pub_means, k) * 100
        rep = grid_mean(legacy_means, k) * 100
        pur = grid_mean(purged_means, k) * 100
        delta = pur - rep
        flag = "FLAG" if abs(delta) > 2.0 else ""
        report["rows"].append({"rung": k, "published_pct": pub,
                               "replicated_pct": rep, "purged_pct": pur,
                               "delta_pp": delta, "flagged": bool(flag)})
        report["parity"][k] = rep - pub
        print(f"{k:>10} {pub:>10.3f} {rep:>11.3f} {pur:>8.3f} "
              f"{delta:>+12.3f} {flag:>6}")
    report["per_language"] = {
        lang: {"legacy": legacy_means[lang], "purged": purged_means[lang]}
        for lang in langs
    }

    run_id = "audit_purged_budget"
    out_dir = os.path.join(RMFS, "results", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(report, f, indent=1)
    git_hash = subprocess.run(["git", "-C", RMFS, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    manifest = {"run_id": run_id, "git": git_hash, "host": socket.gethostname(),
                "when": datetime.datetime.now().isoformat(timespec="seconds"),
                "config": vars(args) | {"dip": dip, "rank": sel_r,
                                        "S": S_SPLITS, "skip_nulls": True},
                "inputs": {"dump": f"{DUMPS}/{tag}/layer{dip:03d}.npz",
                           "published": f"{PUBLISHED}/{tag}/budget.json"},
                "precision": "fp32 load, fp64 ladder (legacy convention)"}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n[done] -> {out_dir}/result.json")


if __name__ == "__main__":
    main()
