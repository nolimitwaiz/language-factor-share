#!/usr/bin/env python
"""Phase 0.3a: quantify document leakage in the legacy misalignment-budget
splits. Pure CPU on the split generator + DOCUMENT_IDS.tsv; no model, no
embeddings.

The legacy splits (`legacy/budget_real.py`) stratify by English source
sentence-length terciles and permute within each stratum
(`stratified_split`), with `rng = np.random.default_rng(0)` and S=20
cross-fitting splits (plus 5 rank-selection splits at seed 0). Document
identity is never consulted. This script replicates that generator
verbatim (same seeds, same order of rng consumption) and joins the splits
against NTREX document IDs to measure:

  doc_straddle_rate : fraction of documents with sentences on BOTH sides
  heldout_leak_rate : fraction of held-out sentences sharing a document
                      with at least one training sentence

at both protocols (n=300 development, n=1500 campaign).

Output: printed table + ledgers/ERROR_LEDGER.md entry is written by hand
from these numbers (rule: the ledger entry is authored, not templated).
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NTREX_SRC = os.path.join(ROOT, "data/raw/NTREX/NTREX-128/newstest2019-src.eng.txt")
DOC_IDS = os.path.join(ROOT, "data/raw/NTREX/DOCUMENT_IDS.tsv")


def length_strata(n_sents: int) -> np.ndarray:
    """Verbatim replication of legacy/budget_real.py:length_strata."""
    with open(NTREX_SRC) as f:
        lens = np.array([len(line.strip()) for line in f][:n_sents])
    q = np.quantile(lens, [1 / 3, 2 / 3])
    return np.digitize(lens, q)


def stratified_split(strata: np.ndarray, rng: np.random.Generator):
    """Verbatim replication of legacy/budget_real.py:stratified_split."""
    tr: list[int] = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        idx = rng.permutation(idx)
        tr.extend(idx[: len(idx) // 2])
    tr_arr = np.array(sorted(tr))
    te = np.setdiff1d(np.arange(len(strata)), tr_arr)
    return tr_arr, te


def docs(n_sents: int) -> np.ndarray:
    with open(DOC_IDS) as f:
        ids = [line.strip().split("\t")[0] for line in f][:n_sents]
    return np.array(ids)


def audit(n_sents: int, n_splits: int = 20, seed: int = 0) -> dict:
    strata = length_strata(n_sents)
    d = docs(n_sents)
    # Replicates the legacy cross-fitting loop: one Generator seeded once,
    # consumed across all S splits in order (budget_real.py line 158).
    rng = np.random.default_rng(seed)
    straddle, leak = [], []
    for _ in range(n_splits):
        tr, te = stratified_split(strata, rng)
        tr_docs, te_docs = set(d[tr]), set(d[te])
        both = tr_docs & te_docs
        straddle.append(len(both) / len(set(d)))
        leak.append(float(np.mean([d[i] in tr_docs for i in te])))
    return {
        "n_sents": n_sents,
        "n_docs": len(set(d)),
        "n_splits": n_splits,
        "doc_straddle_rate": (float(np.mean(straddle)), float(np.min(straddle)),
                              float(np.max(straddle))),
        "heldout_leak_rate": (float(np.mean(leak)), float(np.min(leak)),
                              float(np.max(leak))),
    }


def purged_split(d: np.ndarray, strata: np.ndarray, rng: np.random.Generator):
    """Document-purged counterpart: split DOCUMENTS 50/50, stratifying each
    document by its median sentence-length stratum so the marginal
    stratification intent survives."""
    uniq = np.array(sorted(set(d)))
    doc_stratum = np.array(
        [int(np.median(strata[d == doc])) for doc in uniq]
    )
    tr_docs: list[str] = []
    for s in np.unique(doc_stratum):
        idx = np.where(doc_stratum == s)[0]
        idx = rng.permutation(idx)
        tr_docs.extend(uniq[idx[: len(idx) // 2]])
    tr_set = set(tr_docs)
    tr = np.array([i for i in range(len(d)) if d[i] in tr_set])
    te = np.setdiff1d(np.arange(len(d)), tr)
    return tr, te


def main() -> None:
    print("Legacy split generator, replicated verbatim, joined with document IDs\n")
    rows = []
    for n in (300, 1500):
        r = audit(n)
        rows.append(r)
        sm, s0, s1 = r["doc_straddle_rate"]
        lm, l0, l1 = r["heldout_leak_rate"]
        print(f"  n={n:5d}  docs={r['n_docs']:3d}  S={r['n_splits']}")
        print(f"    documents straddling train/held-out: mean {sm:.1%}  "
              f"range [{s0:.1%}, {s1:.1%}]")
        print(f"    held-out sentences sharing a doc with train: mean {lm:.1%}  "
              f"range [{l0:.1%}, {l1:.1%}]\n")

    print("Purged counterpart (documents split, never sentences):")
    for n in (300, 1500):
        strata = length_strata(n)
        d = docs(n)
        rng = np.random.default_rng(0)
        tr, te = purged_split(d, strata, rng)
        tr_docs, te_docs = set(d[tr]), set(d[te])
        assert not (tr_docs & te_docs), "purge failed"
        # balance check: how far the sentence split drifts from 50/50
        print(f"  n={n:5d}: {len(tr)} train / {len(te)} held-out sentences "
              f"({len(tr)/n:.1%}/{len(te)/n:.1%}), straddle 0, "
              f"stratum balance preserved at document level")
    sys.exit(0)


if __name__ == "__main__":
    main()
