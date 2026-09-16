"""NTREX loader with MANDATORY document IDs (CLAUDE.md rule 4, E2 rule).

Every split over NTREX sentences must purge at the document level: the 1997
sentences belong to 123 documents (~16 sentences each), and the Phase 0
audit measured 99.9% of naively held-out sentences sharing a document with
training. The prior campaign's loader guarded the document-ID file with an
existence check and silently dropped the constraint when the file was
absent — that pattern survived a full campaign and is banned here: this
loader HARD-FAILS on a missing or inconsistent document-ID file.

Concept-disjointness note: in the NTREX protocols a concept IS a sentence
index (one parallel sentence per concept), so a document-purged sentence
split is automatically concept-disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["NTREXData", "load_ntrex", "purged_concept_splits"]


@dataclass(frozen=True)
class NTREXData:
    """Parallel sentences with their document identities."""

    sentences: dict            # lang -> list[str], all length n
    doc_ids: tuple             # length n, document identity per sentence
    n: int
    n_documents: int


def load_ntrex(root: str | Path, doc_ids_path: str | Path,
               langs: list[str], n_sents: int,
               pivot_file: str = "newstest2019-src.eng.txt",
               ref_pattern: str = "newstest2019-ref.{lang}.txt") -> NTREXData:
    """Load `n_sents` parallel sentences for `langs` (+ implicit 'eng').

    Hard-fails on: missing document-ID file, missing language file, or a
    length mismatch between text and document IDs. No silent fallbacks."""
    root = Path(root)
    doc_p = Path(doc_ids_path)
    if not doc_p.exists():
        raise FileNotFoundError(
            f"DOCUMENT_IDS file not found: {doc_p} — document identity is "
            f"mandatory for every NTREX split (rule 4 / E2); refusing to "
            f"load without it")
    with open(doc_p) as f:
        doc_ids = tuple(line.strip().split("\t")[0] for line in f)[:n_sents]
    if len(doc_ids) < n_sents:
        raise ValueError(
            f"DOCUMENT_IDS has {len(doc_ids)} rows but n_sents={n_sents}")

    sentences: dict = {}
    for lang in ["eng", *[lg for lg in langs if lg != "eng"]]:
        p = root / (pivot_file if lang == "eng"
                    else ref_pattern.format(lang=lang))
        if not p.exists():
            raise FileNotFoundError(
                f"NTREX file for language {lang!r} not found: {p}")
        with open(p) as f:
            rows = [line.strip() for line in f][:n_sents]
        if len(rows) < n_sents:
            raise ValueError(
                f"{p.name} has {len(rows)} sentences, need {n_sents}")
        sentences[lang] = rows
    return NTREXData(sentences=sentences, doc_ids=doc_ids, n=n_sents,
                     n_documents=len(set(doc_ids)))


def purged_concept_splits(doc_ids, n_splits: int, seed: int,
                          train_frac: float = 0.5,
                          strata: np.ndarray | None = None):
    """Document-purged concept splits: DOCUMENTS are assigned to train or
    held-out (optionally stratified by a per-document median stratum), and
    sentences follow their document. No document ever straddles.

    Returns a list of (train_idx, test_idx) integer arrays, generated
    sequentially from one Generator so the split set is reproducible from
    the single recorded seed."""
    doc_arr = np.asarray(doc_ids)
    n = len(doc_arr)
    uniq = np.array(sorted(set(doc_arr.tolist())))
    if strata is not None:
        strata = np.asarray(strata)
        doc_stratum = np.array(
            [int(np.median(strata[doc_arr == d])) for d in uniq])
    else:
        doc_stratum = np.zeros(len(uniq), dtype=int)

    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_splits):
        tr_docs: list = []
        for s in np.unique(doc_stratum):
            idx = np.where(doc_stratum == s)[0]
            idx = rng.permutation(idx)
            k = int(round(len(idx) * train_frac))
            tr_docs.extend(uniq[idx[:k]])
        tr_set = set(tr_docs)
        tr = np.array([i for i in range(n) if doc_arr[i] in tr_set])
        te = np.setdiff1d(np.arange(n), tr)
        assert not (set(doc_arr[tr]) & set(doc_arr[te]))
        splits.append((tr, te))
    return splits
