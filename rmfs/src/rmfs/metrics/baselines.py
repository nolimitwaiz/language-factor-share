"""Baseline metrics for the tournament (Phase 1.6; prereg §3.3 candidates).

Three strict conventions, each earned by a prior-campaign incident:

1. SHARED PREPROCESSING (CLAUDE.md rule 6). `mexa_shared` and `aar_shared`
   are exact ports of the legacy implementations and run on the identical
   embeddings every other candidate sees; the unit battery holds them to
   exact parity with `legacy/study_metrics.py`. The OFFICIAL-pooling MEXA
   (position-weighted token averaging per Kargaran et al.) is a separate
   BRIDGE column produced by `official_position_weights` at extraction time
   and is never mixed into shared-prep comparisons.

2. PROBES REPORT MDL AND SELECTIVITY, NOT RAW ACCURACY. The campaign
   measured language identity at 87.5-88.6% linear-probe accuracy at every
   layer of a 29-layer model (chance 4.2%): raw accuracy has no dynamic
   range as a per-layer metric. Minimum description length (prequential
   coding) and a shuffled-label control task are the standard correctives.

3. Every metric returns a typed result with its diagnostics. No bare
   floats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RetrievalResult", "TailResult", "CKAResult", "ProbeResult",
           "mexa_shared", "aar_shared", "cka_linear",
           "official_position_weights", "language_probe"]


@dataclass(frozen=True)
class RetrievalResult:
    value: float               # fraction of mutually-dominant parallel pairs
    n: int
    prep: str                  # "shared" | "official-bridge"


@dataclass(frozen=True)
class TailResult:
    tail_mean: float           # mean margin of the worst q-quantile
    mean: float                # mean margin over all sentences
    q: float
    k: int                     # tail size in sentences
    n: int


@dataclass(frozen=True)
class CKAResult:
    value: float
    n: int


@dataclass(frozen=True)
class ProbeResult:
    accuracy: float            # kept for reference, NOT the headline
    mdl_bits: float            # prequential codelength of the labels
    mdl_bits_control: float    # same probe on shuffled labels
    selectivity: float         # control minus real, in bits (higher = real
    #                            structure, not probe capacity)
    n_train: int
    n_test: int


def _norm(E: np.ndarray) -> np.ndarray:
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def mexa_shared(E_en: np.ndarray, E_l: np.ndarray) -> RetrievalResult:
    """Exact port of legacy mexa_score: fraction of sentences whose parallel
    pair strictly dominates both its row and its column."""
    S = _norm(np.asarray(E_en, np.float64)) @ _norm(
        np.asarray(E_l, np.float64)).T
    n = len(S)
    d = np.diag(S)
    row_ok = d > (np.where(np.eye(n, dtype=bool), -np.inf, S)).max(1)
    col_ok = d > (np.where(np.eye(n, dtype=bool), -np.inf, S)).max(0)
    return RetrievalResult(value=float((row_ok & col_ok).mean()), n=n,
                           prep="shared")


def aar_shared(E_en: np.ndarray, E_l: np.ndarray,
               q: float = 0.10) -> TailResult:
    """Exact port of legacy aar: mean retrieval margin of the worst
    q-quantile of sentences (margins may be negative; that is the point)."""
    S = _norm(np.asarray(E_l, np.float64)) @ _norm(
        np.asarray(E_en, np.float64)).T
    n = len(S)
    d = np.diag(S).copy()
    off = np.where(np.eye(n, dtype=bool), -np.inf, S).max(1)
    margins = d - off
    k = max(1, int(np.floor(q * n)))
    worst = np.sort(margins)[:k]
    return TailResult(tail_mean=float(worst.mean()),
                      mean=float(margins.mean()), q=q, k=k, n=n)


def cka_linear(X: np.ndarray, Z: np.ndarray) -> CKAResult:
    """Linear CKA (Kornblith et al. 2019), centered. Invariant to orthogonal
    rotation and isotropic scaling BY CONSTRUCTION — the battery scores that
    blindness; the misalignment budget says it costs ~0.7% on real data."""
    X = np.asarray(X, np.float64)
    Z = np.asarray(Z, np.float64)
    Xc, Zc = X - X.mean(0), Z - Z.mean(0)
    num = np.linalg.norm(Xc.T @ Zc, "fro") ** 2
    den = (np.linalg.norm(Xc.T @ Xc, "fro")
           * np.linalg.norm(Zc.T @ Zc, "fro"))
    return CKAResult(value=float(num / den) if den > 0 else 0.0, n=len(X))


def official_position_weights(n_tokens: int) -> np.ndarray:
    """Position weights for the OFFICIAL MEXA pooling bridge: weighted token
    averaging with weight proportional to position index (later tokens carry
    more context), per the MEXA reference implementation. Applied at
    EXTRACTION time on token states; the result feeds `mexa_shared`'s
    scoring rule but is reported only in the clearly-labeled bridge column
    (rule 6). This function is the single definition both the extraction
    code and its tests import."""
    if n_tokens < 1:
        raise ValueError("n_tokens must be >= 1")
    w = np.arange(1, n_tokens + 1, dtype=np.float64)
    return w / w.sum()


def _prequential_bits(X_tr, y_tr, X_te, y_te, n_classes, seed, blocks=8):
    """Prequential (online) codelength of test labels under a logistic probe
    trained on incrementally larger prefixes — the MDL variant that
    penalizes probe capacity naturally (Voita & Titov 2020 style)."""
    from sklearn.linear_model import LogisticRegression

    order = np.random.default_rng(seed).permutation(len(X_te))
    X_te, y_te = X_te[order], y_te[order]
    edges = np.unique(np.linspace(0, len(X_te), blocks + 1).astype(int))
    bits = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo == 0:
            # first block: uniform code
            bits += (hi - lo) * np.log2(n_classes)
            continue
        clf = LogisticRegression(max_iter=500, random_state=seed).fit(
            np.vstack([X_tr, X_te[:lo]]), np.concatenate([y_tr, y_te[:lo]]))
        p = clf.predict_proba(X_te[lo:hi])
        cls_index = {c: j for j, c in enumerate(clf.classes_)}
        rows = np.array([p[r, cls_index[y]] for r, y in enumerate(y_te[lo:hi])])
        bits += float(-np.log2(np.clip(rows, 1e-12, None)).sum())
    return bits


def language_probe(E_by_lang: dict, train_idx: np.ndarray,
                   test_idx: np.ndarray, seed: int = 13) -> ProbeResult:
    """Language-identity probe with MDL and a shuffled-label control task.

    E_by_lang: {lang: (N, D)} embeddings over the SAME concept indices.
    train/test indices must be document-purged upstream (rule 4); this
    function trusts and does not re-derive them."""
    langs = sorted(E_by_lang)
    n_cls = len(langs)
    X_tr = np.vstack([E_by_lang[lg][train_idx] for lg in langs])
    X_te = np.vstack([E_by_lang[lg][test_idx] for lg in langs])
    y_tr = np.concatenate([[i] * len(train_idx) for i in range(n_cls)])
    y_te = np.concatenate([[i] * len(test_idx) for i in range(n_cls)])

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=500, random_state=seed).fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))

    bits = _prequential_bits(X_tr, y_tr, X_te, y_te, n_cls, seed)
    g = np.random.default_rng(seed)
    y_tr_c, y_te_c = g.permutation(y_tr), g.permutation(y_te)
    bits_c = _prequential_bits(X_tr, y_tr_c, X_te, y_te_c, n_cls, seed)

    return ProbeResult(accuracy=acc, mdl_bits=float(bits),
                       mdl_bits_control=float(bits_c),
                       selectivity=float(bits_c - bits),
                       n_train=len(y_tr), n_test=len(y_te))
