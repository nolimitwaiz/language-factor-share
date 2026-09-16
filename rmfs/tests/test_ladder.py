"""Battery for the ladder port (Phase 1.5): rung recovery on synthetic
injections, exact parity with the legacy implementation, purged splits,
one-SE conservatism, and the hard-fail loader.

The rung-recovery tests are the discrete signature cells of the prereg
(`rung_le` / `rung_ge`): an offset-only world must select M1, a rotated
world M3, a warped world M5, an untouched world M0. These are the semantics
K carries into RMFS.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rmfs.components.ladder import (
    RUNGS,
    consensus_target,
    fit_ladder,
    gpa,
    ladder_purged,
    select_rung,
)
from rmfs.data.ntrex import load_ntrex, purged_concept_splits
from rmfs.utils.seeding import rng

R_DIM = 12
NOISE = 0.02


def latent(n=240, seed=13):
    return rng(seed).normal(size=(n, R_DIM))


def _split(Z, X, frac=0.5):
    k = int(len(Z) * frac)
    return X[:k], Z[:k], X[k:], Z[k:]


def _select_for(X, Z, seed=42, n_splits=8):
    # synthetic "documents" of 8 consecutive concepts, so purging is active
    doc_ids = [f"d{i // 8}" for i in range(len(Z))]
    per_split = ladder_purged(X, Z, doc_ids, n_splits=n_splits, seed=seed)
    return select_rung(per_split)


# ------------------------------------------------------------ rung recovery
def test_identity_selects_m0():
    Z = latent()
    X = Z + rng(1).normal(0, NOISE, Z.shape)
    assert _select_for(X, Z).rung == "M0"


def test_offset_selects_m1():
    Z = latent()
    b = rng(2).normal(0, 1.0, size=(1, R_DIM))
    X = Z + b + rng(3).normal(0, NOISE, Z.shape)
    sel = _select_for(X, Z)
    assert sel.rung == "M1"
    assert sel.K == pytest.approx(0.2)


def test_scale_selects_m2():
    Z = latent()
    X = 1.7 * Z + 0.5 + rng(4).normal(0, NOISE, Z.shape)
    assert _select_for(X, Z).rung == "M2"


def test_rotation_selects_m3():
    Z = latent()
    Q = np.linalg.qr(rng(5).normal(size=(R_DIM, R_DIM)))[0]
    X = Z @ Q + rng(6).normal(0, NOISE, Z.shape)
    assert _select_for(X, Z).rung == "M3"


def test_general_linear_selects_m4():
    Z = latent()
    g = rng(7)
    A = np.eye(R_DIM) + 0.5 * g.normal(size=(R_DIM, R_DIM))
    X = Z @ A + g.normal(0, NOISE, Z.shape)
    assert _select_for(X, Z).rung == "M4"


def test_exploitable_nonlinear_warp_selects_m5():
    # cubic warp: strong un-saturating nonlinearity KRR can exploit at this
    # n (measured mean errors: M5 106 vs M4 199)
    Z = latent()
    X = Z + 0.25 * Z**3 + rng(8).normal(0, NOISE, Z.shape)
    assert _select_for(X, Z).rung == "M5"


def test_weak_warp_stays_conservative_below_m5():
    # A saturating tanh warp at n~120 leaves the kernel UNDERFIT out of
    # sample (M5 42 vs M4 17 on this seed): the one-SE rule refuses the
    # kernel unless it earns selection. This mirrors the real-data finding
    # that M5's held-out share is negative, and is conservatism, not
    # blindness — estimator_notes.md N4. Battery calibration must therefore
    # verify injected warp magnitudes are KRR-exploitable at protocol n,
    # else the signature cell tests estimator power, not metric blindness.
    Z = latent()
    X = Z + 0.4 * np.tanh(2.0 * Z) + rng(8).normal(0, NOISE, Z.shape)
    sel = _select_for(X, Z)
    assert sel.rung in ("M1", "M2", "M3", "M4")


def test_selection_uses_errors_not_shares():
    # shares may be negative (overfit); selection must key on held-out error
    Z = latent()
    X = Z + rng(9).normal(0, NOISE, Z.shape)
    sel = _select_for(X, Z)
    assert set(sel.mean_errors) == set(RUNGS)
    assert sel.q_K == pytest.approx(1.0 - sel.K)


# ------------------------------------------------------------ legacy parity
def _legacy():
    root = Path(__file__).resolve().parents[1] / "legacy"
    pkg = types.ModuleType("legacypkg_t")
    pkg.__path__ = [str(root)]
    sys.modules["legacypkg_t"] = pkg
    for name in ("gpa", "ladder"):
        spec = importlib.util.spec_from_file_location(
            f"legacypkg_t.{name}", root / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"legacypkg_t.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["legacypkg_t.ladder"]


def test_fit_ladder_exact_parity_with_legacy():
    """The Phase-1 acceptance asks for parity within 1pp per rung; the port
    keeps the rung mathematics identical, so we hold it to EXACT parity on
    shared inputs (same seed, same fold generator)."""
    leg = _legacy()
    g = rng(71)
    Z = g.normal(size=(160, R_DIM))
    A = np.eye(R_DIM) + 0.3 * g.normal(size=(R_DIM, R_DIM))
    X = Z @ A + g.normal(0, 0.05, Z.shape)
    Xtr, Ztr, Xte, Zte = _split(Z, X)
    ours = fit_ladder(Xtr, Ztr, Xte, Zte, seed=3)
    theirs = leg.fit_ladder(Xtr, Ztr, Xte, Zte, seed=3)
    for k in ours["shares"]:
        assert ours["shares"][k] == pytest.approx(theirs["shares"][k],
                                                  abs=1e-12)
    for k in ours["errors"]:
        assert ours["errors"][k] == pytest.approx(theirs["errors"][k],
                                                  rel=1e-12)


def test_gpa_consensus_parity_with_legacy():
    root = Path(__file__).resolve().parents[1] / "legacy"
    spec = importlib.util.spec_from_file_location(
        "legacypkg_t.gpa", root / "gpa.py")
    theirs = importlib.util.module_from_spec(spec)
    sys.modules["legacypkg_t.gpa"] = theirs
    spec.loader.exec_module(theirs)
    g = rng(42)
    Z = g.normal(size=(80, R_DIM))
    X_by = {}
    for i, lang in enumerate(["aa", "bb", "cc"]):
        Q = np.linalg.qr(g.normal(size=(R_DIM, R_DIM)))[0]
        X_by[lang] = (1 + 0.1 * i) * Z @ Q + g.normal(0, 0.05, Z.shape)
    ours_g, theirs_g = gpa(X_by), theirs.gpa(X_by)
    assert np.allclose(ours_g["Z_train"], theirs_g["Z_train"], atol=1e-10)
    tgt_o = consensus_target(ours_g, X_by)
    tgt_t = theirs.consensus_target(theirs_g, X_by)
    assert np.allclose(tgt_o, tgt_t, atol=1e-10)


# ------------------------------------------------------- purged splits/loader
def test_purged_splits_never_straddle():
    doc_ids = [f"d{i // 16}" for i in range(320)]
    for tr, te in purged_concept_splits(doc_ids, n_splits=10, seed=13):
        assert not (set(np.array(doc_ids)[tr]) & set(np.array(doc_ids)[te]))
        assert len(tr) + len(te) == 320


def test_purged_splits_reproducible_from_seed():
    doc_ids = [f"d{i // 16}" for i in range(320)]
    a = purged_concept_splits(doc_ids, n_splits=4, seed=42)
    b = purged_concept_splits(doc_ids, n_splits=4, seed=42)
    assert len(a) == len(b)
    # index loop, not zip(strict=): local dev python is 3.9 (strict= is
    # 3.10+) while ruff targets 3.11 and flags bare zip — this form
    # satisfies both until the local env catches up to pyproject
    for i in range(len(a)):
        assert np.array_equal(a[i][0], b[i][0])


def test_loader_hard_fails_without_doc_ids(tmp_path):
    (tmp_path / "newstest2019-src.eng.txt").write_text("a\nb\n")
    with pytest.raises(FileNotFoundError, match="mandatory"):
        load_ntrex(tmp_path, tmp_path / "DOCUMENT_IDS.tsv", [], n_sents=2)


def test_loader_hard_fails_on_length_mismatch(tmp_path):
    (tmp_path / "newstest2019-src.eng.txt").write_text("a\nb\nc\n")
    (tmp_path / "DOCUMENT_IDS.tsv").write_text("d1\nd1\n")
    with pytest.raises(ValueError, match="rows"):
        load_ntrex(tmp_path, tmp_path / "DOCUMENT_IDS.tsv", [], n_sents=3)


def test_loader_reads_docs_and_languages(tmp_path):
    (tmp_path / "newstest2019-src.eng.txt").write_text("a\nb\nc\n")
    (tmp_path / "newstest2019-ref.deu.txt").write_text("x\ny\nz\n")
    (tmp_path / "DOCUMENT_IDS.tsv").write_text("d1\td\n" * 2 + "d2\td\n")
    d = load_ntrex(tmp_path, tmp_path / "DOCUMENT_IDS.tsv", ["deu"],
                   n_sents=3)
    assert d.n_documents == 2
    assert d.sentences["deu"] == ["x", "y", "z"]
