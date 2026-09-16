#!/usr/bin/env python3
"""Battery runner: apply the frozen injection set to Phase-0 dumps and score
every instrument per cell. Emits numbers only — judgment happens against
prereg/PREREG_BATTERY.md (FROZEN 2026-07-15).

Cell tiers (frozen):
  PANEL cells: curated tier panel + eng; full instruments incl. ladder
               budget (r=32, S=5 splits).
  FULL cells:  all languages, one magnitude per injection; pipeline metrics
               only.
Layers: model's LFS-dip layer (primary); I3 + I10 also at L0 and final.

Usage: python src/battery/score.py --model_tag Qwen3-1.7B-Base
Outputs -> results/battery/<tag>/cells.jsonl (+ calibration.json)
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from pilot_metrics import aar, lfs, mexa_score  # noqa: E402
from cluster_grid import TIER  # noqa: E402
from src.battery import inject  # noqa: E402
from src.common import ledger  # noqa: E402
from src.geometry.gpa import (consensus_target, gpa, orthogonal_procrustes,  # noqa: E402
                              relative_rotation_angles, similarity_procrustes)
from src.geometry.ladder import cka_linear, cka_rbf, fit_ladder  # noqa: E402
from src.geometry.subspace import pooled_pca  # noqa: E402

DUMPS = os.path.join(ROOT, 'results', 'dumps')
OUT = os.path.join(ROOT, 'results', 'battery')
R_BATTERY = 32          # frozen: battery cells run at the grid midpoint rank
S_SPLITS = 5            # frozen: battery cells use S=5 splits
PIVOT = 'eng'


def cell_seed(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def load_layer(tag, layer):
    z = np.load(os.path.join(DUMPS, tag, f'layer{layer:03d}.npz'),
                allow_pickle=False)
    X = z['X'].astype(np.float32)                  # [L, N, D]
    langs = [str(x) for x in z['langs']]
    return {lg: X[i] for i, lg in enumerate(langs)}


# ---------------------------------------------------------------- pipeline

def zscore(E):
    allE = np.concatenate(list(E.values()), 0)
    mu, sd = allE.mean(0), allE.std(0) + 1e-9
    return {lg: (v - mu) / sd for lg, v in E.items()}


def pipeline_metrics(E):
    """Standard pipeline on a dict of pre-z embeddings: z-score -> LFS,
    mean MEXA, mean AaR@10."""
    Ez = zscore(E)
    langs = list(Ez)
    X = np.stack([Ez[lg] for lg in langs], 0)
    out = {'lfs': lfs(X)['lfs']}
    mex, a10 = [], []
    for lg in langs:
        if lg == PIVOT:
            continue
        mex.append(mexa_score(Ez[PIVOT], Ez[lg]))
        a, _ = aar(Ez[PIVOT], Ez[lg])
        a10.append(a)
    out['mexa_mean'] = float(np.mean(mex))
    out['aar10_mean'] = float(np.mean(a10))
    return out


def cvp(E, E0, targets):
    """Concept-variance preservation: trace of per-language centered
    covariance, injected / baseline, averaged over injected languages."""
    r = [float(E[l].var(0).sum() / (E0[l].var(0).sum() + 1e-12))
         for l in targets]
    return float(np.mean(r))


def effective_rank(E, targets):
    """Mean effective rank of per-language centered clouds."""
    out = []
    for l in targets:
        Xc = E[l] - E[l].mean(0)
        s = np.linalg.svd(Xc, compute_uv=False)
        p = s / s.sum()
        p = p[p > 1e-12]
        out.append(float(np.exp(-(p * np.log(p)).sum())))
    return float(np.mean(out))


def probe_accuracy(E, targets, seed=0):
    """Linear language probe on r=32 subspace coords, concept-grouped split
    (train on half the concepts, test on the other half)."""
    from sklearn.linear_model import LogisticRegression
    sub = {l: E[l] for l in targets}
    flat = np.concatenate(list(sub.values()), 0)
    pca = pooled_pca(flat, R_BATTERY)
    N = len(next(iter(sub.values())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    tr_c, te_c = perm[:N // 2], perm[N // 2:]
    Xtr, ytr, Xte, yte = [], [], [], []
    for i, l in enumerate(targets):
        Zc = (sub[l] - pca['mean']) @ pca['basis']
        Xtr.append(Zc[tr_c]); ytr += [i] * len(tr_c)
        Xte.append(Zc[te_c]); yte += [i] * len(te_c)
    clf = LogisticRegression(max_iter=500).fit(np.concatenate(Xtr), ytr)
    return float(clf.score(np.concatenate(Xte), yte))


def alignment_gap(E):
    """DIAGNOSTIC ONLY (never 'rotation share'): raw LFS minus LFS after
    per-language orthogonal Procrustes of each language's centered cloud to
    the pivot's centered cloud (centroids preserved)."""
    Ez = zscore(E)
    langs = list(Ez)
    raw = lfs(np.stack([Ez[lg] for lg in langs], 0))['lfs']
    piv_c = Ez[PIVOT] - Ez[PIVOT].mean(0)
    aligned = {}
    for lg in langs:
        if lg == PIVOT:
            aligned[lg] = Ez[lg]
            continue
        c = Ez[lg].mean(0)
        R = orthogonal_procrustes(Ez[lg] - c, piv_c)
        aligned[lg] = (Ez[lg] - c) @ R + c
    al = lfs(np.stack([aligned[lg] for lg in langs], 0))['lfs']
    return {'lfs_raw': float(raw), 'lfs_aligned': float(al),
            'alignment_gap': float(raw - al)}


def cka_gap(E, targets):
    """Mean (linear CKA - RBF CKA) of each injected language vs the pivot."""
    gaps = []
    for l in targets:
        gaps.append(cka_linear(E[l], E[PIVOT]) - cka_rbf(E[l], E[PIVOT]))
    return float(np.mean(gaps))


# ------------------------------------------------------------------ budget

def budget_panel(E, seed=0, n_splits=S_SPLITS, r=R_BATTERY):
    """Ladder budget over panel languages: pooled PCA (on the injected data
    — end-to-end), GPA consensus on train concepts, M0-M5 per language,
    S splits. Returns mean shares across languages and splits."""
    langs = sorted(E)
    flat = np.concatenate([E[l] for l in langs], 0)
    pca = pooled_pca(flat, r)
    Zs = {l: ((E[l] - pca['mean']) @ pca['basis']).astype(np.float64)
          for l in langs}
    N = len(next(iter(Zs.values())))
    rng = np.random.default_rng(seed)
    acc = {}
    for s in range(n_splits):
        perm = rng.permutation(N)
        tr_c, te_c = perm[:N // 2], perm[N // 2:]
        g = gpa({l: Zs[l][tr_c] for l in langs})
        Z_te = consensus_target(g, {l: Zs[l][te_c] for l in langs})
        Z_tr = g['Z_train']
        for l in langs:
            res = fit_ladder(Zs[l][tr_c], Z_tr, Zs[l][te_c], Z_te,
                             seed=seed + s)
            for k, v in res['shares'].items():
                acc.setdefault(k, []).append(v)
    return {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v))}
            for k, v in acc.items()}


# ------------------------------------------------------------- calibration

def recover_theta(E0, E1, basis, mean, targets, center=False):
    """I4/I4b calibration: Procrustes on injection-basis coords, baseline ->
    injected; mean plane angle per language. center=True (I4b) centers each
    cloud first — required because I4b rotates about the language's own
    centroid, so the recovery must too."""
    out = {}
    for l in targets:
        Z0 = (E0[l].astype(np.float64) - mean) @ basis
        Z1 = (E1[l].astype(np.float64) - mean) @ basis
        if center:
            Z0, Z1 = Z0 - Z0.mean(0), Z1 - Z1.mean(0)
        R = orthogonal_procrustes(Z0, Z1)
        ang, _ = relative_rotation_angles(R, np.eye(basis.shape[1]))
        planes = ang[ang > 0.02]
        out[l] = float(np.median(planes)) if len(planes) else 0.0
    return out


def recover_scale(E0, E1, targets):
    """I2 calibration: similarity-Procrustes scale, baseline -> injected."""
    return {l: float(similarity_procrustes(E0[l].astype(np.float64),
                                           E1[l].astype(np.float64))[0])
            for l in targets}


# ------------------------------------------------------------------- cells

def build_cells():
    """(injection, magnitude, full-tier magnitude marker) per prereg §2 +
    Addendum A1 (2026-07-16): I4b, I8b."""
    return [
        ('I1', [0.25, 0.5, 1.0], 0.5),
        ('I2', [0.25, 0.5, 1.0], 0.5),
        ('I3', [None], None),                      # handled separately: full-D
        ('I4', [0.1, 0.2, 0.4, 0.8], 0.4),
        ('I4b', [0.1, 0.2, 0.4, 0.8], 0.4),        # A1: centroid-preserving
        ('I5', [0.25, 0.5], 0.5),
        ('I6', [0.25, 0.5], 0.5),
        ('I7', [0.5, 1.0], 1.0),
        ('I8', [0.3, 0.6, 0.9], 0.6),
        ('I8b', [0.3, 0.6, 0.9], 0.6),             # A1: global-centroid collapse
        ('I9', [None], None),
        ('I10', [None, None], None),               # twice (noise floor)
    ]


def apply_injection(inj, E, mag, rng, basis=None, mean=None):
    if inj == 'I1':
        return inject.i1_offset(E, mag, rng)
    if inj == 'I2':
        return inject.i2_scale(E, mag, rng)
    if inj == 'I3':
        return inject.i3_global_rotation(E, rng)
    if inj == 'I4':
        return inject.i4_lang_rotation(E, mag, R_BATTERY, basis, mean, rng)
    if inj == 'I4b':
        return inject.i4b_lang_rotation_centered(E, mag, R_BATTERY, basis, rng)
    if inj == 'I8b':
        return inject.i8b_global_collapse(E, mag)
    if inj == 'I5':
        return inject.i5_shear(E, mag, R_BATTERY, basis, mean, rng)
    if inj == 'I6':
        return inject.i6_interaction(E, mag, 4, R_BATTERY, basis, mean, rng)
    if inj == 'I7':
        return inject.i7_warp(E, mag, R_BATTERY, basis, mean, rng)
    if inj == 'I8':
        return inject.i8_collapse(E, mag)
    if inj == 'I9':
        return inject.i9_permute(E, rng)
    if inj == 'I10':
        return inject.i10_identity(E)
    raise ValueError(inj)


def score_cell(tag, layer, inj, mag, rep, E0, panel, basis, mean,
               run_budget):
    seed = cell_seed(tag, layer, inj, mag, rep)
    rng = np.random.default_rng(seed)
    src = {l: E0[l] for l in panel} if run_budget else E0
    # I10 noise floor: independently resampled sentence halves
    if inj == 'I10':
        N = len(next(iter(src.values())))
        half = np.random.default_rng(seed + rep).permutation(N)[:N // 2]
        src = {l: v[half] for l, v in src.items()}
    E1, meta = apply_injection(inj, src, mag, rng, basis, mean)
    targets = [l for l in src if l != PIVOT]

    row = {'model': tag, 'layer': layer, 'injection': inj, 'magnitude': mag,
           'rep': rep, 'seed': seed, 'tier': 'panel' if run_budget else 'full',
           **pipeline_metrics(E1),
           'cvp': cvp(E1, src, targets),
           'eff_rank': effective_rank(E1, targets)}
    if run_budget:
        row.update(alignment_gap(E1))
        row['probe_acc'] = probe_accuracy(E1, targets, seed)
        row['cka_gap'] = cka_gap(E1, targets)
        row['budget'] = budget_panel(E1, seed=seed)
        if inj in ('I4', 'I4b'):
            rec = recover_theta(src, E1, basis, mean, targets,
                                center=(inj == 'I4b'))
            row['recovered_theta'] = {'mean': float(np.mean(list(rec.values()))),
                                      'per_lang': rec}
        if inj == 'I2':
            rec = recover_scale(src, E1, targets)
            err = [abs(rec[l] - meta['scales'][l]) for l in rec]
            row['scale_recovery_abs_err'] = float(np.mean(err))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_tag', required=True)
    ap.add_argument('--skip_full', action='store_true')
    ap.add_argument('--cells', default=None,
                    help='comma-separated injection subset (patch runs); '
                         'output goes to cells_<suffix>.jsonl')
    ap.add_argument('--out_suffix', default=None)
    args = ap.parse_args()
    tag = args.model_tag
    only = set(args.cells.split(',')) if args.cells else None

    meta = json.load(open(os.path.join(DUMPS, tag, 'meta.json')))
    dip = meta['dip_layer']
    layers = meta['layers']
    out_dir = os.path.join(OUT, tag)
    os.makedirs(out_dir, exist_ok=True)
    rec = ledger.open_run('battery', {'model': tag, 'dip': dip,
                                      'r': R_BATTERY, 'S': S_SPLITS})
    rows = []
    try:
        E0 = load_layer(tag, dip)
        panel = [PIVOT] + sorted(l for l in TIER if l in E0)
        print(f'[battery] {tag} dip=L{dip} panel={len(panel)} langs '
              f'(of {len(E0)})', flush=True)
        # injection basis from UNINJECTED panel data (frozen)
        flat = np.concatenate([E0[l] for l in panel], 0)
        pca = pooled_pca(flat, R_BATTERY)
        basis, mean = pca['basis'], pca['mean']
        pca_full = None

        # baseline reference cell (panel + full)
        for run_budget in (True, False):
            src = {l: E0[l] for l in panel} if run_budget else E0
            row = {'model': tag, 'layer': dip, 'injection': 'I0-baseline',
                   'magnitude': None, 'rep': 0,
                   'tier': 'panel' if run_budget else 'full',
                   **pipeline_metrics(src),
                   'cvp': 1.0,
                   'eff_rank': effective_rank(src,
                                              [l for l in src if l != PIVOT])}
            if run_budget:
                row.update(alignment_gap(src))
                row['probe_acc'] = probe_accuracy(
                    src, [l for l in src if l != PIVOT])
                row['budget'] = budget_panel(src)
            rows.append(row)
            print(f"[cell] I0-baseline {row['tier']} lfs={row['lfs']:.4f}",
                  flush=True)

        for inj, mags, full_mag in build_cells():
            if only and inj not in only:
                continue
            for rep, mag in enumerate(mags):
                rows.append(score_cell(tag, dip, inj, mag, rep, E0, panel,
                                       basis, mean, run_budget=True))
                print(f'[cell] {inj} mag={mag} panel done', flush=True)
            if not args.skip_full and full_mag is not None or inj in ('I3', 'I9', 'I10'):
                if not args.skip_full:
                    if inj in ('I4', 'I5', 'I6', 'I7') and pca_full is None:
                        pca_full = pooled_pca(
                            np.concatenate(list(E0.values()), 0), R_BATTERY)
                    b, m = ((pca_full['basis'], pca_full['mean'])
                            if pca_full is not None else (basis, mean))
                    rows.append(score_cell(tag, dip, inj, full_mag, 99, E0,
                                           panel, b, m, run_budget=False))
                    print(f'[cell] {inj} mag={full_mag} full done', flush=True)

        # I3 + I10 pipeline-invariance checks at L0 and final layer
        if only is None:
            for extra_layer in (layers[0], layers[-1]):
                if extra_layer == dip:
                    continue
                E_x = load_layer(tag, extra_layer)
                for inj in ('I3', 'I10'):
                    rows.append(score_cell(tag, extra_layer, inj, None, 0,
                                           E_x, panel, None, None,
                                           run_budget=False))
                print(f'[layer {extra_layer}] I3/I10 invariance cells done',
                      flush=True)

        fname = (f'cells_{args.out_suffix}.jsonl' if args.out_suffix
                 else 'cells.jsonl')
        with open(os.path.join(out_dir, fname), 'w') as f:
            for r_ in rows:
                f.write(json.dumps(r_) + '\n')
        ledger.close_run(rec, 'done', {'n_cells': len(rows),
                                       'out': out_dir})
        print(f'[done] {len(rows)} cells -> {out_dir}/cells.jsonl')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
