#!/usr/bin/env python3
"""Real-data misalignment budget on the Phase-0 dumps (prereg §4, frozen).

Per model, at the LFS-dip layer, over ALL dumped languages:
  1. Rank selection on the rank grid {16, 32, 64}: held-out M4 error,
     one-SE rule (never on downstream numbers).
  2. Cross-fitted ladder budget: S=20 length-stratified concept splits,
     GPA consensus on train, M0-M5 per language, shares on test.
  3. Permutation nulls: B=200 for M3/M4 (M5 rung skipped in refits),
     B=50 for M5 (shared across splits); per-language null percentiles.

Emits numbers only -> results/budget/<tag>/budget.json
Usage: python src/analysis/budget_real.py --model_tag Qwen3-1.7B-Base
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from src.common import ledger  # noqa: E402
from src.geometry.gpa import consensus_target, gpa  # noqa: E402
from src.geometry.ladder import fit_ladder, permutation_null  # noqa: E402
from src.geometry.subspace import mp_rank_ceiling, pooled_pca  # noqa: E402

DUMPS = os.path.join(ROOT, 'results', 'dumps')
OUT = os.path.join(ROOT, 'results', 'budget')
NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')
RANK_GRID_SMALL = (16, 32, 64)      # n=300 prototypes
RANK_GRID_LARGE = (64, 128, 256)    # campaign n>=1000 (spec §4)
S_SPLITS = 20
B_NULL = 200
B_NULL_M5 = 50
PIVOT = 'eng'


def load_layer(tag, layer):
    z = np.load(os.path.join(DUMPS, tag, f'layer{layer:03d}.npz'),
                allow_pickle=False)
    X = z['X'].astype(np.float32)
    langs = [str(x) for x in z['langs']]
    return {lg: X[i] for i, lg in enumerate(langs)}


def length_strata(n_sents):
    """Sentence-length terciles of the English source (frozen stratifier)."""
    src = os.path.join(NTREX, 'newstest2019-src.eng.txt')
    with open(src) as f:
        lens = np.array([len(l.strip()) for l in f][:n_sents])
    q = np.quantile(lens, [1 / 3, 2 / 3])
    return np.digitize(lens, q)


def stratified_split(strata, rng):
    tr = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        idx = rng.permutation(idx)
        tr.extend(idx[:len(idx) // 2])
    tr = np.array(sorted(tr))
    te = np.setdiff1d(np.arange(len(strata)), tr)
    return tr, te


def select_rank(Zs_by_rank, langs, strata, seed=0):
    """Held-out M4 error per rank over 5 quick splits; one-SE rule."""
    rng = np.random.default_rng(seed)
    splits = [stratified_split(strata, rng) for _ in range(5)]
    stats = {}
    for r, Zs in Zs_by_rank.items():
        errs = []
        for tr_c, te_c in splits:
            g = gpa({l: Zs[l][tr_c] for l in langs})
            Z_te = consensus_target(g, {l: Zs[l][te_c] for l in langs})
            for l in langs[::4]:                     # every 4th lang: speed
                res = fit_ladder(Zs[l][tr_c], g['Z_train'], Zs[l][te_c],
                                 Z_te, fit_m5=False)
                errs.append(res['errors']['M4'] / max(res['errors']['M0'],
                                                      1e-12))
        stats[r] = (float(np.mean(errs)), float(np.std(errs) / np.sqrt(len(errs))))
    best_r = min(stats, key=lambda r: stats[r][0])
    thresh = stats[best_r][0] + stats[best_r][1]
    selected = min(r for r in stats if stats[r][0] <= thresh)
    return selected, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_tag', required=True)
    ap.add_argument('--rank_mode', default='selected',
                    choices=['selected', 'fixed'],
                    help='A2 dual-rank reporting: one-SE-selected or fixed')
    ap.add_argument('--fixed_rank', type=int, default=64)
    ap.add_argument('--subspace', default='raw', choices=['raw', 'centered'],
                    help='A2 variant: basis from raw pooled data or from '
                         'language-centered pooled data (removes offset '
                         'directions from the basis)')
    ap.add_argument('--skip_nulls', action='store_true')
    args = ap.parse_args()
    tag = args.model_tag

    meta = json.load(open(os.path.join(DUMPS, tag, 'meta.json')))
    dip = meta['dip_layer']
    E = load_layer(tag, dip)
    langs = sorted(E)
    strata = length_strata(meta['n_sents'])
    suffix = f"_{args.rank_mode}_{args.subspace}"
    out_dir = os.path.join(OUT, tag + suffix if suffix != '_selected_raw'
                           else tag)
    os.makedirs(out_dir, exist_ok=True)
    rec = ledger.open_run('budget_real', {'model': tag, 'dip': dip,
                                          'S': S_SPLITS, 'B': B_NULL,
                                          'rank_mode': args.rank_mode,
                                          'subspace': args.subspace})
    try:
        flat = np.concatenate([E[l] for l in langs], 0)
        if args.subspace == 'centered':
            # A2: basis from language-centered data; coordinates project the
            # RAW embeddings (offsets survive as coordinates, but offset
            # directions no longer dominate the basis)
            flat_basis = np.concatenate(
                [E[l] - E[l].mean(0) for l in langs], 0)
        else:
            flat_basis = flat
        rank_grid = (RANK_GRID_LARGE if meta['n_sents'] >= 1000
                     else RANK_GRID_SMALL)
        n_eff = flat_basis.shape[0]
        if args.rank_mode == 'fixed':
            pca = pooled_pca(flat_basis, args.fixed_rank)
            mu_raw = flat.mean(0)
            Zs = {l: ((E[l] - mu_raw) @ pca['basis']).astype(np.float64)
                  for l in langs}
            sel_r, rank_stats = args.fixed_rank, {}
            mp_k, mp_edge_v, mp_s2 = mp_rank_ceiling(pca['eig'], n_eff,
                                                     flat_basis.shape[1])
        else:
            Zs_by_rank = {}
            pcas = {}
            mu_raw = flat.mean(0)
            for r in rank_grid:
                pca = pooled_pca(flat_basis, r)
                pcas[r] = pca
                Zs_by_rank[r] = {l: ((E[l] - mu_raw) @ pca['basis'])
                                 .astype(np.float64) for l in langs}
            mp_k, mp_edge_v, mp_s2 = mp_rank_ceiling(
                pcas[rank_grid[-1]]['eig'], n_eff, flat_basis.shape[1])
            sel_r, rank_stats = select_rank(Zs_by_rank, langs, strata)
            Zs = Zs_by_rank[sel_r]
        print(f'[rank] {tag}{suffix}: r={sel_r} '
              f'(stats={rank_stats}, MP ceiling={mp_k})', flush=True)

        # cross-fitted budget, S=20 stratified splits
        rng = np.random.default_rng(0)
        per_lang = {l: [] for l in langs}
        splits = []
        for s in range(S_SPLITS):
            tr_c, te_c = stratified_split(strata, rng)
            splits.append((tr_c, te_c))
            g = gpa({l: Zs[l][tr_c] for l in langs})
            Z_te = consensus_target(g, {l: Zs[l][te_c] for l in langs})
            for l in langs:
                res = fit_ladder(Zs[l][tr_c], g['Z_train'], Zs[l][te_c],
                                 Z_te, seed=s)
                per_lang[l].append(res['shares'])
            print(f'[split {s + 1}/{S_SPLITS}] done', flush=True)

        # permutation nulls on split 0 (frozen: shared across splits)
        nulls = {}
        if args.skip_nulls:
            for l in langs:
                nulls[l] = {'M3_p95': None, 'M4_p95': None, 'M5_p95': None}
        tr_c, te_c = splits[0]
        g0 = gpa({l: Zs[l][tr_c] for l in langs})
        Z_te0 = consensus_target(g0, {l: Zs[l][te_c] for l in langs})
        for i, l in enumerate(langs):
            if args.skip_nulls:
                break
            n34 = permutation_null(Zs[l][tr_c], g0['Z_train'], Zs[l][te_c],
                                   Z_te0, B=B_NULL, rungs=('M3', 'M4'),
                                   seed=1000 + i)
            n5 = permutation_null(Zs[l][tr_c], g0['Z_train'], Zs[l][te_c],
                                  Z_te0, B=B_NULL_M5, rungs=('M5',),
                                  seed=2000 + i)
            nulls[l] = {
                'M3_p95': float(np.percentile(n34['M3'], 95)),
                'M4_p95': float(np.percentile(n34['M4'], 95)),
                'M5_p95': float(np.percentile(n5['M5'], 95)),
            }
            if (i + 1) % 16 == 0:
                print(f'[null] {i + 1}/{len(langs)} langs', flush=True)

        summary = {}
        for l in langs:
            keys = per_lang[l][0].keys()
            summary[l] = {k: {'mean': float(np.mean([d[k] for d in per_lang[l]])),
                              'sd': float(np.std([d[k] for d in per_lang[l]]))}
                          for k in keys}
            for rung in ('M3', 'M4', 'M5'):
                p95 = nulls[l][f'{rung}_p95']
                m = summary[l][f'share_{rung}']['mean']
                summary[l][f'share_{rung}']['exceeds_null_p95'] = \
                    (bool(m > p95) if p95 is not None else None)
                summary[l][f'share_{rung}']['null_p95'] = p95

        out = {'model': tag, 'dip_layer': dip, 'selected_rank': sel_r,
               'rank_mode': args.rank_mode, 'subspace': args.subspace,
               'rank_stats': {str(k): v for k, v in rank_stats.items()},
               'mp_ceiling': mp_k, 'mp_edge': mp_edge_v, 'mp_sigma2': mp_s2,
               'n_langs': len(langs), 'S': S_SPLITS,
               'B_null': B_NULL, 'B_null_M5': B_NULL_M5,
               'per_language': summary}
        with open(os.path.join(out_dir, 'budget.json'), 'w') as f:
            json.dump(out, f, indent=1)
        ledger.close_run(rec, 'done', {'out': out_dir,
                                       'selected_rank': sel_r})
        print(f'[done] -> {out_dir}/budget.json')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
