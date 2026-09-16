#!/usr/bin/env python3
"""Two estimator checks that the existing battery does not cover.

Both operate on saved campaign representations and change nothing in the
metric definition. They test claims the report makes rather than measuring
new models.

1. MONOLINGUAL NULL. One language's sentences are split into disjoint
   halves and relabelled as two pseudo-languages, with concept pairing
   assigned arbitrarily. There is no true language difference and no true
   concept correspondence, so any LFS the estimator reports is finite
   sample noise. Compared against a real two-language measurement at
   matched L, N, and D, this gives the noise floor for the reported
   fingerprint values. Repeated at L=12 to match the language-count regime
   of the real grid.

   Claim under test: the measured language share reflects language
   structure rather than an artifact of the estimator.

2. SIZE INVARIANCE. LFS is a variance share and should not move with the
   number of evaluation sentences; MEXA is retrieval based and is expected
   to fall as the candidate set grows. The report states this
   (Figure 5) from a single model at two sizes. Here it is measured on the
   campaign dumps at N in {100, 300, 1000} on a fixed language set.

Usage:
    python -m src.analysis.estimator_nulls --model_tag Qwen3-1.7B-Base
Outputs: results/estimator_nulls/<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from pilot_metrics import lfs, mexa_score  # noqa: E402
from src.common import ledger  # noqa: E402

DUMPS = os.path.join(ROOT, 'results', 'dumps')
OUT = os.path.join(ROOT, 'results', 'estimator_nulls')
PIVOT = 'eng'
N_REPEATS = 20          # independent random splits/subsamples per condition


def load_layer(tag, layer):
    z = np.load(os.path.join(DUMPS, tag, f'layer{layer:03d}.npz'),
                allow_pickle=False)
    X = z['X'].astype(np.float32)                     # [L, N, D]
    langs = [str(x) for x in z['langs']]
    return X, langs


def lfs_of(block):
    """Standard pipeline on a [L, N, D] block: per-dimension z-score across
    all (language, sentence) rows, then the two-way decomposition. Identical
    to the measurement path in cluster_grid.py."""
    L, N, D = block.shape
    flat = block.reshape(L * N, D)
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    return lfs(((block - mu) / sd))['lfs']


def monolingual_null(X, langs, n_pseudo, n_sents, rng, source=PIVOT):
    """Split one language's sentences into `n_pseudo` disjoint groups and
    treat them as languages, with concept index assigned arbitrarily.
    No true language difference, no true concept correspondence."""
    i_src = langs.index(source)
    rows = X[i_src]                                   # [N_total, D]
    need = n_pseudo * n_sents
    if len(rows) < need:
        return None
    perm = rng.permutation(len(rows))[:need]
    block = rows[perm].reshape(n_pseudo, n_sents, -1)
    return lfs_of(block)


def real_matched(X, langs, n_pseudo, n_sents, rng):
    """Real languages at matched L, N, D: the comparison condition."""
    choices = [i for i, l in enumerate(langs) if l != PIVOT]
    if len(choices) + 1 < n_pseudo:
        return None
    idx = [langs.index(PIVOT)] + list(rng.choice(choices, n_pseudo - 1,
                                                 replace=False))
    sent = rng.permutation(X.shape[1])[:n_sents]
    return lfs_of(X[np.ix_(idx, sent)])


def size_invariance(X, langs, sizes, rng, n_langs=12):
    """LFS and mean MEXA on a FIXED language set at several evaluation
    sizes. Sentences at each size are drawn from one shuffled order so the
    smaller sets are nested inside the larger ones."""
    choices = [i for i, l in enumerate(langs) if l != PIVOT]
    idx = [langs.index(PIVOT)] + list(rng.choice(choices, n_langs - 1,
                                                 replace=False))
    order = rng.permutation(X.shape[1])
    out = {}
    for n in sizes:
        if n > len(order):
            continue
        sent = order[:n]
        block = X[np.ix_(idx, sent)]
        L, N, D = block.shape
        flat = block.reshape(L * N, D)
        mu, sd = flat.mean(0), flat.std(0) + 1e-9
        bz = (block - mu) / sd
        mex = [mexa_score(bz[0], bz[j]) for j in range(1, L)]
        out[str(n)] = {'lfs': float(lfs(bz)['lfs']),
                       'mexa_mean': float(np.mean(mex))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    tag = args.model_tag

    meta = json.load(open(os.path.join(DUMPS, tag, 'meta.json')))
    dip = meta['dip_layer']
    os.makedirs(OUT, exist_ok=True)
    rec = ledger.open_run('estimator_nulls', {'model': tag, 'dip': dip,
                                              'repeats': N_REPEATS},
                          seed=args.seed)
    try:
        X, langs = load_layer(tag, dip)
        n_total = X.shape[1]
        print(f'[{tag}] dip L{dip}, {len(langs)} languages, '
              f'{n_total} sentences', flush=True)
        res = {'model': tag, 'dip_layer': dip, 'n_sents_available': n_total,
               'n_languages': len(langs), 'repeats': N_REPEATS,
               'seed': args.seed, 'monolingual_null': {},
               'size_invariance': {}}

        # ---- test 1: monolingual null vs real, at two language counts ----
        for n_pseudo, n_sents in ((2, 700), (12, 120)):
            rng = np.random.default_rng(args.seed)
            null_vals, real_vals = [], []
            for _ in range(N_REPEATS):
                v = monolingual_null(X, langs, n_pseudo, n_sents, rng)
                if v is not None:
                    null_vals.append(v)
                r = real_matched(X, langs, n_pseudo, n_sents, rng)
                if r is not None:
                    real_vals.append(r)
            key = f'L{n_pseudo}_N{n_sents}'
            res['monolingual_null'][key] = {
                'null_mean': float(np.mean(null_vals)),
                'null_sd': float(np.std(null_vals)),
                'null_max': float(np.max(null_vals)),
                'real_mean': float(np.mean(real_vals)),
                'real_sd': float(np.std(real_vals)),
                'real_min': float(np.min(real_vals)),
                'separation': float(np.min(real_vals) - np.max(null_vals)),
            }
            d = res['monolingual_null'][key]
            print(f'  monolingual null {key}: null {d["null_mean"]:.4f} '
                  f'+-{d["null_sd"]:.4f} (max {d["null_max"]:.4f}) vs real '
                  f'{d["real_mean"]:.4f} +-{d["real_sd"]:.4f} '
                  f'(min {d["real_min"]:.4f}); separation '
                  f'{d["separation"]:+.4f}', flush=True)

        # ---- test 2: size invariance ----
        rng = np.random.default_rng(args.seed + 1000)
        runs = [size_invariance(X, langs, [100, 300, 1000], rng)
                for _ in range(N_REPEATS)]
        for n in ('100', '300', '1000'):
            if n in runs[0]:
                lf = [r[n]['lfs'] for r in runs]
                mx = [r[n]['mexa_mean'] for r in runs]
                res['size_invariance'][n] = {
                    'lfs_mean': float(np.mean(lf)), 'lfs_sd': float(np.std(lf)),
                    'mexa_mean': float(np.mean(mx)), 'mexa_sd': float(np.std(mx))}
                print(f'  N={n:>4}: LFS {np.mean(lf):.4f} +-{np.std(lf):.4f} | '
                      f'MEXA {np.mean(mx):.4f} +-{np.std(mx):.4f}', flush=True)
        s = res['size_invariance']
        if '100' in s and '1000' in s:
            res['lfs_drift_100_to_1000'] = s['1000']['lfs_mean'] - s['100']['lfs_mean']
            res['mexa_drift_100_to_1000'] = s['1000']['mexa_mean'] - s['100']['mexa_mean']
            print(f'  drift 100->1000: LFS {res["lfs_drift_100_to_1000"]:+.4f}, '
                  f'MEXA {res["mexa_drift_100_to_1000"]:+.4f}', flush=True)

        with open(os.path.join(OUT, f'{tag}.json'), 'w') as f:
            json.dump(res, f, indent=1)
        ledger.close_run(rec, 'done', {'out': os.path.join(OUT, f'{tag}.json')})
        print(f'[done] -> {OUT}/{tag}.json')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
