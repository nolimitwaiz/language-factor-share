#!/usr/bin/env python3
"""Campaign follow-ups that need only the dumps (prereg A2):

1. AaR tail curves AaR@{1,5,10,20} per language from the saved per-sentence
   margins at the dip layer (valid at n=1500; at n=300 AaR@1 was 3 sentences).
2. ZCA-whitened pipeline variant (motivated by battery I3 basis-dependence):
   recompute LFS and mean MEXA at the dip layer after per-layer ZCA
   whitening. A2 frozen prediction: cross-model dip-LFS ranking preserved
   (Spearman > 0.8 vs native basis); absolute values may shift.

Usage: python src/analysis/tail_whiten.py --model_tag <tag>
Outputs -> results/tailcurves/<tag>.json, results/whitened/<tag>.json
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
QS = (1, 5, 10, 20)
PIVOT = 'eng'


def tail_curves(tag, dip):
    z = np.load(os.path.join(DUMPS, tag, f'margins{dip:03d}.npz'))
    out = {}
    for lg in z.files:
        m = np.sort(z[lg])
        n = len(m)
        out[lg] = {f'aar{q}': float(m[:max(1, int(n * q / 100))].mean())
                   for q in QS}
        out[lg]['margin_mean'] = float(m.mean())
    return out


def zca_whiten(flat, eps=1e-5):
    mu = flat.mean(0)
    Xc = flat - mu
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    ev, V = np.linalg.eigh(cov)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(ev, eps))) @ V.T
    return mu, W


def whitened_metrics(tag, dip):
    z = np.load(os.path.join(DUMPS, tag, f'layer{dip:03d}.npz'))
    X = z['X'].astype(np.float32)                    # [L, N, D]
    langs = [str(x) for x in z['langs']]
    L, N, D = X.shape
    flat = X.reshape(L * N, D).astype(np.float64)
    mu, W = zca_whiten(flat)
    Xw = ((flat - mu) @ W).reshape(L, N, D).astype(np.float32)
    res = {'lfs_white': lfs(Xw)['lfs']}
    i_en = langs.index(PIVOT)
    mex = [mexa_score(Xw[i_en], Xw[i]) for i, lg in enumerate(langs)
           if lg != PIVOT]
    res['mexa_mean_white'] = float(np.mean(mex))
    # native-basis reference through the standard pipeline (z-score)
    muz, sdz = flat.mean(0), flat.std(0) + 1e-9
    Xz = ((flat - muz) / sdz).reshape(L, N, D).astype(np.float32)
    res['lfs_native'] = lfs(Xz)['lfs']
    mex_n = [mexa_score(Xz[i_en], Xz[i]) for i, lg in enumerate(langs)
             if lg != PIVOT]
    res['mexa_mean_native'] = float(np.mean(mex_n))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_tag', required=True)
    args = ap.parse_args()
    tag = args.model_tag
    meta = json.load(open(os.path.join(DUMPS, tag, 'meta.json')))
    dip = meta['dip_layer']
    rec = ledger.open_run('tail_whiten', {'model': tag, 'dip': dip})
    try:
        os.makedirs(os.path.join(ROOT, 'results', 'tailcurves'), exist_ok=True)
        os.makedirs(os.path.join(ROOT, 'results', 'whitened'), exist_ok=True)
        tc = tail_curves(tag, dip)
        with open(os.path.join(ROOT, 'results', 'tailcurves',
                               f'{tag}.json'), 'w') as f:
            json.dump({'model': tag, 'dip': dip, 'per_language': tc}, f)
        agg = {f'aar{q}': float(np.mean([v[f'aar{q}'] for v in tc.values()]))
               for q in QS}
        print(f'[tail] {tag}: ' + ' '.join(f'{k}={v:+.4f}'
                                           for k, v in agg.items()))
        wm = whitened_metrics(tag, dip)
        with open(os.path.join(ROOT, 'results', 'whitened',
                               f'{tag}.json'), 'w') as f:
            json.dump({'model': tag, 'dip': dip, **wm}, f)
        print(f"[whiten] {tag}: LFS {wm['lfs_native']:.4f}->"
              f"{wm['lfs_white']:.4f}  MEXA {wm['mexa_mean_native']:.3f}->"
              f"{wm['mexa_mean_white']:.3f}")
        ledger.close_run(rec, 'done')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
