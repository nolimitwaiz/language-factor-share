#!/usr/bin/env python3
"""THE TEETH (prereg §5, frozen 2026-07-15): partial Spearman of per-language
rotation share and M4 (linear) share at the dip layer vs transfer alpha,
controlling the standard observables (log tokens, macro-family, script,
fertility) AND per-language alignment at the LFS-selected layer, pooled over
the 3 prototype models with model fixed effects.

Pre-registered sign: NEGATIVE. p < 0.05, BH over the 2-test family
{rotation share, M4 share} at the dip layer. Either outcome publishes.

Method: rank-transform target and alpha; residualize both on controls
(model + family + script dummies, log_tokens, fertility, mexa); Spearman of
residuals (partial Spearman), language-cluster bootstrap CI.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from fertility import flores_to_ntrex  # noqa: E402

MODELS = ['Qwen3-1.7B-Base', 'OLMo-2-0425-1B', 'bloom-1b7']


def load_budget_rows():
    rows = []
    for tag in MODELS:
        d = json.load(open(os.path.join(ROOT, 'results', 'budget', tag,
                                        'budget.json')))
        for lang, v in d['per_language'].items():
            rows.append({'model': tag, 'ntrex_code': lang,
                         'rot_share': v['share_M3']['mean'],
                         'm4_share': v['share_M4']['mean']})
    return pd.DataFrame(rows)


def partial_spearman(df, target, controls_num, controls_cat, y='alpha',
                     n_boot=2000, seed=0):
    d = df.dropna(subset=[target, y] + controls_num).copy()
    rt = stats.rankdata(d[target])
    ry = stats.rankdata(d[y])
    X = pd.get_dummies(d[controls_cat], drop_first=True, dtype=float)
    for c in controls_num:
        X[c] = stats.rankdata(d[c])
    X = np.column_stack([np.ones(len(d)), X.values])
    bt, *_ = np.linalg.lstsq(X, rt, rcond=None)
    by, *_ = np.linalg.lstsq(X, ry, rcond=None)
    ut, uy = rt - X @ bt, ry - X @ by
    rho = stats.pearsonr(ut, uy)[0]
    # language-cluster bootstrap
    rng = np.random.default_rng(seed)
    langs = d.ntrex_code.values
    ul = np.unique(langs)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ul, size=len(ul), replace=True)
        idx = np.concatenate([np.where(langs == l)[0] for l in pick])
        if len(np.unique(ut[idx])) < 3:
            continue
        boots.append(stats.pearsonr(ut[idx], uy[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # analytic p (residual df adjusted for controls)
    n, k = len(d), X.shape[1]
    t = rho * np.sqrt(max(n - k - 2, 1) / max(1 - rho ** 2, 1e-12))
    p = 2 * stats.t.sf(abs(t), max(n - k - 2, 1))
    return {'target': target, 'rho_partial': round(float(rho), 4),
            'ci95': [round(float(lo), 4), round(float(hi), 4)],
            'p': float(p), 'n': int(n)}


def main():
    alphas = pd.read_csv(os.path.join(ROOT, 'results', 'deflation',
                                      'attribution', 'alphas.csv'))
    n2f = {}
    for fc in alphas.flores_code.unique():
        n2f.setdefault(flores_to_ntrex(fc), fc)
    bud = load_budget_rows()
    bud['flores_code'] = bud.ntrex_code.map(n2f)
    df = bud.merge(alphas, on=['model', 'flores_code'], how='inner')
    print(f'[teeth] merged rows: {len(df)} '
          f'({df.model.value_counts().to_dict()})')

    res = []
    for target in ('rot_share', 'm4_share'):
        r = partial_spearman(
            df, target,
            controls_num=['log_tokens', 'fertility', 'mexa'],
            controls_cat=['model', 'macro_family_g', 'script_g'])
        res.append(r)
    # BH over the 2-test frozen family
    ps = sorted((r['p'], i) for i, r in enumerate(res))
    for rank, (p, i) in enumerate(ps, 1):
        res[i]['p_bh'] = min(p * len(res) / rank, 1.0)
    for r in res:
        r['prereg_negative_and_sig'] = bool(r['rho_partial'] < 0
                                            and r['p_bh'] < 0.05)
        print(r)
    out = os.path.join(ROOT, 'results', 'budget', 'teeth.json')
    with open(out, 'w') as f:
        json.dump({'spec': 'prereg §5 frozen 2026-07-15', 'results': res},
                  f, indent=1)
    print(f'[done] -> {out}')


if __name__ == '__main__':
    main()
