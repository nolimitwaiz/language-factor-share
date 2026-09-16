#!/usr/bin/env python3
"""Shrinkage diagnostics (advisor round 3):
  - tau2 and weight distribution per macro-family (is EB collapsing to means?)
  - within-family correlation (family-demean metric AND raw alpha — the
    stringent check shrinkage cannot flatter)
  - disattenuated correlation vs raw alphas (reliability route, no family means)
  - Delta(AaR-MEXA) under every scheme (is the comparison stable?)
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attribution as A

rng = np.random.default_rng(7)

df = A.assemble()
m, d = A.fit(df)
acc = d.acc.clip(0.26, 0.99)
p = (acc - 0.25) / 0.75
dyda = 1.0 / (0.75 * p * (1 - p))
d['s2'] = ((acc * (1 - acc) / 300.0) * dyda ** 2).values
d = d.dropna(subset=['mexa', 'aar10', 'alpha'])

# ---- 1. tau2 / weights ------------------------------------------------------
rows = []
for fam, g in d.groupby('macro_family_g'):
    tau2 = g.alpha.var(ddof=1) - g.s2.mean()
    w = np.maximum(tau2, 1e-4) / (np.maximum(tau2, 1e-4) + g.s2)
    rows.append({'family': fam, 'n': len(g), 'var_alpha': g.alpha.var(ddof=1),
                 'mean_s2': g.s2.mean(), 'median_s2': g.s2.median(),
                 'tau2_raw': tau2, 'median_w': float(np.median(w))})
fam_tab = pd.DataFrame(rows)
print('=== tau2 / weight table (per macro-family) ===')
print(fam_tab.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
print(f"\nOVERALL median weight w: "
      f"{fam_tab.median_w.median():.4f}  (advisor threshold: <0.1 = collapse)")

# ---- 2. correlations under each scheme --------------------------------------
def spear(a, b):
    return float(stats.spearmanr(a, b).statistic)

schemes = {}
schemes['raw_alpha'] = {'aar10': spear(d.aar10, d.alpha),
                        'mexa': spear(d.mexa, d.alpha)}
# within-family (demean both metric and RAW alpha by macro-family)
dm = d.copy()
for col in ('alpha', 'mexa', 'aar10'):
    dm[col + '_dm'] = dm[col] - dm.groupby('macro_family_g')[col].transform('mean')
schemes['within_family'] = {'aar10': spear(dm.aar10_dm, dm.alpha_dm),
                            'mexa': spear(dm.mexa_dm, dm.alpha_dm)}
# disattenuated (global reliability; mean- and median-s2 variants)
var_a = d.alpha.var(ddof=1)
for name, sbar in (('mean', d.s2.mean()), ('median', d.s2.median())):
    tau2_g = max(var_a - sbar, 1e-6)
    rel = tau2_g / (tau2_g + sbar)
    schemes[f'disattenuated_{name}'] = {
        'reliability': rel,
        'aar10': min(schemes['raw_alpha']['aar10'] / np.sqrt(rel), 1.0),
        'mexa': min(schemes['raw_alpha']['mexa'] / np.sqrt(rel), 1.0)}
print('\n=== correlations under each scheme ===')
for k, v in schemes.items():
    extra = f" (reliability={v['reliability']:.3f})" if 'reliability' in v else ''
    print(f"{k:22s} AaR={v['aar10']:.3f}  MEXA={v['mexa']:.3f}  "
          f"delta={v['aar10']-v['mexa']:+.3f}{extra}")

# ---- 3. bootstrap Delta under raw and within-family schemes -----------------
langs = d.flores_code.unique()
groups = {lg: g for lg, g in dm.groupby('flores_code')}
out = {}
for scheme, (ca, cm, ct) in {'raw_alpha': ('aar10', 'mexa', 'alpha'),
                             'within_family': ('aar10_dm', 'mexa_dm', 'alpha_dm')}.items():
    deltas = []
    for _ in range(2000):
        pick = rng.choice(langs, size=len(langs), replace=True)
        boot = pd.concat([groups[lg] for lg in pick])
        deltas.append(spear(boot[ca], boot[ct]) - spear(boot[cm], boot[ct]))
    deltas = np.array(deltas)
    out[scheme] = {'delta_mean': float(deltas.mean()),
                   'ci95': [float(np.percentile(deltas, 2.5)),
                            float(np.percentile(deltas, 97.5))],
                   'p_le_0': float((deltas <= 0).mean())}
    print(f"[bootstrap {scheme}] delta={out[scheme]['delta_mean']:+.3f} "
          f"CI95 [{out[scheme]['ci95'][0]:+.3f},{out[scheme]['ci95'][1]:+.3f}] "
          f"P(<=0)={out[scheme]['p_le_0']:.3f}")

res = {'family_table': fam_tab.to_dict('records'), 'schemes': schemes,
       'bootstrap': out}
with open(os.path.join(A.OUT, 'shrinkage_diagnostics.json'), 'w') as f:
    json.dump(res, f, indent=1, default=float)
print(f"\n[done] -> {A.OUT}/shrinkage_diagnostics.json")
