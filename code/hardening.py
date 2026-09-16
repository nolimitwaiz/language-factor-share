#!/usr/bin/env python3
"""
Hardening tests (advisor-mandated, all on existing data):
  1. Proper EB shrinkage of alphas (sampling-variance-based weights) —
     replaces the near-no-op placeholder.
  2. Cluster bootstrap (resample languages) of the AaR-vs-MEXA difference in
     alpha-correlations -> CI on delta. GATES the word "beats".
  3. Drop-near-floor robustness: refit regression + correlations excluding
     models with aggregate accuracy < 0.30 (BLOOMs, EuroLLM).
  4. EuroLLM own-turf probe: its accuracy on its best-aligned EU languages
     (format/capability vs alignment failure).
Outputs -> results/deflation/attribution/hardening.json + printed summary.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attribution as A

OUT = A.OUT
rng = np.random.default_rng(42)


def eb_shrink(d):
    """Proper EB: w_i = tau2 / (tau2 + s_i2), tau2 estimated per macro-family.
    s_i2 = sampling variance of alpha via binomial acc error through logit."""
    acc = d.acc.clip(0.26, 0.99)
    p = (acc - 0.25) / 0.75
    dyda = 1.0 / (0.75 * p * (1 - p))            # d logit(p)/d acc
    s2 = (acc * (1 - acc) / 300.0) * dyda ** 2   # per-row sampling variance
    d = d.assign(s2=s2.values)
    outs = []
    for fam, g in d.groupby('macro_family_g'):
        tau2 = max(g.alpha.var(ddof=1) - g.s2.mean(), 1e-4)
        w = tau2 / (tau2 + g.s2)
        mu = g.alpha.mean()
        outs.append(g.assign(alpha_eb=mu + w * (g.alpha - mu), eb_w=w))
    return pd.concat(outs)


def corr_pair(d, col='alpha_eb'):
    sub = d.dropna(subset=['mexa', 'aar10', col])
    r_a = stats.spearmanr(sub.aar10, sub[col]).statistic
    r_m = stats.spearmanr(sub.mexa, sub[col]).statistic
    return r_a, r_m, len(sub)


def cluster_bootstrap_delta(d, col='alpha_eb', reps=2000):
    langs = d.flores_code.unique()
    groups = {lg: g for lg, g in d.groupby('flores_code')}
    deltas, r_as, r_ms = [], [], []
    for _ in range(reps):
        pick = rng.choice(langs, size=len(langs), replace=True)
        boot = pd.concat([groups[lg] for lg in pick])
        sub = boot.dropna(subset=['mexa', 'aar10', col])
        if sub.aar10.nunique() < 3:
            continue
        r_a = stats.spearmanr(sub.aar10, sub[col]).statistic
        r_m = stats.spearmanr(sub.mexa, sub[col]).statistic
        deltas.append(r_a - r_m); r_as.append(r_a); r_ms.append(r_m)
    deltas = np.array(deltas)
    return {
        'delta_mean': float(deltas.mean()),
        'delta_ci95': [float(np.percentile(deltas, 2.5)),
                       float(np.percentile(deltas, 97.5))],
        'p_delta_le_0': float((deltas <= 0).mean()),
        'aar_ci95': [float(np.percentile(r_as, 2.5)), float(np.percentile(r_as, 97.5))],
        'mexa_ci95': [float(np.percentile(r_ms, 2.5)), float(np.percentile(r_ms, 97.5))],
        'reps': len(deltas),
    }


def main():
    df = A.assemble()
    m, d = A.fit(df)
    d = eb_shrink(d)
    res = {}

    # --- 1+2: EB alphas and the gated comparison ---------------------------
    r_a, r_m, n = corr_pair(d, 'alpha_eb')
    res['alpha_corr_eb'] = {'aar10': r_a, 'mexa': r_m, 'n': n}
    res['bootstrap'] = cluster_bootstrap_delta(d, 'alpha_eb')
    print(f"\n[EB] alpha-corr: AaR={r_a:.3f}  MEXA={r_m:.3f} (n={n})")
    b = res['bootstrap']
    print(f"[bootstrap] delta(AaR-MEXA) = {b['delta_mean']:+.3f} "
          f"CI95 [{b['delta_ci95'][0]:+.3f}, {b['delta_ci95'][1]:+.3f}] "
          f"P(delta<=0) = {b['p_delta_le_0']:.3f}  ({b['reps']} reps)")

    # --- 3: drop near-floor models ------------------------------------------
    agg = d.groupby('model').acc.mean()
    floor_models = list(agg[agg < 0.30].index)
    d8 = d[~d.model.isin(floor_models)]
    df8 = df[~df.model.isin(floor_models)]
    m8, dd8 = A.fit(df8)
    dd8 = eb_shrink(dd8)
    r_a8, r_m8, n8 = corr_pair(dd8, 'alpha_eb')
    res['drop_floor'] = {
        'dropped': floor_models,
        'beta_tokens': float(m8.params['log_tokens']),
        'beta_tokens_p': float(m8.pvalues['log_tokens']),
        'alpha_corr': {'aar10': r_a8, 'mexa': r_m8, 'n': n8},
        'bootstrap': cluster_bootstrap_delta(dd8, 'alpha_eb'),
    }
    print(f"\n[drop-floor] dropped {floor_models}")
    print(f"  beta_tokens = {m8.params['log_tokens']:+.4f} (p={m8.pvalues['log_tokens']:.2e})")
    print(f"  alpha-corr: AaR={r_a8:.3f} MEXA={r_m8:.3f}")
    b8 = res['drop_floor']['bootstrap']
    print(f"  delta CI95 [{b8['delta_ci95'][0]:+.3f}, {b8['delta_ci95'][1]:+.3f}] "
          f"P(delta<=0)={b8['p_delta_le_0']:.3f}")

    # --- 4: EuroLLM own-turf probe ------------------------------------------
    eu_langs = ['deu_Latn', 'fra_Latn', 'spa_Latn', 'por_Latn', 'ita_Latn',
                'nld_Latn', 'pol_Latn', 'ron_Latn', 'ces_Latn', 'swe_Latn']
    probe = {}
    for tag in ('EuroLLM-1.7B', 'bloom-1b7', 'bloom-7b1', 'Qwen3-1.7B-Base'):
        g = df[(df.model == tag) & (df.flores_code.isin(eu_langs))]
        probe[tag] = {'eu_mean_acc': float(g.acc.mean()), 'n': len(g)}
    res['own_turf_probe'] = probe
    print("\n[own-turf] accuracy on 10 high-resource EU languages:")
    for k, v in probe.items():
        print(f"  {k:18s} {v['eu_mean_acc']:.3f} (n={v['n']})")

    with open(os.path.join(OUT, 'hardening.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f"\n[done] -> {OUT}/hardening.json")


if __name__ == '__main__':
    main()
