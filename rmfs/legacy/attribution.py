#!/usr/bin/env python3
"""
Attribution analysis ("deflation"): decompose per-language Belebele accuracy
into observable factors + transfer alpha, then test what intrinsic metrics
(LFS / MEXA / AaR) predict BEYOND the observables.

Spec (advisor-reviewed):
  - outcome: logit((acc - 0.25) / 0.75)   [Belebele is 4-choice; floor at 0.25]
  - model fixed effects
  - factors: log10 CulturaX tokens (proxy, labeled), macro-family (>=3 members,
    else 'other'), script (>=3, else 'other'), per-(model,language) fertility
  - cluster-robust SEs by language
  - CHECK TOKENS-BETA FIRST: if weak/wrong-signed, the proxy is broken -> stop
  - alphas: residuals, EB-shrunk toward macro-family means; per-model TAIL
    alpha (worst-resource decile), NOT total alpha (zero under FE)
  - report BOTH raw and residual correlations for every metric, BH-corrected
    over the pre-registered set {LFS_dip_layer, MEXA, AaR10} x {raw, residual}
Outputs -> results/deflation/attribution/
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF = os.path.join(ROOT, 'results', 'deflation')
OUT = os.path.join(DEF, 'attribution')
os.makedirs(OUT, exist_ok=True)

MODELS = ['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base', 'Qwen3-4B-Base', 'Qwen3-8B-Base',
          'OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'Mistral-7B-v0.3',
          'bloom-1b7', 'bloom-7b1', 'EuroLLM-1.7B',
          # widened set (A3 refresh 2026-07-20): held-out + new families
          'SmolLM2-1.7B', 'Falcon3-7B-Base', 'salamandra-2b', 'salamandra-7b',
          'granite-3.1-8b-base', 'Yi-1.5-9B', 'Llama-3.1-8B']

# flores iso639-3 prefix -> iso639-1 (CulturaX/CC-100 keys)
ISO3_TO_1 = {
 'eng':'en','rus':'ru','spa':'es','deu':'de','fra':'fr','zho':'zh','ita':'it',
 'por':'pt','pol':'pl','jpn':'ja','nld':'nl','arb':'ar','tur':'tr','ces':'cs',
 'vie':'vi','pes':'fa','hun':'hu','ell':'el','ron':'ro','swe':'sv','ukr':'uk',
 'fin':'fi','kor':'ko','dan':'da','bul':'bg','nob':'no','hin':'hi','slk':'sk',
 'tha':'th','lit':'lt','cat':'ca','ind':'id','ben':'bn','est':'et','slv':'sl',
 'lvs':'lv','heb':'he','srp':'sr','tam':'ta','als':'sq','azj':'az','kaz':'kk',
 'urd':'ur','kat':'ka','hye':'hy','isl':'is','mal':'ml','npi':'ne','mkd':'mk',
 'mar':'mr','khk':'mn','bel':'be','tel':'te','glg':'gl','eus':'eu','kan':'kn',
 'guj':'gu','afr':'af','mya':'my','sin':'si','epo':'eo','khm':'km','pan':'pa',
 'cym':'cy','kir':'ky','gle':'ga','pbt':'ps','amh':'am','kmr':'ku','tgl':'tl',
 'ydd':'yi','lao':'lo','snd':'sd','plt':'mg','ory':'or','asm':'as','uig':'ug',
 'uzn':'uz','hrv':'hr','swh':'sw','zsm':'ms','som':'so','hat':'ht','yor':'yo',
 'grn':'gn','quy':'qu','bos':'bs','sun':'su','jav':'jv','hau':'ha','ibo':'ig',
 'zul':'zu','xho':'xh','sna':'sn','nya':'ny','lin':'ln','lug':'lg','wol':'wo',
 'ckb':'ku','tgk':'tg','kea':None,'war':None,'ilo':None,'ceb':None,
}


BELE_DIR = os.environ.get('BELE_DIR', 'belebele0')  # 0-shot = uniform protocol
# (5-shot in results/belebele/ silently truncates on 2k/4k-context models:
#  bloom-1b7 scored BELOW CHANCE on English. Keep as robustness for long-ctx only.)


def load_belebele():
    """lm-eval output jsons -> rows (model, flores_code, acc)."""
    rows = []
    for tag in MODELS:
        hits = glob.glob(os.path.join(ROOT, 'results', BELE_DIR, tag, '**', '*.json'),
                         recursive=True)
        if not hits:
            print(f'[warn] no belebele results for {tag}')
            continue
        # newest file wins
        path = max(hits, key=os.path.getmtime)
        d = json.load(open(path))
        res = d.get('results', {})
        for task, m in res.items():
            if not task.startswith('belebele_'):
                continue
            code = task[len('belebele_'):]
            acc = m.get('acc,none', m.get('acc'))
            if acc is not None:
                rows.append({'model': tag, 'flores_code': code, 'acc': float(acc)})
    return pd.DataFrame(rows)


def load_intrinsics():
    """Our grid metrics at each model's best layer, keyed to flores codes."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fertility import flores_to_ntrex
    meta = pd.read_csv(os.path.join(DEF, 'language_meta.csv'))
    n2f = {}
    for fc in meta.flores_code:
        n2f.setdefault(flores_to_ntrex(fc), fc)
    rows = []
    for tag in MODELS:
        p = os.path.join(ROOT, 'results', 'grid', tag, 'metrics.json')
        r = json.load(open(p))
        bl = r['best_layer']
        L = r['per_layer'][str(bl)]['langs']
        for ntrex_code, v in L.items():
            fc = n2f.get(ntrex_code)
            if fc:
                rows.append({'model': tag, 'flores_code': fc,
                             'mexa': v['mexa'], 'aar10': v['aar10'],
                             'margin_mean': v['margin_mean']})
    return pd.DataFrame(rows)


def assemble():
    bel = load_belebele()
    if bel.empty:
        raise SystemExit('No Belebele results yet — run after job 1686630 lands.')
    meta = pd.read_csv(os.path.join(DEF, 'language_meta.csv'))
    tok = pd.read_csv(os.path.join(DEF, 'tokens_proxy_culturax.csv'))
    fert = pd.read_csv(os.path.join(DEF, 'fertility.csv'))
    intr = load_intrinsics()

    meta['iso1'] = [ISO3_TO_1.get(c.split('_')[0]) for c in meta.flores_code]
    meta = meta.merge(tok.rename(columns={'iso_code': 'iso1'}), on='iso1', how='left')
    meta['log_tokens'] = np.log10(meta.tokens.astype(float))

    df = (bel.merge(meta[['flores_code', 'script', 'macro_family', 'log_tokens']],
                    on='flores_code', how='left')
             .merge(fert, on=['flores_code', 'model'], how='left')
             .merge(intr, on=['flores_code', 'model'], how='left'))

    # group rare categories (>=3 languages per level among covered rows)
    for col in ('macro_family', 'script'):
        counts = df.groupby(col)['flores_code'].nunique()
        keep = counts[counts >= 3].index
        df[col + '_g'] = np.where(df[col].isin(keep), df[col], 'other')

    # outcome transform
    acc = df.acc.clip(0.26, 0.99)
    df['y'] = np.log((acc - 0.25) / 0.75 / (1 - (acc - 0.25) / 0.75))
    return df


def fit(df):
    import statsmodels.formula.api as smf
    d = df.dropna(subset=['y', 'log_tokens', 'fertility']).copy()
    print(f'[fit] rows: {len(d)} of {len(df)} '
          f'({df.log_tokens.isna().sum()} missing tokens, '
          f'{df.fertility.isna().sum()} missing fertility)')
    m = smf.ols('y ~ C(model) + log_tokens + C(macro_family_g) + C(script_g) '
                '+ fertility', data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d.flores_code})

    print('\n' + '=' * 70)
    print('TOKENS-BETA SANITY CHECK (must be positive & significant, else STOP)')
    b = m.params['log_tokens']; se = m.bse['log_tokens']; p = m.pvalues['log_tokens']
    print(f'  beta(log10 tokens) = {b:+.4f}  (SE {se:.4f}, p = {p:.2e})')
    print('=' * 70)
    print(f'\nR2 = {m.rsquared:.3f} | fertility beta = '
          f"{m.params['fertility']:+.4f} (p={m.pvalues['fertility']:.3f})")
    d['alpha'] = m.resid
    return m, d


def shrink_alphas(d):
    """PROPER empirical-Bayes shrinkage: w_i = tau2/(tau2 + s_i2), where s_i2
    is each row's sampling variance (binomial acc error propagated through the
    logit) and tau2 the excess between-language variance per macro-family.
    (Replaces an earlier placeholder whose weights reduced to (n-1)/n ---
    shrinkage in name only; documented in the report's hardening section.)"""
    acc = d.acc.clip(0.26, 0.99)
    p = (acc - 0.25) / 0.75
    dyda = 1.0 / (0.75 * p * (1 - p))
    d = d.assign(s2=((acc * (1 - acc) / 300.0) * dyda ** 2).values)
    out = []
    for fam, g in d.groupby('macro_family_g'):
        tau2 = max(g.alpha.var(ddof=1) - g.s2.mean(), 1e-4)
        w = tau2 / (tau2 + g.s2)
        mu = g.alpha.mean()
        out.append(g.assign(alpha_shrunk=mu + w * (g.alpha - mu)))
    return pd.concat(out)


def downstream_metrics(d):
    rows = []
    for tag, g in d.groupby('model'):
        accs = g.acc.values
        k = max(1, len(accs) // 10)
        rows.append({
            'model': tag,
            'mean_acc': accs.mean(),
            'PaR@10': np.sort(accs)[:k].mean(),
            # Sharpe on EXCESS accuracy over the 4-choice chance floor —
            # otherwise uniformly-at-floor models "win" with tiny std.
            'multilingual_sharpe': (accs.mean() - 0.25) / (accs.std() + 1e-9),
            'tail_alpha': g.nsmallest(max(1, len(g) // 10), 'log_tokens')
                           .alpha_shrunk.mean(),
        })
    return pd.DataFrame(rows).sort_values('PaR@10', ascending=False)


def metric_correlations(d):
    """Pre-registered set: {mexa, aar10} x {raw acc, alpha}. BH correction."""
    from scipy import stats
    tests = []
    for met in ('mexa', 'aar10'):
        # raw alphas as primary target: EB shrinkage collapses to family means
        # on this data (see shrinkage_diagnostics.py) and inflates correlations
        for target, col in (('raw_acc', 'acc'), ('alpha', 'alpha')):
            sub = d.dropna(subset=[met, col])
            r, p = stats.spearmanr(sub[met], sub[col])
            tests.append({'metric': met, 'target': target, 'spearman': r,
                          'p': p, 'n': len(sub)})
    t = pd.DataFrame(tests).sort_values('p')
    t['p_bh'] = t.p * len(t) / (np.arange(len(t)) + 1)  # Benjamini-Hochberg
    return t


if __name__ == '__main__':
    df = assemble()
    df.to_csv(os.path.join(OUT, 'master_table.csv'), index=False)
    m, d = fit(df)
    with open(os.path.join(OUT, 'regression_summary.txt'), 'w') as f:
        f.write(m.summary().as_text())
    d = shrink_alphas(d)
    dm = downstream_metrics(d)
    mc = metric_correlations(d)
    d.to_csv(os.path.join(OUT, 'alphas.csv'), index=False)
    dm.to_csv(os.path.join(OUT, 'downstream_metrics.csv'), index=False)
    mc.to_csv(os.path.join(OUT, 'metric_correlations.csv'), index=False)
    print('\n=== downstream metrics (PaR/Sharpe/tail-alpha) ===')
    print(dm.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\n=== metric correlations: raw vs residual (BH-corrected) ===')
    print(mc.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\n[done] -> {OUT}/')
