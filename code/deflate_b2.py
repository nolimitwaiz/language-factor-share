#!/usr/bin/env python3
"""Deflate B2': partial observables (log tokens, macro-family, script,
fertility) out of BOTH R_content and MEXA, then correlate residuals.
Also: sequential-partialling 'waterfall' numbers for the Table-3 deflation
(metric vs Belebele accuracy), for the Barra-style figure."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF = os.path.join(ROOT, 'results', 'deflation')
MODELS = ['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base', 'Qwen3-4B-Base', 'OLMo-2-0425-1B']

# NTREX ctx code -> FLORES code (for joining meta/tokens/fertility)
N2F = {'eng':'eng_Latn','deu':'deu_Latn','fra':'fra_Latn','spa':'spa_Latn',
 'por':'por_Latn','ita':'ita_Latn','nld':'nld_Latn','swe':'swe_Latn',
 'pol':'pol_Latn','ces':'ces_Latn','ron':'ron_Latn','rus':'rus_Cyrl',
 'ukr':'ukr_Cyrl','ell':'ell_Grek','tur':'tur_Latn','vie':'vie_Latn',
 'ind':'ind_Latn','zho-CN':'zho_Hans','jpn':'jpn_Jpan','kor':'kor_Hang',
 'arb':'arb_Arab','heb':'heb_Hebr','fas':'pes_Arab','hin':'hin_Deva',
 'ben':'ben_Beng','tam':'tam_Taml','tel':'tel_Telu','mar':'mar_Deva',
 'urd':'urd_Arab','tha':'tha_Thai','khm':'khm_Khmr','mya':'mya_Mymr',
 'swa':'swh_Latn','amh':'amh_Ethi','som':'som_Latn','hau':'hau_Latn',
 'zul':'zul_Latn','kat':'kat_Geor','hye':'hye_Armn','aze-Latn':'azj_Latn',
 'uzb':'uzn_Latn'}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution import ISO3_TO_1


def build():
    meta = pd.read_csv(os.path.join(DEF, 'language_meta.csv'))
    tok = pd.read_csv(os.path.join(DEF, 'tokens_proxy_culturax.csv'))
    fert = pd.read_csv(os.path.join(DEF, 'fertility.csv'))
    meta['iso1'] = [ISO3_TO_1.get(c.split('_')[0]) for c in meta.flores_code]
    meta = meta.merge(tok.rename(columns={'iso_code': 'iso1'}), on='iso1', how='left')
    meta['log_tokens'] = np.log10(meta.tokens.astype(float))
    rows = []
    for m in MODELS:
        r = json.load(open(os.path.join(ROOT, 'results', 'transfer313', f'{m}_v2.json')))
        g = json.load(open(os.path.join(ROOT, 'results', 'grid', m, 'metrics.json')))
        L = g['per_layer'][str(g['best_layer'])]['langs']
        for lg, v in r['results'].items():
            if lg == 'eng' or lg not in L:
                continue
            fc = N2F.get(lg)
            rows.append({'model': m, 'ntrex': lg, 'flores_code': fc,
                         'R_content': v['R_content'], 'mexa': L[lg]['mexa']})
    df = pd.DataFrame(rows)
    df = df.merge(meta[['flores_code', 'script', 'macro_family', 'log_tokens']],
                  on='flores_code', how='left')
    df = df.merge(fert, on=['flores_code', 'model'], how='left')
    for col in ('macro_family', 'script'):
        counts = df.groupby(col)['flores_code'].nunique()
        keep = counts[counts >= 3].index
        df[col + '_g'] = np.where(df[col].isin(keep), df[col], 'other')
    return df


def residualize(d, ycol):
    import statsmodels.formula.api as smf
    m = smf.ols(f'{ycol} ~ C(model) + log_tokens + C(macro_family_g) '
                '+ C(script_g) + fertility', data=d).fit()
    return m.resid


def main():
    df = build().dropna(subset=['R_content', 'mexa', 'log_tokens', 'fertility'])
    print(f'[data] {len(df)} (model,language) rows '
          f'({df.flores_code.nunique()} languages x {df.model.nunique()} models)')
    d = df.copy()
    d['r_R'] = residualize(d, 'R_content')
    d['r_M'] = residualize(d, 'mexa')
    print('\n=== B2 deflated: Spearman(resid MEXA, resid R_content) ===')
    out = {}
    for m in MODELS:
        g = d[d.model == m]
        raw = stats.spearmanr(g.mexa, g.R_content)
        defl = stats.spearmanr(g.r_M, g.r_R)
        out[m] = {'raw': float(raw.statistic), 'deflated': float(defl.statistic),
                  'p_deflated': float(defl.pvalue), 'n': len(g)}
        print(f'{m:18s} raw={raw.statistic:+.3f} -> deflated={defl.statistic:+.3f} '
              f'(p={defl.pvalue:.2e}, n={len(g)})')
    pooled = stats.spearmanr(d.r_M, d.r_R)
    out['pooled'] = {'deflated': float(pooled.statistic), 'p': float(pooled.pvalue),
                     'n': len(d)}
    print(f"POOLED deflated: {pooled.statistic:+.3f} (p={pooled.pvalue:.2e}, n={len(d)})")

    # ---- waterfall numbers for Table-3 deflation (metric vs Belebele acc) ----
    mt = pd.read_csv(os.path.join(DEF, 'attribution', 'master_table.csv'))
    mt = mt.dropna(subset=['acc', 'mexa', 'aar10', 'log_tokens', 'fertility'])
    steps = [('raw', []), ('- data size', ['log_tokens']),
             ('- family', ['log_tokens', 'C(macro_family_g)']),
             ('- script', ['log_tokens', 'C(macro_family_g)', 'C(script_g)']),
             ('- fertility', ['log_tokens', 'C(macro_family_g)', 'C(script_g)',
                              'fertility'])]
    import statsmodels.formula.api as smf
    wf = {}
    for met in ('mexa', 'aar10'):
        seq = []
        for name, terms in steps:
            if not terms:
                r = stats.spearmanr(mt[met], mt.acc).statistic
            else:
                rhs = ' + '.join(['C(model)'] + terms)
                ra = smf.ols(f'acc_l ~ {rhs}',
                             data=mt.assign(acc_l=mt.acc)).fit().resid
                rm = smf.ols(f'{met} ~ {rhs}', data=mt).fit().resid
                r = stats.spearmanr(rm, ra).statistic
            seq.append((name, float(r)))
        wf[met] = seq
        print(f"\nwaterfall {met}: " + '  '.join(f'{n}={v:+.3f}' for n, v in seq))
    json.dump({'b2_deflated': out, 'waterfall': wf},
              open(os.path.join(DEF, 'attribution', 'b2_deflated.json'), 'w'), indent=1)
    print(f"\n[done] -> {DEF}/attribution/b2_deflated.json")


if __name__ == '__main__':
    main()
