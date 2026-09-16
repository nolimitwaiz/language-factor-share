#!/usr/bin/env python3
"""
PILOT: LFM+AaR metric suite vs MEXA baseline, on NTREX-128 parallel text.

Extracts per-layer sentence embeddings from a small multilingual LM (MPS),
then computes, per layer:
  - MEXA        : Kargaran et al. alignment score (diag strictly dominates
                  row+column of the EN<->L cosine matrix)  [baseline]
  - LFS         : Language-Factor Share — variance decomposition of matched
                  sentence reps into concept vs language components  [ours, M1]
  - HUB         : English-hub test — CV-R^2 of predicting X_L from English
                  reps vs from the latent concept factor (leave-L-out mean)
                  [ours, M2]
  - AaR@10      : Alignment-at-Risk — mean retrieval margin of the worst
                  decile of sentences (tail, not mean)  [ours, M3]

Outputs: results/pilot/<model_tag>/metrics.json + figures.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')

# preference list: (ntrex code, tier). Falls back gracefully if file missing.
LANGS = [
    ('deu', 'high'), ('fra', 'high'), ('spa', 'high'), ('por', 'high'),
    ('ita', 'high'), ('nld', 'high'), ('rus', 'high'), ('zho-CN', 'high'),
    ('jpn', 'high'), ('arb', 'high'),
    ('ces', 'mid'), ('pol', 'mid'), ('ukr', 'mid'), ('tur', 'mid'),
    ('vie', 'mid'), ('ind', 'mid'), ('hin', 'mid'), ('ben', 'mid'),
    ('swa', 'low'), ('amh', 'low'), ('som', 'low'), ('hau', 'low'),
    ('zul', 'low'), ('khm', 'low'), ('mya', 'low'), ('kat', 'low'),
]
PIVOT = 'eng'


def load_parallel(n_sents):
    def path(code):
        return os.path.join(NTREX, f'newstest2019-ref.{code}.txt')
    # NTREX pivot: English source file is newstest2019-src.eng.txt
    src = os.path.join(NTREX, 'newstest2019-src.eng.txt')
    if not os.path.exists(src):
        src = path('eng')
    with open(src) as f:
        eng = [l.strip() for l in f][:n_sents]
    data = {'eng': eng}
    tiers = {'eng': 'pivot'}
    for code, tier in LANGS:
        p = path(code)
        if not os.path.exists(p):
            print(f'[skip] {code} (no file)')
            continue
        with open(p) as f:
            data[code] = [l.strip() for l in f][:n_sents]
        tiers[code] = tier
    return data, tiers


def embed_all(data, model_name, device, batch_size=16, max_len=128, dtype=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:            # Mistral/EuroLLM ship without one
        tok.pad_token = tok.eos_token
    if dtype in ('fp16', 'float16'):     # e.g. Turing GPUs: no bf16 support
        dtype = torch.float16
    elif dtype in ('bf16', 'bfloat16'):
        dtype = torch.bfloat16
    elif dtype in ('fp32', 'float32'):
        dtype = torch.float32
    else:                                # auto
        dtype = torch.float16 if device == 'mps' else \
            torch.bfloat16 if device == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(model_name, dtype=dtype,
                                      output_hidden_states=True).to(device).eval()
    out = {}
    for lang, sents in data.items():
        embs = None
        for i in range(0, len(sents), batch_size):
            batch = sents[i:i + batch_size]
            enc = tok(batch, return_tensors='pt', padding=True,
                      truncation=True, max_length=max_len).to(device)
            with torch.no_grad():
                hs = model(**enc).hidden_states       # tuple (L+1) of [B,T,D]
            # pool in fp32: massive-activation outliers overflow fp16 sums
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = [(h.float() * mask).sum(1) / mask.sum(1) for h in hs]
            pooled = torch.stack(pooled, 1).float().cpu().numpy()    # [B, L+1, D]
            embs = pooled if embs is None else np.concatenate([embs, pooled], 0)
        out[lang] = embs                                             # [N, L+1, D]
        print(f'[embed] {lang}: {embs.shape}')
    return out


def _norm(M):
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-9)


def mexa_score(E_en, E_l):
    """Fraction of sentences whose parallel pair strictly dominates row & col."""
    S = _norm(E_en) @ _norm(E_l).T
    n = len(S)
    d = np.diag(S)
    row_ok = d > (np.where(np.eye(n, dtype=bool), -np.inf, S)).max(1)
    col_ok = d > (np.where(np.eye(n, dtype=bool), -np.inf, S)).max(0)
    return float((row_ok & col_ok).mean())


def aar(E_en, E_l, q=0.10):
    """Alignment-at-Risk: mean retrieval margin of the worst q-quantile."""
    S = _norm(E_l) @ _norm(E_en).T
    n = len(S)
    d = np.diag(S).copy()
    off = np.where(np.eye(n, dtype=bool), -np.inf, S).max(1)
    margins = d - off
    k = max(1, int(np.floor(q * n)))
    worst = np.sort(margins)[:k]
    return float(worst.mean()), float(margins.mean())


def lfs(X):
    """Language-Factor Share. X: [n_lang, N, D] matched reps (z-scored dims)."""
    L, N, D = X.shape
    mu = X.mean((0, 1))
    mu_c = X.mean(0)                    # [N, D] concept means
    mu_l = X.mean(1)                    # [L, D] language means
    ss_concept = L * ((mu_c - mu) ** 2).sum()
    ss_lang = N * ((mu_l - mu) ** 2).sum()
    resid = X - mu_c[None] - mu_l[:, None] + mu
    ss_resid = (resid ** 2).sum()
    tot = ss_concept + ss_lang + ss_resid
    return {'lfs': float(ss_lang / (ss_lang + ss_concept)),
            'var_lang': float(ss_lang / tot), 'var_concept': float(ss_concept / tot),
            'var_resid': float(ss_resid / tot)}


def hub_test(X, langs, n_pc=64, folds=2, seed=0):
    """M2: for each lang, CV-R^2 predicting X_l from English vs latent factor
    (mean of all other non-English langs). Positive advantage => latent hub."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    L, N, D = X.shape
    flat = X.reshape(L * N, D)
    Z = PCA(n_components=min(n_pc, N - 2), random_state=seed).fit_transform(flat)
    Z = Z.reshape(L, N, -1)
    i_en = langs.index('eng')
    out = {}
    for i, lg in enumerate(langs):
        if lg == 'eng':
            continue
        others = [j for j in range(L) if j not in (i, i_en)]
        latent = Z[others].mean(0)
        y = Z[i]
        r2 = {}
        for name, Xsrc in (('english', Z[i_en]), ('latent', latent)):
            preds = np.zeros_like(y)
            for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Xsrc):
                m = Ridge(alpha=1.0).fit(Xsrc[tr], y[tr])
                preds[te] = m.predict(Xsrc[te])
            ss_res = ((y - preds) ** 2).sum()
            ss_tot = ((y - y.mean(0)) ** 2).sum()
            r2[name] = float(1 - ss_res / ss_tot)
        out[lg] = {'r2_english': r2['english'], 'r2_latent': r2['latent'],
                   'latent_advantage': r2['latent'] - r2['english']}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--n_sents', type=int, default=100)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tag = args.model.split('/')[-1]
    out_dir = args.out or os.path.join(ROOT, 'results', 'pilot', tag)
    os.makedirs(out_dir, exist_ok=True)

    data, tiers = load_parallel(args.n_sents)
    embs = embed_all(data, args.model, args.device)
    langs = list(embs.keys())
    n_layers = embs[PIVOT].shape[1]

    results = {'model': args.model, 'n_sents': args.n_sents, 'tiers': tiers,
               'n_layers': n_layers, 'per_layer': {}}

    for layer in range(n_layers):
        E = {lg: embs[lg][:, layer, :] for lg in langs}
        # z-score dims across all (lang, sent) samples — anisotropy guard
        allE = np.concatenate(list(E.values()), 0)
        mu, sd = allE.mean(0), allE.std(0) + 1e-9
        Ez = {lg: (E[lg] - mu) / sd for lg in langs}

        X = np.stack([Ez[lg] for lg in langs], 0)
        layer_res = {'lfs': lfs(X), 'langs': {}}

        for lg in langs:
            if lg == PIVOT:
                continue
            mx = mexa_score(Ez[PIVOT], Ez[lg])
            a10, amean = aar(Ez[PIVOT], Ez[lg])
            layer_res['langs'][lg] = {'mexa': mx, 'aar10': a10, 'margin_mean': amean}
        results['per_layer'][layer] = layer_res
        best = np.mean([v['mexa'] for v in layer_res['langs'].values()])
        print(f'[layer {layer:2d}] LFS={layer_res["lfs"]["lfs"]:.3f} '
              f'mean-MEXA={best:.3f}')

    # hub test at the layer with max mean MEXA (the "shared" layer)
    mean_mexa = {l: np.mean([v['mexa'] for v in r['langs'].values()])
                 for l, r in results['per_layer'].items()}
    best_layer = max(mean_mexa, key=mean_mexa.get)
    results['best_layer'] = int(best_layer)
    E = {lg: embs[lg][:, best_layer, :] for lg in langs}
    allE = np.concatenate(list(E.values()), 0)
    mu, sd = allE.mean(0), allE.std(0) + 1e-9
    X = np.stack([(E[lg] - mu) / sd for lg in langs], 0)
    print(f'[hub] running at best layer {best_layer} ...')
    results['hub_at_best_layer'] = hub_test(X, langs)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'[done] -> {out_dir}/metrics.json')

    make_figures(results, out_dir)


def make_figures(res, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layers = sorted(int(k) for k in res['per_layer'].keys())
    tiers = res['tiers']
    tier_colors = {'high': '#4477aa', 'mid': '#ccbb44', 'low': '#ee6677'}

    # 1: MEXA by layer, per tier
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for tier in ('high', 'mid', 'low'):
        lgs = [lg for lg, t in tiers.items() if t == tier]
        for lg in lgs:
            curve = [res['per_layer'][str(l) if str(l) in res['per_layer'] else l]['langs'][lg]['mexa'] for l in layers]
            axes[0].plot(layers, curve, color=tier_colors[tier], alpha=0.5, lw=1)
        if lgs:
            mean_curve = np.mean([[res['per_layer'][str(l) if str(l) in res['per_layer'] else l]['langs'][lg]['mexa']
                                   for l in layers] for lg in lgs], 0)
            axes[0].plot(layers, mean_curve, color=tier_colors[tier], lw=2.5, label=tier)
    axes[0].set_title('MEXA by layer (per resource tier)')
    axes[0].set_xlabel('layer'); axes[0].legend()

    # 2: LFS curve
    lfs_curve = [res['per_layer'][str(l) if str(l) in res['per_layer'] else l]['lfs']['lfs'] for l in layers]
    axes[1].plot(layers, lfs_curve, lw=2.5, color='#228833')
    axes[1].set_title('Language-Factor Share by layer (ours, M1)')
    axes[1].set_xlabel('layer'); axes[1].set_ylabel('LFS')

    # 3: tail vs mean at best layer
    bl = res['best_layer']
    blk = str(bl) if str(bl) in res['per_layer'] else bl
    langs_d = res['per_layer'][blk]['langs']
    for lg, v in langs_d.items():
        c = tier_colors.get(tiers.get(lg, 'mid'), 'gray')
        axes[2].scatter(v['margin_mean'], v['aar10'], color=c, s=40)
        axes[2].annotate(lg, (v['margin_mean'], v['aar10']), fontsize=7, alpha=0.8)
    lims = axes[2].get_xlim()
    axes[2].plot(lims, lims, ls='--', c='gray', lw=1)
    axes[2].set_title(f'AaR@10 (tail) vs mean margin — layer {bl}')
    axes[2].set_xlabel('mean margin'); axes[2].set_ylabel('worst-decile margin')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_pilot_overview.png'), dpi=140)
    plt.close()
    print(f'[fig] -> {out_dir}/fig_pilot_overview.png')


if __name__ == '__main__':
    main()
