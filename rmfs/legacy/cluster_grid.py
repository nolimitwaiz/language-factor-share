#!/usr/bin/env python3
"""
CLUSTER GRID: LFM+AaR metric suite over a model grid x all NTREX-128 languages.

Per model (one job-array task each):
  - per-layer LFS, per-language MEXA / AaR@10 / mean margin
  - hub test at the best (most-shared) layer WITH controls:
      (a) latent-vs-English (standardized predictors)
      (b) donor ranking: single-language donors incl. English — no smoothness
          confound; if eng is rarely the best donor, the hub is not English.
Outputs -> results/grid/<model_tag>/metrics.json  (small; embeddings not kept)

Usage (SLURM array): python cluster_grid.py --model_idx $SLURM_ARRAY_TASK_ID
       (local test) : python cluster_grid.py --model Qwen/Qwen3-0.6B-Base --n_sents 50 --max_langs 8
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_metrics import embed_all, mexa_score, aar, lfs, _norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')

MODELS = [
    'Qwen/Qwen3-0.6B-Base',        # scaling curve, multilingual-ish
    'Qwen/Qwen3-1.7B-Base',
    'Qwen/Qwen3-4B-Base',
    'Qwen/Qwen3-8B-Base',
    'allenai/OLMo-2-0425-1B',      # OPEN DATA -> deflation anchor
    'allenai/OLMo-2-1124-7B',      # OPEN DATA -> deflation anchor
    'mistralai/Mistral-7B-v0.3',   # English-centric contrast
    'bigscience/bloom-1b7',        # multilingual by design, OPEN DATA
    'bigscience/bloom-7b1',
    'utter-project/EuroLLM-1.7B',  # multilingual by design (EU)
    'utter-project/EuroLLM-9B',
]

# donor panel for the hub donor-ranking control (family/script spread)
DONORS = ['eng', 'deu', 'fra', 'spa', 'rus', 'zho-CN', 'arb', 'hin',
          'tur', 'vie', 'swa', 'ind']

TIER = {  # curated subset; everything else 'unk' (slice later)
    'deu': 'high', 'fra': 'high', 'spa': 'high', 'por': 'high', 'ita': 'high',
    'nld': 'high', 'rus': 'high', 'zho-CN': 'high', 'jpn': 'high', 'arb': 'high',
    'ces': 'mid', 'pol': 'mid', 'ukr': 'mid', 'tur': 'mid', 'vie': 'mid',
    'ind': 'mid', 'hin': 'mid', 'ben': 'mid',
    'swa': 'low', 'amh': 'low', 'som': 'low', 'hau': 'low', 'zul': 'low',
    'khm': 'low', 'mya': 'low', 'kat': 'low',
}


def load_all_ntrex(n_sents, max_langs=None):
    src = os.path.join(NTREX, 'newstest2019-src.eng.txt')
    with open(src) as f:
        eng = [l.strip() for l in f][:n_sents]
    data = {'eng': eng}
    files = sorted(f for f in os.listdir(NTREX) if f.startswith('newstest2019-ref.'))
    for fn in files:
        code = fn[len('newstest2019-ref.'):-len('.txt')]
        if code == 'eng':
            continue
        with open(os.path.join(NTREX, fn)) as f:
            sents = [l.strip() for l in f][:n_sents]
        if len(sents) == len(eng) and all(sents):
            data[code] = sents
        if max_langs and len(data) - 1 >= max_langs:
            break
    print(f'[data] {len(data) - 1} languages + eng pivot, {len(eng)} sents')
    return data


def cv_r2(Xsrc, y, folds=3, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    # standardize predictor columns -> comparable ridge shrinkage across arms
    Xs = (Xsrc - Xsrc.mean(0)) / (Xsrc.std(0) + 1e-9)
    preds = np.zeros_like(y)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Xs):
        m = Ridge(alpha=1.0).fit(Xs[tr], y[tr])
        preds[te] = m.predict(Xs[te])
    return float(1 - ((y - preds) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def hub_with_controls(X, langs, n_pc=64, seed=0):
    """X: [L, N, D] z-scored. Returns per-target: latent vs english R2 (std'ized)
    + donor ranking over the DONORS panel."""
    from sklearn.decomposition import PCA
    L, N, D = X.shape
    assert np.isfinite(X).all(), 'non-finite values reached hub_with_controls'
    # np.errstate: Apple Accelerate BLAS raises spurious FPE flags in matmul
    # on verified-finite input (numpy/arm64 quirk); harmless, silence locally.
    with np.errstate(all='ignore'):
        Z = PCA(n_components=min(n_pc, N - 2), random_state=seed).fit_transform(
            X.reshape(L * N, D)).reshape(L, N, -1)
    idx = {lg: i for i, lg in enumerate(langs)}
    donors = [d for d in DONORS if d in idx]
    out = {}
    for i, lg in enumerate(langs):
        if lg == 'eng':
            continue
        y = Z[i]
        others = [j for j in range(L) if j not in (i, idx['eng'])]
        latent = Z[others].mean(0)
        res = {'r2_latent': cv_r2(latent, y), 'r2_english': cv_r2(Z[idx['eng']], y)}
        res['latent_advantage'] = res['r2_latent'] - res['r2_english']
        ranking = {}
        for d in donors:
            if d == lg:
                continue
            ranking[d] = cv_r2(Z[idx[d]], y)
        res['donor_r2'] = ranking
        res['best_donor'] = max(ranking, key=ranking.get) if ranking else None
        out[lg] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--model_idx', type=int, default=None)
    ap.add_argument('--n_sents', type=int, default=300)
    ap.add_argument('--max_langs', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--dtype', default=None, help='fp16|bf16|fp32 (default: auto)')
    args = ap.parse_args()

    model = args.model or MODELS[args.model_idx]
    if args.device is None:
        import torch
        args.device = ('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available() else 'cpu')
    tag = model.split('/')[-1]
    out_dir = os.path.join(ROOT, 'results', 'grid', tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f'[run] {model} on {args.device}')

    data = load_all_ntrex(args.n_sents, args.max_langs)
    embs = embed_all(data, model, args.device, batch_size=args.batch_size,
                     dtype=args.dtype)
    langs = list(embs.keys())
    n_layers = embs['eng'].shape[1]

    results = {'model': model, 'n_sents': args.n_sents, 'n_layers': n_layers,
               'langs': langs, 'tiers': {lg: TIER.get(lg, 'unk') for lg in langs},
               'per_layer': {}}

    for layer in range(n_layers):
        E = {lg: embs[lg][:, layer, :].astype(np.float32) for lg in langs}
        allE = np.concatenate(list(E.values()), 0)
        mu, sd = allE.mean(0), allE.std(0) + 1e-9
        Ez = {lg: (E[lg] - mu) / sd for lg in langs}
        X = np.stack([Ez[lg] for lg in langs], 0)
        lr = {'lfs': lfs(X), 'langs': {}}
        for lg in langs:
            if lg == 'eng':
                continue
            a10, amean = aar(Ez['eng'], Ez[lg])
            lr['langs'][lg] = {'mexa': mexa_score(Ez['eng'], Ez[lg]),
                               'aar10': a10, 'margin_mean': amean}
        results['per_layer'][layer] = lr
        print(f"[layer {layer:2d}] LFS={lr['lfs']['lfs']:.3f} "
              f"meanMEXA={np.mean([v['mexa'] for v in lr['langs'].values()]):.3f}",
              flush=True)

    mean_mexa = {l: np.mean([v['mexa'] for v in r['langs'].values()])
                 for l, r in results['per_layer'].items()}
    best_layer = max(mean_mexa, key=mean_mexa.get)
    results['best_layer'] = int(best_layer)

    E = {lg: embs[lg][:, best_layer, :].astype(np.float32) for lg in langs}
    allE = np.concatenate(list(E.values()), 0)
    mu, sd = allE.mean(0), allE.std(0) + 1e-9
    X = np.stack([(E[lg] - mu) / sd for lg in langs], 0)
    print(f'[hub] at best layer {best_layer} with donor controls ...', flush=True)
    results['hub_at_best_layer'] = hub_with_controls(X, langs)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'[done] -> {out_dir}/metrics.json')


if __name__ == '__main__':
    main()
