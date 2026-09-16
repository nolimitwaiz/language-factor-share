#!/usr/bin/env python3
"""Cross-model comparison of the LFM+AaR metric suite (results/grid/*)."""
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, 'results', 'grid')

ORDER = ['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base', 'Qwen3-4B-Base', 'Qwen3-8B-Base',
         'OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'Mistral-7B-v0.3',
         'bloom-1b7', 'bloom-7b1', 'EuroLLM-1.7B']
KIND = {'Qwen3-0.6B-Base': 'multi-ish', 'Qwen3-1.7B-Base': 'multi-ish',
        'Qwen3-4B-Base': 'multi-ish', 'Qwen3-8B-Base': 'multi-ish',
        'OLMo-2-0425-1B': 'EN-centric', 'OLMo-2-1124-7B': 'EN-centric',
        'Mistral-7B-v0.3': 'EN-centric',
        'bloom-1b7': 'multi-design', 'bloom-7b1': 'multi-design',
        'EuroLLM-1.7B': 'multi-design(EU)'}


def load():
    out = {}
    for tag in ORDER:
        p = os.path.join(GRID, tag, 'metrics.json')
        if os.path.exists(p):
            out[tag] = json.load(open(p))
    return out


def summarize(res):
    rows = []
    for tag, r in res.items():
        layers = sorted(int(k) for k in r['per_layer'])
        lfs = np.array([r['per_layer'][str(l)]['lfs']['lfs'] for l in layers])
        mexa = np.array([np.mean([v['mexa'] for v in r['per_layer'][str(l)]['langs'].values()])
                         for l in layers])
        bl = r['best_layer']
        L = r['per_layer'][str(bl)]['langs']
        tiers = r['tiers']
        tier_stats = {}
        for tier in ('high', 'mid', 'low'):
            lgs = [lg for lg in L if tiers.get(lg) == tier]
            tier_stats[tier] = {
                'mexa': float(np.mean([L[lg]['mexa'] for lg in lgs])),
                'aar10': float(np.mean([L[lg]['aar10'] for lg in lgs])),
            }
        hub = r['hub_at_best_layer']
        advs = [v['latent_advantage'] for v in hub.values()]
        engbest = sum(v['best_donor'] == 'eng' for v in hub.values())
        rows.append({
            'model': tag, 'kind': KIND.get(tag, '?'),
            'n_layers': len(layers) - 1, 'best_layer': bl,
            'rel_depth': bl / (len(layers) - 1),
            'lfs_min': float(lfs.min()),
            'lfs_dip': float(lfs[0] - lfs.min()),
            'peak_mexa': float(mexa.max()),
            'mexa_high': tier_stats['high']['mexa'],
            'mexa_mid': tier_stats['mid']['mexa'],
            'mexa_low': tier_stats['low']['mexa'],
            'aar_mid': tier_stats['mid']['aar10'],
            'latent_adv': float(np.mean(advs)),
            'eng_best_donor': f"{engbest}/{len(hub)}",
        })
    return rows


def figures(res, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(res)))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for (tag, r), c in zip(res.items(), colors):
        layers = sorted(int(k) for k in r['per_layer'])
        x = np.array(layers) / max(layers)
        lfs = [r['per_layer'][str(l)]['lfs']['lfs'] for l in layers]
        mexa = [np.mean([v['mexa'] for v in r['per_layer'][str(l)]['langs'].values()])
                for l in layers]
        axes[0].plot(x, lfs, color=c, lw=2, label=tag)
        axes[1].plot(x, mexa, color=c, lw=2, label=tag)
    axes[0].set_title('LFS by relative depth (M1): the depth profile')
    axes[0].set_xlabel('layer / n_layers'); axes[0].set_ylabel('language-factor share')
    axes[0].legend(fontsize=7)
    axes[1].set_title('mean MEXA by relative depth')
    axes[1].set_xlabel('layer / n_layers'); axes[1].legend(fontsize=7)
    # tier bars at best layer
    tags = list(res)
    width = 0.25
    for i, tier in enumerate(('high', 'mid', 'low')):
        vals = []
        for tag in tags:
            r = res[tag]; bl = r['best_layer']
            L = r['per_layer'][str(bl)]['langs']; tiers = r['tiers']
            lgs = [lg for lg in L if tiers.get(lg) == tier]
            vals.append(np.mean([L[lg]['mexa'] for lg in lgs]))
        axes[2].bar(np.arange(len(tags)) + (i - 1) * width, vals, width,
                    label=tier, color=['#4477aa', '#ccbb44', '#ee6677'][i])
    axes[2].set_xticks(range(len(tags)))
    axes[2].set_xticklabels([t.replace('-Base', '') for t in tags],
                            rotation=25, fontsize=7, ha='right')
    axes[2].set_title('MEXA by resource tier @ best layer')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig_cross_model.png'), dpi=140)
    print(f'[fig] -> {out}/fig_cross_model.png')


if __name__ == '__main__':
    res = load()
    rows = summarize(res)
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False,
                       float_format=lambda v: f'{v:.3f}'))
    df.to_csv(os.path.join(GRID, 'cross_model_summary.csv'), index=False)
    figures(res, GRID)
