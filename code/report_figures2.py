#!/usr/bin/env python3
"""Additional report figures: language x model heatmap, mix-vs-scale scatter."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, 'results', 'grid')
FIGS = os.path.join(ROOT, 'paper', 'figs')

ORDER = ['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base', 'Qwen3-4B-Base', 'Qwen3-8B-Base',
         'OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'Mistral-7B-v0.3',
         'EuroLLM-1.7B', 'bloom-1b7', 'bloom-7b1']
SHORT = ['Qwen3\n0.6B', 'Qwen3\n1.7B', 'Qwen3\n4B', 'Qwen3\n8B',
         'OLMo-2\n1B', 'OLMo-2\n7B', 'Mistral\n7B', 'EuroLLM\n1.7B',
         'BLOOM\n1.7B', 'BLOOM\n7B']
PARAMS = [0.6, 1.7, 4.0, 8.0, 1.0, 7.0, 7.0, 1.7, 1.7, 7.0]
KIND = ['multi', 'multi', 'multi', 'multi', 'EN-centric', 'EN-centric',
        'EN-centric', 'EU-multi', 'BLOOM-multi', 'BLOOM-multi']
KCOLOR = {'multi': '#2171b5', 'EN-centric': '#d62728',
          'EU-multi': '#2ca02c', 'BLOOM-multi': '#9467bd'}

res = {t: json.load(open(os.path.join(GRID, t, 'metrics.json'))) for t in ORDER}

# ---- F6: language x model MEXA heatmap --------------------------------------
langs_all = [lg for lg in res[ORDER[0]]['per_layer']['0']['langs']]
M = np.zeros((len(langs_all), len(ORDER)))
for j, t in enumerate(ORDER):
    r = res[t]; bl = r['best_layer']
    L = r['per_layer'][str(bl)]['langs']
    for i, lg in enumerate(langs_all):
        M[i, j] = L.get(lg, {}).get('mexa', np.nan)
order_rows = np.argsort(-np.nanmean(M, 1))
M_sorted = M[order_rows]
langs_sorted = [langs_all[i] for i in order_rows]

tiers = res[ORDER[0]]['tiers']
fig, axes = plt.subplots(1, 2, figsize=(13, 12), gridspec_kw={'width_ratios': [1, 1.15]})
im0 = axes[0].imshow(M_sorted, aspect='auto', cmap='viridis', vmin=0, vmax=1)
axes[0].set_xticks(range(len(ORDER))); axes[0].set_xticklabels(SHORT, fontsize=7)
axes[0].set_yticks([])
axes[0].set_ylabel(f'{len(langs_sorted)} languages, sorted by cross-model mean alignment')
axes[0].set_title('(a) MEXA @ best layer — all languages')

tier_langs = [lg for lg in langs_sorted if tiers.get(lg) in ('high', 'mid', 'low')]
Mt = np.array([M_sorted[langs_sorted.index(lg)] for lg in tier_langs])
im1 = axes[1].imshow(Mt, aspect='auto', cmap='viridis', vmin=0, vmax=1)
axes[1].set_xticks(range(len(ORDER))); axes[1].set_xticklabels(SHORT, fontsize=7)
axes[1].set_yticks(range(len(tier_langs)))
tcol = {'high': '#4477aa', 'mid': '#997700', 'low': '#cc3311'}
axes[1].set_yticklabels([f"{lg}  [{tiers[lg]}]" for lg in tier_langs], fontsize=6.5)
for tick, lg in zip(axes[1].get_yticklabels(), tier_langs):
    tick.set_color(tcol[tiers[lg]])
axes[1].set_title('(b) curated tier subset (labelled)')
fig.colorbar(im1, ax=axes, fraction=0.02, pad=0.01, label='MEXA')
plt.savefig(os.path.join(FIGS, 'fig6_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# ---- F7: mix vs scale — dip depth vs params for all models ------------------
plt.figure(figsize=(7.6, 5))
dips = []
for t in ORDER:
    r = res[t]
    layers = sorted(int(k) for k in r['per_layer'])
    lfs = np.array([r['per_layer'][str(l)]['lfs']['lfs'] for l in layers])
    dips.append(lfs[0] - lfs.min())
for fam, idxs, ls in (('Qwen3', [0, 1, 2, 3], '-'), ('OLMo-2', [4, 5], '-'),
                      ('BLOOM', [8, 9], '-')):
    plt.plot([PARAMS[i] for i in idxs], [dips[i] for i in idxs],
             ls, color=KCOLOR[KIND[idxs[0]]], lw=1.5, alpha=0.55, zorder=1)
seen = set()
for i, t in enumerate(ORDER):
    lbl = KIND[i] if KIND[i] not in seen else None
    seen.add(KIND[i])
    plt.scatter(PARAMS[i], dips[i], s=90, color=KCOLOR[KIND[i]], zorder=2, label=lbl)
    plt.annotate(SHORT[i].replace('\n', ' '), (PARAMS[i], dips[i]),
                 textcoords='offset points', xytext=(7, 4), fontsize=7.5)
plt.xscale('log'); plt.xticks([0.6, 1, 1.7, 4, 7, 8], ['0.6B', '1B', '1.7B', '4B', '7B', '8B'])
plt.xlabel('parameters'); plt.ylabel('LFS dip depth (first-layer LFS minus minimum)')
plt.title('Dip depth against parameter count\n'
          '(between-family spread at a fixed size exceeds the within-family scaling slope)')
plt.legend(fontsize=8)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig7_mix_vs_scale.png'), dpi=160)
plt.close()
print('wrote fig6_heatmap.png, fig7_mix_vs_scale.png')
