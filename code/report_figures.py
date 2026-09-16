#!/usr/bin/env python3
"""Build all figures for the analysis report -> paper/figs/."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, 'results', 'grid')
FIGS = os.path.join(ROOT, 'paper', 'figs')
os.makedirs(FIGS, exist_ok=True)

ORDER = ['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base', 'Qwen3-4B-Base', 'Qwen3-8B-Base',
         'OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'Mistral-7B-v0.3',
         'EuroLLM-1.7B', 'bloom-1b7', 'bloom-7b1']    # blooms = fp32/bf16 only
LABEL = {'Qwen3-0.6B-Base': 'Qwen3 0.6B', 'Qwen3-1.7B-Base': 'Qwen3 1.7B',
         'Qwen3-4B-Base': 'Qwen3 4B', 'Qwen3-8B-Base': 'Qwen3 8B',
         'OLMo-2-0425-1B': 'OLMo-2 1B (EN-centric)',
         'OLMo-2-1124-7B': 'OLMo-2 7B (EN-centric)',
         'Mistral-7B-v0.3': 'Mistral 7B (EN-centric)',
         'EuroLLM-1.7B': 'EuroLLM 1.7B (EU-multi)',
         'bloom-1b7': 'BLOOM 1.7B (multi)', 'bloom-7b1': 'BLOOM 7B (multi)'}
COLOR = {'Qwen3-0.6B-Base': '#c6dbef', 'Qwen3-1.7B-Base': '#6baed6',
         'Qwen3-4B-Base': '#2171b5', 'Qwen3-8B-Base': '#08306b',
         'OLMo-2-0425-1B': '#fb6a4a', 'OLMo-2-1124-7B': '#a50f15',
         'Mistral-7B-v0.3': '#ff7f0e', 'EuroLLM-1.7B': '#2ca02c',
         'bloom-1b7': '#bcbddc', 'bloom-7b1': '#54278f'}

res = {t: json.load(open(os.path.join(GRID, t, 'metrics.json'))) for t in ORDER
       if os.path.exists(os.path.join(GRID, t, 'metrics.json'))}


def curves(r):
    layers = sorted(int(k) for k in r['per_layer'])
    x = np.array(layers) / max(layers)
    lfs = np.array([r['per_layer'][str(l)]['lfs']['lfs'] for l in layers])
    mexa = np.array([np.mean([v['mexa'] for v in r['per_layer'][str(l)]['langs'].values()])
                     for l in layers])
    return x, lfs, mexa


# ---- F1: LFS depth profile + MEXA mirror -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for t, r in res.items():
    x, lfs, mexa = curves(r)
    axes[0].plot(x, lfs, color=COLOR[t], lw=2.2, label=LABEL[t])
    axes[1].plot(x, mexa, color=COLOR[t], lw=2.2, label=LABEL[t])
axes[0].set_xlabel('relative depth (layer / n\\_layers)')
axes[0].set_ylabel('Language-Factor Share')
axes[0].set_title('(a) LFS depth profile: language vs concept variance')
axes[0].legend(fontsize=8, loc='lower left')
axes[1].set_xlabel('relative depth')
axes[1].set_ylabel('mean MEXA (128 languages)')
axes[1].set_title('(b) MEXA alignment (mirror image)')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig1_lfs_depth_profile.png'), dpi=160)
plt.close()

# ---- F2: scaling + tier bars ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
qwen = ['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base', 'Qwen3-4B-Base', 'Qwen3-8B-Base']
params = [0.6, 1.7, 4.0, 8.0]
dip = []
low = []
for t in qwen:
    x, lfs, mexa = curves(res[t])
    dip.append(lfs[0] - lfs.min())
    bl = res[t]['best_layer']
    L = res[t]['per_layer'][str(bl)]['langs']
    tiers = res[t]['tiers']
    low.append(np.mean([L[lg]['mexa'] for lg in L if tiers.get(lg) == 'low']))
ax2 = axes[0].twinx()
axes[0].plot(params, dip, 'o-', color='#2171b5', lw=2, label='LFS dip depth')
ax2.plot(params, low, 's--', color='#ee6677', lw=2, label='low-resource MEXA')
axes[0].set_xscale('log'); axes[0].set_xticks(params)
axes[0].set_xticklabels(['0.6B', '1.7B', '4B', '8B'])
axes[0].set_xlabel('Qwen3 parameters'); axes[0].set_ylabel('LFS dip depth', color='#2171b5')
ax2.set_ylabel('low-tier MEXA', color='#ee6677')
axes[0].set_title('(a) Scale deepens the shared concept space')
h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
axes[0].legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')

width = 0.25
tags = list(res)
for i, (tier, c) in enumerate(zip(('high', 'mid', 'low'),
                                  ('#4477aa', '#ccbb44', '#ee6677'))):
    vals = []
    for t in tags:
        r = res[t]; bl = r['best_layer']
        L = r['per_layer'][str(bl)]['langs']; tiers = r['tiers']
        vals.append(np.mean([L[lg]['mexa'] for lg in L if tiers.get(lg) == tier]))
    axes[1].bar(np.arange(len(tags)) + (i - 1) * width, vals, width, label=tier, color=c)
axes[1].set_xticks(range(len(tags)))
axes[1].set_xticklabels([LABEL[t] for t in tags], rotation=18, fontsize=7, ha='right')
axes[1].set_ylabel('MEXA @ best layer')
axes[1].set_title('(b) Training mix is visible per tier')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig2_scaling_tiers.png'), dpi=160)
plt.close()

# ---- F3: tail vs mean (AaR) for best Qwen3-4B -------------------------------
r = res['Qwen3-4B-Base']
bl = r['best_layer']; L = r['per_layer'][str(bl)]['langs']; tiers = r['tiers']
tc = {'high': '#4477aa', 'mid': '#ccbb44', 'low': '#ee6677', 'unk': '#bbbbbb'}
plt.figure(figsize=(6.4, 5))
for lg, v in L.items():
    plt.scatter(v['margin_mean'], v['aar10'], s=22,
                color=tc.get(tiers.get(lg, 'unk'), '#bbbbbb'), alpha=0.8)
lims = plt.xlim()
plt.plot(lims, lims, ls='--', c='gray', lw=1)
plt.axhline(0, color='k', lw=0.6)
for tier in ('high', 'mid', 'low'):
    plt.scatter([], [], color=tc[tier], label=tier)
plt.scatter([], [], color='#bbbbbb', label='other')
plt.legend(fontsize=8)
plt.xlabel('mean retrieval margin'); plt.ylabel('AaR@10 (worst decile)')
plt.title(f'Alignment-at-Risk vs mean — Qwen3-4B, layer {bl}\n(all languages fall below the diagonal)')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig3_aar_tail.png'), dpi=160)
plt.close()

# ---- F4: N=500 donor ranking (hub analysis) ---------------------------------
p500 = os.path.join(GRID, 'Qwen3-0.6B-Base-N500-local', 'metrics.json')
if os.path.exists(p500):
    r5 = json.load(open(p500))
    hub = r5['hub_at_best_layer']; tiers5 = r5['tiers']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    tiers_x = ['high', 'mid', 'low']
    fr = []
    for tier in tiers_x:
        tg = [lg for lg in hub if tiers5.get(lg) == tier]
        fr.append(np.mean([hub[lg]['best_donor'] == 'eng' for lg in tg]))
    axes[0].bar(tiers_x, fr, color=['#4477aa', '#ccbb44', '#ee6677'])
    axes[0].set_ylabel('P(English is best single donor)')
    axes[0].set_title('(a) English hub-ness is resource-graded\n(Qwen3-0.6B, N=500, 127 languages)')
    axes[0].set_ylim(0, 1)
    from collections import Counter
    cnt = Counter(v['best_donor'] for v in hub.values()).most_common(8)
    axes[1].bar([c[0] for c in cnt], [c[1] for c in cnt], color='#2171b5')
    axes[1].set_ylabel('# target languages'); axes[1].set_title('(b) Best single donor across all targets')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'fig4_hub_donors.png'), dpi=160)
    plt.close()

# ---- F5: MEXA N-sensitivity vs LFS stability --------------------------------
pilot = json.load(open(os.path.join(ROOT, 'results', 'pilot', 'Qwen3-0.6B-Base', 'metrics.json')))
n300 = res['Qwen3-0.6B-Base']
plt.figure(figsize=(6.8, 4.2))
for r_, n, c in ((pilot, 'N=100 (26 langs)', '#6baed6'), (n300, 'N=300 (128 langs)', '#2171b5')):
    layers = sorted(int(k) for k in r_['per_layer'])
    x = np.array(layers) / max(layers)
    mexa = [np.mean([v['mexa'] for v in r_['per_layer'][str(l)]['langs'].values()]) for l in layers]
    lfs = [r_['per_layer'][str(l)]['lfs']['lfs'] for l in layers]
    plt.plot(x, mexa, color=c, lw=2, label=f'MEXA {n}')
    plt.plot(x, lfs, color=c, lw=2, ls='--', label=f'LFS {n}')
plt.xlabel('relative depth'); plt.legend(fontsize=8)
plt.title('MEXA depends on candidate-set size; LFS is a variance share\n(same model: Qwen3-0.6B)')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig5_n_sensitivity.png'), dpi=160)
plt.close()

print('figures ->', FIGS)
for f in sorted(os.listdir(FIGS)):
    print(' ', f)
