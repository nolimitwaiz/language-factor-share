#!/usr/bin/env python3
"""Round-3 report figures: B2' scatter, deflation waterfall, two-events, risk-return."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'paper', 'figs')
DEF = os.path.join(ROOT, 'results', 'deflation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deflate_b2 import build, residualize, MODELS

# ---- F10: B2' scatter — raw (4B) + deflated (pooled residuals) --------------
df = build().dropna(subset=['R_content', 'mexa', 'log_tokens', 'fertility'])
d = df.copy()
d['r_R'] = residualize(d, 'R_content')
d['r_M'] = residualize(d, 'mexa')
from scipy import stats
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
g4 = d[d.model == 'Qwen3-4B-Base']
axes[0].scatter(g4.mexa, g4.R_content, s=30, c='#2171b5', alpha=0.8)
for _, row in g4.iterrows():
    if row.ntrex in ('deu', 'zho-CN', 'hin', 'swa', 'amh', 'mya', 'vie'):
        axes[0].annotate(row.ntrex, (row.mexa, row.R_content),
                         textcoords='offset points', xytext=(5, 3), fontsize=7)
r = stats.spearmanr(g4.mexa, g4.R_content).statistic
axes[0].set_xlabel('intrinsic alignment (MEXA @ best layer)')
axes[0].set_ylabel('content-specific context benefit  $R_{content}$')
axes[0].set_title(f'(a) Raw: Qwen3-4B, 38 languages  (Spearman = {r:.2f})')
COL = {'Qwen3-0.6B-Base': '#c6dbef', 'Qwen3-1.7B-Base': '#6baed6',
       'Qwen3-4B-Base': '#2171b5', 'OLMo-2-0425-1B': '#d62728'}
for m in MODELS:
    g = d[d.model == m]
    axes[1].scatter(g.r_M, g.r_R, s=22, c=COL[m], alpha=0.75,
                    label=m.replace('-Base', ''))
rp = stats.spearmanr(d.r_M, d.r_R)
axes[1].axhline(0, c='gray', lw=0.6); axes[1].axvline(0, c='gray', lw=0.6)
axes[1].set_xlabel('alignment residual (observables partialled out)')
axes[1].set_ylabel('$R_{content}$ residual')
axes[1].set_title(f'(b) Deflated, pooled: Spearman = {rp.statistic:.2f} '
                  f'(p = {rp.pvalue:.0e}, n = {len(d)})')
axes[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig10_b2scatter.png'), dpi=160)
plt.close()

# ---- F11: deflation waterfall ------------------------------------------------
wf = json.load(open(os.path.join(DEF, 'attribution', 'b2_deflated.json')))['waterfall']
names = [n for n, _ in wf['mexa']]
vm = [v for _, v in wf['mexa']]; va = [v for _, v in wf['aar10']]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(names))
for i in range(len(names)):
    if i == 0:
        ax.bar(x[i], vm[i], 0.55, color='#9ecae1')
    else:
        ax.bar(x[i], vm[i], 0.55, color='#9ecae1')
        ax.bar(x[i], vm[i-1]-vm[i], 0.55, bottom=vm[i], color='#de2d26', alpha=0.75)
        ax.text(x[i], vm[i-1]+0.005, f'−{vm[i-1]-vm[i]:.3f}', ha='center',
                fontsize=8, color='#a50f15')
    ax.text(x[i], vm[i]-0.03, f'{vm[i]:.2f}', ha='center', fontsize=9,
            color='white', fontweight='bold')
ax.plot(x, va, 'o--', color='#08306b', lw=1.4, ms=5, label='AaR@10 (same steps)')
ax.axhspan(0.19, 0.31, color='#fdd0a2', alpha=0.5,
           label='FE-logit residual range (stricter estimator, Table 3)')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Spearman correlation with Belebele accuracy')
ax.set_title('Where the correlation goes: sequential partialling of observables (MEXA bars)')
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig11_waterfall.png'), dpi=160)
plt.close()

# ---- F12: two events, one depth axis (Qwen3-4B) ------------------------------
r = json.load(open(os.path.join(ROOT, 'results', 'lens312', 'Qwen3-4B-Base_meanpool.json')))
inp = np.mean([r['curves'][l]['input_script'] for l in r['curves']], 0)
lfs = np.array(r['lfs'])
x1 = np.arange(len(inp)) / (len(inp) - 1)
x2 = np.arange(len(lfs)) / (len(lfs) - 1)
fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax1.plot(x1, inp, color='#d62728', lw=2.2, label='input-script token mass (lens)')
ax1.set_ylabel('input-script mass (lens probe)', color='#d62728')
ax2 = ax1.twinx()
ax2.plot(x2, lfs, color='#2171b5', lw=2.2, label='Language-Factor Share')
ax2.set_ylabel('LFS (variance share)', color='#2171b5')
v1 = x1[int(np.argmin(inp))]; v2 = x2[int(np.argmin(lfs))]
ax1.axvline(v1, color='#d62728', ls=':', lw=1.2)
ax1.axvline(v2, color='#2171b5', ls=':', lw=1.2)
ax1.annotate('EVENT 1\nscript abandoned', (v1, 0.55), ha='center', fontsize=9,
             color='#a50f15')
ax1.annotate('EVENT 2\nconcepts converge', (v2, 0.75), ha='center', fontsize=9,
             color='#08306b')
ax1.set_xlabel('relative depth (layer / n_layers)')
ax1.set_title('The multilingual funnel has two events (Qwen3-4B; both instruments,\n'
              'matched pooling; gap pre-registered and confirmed in all four models)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig12_twoevents.png'), dpi=160)
plt.close()

# ---- F13: risk-return scatter (mean acc vs PaR@10) ---------------------------
dm = pd.read_csv(os.path.join(DEF, 'attribution', 'downstream_metrics.csv'))
plt.figure(figsize=(7, 5))
plt.scatter(dm.mean_acc, dm['PaR@10'], s=60, c='#2171b5')
for _, row in dm.iterrows():
    plt.annotate(row.model.replace('-Base', ''), (row.mean_acc, row['PaR@10']),
                 textcoords='offset points', xytext=(6, 4), fontsize=7.5)
plt.axvline(0.25, c='k', ls='--', lw=0.8); plt.axhline(0.25, c='k', ls='--', lw=0.8)
plt.text(0.252, 0.20, 'chance', fontsize=8)
plt.xlabel('mean accuracy across languages ("return")')
plt.ylabel('PaR@10: worst-decile language accuracy ("tail risk")')
plt.title('Risk–return view of the model grid (0-shot Belebele, 122 languages)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig13_riskreturn.png'), dpi=160)
plt.close()
print('wrote fig10_b2scatter, fig11_waterfall, fig12_twoevents, fig13_riskreturn')
