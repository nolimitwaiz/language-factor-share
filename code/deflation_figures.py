#!/usr/bin/env python3
"""Deflation/attribution figures -> paper/figs/ (run after attribution.py)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATT = os.path.join(ROOT, 'results', 'deflation', 'attribution')
FIGS = os.path.join(ROOT, 'paper', 'figs')

mc = pd.read_csv(os.path.join(ATT, 'metric_correlations.csv'))
al = pd.read_csv(os.path.join(ATT, 'alphas.csv'))
dm = pd.read_csv(os.path.join(ATT, 'downstream_metrics.csv'))

# ---- F8: the money plot — raw vs residual correlation per metric ------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
metrics = ['mexa', 'aar10']
labels = {'mexa': 'MEXA\n(reference)', 'aar10': 'AaR@10\n(tail measure)'}
x = np.arange(len(metrics))
raw = [mc[(mc.metric == m) & (mc.target == 'raw_acc')].spearman.iloc[0] for m in metrics]
res = [mc[(mc.metric == m) & (mc.target == 'alpha')].spearman.iloc[0] for m in metrics]
b1 = axes[0].bar(x - 0.18, raw, 0.36, label='raw correlation (what the field reports)',
                 color='#9ecae1')
b2 = axes[0].bar(x + 0.18, res, 0.36, label='residual correlation (after\n'
                 'data size, family, script, fertility)', color='#08519c')
for bars in (b1, b2):
    for b in bars:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                     f'{b.get_height():.2f}', ha='center', fontsize=9)
axes[0].set_xticks(x); axes[0].set_xticklabels([labels[m] for m in metrics])
axes[0].set_ylabel('Spearman correlation with Belebele accuracy')
axes[0].set_ylim(0, 1)
axes[0].set_title('(a) Raw vs residual (confounder-adjusted) metric validity\n(EB-shrunk residuals; estimator range 0.19 to 0.31)')
axes[0].legend(fontsize=8, loc='upper right')

# ---- (b) alpha vs metric scatter for our metric ------------------------------
sub = al.dropna(subset=['aar10', 'alpha'])
axes[1].scatter(sub.aar10, sub.alpha, s=14, alpha=0.5, c='#08519c')
axes[1].axhline(0, color='gray', lw=0.7)
axes[1].set_xlabel('AaR@10 (intrinsic, per model x language)')
axes[1].set_ylabel('transfer residual (raw)')
rho = sub[['aar10', 'alpha']].corr(method='spearman').iloc[0, 1]
axes[1].set_title(f'(b) AaR@10 vs transfer residual (rho = {rho:.2f}, n = {len(sub)})')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig8_deflation.png'), dpi=160)
plt.close()

# ---- F9: PaR@10 vs mean accuracy (downstream tail table, visual) ------------
plt.figure(figsize=(7, 4.6))
order = dm.sort_values('mean_acc')
y = np.arange(len(order))
plt.barh(y + 0.18, order.mean_acc, 0.36, label='mean accuracy', color='#9ecae1')
plt.barh(y - 0.18, order['PaR@10'], 0.36, label='PaR@10 (worst decile)', color='#cb181d')
plt.axvline(0.25, color='k', lw=0.8, ls='--')
plt.text(0.252, len(order) - 0.4, 'chance', fontsize=8)
plt.yticks(y, order.model, fontsize=8)
plt.xlabel('Belebele accuracy (0-shot, 122 languages)')
plt.title('Mean accuracy vs Performance-at-Risk: the tail lags everywhere')
plt.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig9_par.png'), dpi=160)
plt.close()
print('wrote fig8_deflation.png, fig9_par.png')
