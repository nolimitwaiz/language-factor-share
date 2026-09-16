#!/usr/bin/env python3
"""Flat pipeline-style figures for the computation steps of LFS, with notation.
Style: light-blue boxes, navy labels above, arrows between stages, white background."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'docs', 'study', 'figs', 'pipeline'); os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, ORANGE = '#1b365d', '#178a7a', '#d9a300'
BLUE_F, BLUE_E = '#dbeafe', '#2c4a7a'
TEAL_F, TEAL_E = '#cdeee8', '#178a7a'
OR_F, OR_E = '#ffe4a8', '#d9a300'
GREY = '#6b7280'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})

def fig_ax(w=14, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off'); return fig, ax
def label(ax, x, y, text, size=17, color=NAVY, ha='center', weight='bold'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=size, color=color, weight=weight)
def note(ax, x, y, text, size=13, color=NAVY, ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=size, color=color)
def box(ax, x, y, w, h, fc=BLUE_F, ec=BLUE_E, lw=1.6, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))
def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18, lw=lw, color=color, shrinkA=0, shrinkB=0))
def grid(ax, x, y, w, h, rows, cols, fc='white', ec=BLUE_E, lw=1.2, hl=None, hl_fc=OR_F):
    cw, ch = w / cols, h / rows
    for i in range(rows):
        for j in range(cols):
            f = hl_fc if hl is not None and (i, j) == hl else fc
            ax.add_patch(Rectangle((x + j * cw, y + (rows - 1 - i) * ch), cw, ch, fc=f, ec=ec, lw=lw))
def slab(ax, x, y, w, h, d=0.22, fc=BLUE_F, ec=BLUE_E):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.4))
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)], closed=True, fc='#c7dcf5', ec=ec, lw=1.4))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)], closed=True, fc='#b3cdee', ec=ec, lw=1.4))
def save(fig, name): fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------------- 1. sentence -> one vector per layer (their sketch, with notation)
fig, ax = fig_ax(16.6, 5.4)
label(ax, 1.7, 4.8, 'Token vectors', 15)
for k, tok in enumerate(['The', 'cat', 'sits']):
    box(ax, 0.6 + k * 0.78, 2.45, 0.7, 0.7); note(ax, 0.95 + k * 0.78, 2.8, tok, 15)
note(ax, 1.7, 1.55, r'$x_1, \dots, x_T \in \mathbb{R}^{D}$', 15); note(ax, 1.7, 1.05, 'T tokens, D = 4,096', 12, GREY)
arrow(ax, 3.1, 2.8, 3.8, 2.8)
label(ax, 5.2, 4.8, 'Transformer blocks', 15)
for k, (yy, fc) in enumerate([(3.7, BLUE_F), (2.55, OR_F), (1.4, BLUE_F)]):
    slab(ax, 4.1, yy, 2.0, 0.42, fc=fc, ec=(OR_E if fc == OR_F else BLUE_E))
note(ax, 5.2, 3.35, r'$\vdots$', 16); note(ax, 5.2, 2.2, r'$\vdots$', 16)
note(ax, 5.2, 0.85, r'blocks $k = 1 \dots K$;  layer 0 = embeddings', 12, GREY)
label(ax, 8.6, 4.8, 'Hidden states', 15)
for k, yy in enumerate([3.45, 2.3, 1.15]):
    grid(ax, 7.7, yy, 1.8, 0.85, 3, 4); arrow(ax, 6.35, [3.91, 2.76, 1.61][k], 7.65, yy + 0.42, color=(ORANGE if k == 1 else NAVY))
note(ax, 8.6, 0.6, r'$h^{(k)}_t \in \mathbb{R}^{D}$, one row per token', 12, GREY)
label(ax, 11.6, 4.8, 'Mean over tokens', 15)
for k, yy in enumerate([3.45, 2.3, 1.15]):
    arrow(ax, 9.6, yy + 0.42, 10.7, yy + 0.42, color=TEAL); grid(ax, 10.75, yy + 0.25, 1.7, 0.35, 1, 4, fc=TEAL_F, ec=TEAL_E)
note(ax, 11.6, 0.6, r'$\bar h^{(k)} = \frac{1}{T}\sum_{t=1}^{T} h^{(k)}_t$  (fp32)', 13, NAVY)
label(ax, 15.0, 4.8, 'One vector per layer', 15)
for k, yy in enumerate([3.45, 2.3, 1.15]):
    arrow(ax, 12.55, yy + 0.42, 13.9, yy + 0.42, color=TEAL); grid(ax, 13.95, yy + 0.25, 1.7, 0.35, 1, 4, fc=TEAL_F, ec=TEAL_E)
note(ax, 14.9, 0.6, r'$\bar h^{(k)} \in \mathbb{R}^{D}$ for $k = 0 \dots K$', 12, GREY)
save(fig, 'pl1_sentence_to_vectors.png')

# ---------------- 2. the grid: languages x sentences, one grid per layer
fig, ax = fig_ax(14, 5.6)
label(ax, 1.9, 5.1, 'Parallel sentences', 16)
langs = ['English', 'German', 'Arabic', '...', 'lang. L']; sents = ['sent. 1', 'sent. 2', '...', 'sent. N']
for i, l in enumerate(langs):
    box(ax, 0.5, 3.9 - i * 0.62, 1.15, 0.5); note(ax, 1.075, 4.15 - i * 0.62, l, 11.5)
    box(ax, 1.85, 3.9 - i * 0.62, 1.15, 0.5, fc='white'); note(ax, 2.425, 4.15 - i * 0.62, ['The cat sits', 'Die Katze sitzt', 'Arabic text', '...', 'same sentence'][i], 10.5, GREY)
note(ax, 1.75, 0.95, r'$s_{\ell c}$ = sentence $c$ in language $\ell$', 13); note(ax, 1.75, 0.5, '128 languages x 300 sentences, translated', 11.5, GREY)
arrow(ax, 3.2, 2.7, 4.0, 2.7)
label(ax, 5.1, 4.55, 'Run each one', 15); slab(ax, 4.3, 2.3, 1.6, 0.8); note(ax, 5.1, 2.7, 'model,\nfrozen', 12)
note(ax, 5.0, 1.55, 'one pass per text,\nmean over tokens', 11.5, GREY)
arrow(ax, 6.15, 2.7, 6.75, 2.7)
label(ax, 9.6, 5.1, 'The grid at one layer', 16)
gx, gy, gw, gh = 7.5, 1.15, 4.3, 3.2
grid(ax, gx, gy, gw, gh, 5, 6, hl=(1, 1))
for i, l in enumerate(['eng', 'deu', 'arb', '...', 'L']): note(ax, gx - 0.12, gy + gh - (i + 0.5) * gh / 5, l, 11, ha='right')
for j, s in enumerate(['1', '2', '3', '4', '...', 'N']): note(ax, gx + (j + 0.5) * gw / 6, gy + gh + 0.22, s, 11.5)
note(ax, gx + gw / 2, gy - 0.35, r'cell $(\ell, c)$ holds one vector $\bar h^{(k)}_{\ell c} \in \mathbb{R}^{d}$', 13)
note(ax, gx + gw / 2, gy - 0.8, r'$X^{(k)} \in \mathbb{R}^{L \times N \times d}$,  L = 128, N = 300, d = 4,096 for Qwen3-8B', 13, NAVY)
arrow(ax, 11.95, 2.7, 12.45, 2.7, color=TEAL)
label(ax, 13.15, 5.1, 'Per layer', 16)
for k in range(3): grid(ax, 12.55 + k * 0.18, 2.0 + k * 0.28, 1.0, 1.0, 3, 3, fc='white')
note(ax, 13.15, 1.35, 'one grid for each\nof the K+1 layers', 11.5, GREY)
save(fig, 'pl2_grid.png')

# ---------------- 3. standardization per coordinate
fig, ax = fig_ax(14, 5.2)
label(ax, 2.6, 4.75, 'One coordinate, raw', 16); note(ax, 2.6, 4.3, r'values of coordinate $d$ over all $LN$ cells', 12, GREY)
rng = np.random.default_rng(3); vals = np.clip(rng.normal(224, 162, 12), -80, 560)
for k, v in enumerate(vals): ax.add_patch(Rectangle((0.7 + k * 0.32, 1.45), 0.24, v / 260, fc=BLUE_F, ec=BLUE_E))
ax.plot([0.6, 4.6], [1.45, 1.45], color=GREY, lw=0.8)
note(ax, 2.6, 0.75, r'e.g. coordinate 2276: mean 224, s.d. 162', 12, NAVY); note(ax, 2.6, 0.35, r'coordinate 1082: mean 0.1, s.d. 0.9 (Qwen3-8B, layer 19)', 11, GREY)
arrow(ax, 5.0, 2.7, 5.9, 2.7)
label(ax, 7.0, 4.75, 'Standardize', 16); box(ax, 5.9, 2.1, 2.2, 1.3); note(ax, 7.0, 2.75, r'$z_{\ell c d} = \dfrac{h_{\ell c d} - \bar h_{\cdot\cdot d}}{s_d}$', 15)
note(ax, 7.0, 1.75, 'mean and s.d. over the whole grid, jointly', 10.5, GREY)
note(ax, 7.0, 1.2, 'never per language: that would\nremove the thing we measure', 11, NAVY)
arrow(ax, 8.1, 2.7, 9.0, 2.7)
label(ax, 11.3, 4.75, 'Same coordinate, standardized', 16); note(ax, 11.3, 4.3, 'mean 0, s.d. 1 for every coordinate', 12, GREY)
zs = (vals - vals.mean()) / vals.std()
for k, v in enumerate(zs): ax.add_patch(Rectangle((9.4 + k * 0.32, 2.7), 0.24, v * 0.55, fc=TEAL_F, ec=TEAL_E))
ax.plot([9.3, 13.3], [2.7, 2.7], color=GREY, lw=0.8)
note(ax, 11.3, 0.75, 'why: without this, one loud coordinate carries 92.8 percent of the raw spread', 12, NAVY)
note(ax, 11.3, 0.35, 'and the share reads 0.347 instead of 0.608', 12, NAVY)
save(fig, 'pl3_standardize.png')

# ---------------- 4. three kinds of mean
fig, ax = fig_ax(16, 5.8)
label(ax, 3.4, 5.35, 'Language means (row averages)', 16)
gx, gy, gw, gh = 0.9, 1.7, 3.9, 2.9
grid(ax, gx, gy, gw, gh, 5, 6)
for i in range(5):
    ax.add_patch(Rectangle((gx, gy + gh - (i + 1) * gh / 5), gw, gh / 5, fc='#eaf2fb', ec='none', zorder=0))
    arrow(ax, gx + gw + 0.1, gy + gh - (i + 0.5) * gh / 5, gx + gw + 0.5, gy + gh - (i + 0.5) * gh / 5, color=NAVY, lw=1.5)
    grid(ax, gx + gw + 0.55, gy + gh - (i + 0.85) * gh / 5, 0.5, 0.42, 1, 1, fc=BLUE_F)
note(ax, 3.4, 1.05, r'$\bar z_\ell = \frac{1}{N}\sum_{c=1}^{N} z_{\ell c}$', 15); note(ax, 3.4, 0.5, 'one vector per language: what that language looks like on average', 11, GREY)
label(ax, 9.3, 5.35, 'Sentence means (column averages)', 16)
gx2 = 7.4; grid(ax, gx2, gy, gw, gh, 5, 6)
for j in range(6):
    ax.add_patch(Rectangle((gx2 + j * gw / 6, gy), gw / 6, gh, fc='#e6f6f3', ec='none', zorder=0))
    arrow(ax, gx2 + (j + 0.5) * gw / 6, gy - 0.08, gx2 + (j + 0.5) * gw / 6, gy - 0.4, color=TEAL, lw=1.5)
    grid(ax, gx2 + (j + 0.13) * gw / 6, gy - 0.9, 0.48, 0.42, 1, 1, fc=TEAL_F, ec=TEAL_E)
note(ax, 9.3, 0.35, r'$\bar z_c = \frac{1}{L}\sum_{\ell=1}^{L} z_{\ell c}$   one vector per sentence', 14)
label(ax, 14.0, 5.35, 'Grand mean', 16)
box(ax, 12.9, 1.7, 2.2, 2.9, fc='white')
grid(ax, 13.55, 3.1, 0.9, 0.42, 1, 1, fc='#eeeeee', ec=GREY); note(ax, 14.0, 2.55, r'$\bar z = \frac{1}{LN}\sum_{\ell, c} z_{\ell c}$', 14); note(ax, 14.0, 2.05, 'average of all cells', 11, GREY)
save(fig, 'pl4_means.png')

# ---------------- 5. sums of squares and the identity
fig, ax = fig_ax(14, 5.4)
label(ax, 7, 5.0, 'Split the total spread into three parts (the decomposition identity holds exactly on a balanced grid)')
parts = [('Language part', r'$SS_{lang} = N\sum_{\ell=1}^{L}\|\bar z_\ell - \bar z\|^2$', 'how far the language\naverages sit from the\ngrand mean', BLUE_F, BLUE_E),
         ('Sentence part', r'$SS_{con} = L\sum_{c=1}^{N}\|\bar z_c - \bar z\|^2$', 'how far the sentence\naverages sit from the\ngrand mean', TEAL_F, TEAL_E),
         ('Residual', r'$SS_{res} = \sum_{\ell, c}\|z_{\ell c} - \bar z_\ell - \bar z_c + \bar z\|^2$', 'what is specific to one\n(language, sentence) pair:\ninteraction plus noise', '#f3f4f6', GREY)]
for k, (t, f, d, fc, ec) in enumerate(parts):
    x = 0.5 + k * 4.55
    box(ax, x, 1.5, 4.1, 2.9, fc=fc, ec=ec); label(ax, x + 2.05, 3.95, t, 15); note(ax, x + 2.05, 3.15, f, 14); note(ax, x + 2.05, 2.15, d, 11.5, GREY)
    if k < 2: note(ax, x + 4.33, 2.95, '+', 22)
note(ax, 7, 0.75, r'$\sum_{\ell, c}\|z_{\ell c} - \bar z\|^2 \;=\; SS_{lang} + SS_{con} + SS_{res}$', 16)
note(ax, 7, 0.25, 'the multipliers N and L appear because each language mean stands for N cells and each sentence mean for L cells', 11.5, GREY)
save(fig, 'pl5_sums_of_squares.png')

# ---------------- 6. the share, real numbers, null
d = json.load(open(os.path.join(ROOT, 'results', 'grid', 'Qwen3-8B-Base', 'metrics.json')))
fig, ax = fig_ax(14, 5.2)
label(ax, 3.4, 4.7, 'The Language-Factor Share', 16)
box(ax, 1.2, 2.55, 4.4, 1.6, fc='white'); note(ax, 3.4, 3.35, r'$\mathrm{LFS}^{(k)} = \dfrac{SS_{lang}}{SS_{lang} + SS_{con}}$', 20)
note(ax, 3.4, 2.15, 'the residual is left out on purpose: the share compares the two systematic parts', 11, GREY)
note(ax, 3.4, 1.6, 'reads 1 when only language separates the cells, 0 when only the sentence does', 12, NAVY)
note(ax, 3.4, 0.9, r'expected value under pure noise:  $\dfrac{L-1}{L+N-2} = \dfrac{127}{426} = 0.298$', 13, NAVY)
arrow(ax, 6.2, 3.35, 7.0, 3.35)
label(ax, 10.6, 4.7, 'Real numbers: Qwen3-8B, layer 19', 16)
tot = 55.2 + 35.6; w = 6.0; x0 = 7.6
ax.add_patch(Rectangle((x0, 2.6), w * 55.2 / tot, 1.0, fc=BLUE_F, ec=BLUE_E, lw=1.6)); ax.add_patch(Rectangle((x0 + w * 55.2 / tot, 2.6), w * 35.6 / tot, 1.0, fc=TEAL_F, ec=TEAL_E, lw=1.6))
note(ax, x0 + w * 55.2 / tot / 2, 3.1, r'$SS_{lang}$ = 55.2 million', 13); note(ax, x0 + w * 55.2 / tot + w * 35.6 / tot / 2, 3.1, r'$SS_{con}$ = 35.6 million', 12)
note(ax, 10.6, 1.9, r'$\mathrm{LFS}^{(19)} = \dfrac{55.2}{55.2 + 35.6} = 0.608$', 17)
note(ax, 10.6, 1.05, 'against 0.945 at layer 0 and 0.958 at layer 36; all far above 0.298', 12, NAVY)
save(fig, 'pl6_share.png')

# ---------------- 7. repeat over layers -> depth profile, with definitions
q = []
for l in range(d['n_layers']):
    v = d['per_layer'][str(l)]['lfs']; v = v.get('lfs', v.get('value')) if isinstance(v, dict) else v; q.append(float(v))
q = np.array(q); dip = int(np.argmin(q))
fig = plt.figure(figsize=(14, 5.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 5.4); ax.axis('off')
label(ax, 2.5, 4.9, 'Repeat at every layer')
for k in range(4):
    grid(ax, 0.7 + k * 0.9, 2.3, 0.8, 0.8, 3, 3, fc='white'); note(ax, 1.1 + k * 0.9, 1.95, ['layer 0', 'layer 1', '...', 'layer B'][k], 11, GREY)
note(ax, 2.5, 1.3, r'$\mathrm{LFS}^{(0)}, \mathrm{LFS}^{(1)}, \dots, \mathrm{LFS}^{(K)}$', 14); note(ax, 2.5, 0.8, 'K + 1 numbers, one per layer', 11.5, GREY)
arrow(ax, 4.5, 2.7, 5.0, 2.7)
ax2 = fig.add_axes([0.42, 0.14, 0.32, 0.68]); x = np.arange(len(q)); ax2.plot(x, q, 'o-', color=NAVY, lw=2, ms=4)
ax2.axhline(0.298, color=GREY, ls=':'); ax2.set_ylim(0.2, 1.0); ax2.set_xlabel('layer', color=NAVY); ax2.set_ylabel('LFS', color=NAVY); ax2.tick_params(colors=NAVY)
for s in ax2.spines.values(): s.set_color(BLUE_E)
ax2.annotate('', (dip, q[dip]), (dip, q[0]), arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2)); ax2.text(dip + 0.8, (q[0] + q[dip]) / 2, 'dip depth', color=ORANGE, fontsize=11, va='center')
ax2.text(len(q) - 1, 0.31, 'noise value 0.298', ha='right', fontsize=9, color=GREY); ax2.set_title('the depth profile (Qwen3-8B)', color=NAVY, fontsize=13)
label(ax, 12.0, 4.9, 'Three summary numbers')
box(ax, 10.55, 1.2, 2.9, 3.2, fc='white')
note(ax, 12.0, 3.9, r'minimum layer:  $k^* = \arg\min_k \mathrm{LFS}^{(k)}$', 12); note(ax, 12.0, 3.5, f'= {dip}', 12, GREY)
note(ax, 12.0, 2.95, r'minimum LFS:  $\mathrm{LFS}^{(k^*)}$', 12); note(ax, 12.0, 2.55, f'= {q[dip]:.3f}', 12, GREY)
note(ax, 12.0, 2.0, r'dip depth:  $\mathrm{LFS}^{(0)} - \mathrm{LFS}^{(k^*)}$', 12); note(ax, 12.0, 1.6, f'= {q[0]:.3f} - {q[dip]:.3f} = {q[0]-q[dip]:.3f}', 12, GREY)
save(fig, 'pl7_profile.png')

# ---------------- 8. worked example with notation side by side
fig, ax = fig_ax(15.5, 5.6)
label(ax, 3.2, 5.15, 'A tiny grid (one number per cell)', 16)
vals = [[2, 6, 10], [5, 9, 13]]
gx, gy, cw, ch = 1.2, 2.6, 1.1, 0.75
for j, s in enumerate(['c = 1', 'c = 2', 'c = 3']): note(ax, gx + (j + 0.5) * cw, gy + 2 * ch + 0.25, s, 12)
for i, (l, row) in enumerate(zip(['English', 'Arabic'], vals)):
    note(ax, gx - 0.15, gy + (1.5 - i) * ch, l, 12, ha='right')
    for j, v in enumerate(row):
        ax.add_patch(Rectangle((gx + j * cw, gy + (1 - i) * ch), cw, ch, fc=BLUE_F, ec=BLUE_E)); note(ax, gx + (j + 0.5) * cw, gy + (1.5 - i) * ch, str(v), 15)
    ax.add_patch(Rectangle((gx + 3 * cw + 0.2, gy + (1 - i) * ch), cw, ch, fc=OR_F, ec=OR_E)); note(ax, gx + 3.5 * cw + 0.2, gy + (1.5 - i) * ch, r'$\bar z_\ell$ = ' + str(np.mean(row)), 12)
for j in range(3):
    m = np.mean([vals[0][j], vals[1][j]]); ax.add_patch(Rectangle((gx + j * cw, gy - ch - 0.2), cw, ch, fc=TEAL_F, ec=TEAL_E)); note(ax, gx + (j + 0.5) * cw, gy - ch / 2 - 0.2, r'$\bar z_c$ = ' + f'{m:g}', 11.5)
ax.add_patch(Rectangle((gx + 3 * cw + 0.2, gy - ch - 0.2), cw, ch, fc='#eeeeee', ec=GREY)); note(ax, gx + 3.5 * cw + 0.2, gy - ch / 2 - 0.2, r'$\bar z$ = 7.5', 12)
note(ax, 3.2, 0.55, 'L = 2 languages, N = 3 sentences', 11.5, GREY)
label(ax, 10.7, 5.15, 'The same steps in notation', 16)
lines = [r'$SS_{lang} = N\sum_\ell (\bar z_\ell - \bar z)^2 = 3\,[(6-7.5)^2 + (9-7.5)^2] = 3 \times 4.5 = 13.5$',
         r'$SS_{con} = L\sum_c (\bar z_c - \bar z)^2 = 2\,[(3.5-7.5)^2 + 0 + (11.5-7.5)^2] = 2 \times 32 = 64$',
         r'$SS_{res} = \sum_{\ell,c}(z_{\ell c} - \bar z_\ell - \bar z_c + \bar z)^2 = 0$  (this grid is exactly additive)',
         r'check: $\sum_{\ell,c}(z_{\ell c} - \bar z)^2 = 77.5 = 13.5 + 64 + 0$',
         r'$\mathrm{LFS} = \dfrac{13.5}{13.5 + 64} = 0.17$;   noise value here: $\dfrac{L-1}{L+N-2} = \dfrac{1}{3} = 0.33$']
for k, ln in enumerate(lines):
    box(ax, 6.6, 4.25 - k * 0.85, 8.3, 0.7, fc='white', ec=BLUE_E if k < 4 else OR_E); note(ax, 10.75, 4.6 - k * 0.85, ln, 12)
save(fig, 'pl8_worked_example.png')
print(sorted(os.listdir(OUT)))
