#!/usr/bin/env python3
"""Slide-sized figures for the teaching deck. Real numbers from the paper tables and results."""
import re, os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TAB = os.path.join(ROOT, 'paper', 'iclr2027', 'tables')
OUT = os.path.join(ROOT, 'docs', 'study', 'figs', 'teaching'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 15, 'legend.fontsize': 13})
NAVY, RED, BLUE, GREEN, GREY, ORANGE = '#0b2a4a', '#b30000', '#1f5fa8', '#2e7d32', '#666666', '#e07b00'
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
def rows_of(texfile):
    out = []
    for line in open(os.path.join(TAB, texfile)):
        if '&' in line and '\\\\' in line and not line.strip().startswith(('Model', 'Base', '\\multicolumn')):
            cells = [c.strip().replace('\\', '').replace('$', '') for c in line.split('\\\\')[0].split('&')]
            out.append(cells)
    return out
FAMILY = lambda m: next(f for f in ['Qwen3', 'Qwen2.5', 'OLMo-2', 'Mistral', 'bloom', 'EuroLLM', 'SmolLM2', 'Falcon3', 'salamandra', 'granite', 'Yi', 'Llama', 'xglm', 'Sailor', 'Tower', 'occiglot', 'mGPT'] if f.lower() in m.lower())
FCOL = {'Qwen3': '#08306b', 'Qwen2.5': '#2171b5', 'OLMo-2': '#a50f15', 'Mistral': '#e6550d', 'bloom': '#756bb1', 'EuroLLM': '#31a354', 'SmolLM2': '#fd8d3c', 'Falcon3': '#636363', 'salamandra': '#c51b8a', 'granite': '#7f2704', 'Yi': '#006d2c', 'Llama': '#4292c6', 'xglm': '#969696', 'Sailor': '#bcbddc', 'Tower': '#fdae6b', 'occiglot': '#fc9272', 'mGPT': '#d9d9d9'}

# ---- 1. dip depth per model, primary set
rows = [r for r in rows_of('depth_profiles.tex') if len(r) == 8]
prim = rows[:17]; exp = rows[17:]
fig, ax = plt.subplots(figsize=(12.5, 6.2))
names = [r[0].replace('-Base', '').replace('-v0.3', '') for r in prim]; depth = [float(r[7]) for r in prim]; mins = [float(r[5]) for r in prim]
order = np.argsort(depth)
ax.barh([names[i] for i in order], [depth[i] for i in order], color=[FCOL[FAMILY(prim[i][0])] for i in order], edgecolor='k')
for k, i in enumerate(order): ax.text(depth[i] + 0.005, k, f'{depth[i]:.3f}  (minimum LFS {mins[i]:.2f} at layer {prim[i][2]})', va='center', fontsize=11)
ax.set_xlim(0, 0.5); ax.set_xlabel('dip depth = LFS at layer 0 minus minimum LFS'); ax.set_title('Experiment 1. Every one of the 17 primary models dips and recovers; how far it dips differs nine-fold', fontsize=15)
ax.axvline(0, color='k', lw=0.8); ax.grid(axis='x', alpha=0.3)
fig.tight_layout(); save(fig, 'tf_dip_depth_bar.png')

# ---- 2. FLORES replication scatter
fr = [r for r in rows_of('flores_replication.tex') if len(r) == 8]
fig, ax = plt.subplots(figsize=(8.5, 7))
OFF = {'Qwen3-8B-Base': (-95, 10), 'Llama-3.1-8B': (8, -14), 'salamandra-7b': (-105, 8), 'Qwen3-0.6B-Base': (8, 10), 'Mistral-7B-v0.3': (8, -14),
       'salamandra-2b': (8, -4), 'OLMo-2-1124-7B': (-110, 8), 'Falcon3-7B-Base': (10, -16), 'Yi-1.5-9B': (-72, 12), 'EuroLLM-1.7B': (8, 4),
       'OLMo-2-0425-1B': (8, 8), 'SmolLM2-1.7B': (12, -18), 'bloom-7b1': (-70, 12), 'bloom-1b7': (8, -14), 'granite-3.1-8b-base': (8, -4), 'Qwen3-4B-Base': (8, -4), 'Qwen3-1.7B-Base': (8, 4)}
for r in fr:
    x, y = float(r[1]), float(r[2]); ax.scatter(x, y, s=140, color=FCOL[FAMILY(r[0])], edgecolor='k', zorder=3)
    ax.annotate(r[0].replace('-Base', '').replace('-v0.3', ''), (x, y), xytext=OFF.get(r[0], (8, 4)), textcoords='offset points', fontsize=10)
ax.plot([0, 0.42], [0, 0.42], ls='--', color=GREY); ax.set_xlim(0, 0.42); ax.set_ylim(0, 0.42)
ax.set_xlabel('dip depth on NTREX (news sentences)'); ax.set_ylabel('dip depth on FLORES-200 (Wikipedia sentences)')
ax.set_title('Experiment 3. Same 17 models, a second corpus: 17 of 17 dip and recover;\ndepth ordering agrees at Spearman 0.92', fontsize=15)
ax.text(0.02, 0.39, 'dashed line: identical depth on both corpora', fontsize=11, color=GREY); ax.grid(alpha=0.3)
fig.tight_layout(); save(fig, 'tf_flores_scatter.png')

# ---- 3. subsampling intervals
bs = [r for r in rows_of('dip_bootstrap.tex') if len(r) == 5]
fig, ax = plt.subplots(figsize=(12.5, 6))
names = [r[0].replace('-Base', '').replace('-v0.3', '') for r in bs]; med = [float(r[3]) for r in bs]
lo = [float(re.findall(r'[-\d.]+', r[4])[0]) for r in bs]; hi = [float(re.findall(r'[-\d.]+', r[4])[1]) for r in bs]; grid = [float(r[2]) for r in bs]
y = np.arange(len(bs))[::-1]
ax.hlines(y, lo, hi, color=BLUE, lw=6, alpha=0.6, label='95% interval over 2,000 subsets of 300 sentences')
ax.scatter(med, y, color=BLUE, s=80, zorder=3, label='median of the subsets'); ax.scatter(grid, y, color=RED, marker='|', s=300, zorder=4, label='the paper\'s value (first 300 sentences)')
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlabel('dip depth'); ax.grid(axis='x', alpha=0.3); ax.legend(loc='lower right')
ax.set_title('Experiment 4. How sure are we of each dip depth? Intervals from resampling the sentences (14 models)', fontsize=15)
fig.tight_layout(); save(fig, 'tf_subsampling_intervals.png')

# ---- 4. hub test wins per model
wins = {}
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'grid', '*', 'metrics.json'))):
    m = os.path.basename(os.path.dirname(f))
    if m.startswith('aim2') or 'local' in m: continue
    hub = json.load(open(f)).get('hub_at_best_layer') or {}
    if hub: wins[m] = (sum(1 for r in hub.values() if isinstance(r, dict) and r.get('latent_advantage', 0) > 0), len(hub), float(np.median([r['latent_advantage'] for r in hub.values() if isinstance(r, dict)])))
fig, ax = plt.subplots(figsize=(12.5, 6))
ms = sorted(wins, key=lambda k: wins[k][0]); w = [wins[k][0] for k in ms]
ax.barh([k.replace('-Base', '').replace('-v0.3', '') for k in ms], w, color=[FCOL[FAMILY(k)] for k in ms], edgecolor='k')
for i, k in enumerate(ms): ax.text(w[i] + 1, i, f'{w[i]} of {wins[k][1]}   (median gain +{wins[k][2]:.2f} R²)', va='center', fontsize=11)
ax.set_xlim(0, 185); ax.set_xlabel('languages (out of 127) for which the multilingual average predicts better than English')
ax.set_title('Experiment 6. Is the shared middle English? For 113 to 124 of 127 languages, a blend of the\nother languages predicts a language\'s representations better than English does', fontsize=15)
fig.tight_layout(); save(fig, 'tf_hub_wins.png')

# ---- 5. instruct pairs
ip = [r for r in rows_of('instruct_pairs.tex') if len(r) == 7]
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(ip)); b = [float(r[1]) for r in ip]; i_ = [float(r[2]) for r in ip]
ax.bar(x - 0.2, b, 0.4, color=BLUE, edgecolor='k', label='base model'); ax.bar(x + 0.2, i_, 0.4, color=ORANGE, edgecolor='k', label='instruction-tuned model')
for k in range(len(ip)): ax.text(x[k] - 0.2, b[k] + 0.006, f'{b[k]:.3f}', ha='center', fontsize=10); ax.text(x[k] + 0.2, i_[k] + 0.006, f'{i_[k]:.3f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([r[0].split(' ')[0].replace('-Base', '') for r in ip], rotation=15); ax.set_ylabel('dip depth'); ax.legend()
ax.set_title('Experiment 10. Teaching a model to follow instructions leaves the profile alone: dip depth changes by at most 0.01', fontsize=14)
fig.tight_layout(); save(fig, 'tf_instruct_pairs.png')

# ---- 6. two targets, same measures
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=False)
for ax, title, data in [(axes[0], 'Pooled benchmark accuracy\n(33 models, 70 languages, adjusted)', [('1 - LFS-VC', 0.041, -0.05, 0.12), ('MEXA (reference)', 0.238, 0.18, 0.28), ('AaR@10 (tail)', 0.105, 0.05, 0.14)]),
                        (axes[1], 'Benefit of foreign-language context\n(19 models, adjusted)', [('1 - LFS-VC', 0.508, 0.19, 0.71), ('MEXA (reference)', 0.591, 0.33, 0.82), ('AaR@10 (tail)', 0.443, 0.09, 0.71)])]:
    names = [d[0] for d in data]; v = [d[1] for d in data]; lo = [d[1] - d[2] for d in data]; hi = [d[3] - d[1] for d in data]
    ax.bar(names, v, color=[RED, GREY, BLUE], edgecolor='k', yerr=[lo, hi], capsize=8)
    ax.axhline(0, color='k', lw=0.8); ax.set_ylim(-0.15, 0.9); ax.set_ylabel('Spearman correlation after adjustment'); ax.set_title(title, fontsize=14)
    for i, val in enumerate(v): ax.text(i, val + hi[i] + 0.03, f'{val:.2f}', ha='center', fontsize=13, weight='bold')
fig.suptitle('Experiments 8 and 9. The same measures against two behaviors: nothing predicts pooled test scores well;\nthe sentence-component share does track how much a model uses a foreign-language hint', fontsize=14, y=1.03)
fig.tight_layout(); save(fig, 'tf_two_targets.png')

# ---- 7. unseen families card
fig, ax = plt.subplots(figsize=(12.5, 5.2)); ax.axis('off'); ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.2)
ax.text(6.25, 4.9, 'Experiment 5. Predictions written down BEFORE two new model families were run', ha='center', fontsize=17, weight='bold')
ax.text(0.4, 4.3, 'Frozen prediction', fontsize=14, weight='bold'); ax.text(7.6, 4.3, 'What we then measured', fontsize=14, weight='bold'); ax.text(11.5, 4.3, 'Result', fontsize=14, weight='bold')
rows_ = [('Both families show an interior minimum\nwith both endpoints higher', 'SmolLM2-1.7B and Falcon3-7B:\nboth dip and recover', 'pass'),
         ('SmolLM2 (English-heavy data):\ndip depth at most 0.15', 'dip depth 0.068', 'pass'),
         ('Falcon3 (mixed data): dip depth\nbetween the shallow and deep groups', 'dip depth 0.140', 'pass'),
         ('The training-data coefficient stays\npositive with 12 models', 'coefficient +0.272, p = 4e-6', 'pass')]
for i, (p, o, r) in enumerate(rows_):
    y = 3.6 - i * 0.85
    ax.add_patch(Rectangle((0.2, y - 0.38), 12.1, 0.8, fc='#f7f7f7' if i % 2 else 'white', ec='none'))
    ax.text(0.4, y, p, fontsize=12, va='center'); ax.text(7.6, y, o, fontsize=12, va='center'); ax.text(11.5, y, r.upper(), fontsize=13, va='center', color=GREEN, weight='bold')
ax.text(6.25, 0.15, 'Four of four passed. Writing predictions down first is what stops us from fitting the story to the data.', ha='center', fontsize=12.5, style='italic')
save(fig, 'tf_heldout_card.png')

# ---- 8. what each check catches
fig, ax = plt.subplots(figsize=(14, 5.6)); ax.axis('off'); ax.set_xlim(0, 14); ax.set_ylim(0, 5.6)
ax.text(7, 5.3, 'Experiment 11. Break the representations on purpose and see which check notices', ha='center', fontsize=17, weight='bold')
cols = ['Injected fault', 'LFS', 'MEXA', 'Nested maps', 'CVP (raw variance)']
xs = [0.3, 4.9, 6.9, 8.9, 11.3]
for x, c in zip(xs, cols): ax.text(x, 4.6, c, fontsize=13, weight='bold')
data = [('Per-language sideways shift', 'rises', 'changes', 'recovered (M1)', 'unchanged'),
        ('Rotation, 0.1 to 0.8 rad', 'unchanged*', 'changes', 'recovered (M3)', 'unchanged'),
        ('Uniform shrink of everything', 'BLIND', 'BLIND', 'BLIND', 'fires: (1-m)^2'),
        ('Shrink non-English to center', 'small move', 'within 0.01', 'not the target', 'fires')]
for i, row in enumerate(data):
    y = 3.9 - i * 0.85
    ax.add_patch(Rectangle((0.2, y - 0.35), 13.6, 0.75, fc='#f7f7f7' if i % 2 else 'white', ec='none'))
    for x, c in zip(xs, row):
        col = RED if c.startswith('BLIND') else (GREEN if ('fires' in c or 'recovered' in c) else 'k')
        ax.text(x, y, c, fontsize=12, va='center', color=col, weight='bold' if col != 'k' else 'normal')
ax.text(7, 0.25, '* on the standardized grid. Lesson: a ratio cannot see uniform shrinking, so the raw components must always be printed beside it.', ha='center', fontsize=12, style='italic')
save(fig, 'tf_fault_matrix.png')

# ---- 9. hint test schematic
fig, ax = plt.subplots(figsize=(13, 5.6)); ax.axis('off'); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6)
ax.text(6.5, 5.3, 'Experiment 8. The hint test: does a first sentence in another language help predict the next English sentence?', ha='center', fontsize=15, weight='bold')
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06', fc=fc, ec='#333', lw=1.5))
    ax.text(x + w / 2, y + h - 0.35, title, ha='center', fontsize=13, weight='bold'); ax.text(x + w / 2, y + h / 2 - 0.2, body, ha='center', va='center', fontsize=11.5)
box(0.3, 2.6, 3.9, 2.3, 'Matched context', 'Sentence 1 of the same news story,\nin German (or any of 41 languages)\n+ Sentence 2 in English', '#e2f0d9')
box(0.3, 0.2, 3.9, 2.3, 'Mismatched context', 'Sentence 1 from a DIFFERENT story,\nsame language, same length\n+ the same Sentence 2 in English', '#f8d7da')
box(4.8, 1.4, 3.6, 2.3, 'Score', 'How surprised is the model by\nSentence 2? (its loss on the true\nwords, with each context)\n\nbenefit = loss(mismatched)\n            - loss(matched)', '#fff3cd')
box(9.0, 1.4, 3.7, 2.3, 'Result over 19 models', 'Models with a smaller language\nshare at the dip layer get more\nbenefit from the foreign hint:\nSpearman 0.51, interval [0.19, 0.71],\nafter adjusting for data, family,\nscript, and tokenizer', '#dbe9f6')
for (x1, y1, x2, y2) in [(4.2, 3.75, 4.8, 2.9), (4.2, 1.35, 4.8, 2.2), (8.4, 2.55, 9.0, 2.55)]:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=20, lw=2, color='#333'))
save(fig, 'tf_hint_test.png')

# ---- 10. the proposal's three questions and our three answers (clean version)
fig, ax = plt.subplots(figsize=(15, 6)); ax.axis('off'); ax.set_xlim(0, 15); ax.set_ylim(0, 6)
ax.text(7.5, 5.7, 'The proposal asked three things about the inside of the model. Here is what the ruler found.', ha='center', fontsize=16, weight='bold')
qa = [('How strong is the language\npart, layer by layer?', 'Large everywhere and smallest\nin the middle: 0.95 at the door,\n0.61 at the dip, 0.96 at the exit\n(Qwen3-8B).\nSame shape in all 36 measured\nprofiles (33 models).', '#fff3cd'),
      ('Is the shared middle really\nshared, and is it English?', 'Shared: yes. English: no.\nA blend of the other languages\npredicts each language better\nthan English does, for 113 to\n124 of 127 languages.', '#e2f0d9'),
      ('Is the map between languages\nsimple or complicated?', 'Simple. A sideways shift plus\na stretch explains 63 to 97\npercent of the difference;\nrotation adds under 1 percent.', '#dbe9f6')]
for i, (q, a, fc) in enumerate(qa):
    x = 0.4 + i * 4.85
    ax.add_patch(FancyBboxPatch((x, 0.4), 4.45, 4.7, boxstyle='round,pad=0.06', fc=fc, ec='#333', lw=1.5))
    ax.text(x + 2.225, 4.55, q, ha='center', va='center', fontsize=13.5, weight='bold'); ax.text(x + 2.225, 2.2, a, ha='center', va='center', fontsize=12.5)
save(fig, 'tf_three_answers.png')
print(sorted(os.listdir(OUT)))
