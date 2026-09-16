#!/usr/bin/env python3
"""Flat-style concept figures for the deck (problem, proposal, two pictures, answers, limits, next aim)."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'docs', 'study', 'figs', 'pipeline'); os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, ORANGE, GREY, RED = '#1b365d', '#178a7a', '#d9a300', '#6b7280', '#b23a3a'
BLUE_F, BLUE_E, TEAL_F, TEAL_E, OR_F, OR_E = '#dbeafe', '#2c4a7a', '#cdeee8', '#178a7a', '#ffe4a8', '#d9a300'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
def fig_ax(w=14, h=5.4):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off'); return fig, ax
def label(ax, x, y, s, size=16, color=NAVY, ha='center'): ax.text(x, y, s, ha=ha, va='center', fontsize=size, color=color, weight='bold')
def note(ax, x, y, s, size=12.5, color=NAVY, ha='center'): ax.text(x, y, s, ha=ha, va='center', fontsize=size, color=color)
def box(ax, x, y, w, h, fc=BLUE_F, ec=BLUE_E, lw=1.6): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec=ec, lw=lw))
def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18, lw=lw, color=color, shrinkA=0, shrinkB=0))
def save(fig, name): fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---- C1 the problem: real 0-shot Belebele bars, flat style
f = glob.glob(os.path.join(ROOT, 'results', 'belebele0', 'Qwen3-8B-Base', '*', 'results_*.json'))[0]
res = json.load(open(f))['results']
want = [('eng_Latn', 'English'), ('deu_Latn', 'German'), ('fra_Latn', 'French'), ('spa_Latn', 'Spanish'), ('rus_Cyrl', 'Russian'), ('arb_Arab', 'Arabic'), ('pes_Arab', 'Persian'), ('swh_Latn', 'Swahili'), ('khm_Khmr', 'Khmer'), ('amh_Ethi', 'Amharic')]
names, accs = zip(*[(n, float(res['belebele_' + c]['acc,none'])) for c, n in want if 'belebele_' + c in res])
fig, ax = fig_ax(14, 5.4)
label(ax, 4.0, 5.0, 'The same reading test, ten languages, one model (Qwen3-8B, zero-shot Belebele)', 15)
x0, w = 0.9, 0.62
for i, (n, a) in enumerate(zip(names, accs)):
    x = x0 + i * 1.28; h = a * 3.6
    ax.add_patch(Rectangle((x, 0.9), w, h, fc=(BLUE_F if i == 0 else '#eef4fb'), ec=BLUE_E, lw=1.4))
    note(ax, x + w / 2, 0.9 + h + 0.18, f'{a:.2f}', 12); note(ax, x + w / 2, 0.6, n, 11, GREY)
ax.plot([0.7, 13.6], [0.9 + 0.25 * 3.6, 0.9 + 0.25 * 3.6], color=GREY, ls=':', lw=1.2); note(ax, 13.5, 0.9 + 0.25 * 3.6 + 0.2, 'guessing: 0.25', 10.5, GREY, ha='right')
note(ax, 7.0, 0.15, 'The model is a box of numbers nobody wrote by hand. To close this gap, Aim 1 first asks what happens inside.', 12.5, NAVY)
save(fig, 'c1_problem.png')

# ---- C2 the proposal's three stages with the open question
fig, ax = fig_ax(14, 5.2)
label(ax, 7, 4.85, 'The proposal\'s picture of the model (section 3.1.1)', 16)
stages = [('Stage 1: take in the word', 'language-dependent', BLUE_F, BLUE_E), ('Stage 2: think about content', 'should be shared across languages', TEAL_F, TEAL_E), ('Stage 3: pick the output word', 'language-dependent', BLUE_F, BLUE_E)]
for i, (t, s, fc, ec) in enumerate(stages):
    x = 0.6 + i * 4.5; box(ax, x, 2.4, 3.9, 1.6, fc=fc, ec=ec); label(ax, x + 1.95, 3.5, t, 12.5); note(ax, x + 1.95, 2.9, s, 11.5, GREY)
    if i < 2: arrow(ax, x + 3.95, 3.2, x + 4.45, 3.2)
note(ax, 7, 1.6, 'Open question, quoted from the proposal: "It is not clear how strong the language component in the representation is, or should be."', 12.5, RED)
note(ax, 7, 1.05, 'Second question: is the map between the language spaces simple, as it was for word vectors, or complicated at deeper layers?', 12.5, NAVY)
note(ax, 7, 0.45, r'LFS answers the first with one number per layer: $\mathrm{LFS}^{(k)}$ = the language share of the systematic spread.', 12.5, GREY)
save(fig, 'c2_proposal.png')

# ---- C3 two possible pictures (dots: color = language, shape = sentence)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
rng = np.random.default_rng(0)
LC = ['#1f5fa8', '#e07b00', '#2e7d32', '#b23a3a', '#7b4fa0', '#7a5230']; MK = ['o', 's', '^']
ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_title('Picture A: one shared space', color=NAVY, fontsize=16, weight='bold')
ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.9, 0.82, boxstyle='round,pad=0,rounding_size=0.03', fc='#f5f8fc', ec=BLUE_E, lw=1.6))
for j, (cx, cy) in enumerate([(0.3, 0.62), (0.55, 0.35), (0.75, 0.65)]):
    for i, c in enumerate(LC): ax.scatter(cx + rng.normal(0, 0.03), cy + rng.normal(0, 0.03), c=c, marker=MK[j], s=110, edgecolor='k', zorder=3)
ax.text(0.5, 0.0, 'same sentence (shape) lands in the same place, whatever the language (color)', ha='center', fontsize=11, color=GREY)
ax = axes[1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_title('Picture B: an English space with copies', color=NAVY, fontsize=16, weight='bold')
ax.add_patch(FancyBboxPatch((0.05, 0.3), 0.5, 0.6, boxstyle='round,pad=0,rounding_size=0.03', fc=BLUE_F, ec=BLUE_E, lw=1.6)); ax.text(0.3, 0.84, 'English', ha='center', fontsize=12, color=NAVY, weight='bold')
for j, m in enumerate(MK): ax.scatter(0.15 + 0.15 * j, 0.55, c=LC[0], marker=m, s=120, edgecolor='k', zorder=3)
for i, c in enumerate(LC[1:]):
    x0 = 0.62 + 0.18 * (i % 2); y0 = 0.66 - 0.2 * (i // 2)
    ax.add_patch(Rectangle((x0, y0), 0.15, 0.13, fc='white', ec=c, lw=1.6))
    for j, m in enumerate(MK): ax.scatter(x0 + 0.03 + 0.045 * j, y0 + 0.065, c=c, marker=m, s=40, edgecolor='k', zorder=3)
ax.text(0.5, 0.0, 'every other language gets its own smaller copy', ha='center', fontsize=11, color=GREY)
fig.suptitle('Which is true inside the model? Aim 1 asks for a measurement that can tell.', color=NAVY, fontsize=14, y=1.02)
fig.tight_layout(); save(fig, 'c3_two_pictures.png')

# ---- C4 answers to the proposal, with real values
fig, ax = fig_ax(15, 6.2)
label(ax, 7.5, 5.85, 'What the measurement answers, with the numbers behind each answer', 16)
rows = [('How strong is the\nlanguage component?', 'Large everywhere,\nsmallest at an interior layer', 'Qwen3-8B: 0.945 at layer 0, 0.608 at layer 19,\n0.958 at layer 36; same shape in 36 of 36 profiles'),
        ('Is the shared middle\nEnglish?', 'No: a multilingual average\npredicts each language better', r'113 to 124 of 127 languages;' + '\n' + r'median gain +0.06 to +0.16 $R^2$'),
        ('Is the map between\nlanguages simple?', 'Mostly an offset\nplus a scale', '63 to 97 percent of held-out misalignment;\nrotation under 1 percent'),
        ('Does it show\nin behavior?', 'Yes for foreign-language context use;\nno for pooled test scores', 'Spearman 0.51 [0.19, 0.71] on 19 models;\n' + r'0.04 [$-$0.05, 0.12] on the pooled panel')]
for i, (q, a, v) in enumerate(rows):
    y = 4.85 - i * 1.22
    box(ax, 0.4, y - 0.5, 3.4, 1.05, fc=BLUE_F); note(ax, 2.1, y, q, 12)
    arrow(ax, 3.85, y, 4.2, y); box(ax, 4.25, y - 0.5, 4.2, 1.05, fc=TEAL_F, ec=TEAL_E); note(ax, 6.35, y, a, 11.5)
    arrow(ax, 8.5, y, 8.85, y); box(ax, 8.9, y - 0.5, 5.7, 1.05, fc='white', ec=BLUE_E); note(ax, 11.75, y, v, 11)
note(ax, 2.1, 0.3, 'question from the proposal', 11, GREY); note(ax, 6.35, 0.3, 'answer', 11, GREY); note(ax, 11.75, 0.3, 'value, from the stored results', 11, GREY)
save(fig, 'c4_answers.png')

# ---- C5 limits and reporting rule
fig, ax = fig_ax(15, 5.4)
label(ax, 7.5, 5.05, 'Two limits, found on purpose, and the reporting rule they impose', 16)
for k, (sc, x0, lab) in enumerate([(1.0, 1.2, 'before'), (0.3, 4.9, 'after a shrinking objective')]):
    W, H = 2.4 * sc, 1.6 * sc
    ax.add_patch(Rectangle((x0, 2.0), W * 0.45, H, fc=BLUE_F, ec=BLUE_E, lw=1.6)); ax.add_patch(Rectangle((x0 + W * 0.45, 2.0), W * 0.55, H, fc=TEAL_F, ec=TEAL_E, lw=1.6))
    note(ax, x0 + W / 2, 3.85, lab, 12); note(ax, x0 + W / 2, 1.65, 'share = 0.45', 11.5, GREY)
note(ax, 3.6, 1.05, 'Limit 1. A share cannot see uniform shrinking.', 13, NAVY)
note(ax, 3.6, 0.55, 'Real case: raw content variance 13,791 to 699 while LFS moved 0.445 to 0.493.\nSo the raw components are always printed beside the share.', 11, GREY)
box(ax, 7.9, 0.9, 6.6, 3.5, fc='white', ec=BLUE_E)
note(ax, 11.2, 3.95, 'Limit 2. Not a training objective.', 13, NAVY)
note(ax, 11.2, 2.95, 'Objectives that pushed LFS down moved the retrieval-tail\nalignment the other way (Pearson +0.92 over 18\ncondition-seed points); training on LFS itself doubled\nthe dip while accuracy fell.', 11)
note(ax, 11.2, 1.55, 'The tested interventions did not establish that optimizing\nLFS improves downstream behavior. Use it to choose the\nlayer, watch the components, and judge a fix; not as a loss.', 11, GREY)
save(fig, 'c5_limits.png')

# ---- C6 what the next aim gets
fig, ax = fig_ax(14, 4.6)
label(ax, 7, 4.25, 'What this hands to the next aim (improving multilinguality)', 16)
cards = [('Where to act', 'the interior layer with the smallest\nlanguage share; it differs by model\n(layer 19 for Qwen3-8B, layer 2 for Mistral)', BLUE_F, BLUE_E),
         ('What to act on', 'the per-language offset: removing it from\nsaved states deleted 97 percent of the\nlanguage part in a zero-GPU pre-test', TEAL_F, TEAL_E),
         ('How to check a fix', r'print $SS_{lang}$, $SS_{con}$, the residual, and' + '\nthe norm beside the share, and run\nthe behavioral test, never the share alone', OR_F, OR_E)]
for i, (t, b, fc, ec) in enumerate(cards):
    x = 0.5 + i * 4.5; box(ax, x, 0.5, 4.1, 3.2, fc=fc, ec=ec); label(ax, x + 2.05, 3.2, t, 14); note(ax, x + 2.05, 1.9, b, 11.5)
save(fig, 'c6_next_aim.png')
# ---- C7 how we checked it
fig, ax = fig_ax(15, 6.6)
label(ax, 7.5, 6.25, 'How the measurement was checked (each with its result)', 16)
checks = [('Second corpus', 'same 17 models on FLORES-200,\nWikipedia sentences, 116 languages', '17 of 17 dip and recover;\ndepth ordering Spearman 0.92', BLUE_F, BLUE_E),
          ('Resampled sentences', '2,000 subsets of 300 sentences\nfrom 1,500, at the stored layers', 'ordering Spearman >= 0.90 in 2,000\nof 2,000 draws (median 0.996)', TEAL_F, TEAL_E),
          ('Sealed predictions', 'four predictions for two unseen\nmodel families, frozen before the runs', '4 of 4 passed', OR_F, OR_E),
          ('Injected faults', 'shift, rotation, uniform shrink,\nnon-English shrink, into saved states', 'shift and rotation recovered;\nuniform shrink invisible to any ratio', BLUE_F, BLUE_E),
          ('Instruction tuning', 'six base and instruction-tuned\npairs on the same grid', 'dip depth changes by at most 0.01', TEAL_F, TEAL_E)]
W, H = 4.55, 2.55
for i, (t_, how, res, fc, ec) in enumerate(checks):
    row, col = divmod(i, 3)
    x = (0.3 + col * 4.85) if row == 0 else (2.75 + col * 4.85)
    y = 3.25 if row == 0 else 0.35
    box(ax, x, y, W, H, fc=fc, ec=ec); label(ax, x + W/2, y + H - 0.38, t_, 13)
    note(ax, x + W/2, y + H - 1.0, how, 10.5, GREY); note(ax, x + W/2, y + 1.15, 'result', 9.5, GREY); note(ax, x + W/2, y + 0.6, res, 11, NAVY)
save(fig, 'c7_checks.png')
print(sorted(f for f in os.listdir(OUT) if f.startswith('c')))
