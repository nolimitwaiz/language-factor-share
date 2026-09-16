#!/usr/bin/env python3
"""Twelve step-by-step pictures explaining LFS from zero. One idea per picture.
Real numbers from results/grid/Qwen3-8B-Base/metrics.json where marked."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'docs', 'study', 'figs', 'steps'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 12})
RED, BLUE, GREY = '#b30000', '#1f5fa8', '#666666'
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight'); plt.close(fig)
def card(ax, x, y, w, h, values, title, color='#dbe9f6', ec='#1f5fa8', fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', fc=color, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h + 0.03, title, ha='center', fontsize=fs + 1, weight='bold')
    for i, v in enumerate(values):
        ax.text(x + w / 2, y + h - 0.25 - i * 0.34, v, ha='center', fontsize=fs, family='monospace')

# ---------- S1: a sentence becomes number cards
fig, ax = plt.subplots(figsize=(13, 5)); ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis('off')
ax.text(6.5, 4.7, 'Step 1. The machine does not read words. It turns each piece of a sentence into a card full of numbers.', ha='center', fontsize=13, weight='bold')
ax.add_patch(FancyBboxPatch((0.3, 2.0), 3.0, 1.2, boxstyle='round,pad=0.05', fc='#fff3cd', ec='#333', lw=1.5))
ax.text(1.8, 2.6, '"The cat sleeps"', ha='center', fontsize=15)
ax.add_patch(FancyArrowPatch((3.4, 2.6), (4.4, 2.6), arrowstyle='-|>', mutation_scale=22, lw=2, color='#333'))
ax.text(3.9, 2.95, 'cut into\npieces', ha='center', fontsize=9.5)
for i, tok in enumerate(['The', 'cat', 'sleeps']):
    ax.add_patch(Rectangle((4.6 + i * 1.1, 2.25), 0.95, 0.7, fc='white', ec='#333', lw=1.3))
    ax.text(4.6 + i * 1.1 + 0.475, 2.6, tok, ha='center', va='center', fontsize=12)
ax.add_patch(FancyArrowPatch((7.9, 2.6), (8.7, 2.6), arrowstyle='-|>', mutation_scale=22, lw=2, color='#333'))
ax.text(8.3, 2.95, 'each piece\nbecomes', ha='center', fontsize=9.5)
vals = [['0.42', '-1.07', '2.31', '0.08', '...'], ['-0.15', '0.88', '1.92', '-0.61', '...'], ['0.30', '-0.44', '2.05', '0.77', '...']]
for i, (tok, v) in enumerate(zip(['"The"', '"cat"', '"sleeps"'], vals)):
    card(ax, 8.9 + i * 1.4, 1.0, 1.2, 2.2, v, tok, fs=10)
ax.text(11.0, 0.75, 'each card: 4,096 numbers tall', ha='center', fontsize=9.5, color=BLUE)
ax.text(6.5, 0.35, 'Think of each card as a very long list: 4,096 numbers tall. The numbers here are made up to show the shape; the real ones live in the saved files.', ha='center', fontsize=10.5, style='italic')
save(fig, 's01_sentence_to_cards.png')

# ---------- S2: floors
fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis('off')
ax.text(6.5, 5.3, 'Step 2. The machine is a building with floors. Every floor rewrites the cards. We photograph the cards on every floor.', ha='center', fontsize=13, weight='bold')
floors = [(0, 'floor 0: front door', '#fff3cd'), (1, 'floor 1', '#eef'), (2, 'floor 2', '#eef'), (3, '...', 'white'), (4, 'floor 19: the middle', '#e2f0d9'), (5, '...', 'white'), (6, 'floor 36: the exit', '#f8d7da')]
for k, (i, lab, fc) in enumerate(floors):
    y = 0.5 + k * 0.62
    ax.add_patch(Rectangle((1.0, y), 5.5, 0.55, fc=fc, ec='#333', lw=1.2))
    ax.text(3.75, y + 0.275, lab, ha='center', va='center', fontsize=11)
    ax.add_patch(FancyArrowPatch((6.7, y + 0.275), (7.6, y + 0.275), arrowstyle='-|>', mutation_scale=16, lw=1.5, color=GREY))
    ax.add_patch(Rectangle((7.7, y + 0.05), 1.3, 0.45, fc='#dbe9f6', ec=BLUE, lw=1.2))
    ax.text(8.35, y + 0.275, 'photo of cards', ha='center', va='center', fontsize=8.5)
ax.add_patch(FancyArrowPatch((0.6, 0.5), (0.6, 4.7), arrowstyle='-|>', mutation_scale=20, lw=2, color='#333'))
ax.text(0.35, 2.6, 'the sentence climbs up', rotation=90, va='center', fontsize=10)
ax.text(10.9, 3.0, 'Qwen3-8B has 37 floors:\nthe front door (floor 0)\nplus 36 working floors.\n\nA "floor" is what the papers\ncall a layer.\n\nWe never let the machine\nwrite anything. We only\nlook at the cards.', ha='center', va='center', fontsize=11, bbox=dict(boxstyle='round', fc='#f7f7f7', ec='#999'))
save(fig, 's02_floors.png')

# ---------- S3: mean pooling as averaging heights
fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.2]})
ax = axes[0]
h = [1.10, 1.25, 1.45]; names = ['Ali', 'Ben', 'Cara']
ax.bar(names, h, color=['#9ecae1', '#6baed6', '#3182bd'], edgecolor='k')
ax.axhline(np.mean(h), color=RED, lw=2.5, ls='--'); ax.text(2.45, np.mean(h) + 0.02, f'average {np.mean(h):.2f} m', color=RED, ha='right', fontsize=11, weight='bold')
for i, v in enumerate(h): ax.text(i, v + 0.02, f'{v:.2f} m', ha='center', fontsize=10)
ax.set_ylim(0, 1.7); ax.set_ylabel('height'); ax.set_title('An average: three kids, one "typical" height', fontsize=12)
ax.text(1, -0.35, '(1.10 + 1.25 + 1.45) / 3 = 3.80 / 3 = 1.27', ha='center', fontsize=11, transform=ax.transData)
ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')
ax.set_title('Same trick on the cards: average the pieces, one card per sentence', fontsize=12)
rows = [['0.42', '-0.15', '0.30', '0.19'], ['-1.07', '0.88', '-0.44', '-0.21'], ['2.31', '1.92', '2.05', '2.09'], ['0.08', '-0.61', '0.77', '0.08']]
heads = ['"The"', '"cat"', '"sleeps"', 'sentence card']
for j, hd in enumerate(heads):
    x = 0.4 + j * 2.3 + (0.6 if j == 3 else 0)
    ax.text(x + 0.7, 4.3, hd, ha='center', fontsize=11, weight='bold', color=RED if j == 3 else 'k')
    ax.add_patch(Rectangle((x, 1.2), 1.4, 2.9, fc='#f8d7da' if j == 3 else '#dbe9f6', ec=RED if j == 3 else BLUE, lw=1.8))
    for i in range(4):
        ax.text(x + 0.7, 3.7 - i * 0.7, rows[i][j], ha='center', va='center', fontsize=11, family='monospace')
    ax.text(x + 0.7, 1.35, '...', ha='center', fontsize=11)
ax.text(7.35, 2.6, '=', fontsize=20, ha='center', va='center')
ax.text(5.0, 0.7, 'Line 1: (0.42 + (-0.15) + 0.30) / 3 = 0.19.\nSame for all 4,096 lines.', ha='center', fontsize=10.5)
fig.text(0.5, -0.04, 'Step 3. This is "mean pooling": average the piece-cards so every sentence, long or short, in any language, becomes ONE card of the same size.', ha='center', fontsize=12, weight='bold')
fig.tight_layout(); save(fig, 's03_mean_pooling.png')

# ---------- S4: the table
fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis('off')
ax.text(6.5, 5.3, 'Step 4. Put the cards in a table: one row per language, one column per story. Down a column: same story. Across a row: same language.', ha='center', fontsize=12.5, weight='bold')
langs = ['English', 'German', 'Arabic', '...', '(128 rows)']; stories = ['story 1\n"The cat sleeps"', 'story 2\n"Rain is coming"', 'story 3\n"I lost my keys"', '...', '(300 columns)']
x0, y0, cw, ch = 2.2, 0.5, 1.9, 0.8
for j, s in enumerate(stories):
    ax.text(x0 + j * cw + cw / 2, y0 + 5 * ch + 0.1, s, ha='center', fontsize=9.5, weight='bold' if j < 3 else 'normal')
for i, l in enumerate(langs):
    yy = y0 + (4 - i) * ch
    ax.text(x0 - 0.15, yy + ch / 2, l, ha='right', va='center', fontsize=11, weight='bold' if i < 3 else 'normal')
    for j in range(5):
        xx = x0 + j * cw
        if i < 3 and j < 3:
            ax.add_patch(Rectangle((xx + 0.15, yy + 0.08), cw - 0.3, ch - 0.16, fc='#dbe9f6', ec=BLUE, lw=1.2))
            ax.text(xx + cw / 2, yy + ch / 2, 'one card\n(4,096 numbers)', ha='center', va='center', fontsize=8.5)
        else:
            ax.text(xx + cw / 2, yy + ch / 2, '...', ha='center', va='center', fontsize=12, color=GREY)
ax.text(6.5, 0.1, 'Every card in one column says the same thing in a different language. Every card in one row is the same language telling different stories.\nThere is one such table for every floor.', ha='center', fontsize=10.5, style='italic')
save(fig, 's04_table.png')

# ---------- S5: tiny worked table with averages
fig, ax = plt.subplots(figsize=(12, 5.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis('off')
ax.text(6, 5.3, 'Step 5. A tiny pretend table: 2 languages, 3 stories, and ONE number per card instead of 4,096.', ha='center', fontsize=13, weight='bold')
vals = [[2, 6, 10], [5, 9, 13]]; rl = ['English', 'Arabic']; cl = ['story 1', 'story 2', 'story 3']
x0, y0, cw, ch = 2.6, 1.6, 1.6, 0.9
for j, c in enumerate(cl): ax.text(x0 + j * cw + cw / 2, y0 + 2 * ch + 0.15, c, ha='center', fontsize=12, weight='bold')
ax.text(x0 + 3 * cw + cw / 2 + 0.2, y0 + 2 * ch + 0.15, 'language\naverage', ha='center', fontsize=11, color=RED, weight='bold')
for i in range(2):
    yy = y0 + (1 - i) * ch
    ax.text(x0 - 0.2, yy + ch / 2, rl[i], ha='right', va='center', fontsize=12, weight='bold')
    for j in range(3):
        ax.add_patch(Rectangle((x0 + j * cw, yy), cw, ch, fc='#dbe9f6', ec=BLUE, lw=1.5))
        ax.text(x0 + j * cw + cw / 2, yy + ch / 2, str(vals[i][j]), ha='center', va='center', fontsize=18)
    m = np.mean(vals[i])
    ax.add_patch(Rectangle((x0 + 3 * cw + 0.2, yy), cw, ch, fc='#f8d7da', ec=RED, lw=1.5))
    ax.text(x0 + 3 * cw + 0.2 + cw / 2, yy + ch / 2, f'{m:g}', ha='center', va='center', fontsize=18, color=RED)
ax.text(x0 - 0.2, y0 - 0.45, 'story\naverage', ha='right', va='center', fontsize=11, color=BLUE, weight='bold')
for j in range(3):
    m = np.mean([vals[0][j], vals[1][j]])
    ax.add_patch(Rectangle((x0 + j * cw, y0 - 0.95), cw, 0.8, fc='#e2f0d9', ec='#2ca02c', lw=1.5))
    ax.text(x0 + j * cw + cw / 2, y0 - 0.55, f'{m:g}', ha='center', va='center', fontsize=18, color='#2ca02c')
ax.add_patch(Rectangle((x0 + 3 * cw + 0.2, y0 - 0.95), cw, 0.8, fc='#eee', ec='#333', lw=1.5))
ax.text(x0 + 3 * cw + 0.2 + cw / 2, y0 - 0.55, '7.5', ha='center', va='center', fontsize=18)
ax.text(x0 + 3 * cw + 0.2 + cw / 2, y0 - 1.25, 'average of everything', ha='center', fontsize=10)
ax.text(9.9, 3.0, 'English average:\n(2 + 6 + 10) / 3 = 6\nArabic average:\n(5 + 9 + 13) / 3 = 9\n\nStory 1 average: (2 + 5) / 2 = 3.5\nStory 2: (6 + 9) / 2 = 7.5\nStory 3: (10 + 13) / 2 = 11.5\n\nEverything: 45 / 6 = 7.5', ha='left', va='center', fontsize=10.5, family='monospace', bbox=dict(boxstyle='round', fc='#f7f7f7', ec='#999'))
save(fig, 's05_tiny_table.png')

# ---------- S6: spread as squares
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
ax = axes[0]; ax.set_xlim(-1, 13); ax.set_ylim(-1, 6); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('Language spread: how far is each language average from 7.5?', fontsize=11.5, weight='bold')
for k, (lab, dev) in enumerate([('English 6', -1.5), ('Arabic 9', 1.5)]):
    s = abs(dev); x = 2.5 + k * 4.5
    ax.add_patch(Rectangle((x, 0.5), s, s, fc='#f8d7da', ec=RED, lw=2))
    ax.text(x + s / 2, 0.5 + s + 0.25, f'{lab}: away by {dev:+g}', ha='center', fontsize=10)
    ax.text(x + s / 2, 0.5 + s / 2, f'{s}x{s}\n= {s*s:g}', ha='center', va='center', fontsize=10, color=RED)
ax.text(6, -0.6, 'squares: 2.25 + 2.25 = 4.5; times 3 stories = 13.5', ha='center', fontsize=11, color=RED, weight='bold')
ax = axes[1]; ax.set_xlim(-1, 13); ax.set_ylim(-1, 6); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('Meaning spread: how far is each story average from 7.5?', fontsize=11.5, weight='bold')
for k, (lab, dev) in enumerate([('story 1 (3.5)', -4), ('story 2 (7.5)', 0), ('story 3 (11.5)', 4)]):
    s = abs(dev); x = 0.3 + k * 4.4
    if s == 0:
        ax.plot(x + 2, 0.5, 'o', color=BLUE); ax.text(x + 2, 0.9, 'story 2: away by 0\n0x0 = 0', ha='center', fontsize=10, color=BLUE)
    else:
        ax.add_patch(Rectangle((x, 0.5), s, s, fc='#dbe9f6', ec=BLUE, lw=2))
        ax.text(x + s / 2, 0.5 + s + 0.25, f'{lab}: away by {dev:+g}', ha='center', fontsize=10)
        ax.text(x + s / 2, 0.5 + s / 2, f'{s}x{s}\n= {s*s:g}', ha='center', va='center', fontsize=11, color=BLUE)
ax.text(6, -0.6, 'squares: 16 + 0 + 16 = 32; times 2 languages = 64', ha='center', fontsize=11, color=BLUE, weight='bold')
fig.suptitle('Step 6. "Spread" = how far things sit from the middle. We square each distance so it is always positive and big gaps count more.', fontsize=12.5, y=1.02)
fig.tight_layout(); save(fig, 's06_spread_squares.png')

# ---------- S7: the share
fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis('off')
ax.text(6, 4.3, 'Step 7. LFS is a share: the red area out of the red and blue areas together.', ha='center', fontsize=13, weight='bold')
tot = 77.5; w = 9.0; xr = 1.5
ax.add_patch(Rectangle((xr, 1.8), w * 13.5 / tot, 1.2, fc='#f8d7da', ec=RED, lw=2))
ax.add_patch(Rectangle((xr + w * 13.5 / tot, 1.8), w * 64 / tot, 1.2, fc='#dbe9f6', ec=BLUE, lw=2))
ax.text(xr + w * 13.5 / tot / 2, 2.4, 'language\n13.5', ha='center', va='center', fontsize=11, color=RED, weight='bold')
ax.text(xr + w * 13.5 / tot + w * 64 / tot / 2, 2.4, 'meaning  64', ha='center', va='center', fontsize=12, color=BLUE, weight='bold')
ax.text(6, 1.2, 'LFS = 13.5 / (13.5 + 64) = 13.5 / 77.5 = 0.17', ha='center', fontsize=15, weight='bold')
ax.text(6, 0.5, 'Read it: "in this pretend table, 17 percent of the spread is about which language, 83 percent is about which story."\nIf the languages moved farther apart (English 2, Arabic 12...), the red part would grow and LFS would rise.', ha='center', fontsize=10.5)
save(fig, 's07_share.png')

# ---------- S8: real cards are tall: repeat per line and add up
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis('off')
ax.text(6.5, 4.9, 'Step 8. Real cards have 4,096 lines. Do steps 5 to 7 on line 1, then line 2, ... then add all the red parts and all the blue parts.', ha='center', fontsize=12.5, weight='bold')
for k in range(5):
    y = 3.9 - k * 0.62
    lab = ['line 1', 'line 2', 'line 3', '...', 'line 4,096'][k]
    ax.text(0.9, y, lab, ha='right', va='center', fontsize=11)
    r = [0.3, 0.5, 0.2, 0.35, 0.45][k]
    ax.add_patch(Rectangle((1.1, y - 0.2), 3.0 * r, 0.4, fc='#f8d7da', ec=RED, lw=1.2))
    ax.add_patch(Rectangle((1.1 + 3.0 * r, y - 0.2), 3.0 * (1 - r), 0.4, fc='#dbe9f6', ec=BLUE, lw=1.2))
ax.add_patch(FancyArrowPatch((4.6, 2.6), (6.0, 2.6), arrowstyle='-|>', mutation_scale=22, lw=2, color='#333'))
ax.text(5.3, 2.95, 'add up', ha='center', fontsize=11)
ax.add_patch(Rectangle((6.3, 2.0), 6.0 * 0.608, 1.2, fc='#f8d7da', ec=RED, lw=2))
ax.add_patch(Rectangle((6.3 + 6.0 * 0.608, 2.0), 6.0 * 0.392, 1.2, fc='#dbe9f6', ec=BLUE, lw=2))
ax.text(6.3 + 6.0 * 0.608 / 2, 2.6, 'language\n55.2 million', ha='center', va='center', fontsize=11, color=RED, weight='bold')
ax.text(6.3 + 6.0 * 0.608 + 6.0 * 0.392 / 2, 2.6, 'meaning\n35.6 million', ha='center', va='center', fontsize=11, color=BLUE, weight='bold')
ax.text(9.3, 3.5, 'Real Qwen3-8B, floor 19, all 128 languages x 300 stories', ha='center', fontsize=10.5)
ax.text(9.3, 1.45, 'LFS = 55.2 / (55.2 + 35.6) = 0.608', ha='center', fontsize=15, weight='bold')
ax.text(6.5, 0.45, 'One more rule before adding: every line is first put on the same scale (its own average becomes 0, its own typical spread becomes 1).\nOtherwise one loud line drowns the rest: on this floor, line 2276 alone holds 92.8 percent of all the raw spread, and the raw share would read 0.347 instead of 0.608.', ha='center', fontsize=10, style='italic')
save(fig, 's08_add_up_lines.png')

# ---------- S9: the curve and the chance level (real)
d = json.load(open(os.path.join(ROOT, 'results', 'grid', 'Qwen3-8B-Base', 'metrics.json')))
q = []
for l in range(d['n_layers']):
    v = d['per_layer'][str(l)]['lfs']; v = v.get('lfs', v.get('value')) if isinstance(v, dict) else v; q.append(float(v))
q = np.array(q); dip = int(np.argmin(q))
fig, ax = plt.subplots(figsize=(12, 5.6))
ax.plot(range(len(q)), q, 'o-', color='#08306b', lw=2.5, ms=5, label='Qwen3-8B, one LFS per floor (real)')
ax.axhline(0.298, color='grey', ls=':', lw=2); ax.text(36, 0.315, 'shuffled table would give 0.298: the ruler\'s "nothing here" mark', ha='right', fontsize=10, color='grey')
for xi, txt in [(0, f'door {q[0]:.3f}'), (dip, f'middle {q[dip]:.3f}'), (36, f'exit {q[-1]:.3f}')]:
    ax.annotate(txt, (xi, q[xi]), xytext=(xi + (2 if xi < 30 else -2), q[xi] + (0.05 if xi == dip else -0.07)), fontsize=11, weight='bold', color='#08306b', ha='left' if xi < 30 else 'right', arrowprops=dict(arrowstyle='->', color='#08306b'))
ax.annotate('', (dip, q[dip]), (dip, q[0]), arrowprops=dict(arrowstyle='<->', color=RED, lw=2))
ax.text(dip + 0.8, (q[0] + q[dip]) / 2, f'dip depth\n{q[0]-q[dip]:.3f}', color=RED, fontsize=11, weight='bold', va='center')
ax.set_xlabel('floor (layer)'); ax.set_ylabel('LFS'); ax.set_ylim(0.2, 1.0); ax.grid(alpha=0.25); ax.legend(loc='lower left')
ax.set_title('Step 9. Do steps 4 to 8 on every floor. The 37 numbers make the depth profile.', fontsize=13)
fig.tight_layout(); save(fig, 's09_curve_real.png')

# ---------- S10: reading the curve
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.plot(range(len(q)), q, '-', color='#08306b', lw=2.5)
ax.axvspan(-0.5, 4, color='#fff3cd', alpha=0.7); ax.axvspan(13, 25, color='#e2f0d9', alpha=0.7); ax.axvspan(31, 36.5, color='#f8d7da', alpha=0.6)
ax.text(1.75, 0.55, 'DOOR\nthe words are\nstill words, so\nlanguage rules', ha='center', fontsize=10.5)
ax.text(19, 0.45, 'MIDDLE\nthe machine has worked out\nWHAT is being said; the same story\nlands close in every language.\nLanguage never fully leaves:\n0.61 is far above 0.30', ha='center', fontsize=10.5)
ax.text(33.75, 0.55, 'EXIT\nit must answer\nin your language,\nso language\nrules again', ha='center', fontsize=10.5)
ax.set_ylim(0.2, 1.0); ax.set_xlabel('floor (layer)'); ax.set_ylabel('LFS'); ax.grid(alpha=0.25)
ax.set_title('Step 10. How to read the curve, in words', fontsize=13)
fig.tight_layout(); save(fig, 's10_reading_curve.png')

# ---------- S11: what the ruler cannot see
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off'); ax.set_aspect('equal')
ax.text(5, 4.6, 'The share is blind to size', ha='center', fontsize=13, weight='bold')
for k, (sc, x0, lab) in enumerate([(1.0, 0.6, 'before training'), (0.25, 6.4, 'after a bad training run')]):
    W, H = 3.6 * sc, 2.4 * sc
    ax.add_patch(Rectangle((x0, 1.5), W * 0.45, H, fc='#f8d7da', ec=RED, lw=2)); ax.add_patch(Rectangle((x0 + W * 0.45, 1.5), W * 0.55, H, fc='#dbe9f6', ec=BLUE, lw=2))
    ax.text(x0 + W / 2, 1.5 + H + 0.2, lab, ha='center', fontsize=11, weight='bold'); ax.text(x0 + W / 2, 1.1, 'share = 0.45', ha='center', fontsize=11, color=RED)
ax.text(5, 0.3, 'Real case: meaning spread fell from 13,791 to 699 (twenty times smaller)\nwhile LFS went 0.445 to 0.493. Always show the two areas, not just the share.', ha='center', fontsize=10)
ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')
ax.text(5, 4.6, 'Do not train against the ruler', ha='center', fontsize=13, weight='bold')
ax.text(5, 2.6, 'We tried: "machine, make LFS small."\nThe machine found the cheap way:\nit squashed the cards instead of\nbuilding a better shared room.\n\nSo the ruler is a thermometer,\nnot a steering wheel.\nKoehn agreed this is fine.', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', fc='#f7f7f7', ec='#999'))
fig.suptitle('Step 11. Two things the ruler cannot do, found on purpose', fontsize=13, y=1.02)
fig.tight_layout(); save(fig, 's11_cannot.png')

# ---------- S12: from ruler to Aim 1 and Aim 2
fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis('off')
ax.text(6.5, 5.3, 'Step 12. What the ruler gave Aim 1, and what it hands to Aim 2', ha='center', fontsize=13, weight='bold')
def bx(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', fc=fc, ec='#333', lw=1.6))
    ax.text(x + w / 2, y + h - 0.4, title, ha='center', fontsize=12, weight='bold'); ax.text(x + w / 2, y + h / 2 - 0.25, body, ha='center', va='center', fontsize=10)
bx(0.3, 0.6, 3.8, 4.2, 'Aim 1 asked', 'Is there a shared room?\nIs it English?\nHow do languages differ inside?\nDoes any of it show in behavior?\nGive us rulers we can trust.', '#fff3cd')
bx(4.6, 0.6, 3.8, 4.2, 'The ruler answered', 'Yes, in every machine (36 of 36).\nNo, a mix (113 to 124 of 127).\nMostly a shift (63 to 97 percent).\nYes: deeper dip, better hint use (0.51).\nChecked: blind predictions 4 of 4,\nfaults caught, FLORES 17 of 17.', '#e2f0d9')
bx(8.9, 0.6, 3.8, 4.2, 'Aim 2 gets', 'WHERE to act: the middle floor.\nWHAT to act on: the shift.\nHOW to check: the two areas\n(the pieces), not the share.\nRule: the ruler judges the fix,\nit is never the training target.', '#dbe9f6')
ax.add_patch(FancyArrowPatch((4.15, 2.7), (4.55, 2.7), arrowstyle='-|>', mutation_scale=22, lw=2, color='#333'))
ax.add_patch(FancyArrowPatch((8.45, 2.7), (8.85, 2.7), arrowstyle='-|>', mutation_scale=22, lw=2, color='#333'))
save(fig, 's12_aims.png')
print('dip', dip, [round(float(x), 3) for x in (q[0], q[dip], q[-1])]); print(sorted(os.listdir(OUT)))
