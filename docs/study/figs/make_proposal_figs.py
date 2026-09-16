#!/usr/bin/env python3
"""Pictures for Part A (what Aim 1 of the NSF proposal asks) and Part B (how LFS answers it).
Real numbers: Belebele accuracy from results/belebele (lm-evaluation-harness output)."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'docs', 'study', 'figs', 'proposal'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 12})
RED, BLUE, GREEN, GREY = '#b30000', '#1f5fa8', '#2e7d32', '#666666'
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight'); plt.close(fig)
def box(ax, x, y, w, h, title, body, fc, fs=10.5, tfs=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', fc=fc, ec='#333', lw=1.6))
    ax.text(x + w / 2, y + h - 0.42, title, ha='center', fontsize=tfs, weight='bold')
    ax.text(x + w / 2, y + h / 2 - 0.3, body, ha='center', va='center', fontsize=fs)

# ---- P1: the worry, with real Belebele numbers
f = glob.glob(os.path.join(ROOT, 'results', 'belebele0', 'Qwen3-8B-Base', '*', 'results_*.json'))[0]
res = json.load(open(f))['results']
want = [('eng_Latn', 'English'), ('deu_Latn', 'German'), ('fra_Latn', 'French'), ('spa_Latn', 'Spanish'), ('rus_Cyrl', 'Russian'), ('arb_Arab', 'Arabic'), ('pes_Arab', 'Persian'), ('swh_Latn', 'Swahili'), ('amh_Ethi', 'Amharic'), ('khm_Khmr', 'Khmer')]
names, accs = [], []
for code, name in want:
    k = 'belebele_' + code
    if k in res:
        v = res[k].get('acc,none', res[k].get('acc_norm,none'))
        if v is not None: names.append(name); accs.append(float(v))
fig, ax = plt.subplots(figsize=(11, 5))
cols = [BLUE if n == 'English' else ('#6baed6' if i < 7 else '#c6dbef') for i, n in enumerate(names)]
ax.bar(names, accs, color=cols, edgecolor='k')
ax.axhline(0.25, color=GREY, ls=':', lw=1.5); ax.text(len(names) - 0.5, 0.265, 'guessing (1 in 4)', ha='right', color=GREY, fontsize=10)
for i, a in enumerate(accs): ax.text(i, a + 0.015, f'{a:.2f}', ha='center', fontsize=10)
ax.set_ylim(0, 1); ax.set_ylabel('share of reading questions answered correctly')
ax.set_title('The worry. One machine (Qwen3-8B), the same reading test (Belebele, 0-shot), ten languages.', fontsize=13)
ax.text(0.5, -0.2, 'Why is it worse in some languages? To fix it you first have to know what is going on inside. That is Aim 1.', ha='center', transform=ax.transAxes, fontsize=11.5, style='italic')
fig.tight_layout(); save(fig, 'p1_worry.png')

# ---- P2: Koehn's picture of the machine (3.1.1)
fig, ax = plt.subplots(figsize=(13, 5.8)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.8); ax.axis('off')
ax.text(6.5, 5.5, 'The proposal\'s picture of the machine (section 3.1.1): three stages, and two open questions', ha='center', fontsize=13, weight='bold')
box(ax, 0.3, 2.3, 3.8, 2.7, 'Stage 1: take in the word', 'Work out which word this is\nand what it means here.\n\nProposal: this stage is\nLANGUAGE-DEPENDENT.', '#fff3cd')
box(ax, 4.6, 2.3, 3.8, 2.7, 'Stage 2: core thinking', 'Decide what comes next,\nas a CONCEPT, not a word.\n\nProposal: this stage SHOULD BE\nthe same for all languages.', '#e2f0d9')
box(ax, 8.9, 2.3, 3.8, 2.7, 'Stage 3: pick the output word', 'Turn the concept back into\na word of the right language.\n\nProposal: this stage is\nLANGUAGE-DEPENDENT.', '#f8d7da')
for x1, x2 in [(4.15, 4.55), (8.45, 8.85)]:
    ax.add_patch(FancyArrowPatch((x1, 3.65), (x2, 3.65), arrowstyle='-|>', mutation_scale=22, lw=2, color='#333'))
ax.text(6.5, 1.75, 'Open question 1: "It is not clear how strong the language component in the representation is, or should be."', ha='center', fontsize=11.5, color=RED, weight='bold')
ax.text(6.5, 1.15, 'Open question 2: can the spaces of different languages be connected by a simple map (a shift, a rotation),\nas plain word vectors could, or is the map complicated at deeper layers?', ha='center', fontsize=11.5, color=BLUE, weight='bold')
ax.text(6.5, 0.3, '(Quotes and stages are from the proposal, section 3.1.1. Stages 1 and 3 are the door and the exit of our building; stage 2 is the middle floors.)', ha='center', fontsize=10, style='italic')
save(fig, 'p2_three_stages.png')

# ---- P3: the four tasks of Aim 1
fig, ax = plt.subplots(figsize=(13, 7)); ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis('off')
ax.text(6.5, 6.7, 'Aim 1 has four tasks. In plain words:', ha='center', fontsize=14, weight='bold')
box(ax, 0.3, 3.6, 6.0, 2.7, '3.1.1  Mapping between language spaces', 'Do all languages share one room inside?\nHow much of a card is "language" and\nhow much is "meaning"? Is the map from\none language to another simple?', '#fff3cd', fs=10.5)
box(ax, 6.7, 3.6, 6.0, 2.7, '3.1.2  Does it "think in English"?', 'An old method (the logit lens) said yes.\nThe professors doubt it for two reasons:\nthe middle cards are far from every word, and\nEnglish words look close to everything because\nthey are so common. Check this properly.', '#dbe9f6', fs=10.5)
box(ax, 0.3, 0.5, 6.0, 2.7, '3.1.3  Using a hint from another language', '"The pitcher swung too hard. The bat flew\nthrough the air." The first sentence tells you\nwhat "bat" means. Does it still help when the\nfirst sentence is in another language?', '#e2f0d9', fs=10.5)
box(ax, 6.7, 0.5, 6.0, 2.7, '3.1.4  Build rulers (metrics)', 'Make measures that look inside and say\nhow multilingual a machine is. They should track\nreal ability, but tests are unreliable, so the ruler\nmust stand on its own. "We have not singled out\na specific metric": maybe raw distances,\nmaybe how complex the map is.', '#f8d7da', fs=10.5)
save(fig, 'p3_four_tasks.png')

# ---- P4: what a good ruler must do (the professors' wish list)
fig, ax = plt.subplots(figsize=(12, 5.4)); ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis('off')
ax.text(6, 5.1, 'So the proposal is really asking for a ruler that can answer these, floor by floor:', ha='center', fontsize=13, weight='bold')
items = [('1', 'How much of each card is LANGUAGE and how much is MEANING?', 'the language component (3.1.1)'),
         ('2', 'Is the middle stage really shared, and is the shared part English?', 'stage 2, think in English (3.1.1, 3.1.2)'),
         ('3', 'Is the map between languages simple (a shift) or complicated?', 'mapping complexity (3.1.1, 3.1.4)'),
         ('4', 'Does what the ruler sees show up in what the machine can DO?', 'hint use (3.1.3), extrinsic checks (3.1.4)'),
         ('5', 'Can we trust the ruler, and can it be used to steer training?', 'metrics as feedback or objective (3.1.4)')]
for i, (n, q, src) in enumerate(items):
    y = 4.2 - i * 0.85
    ax.add_patch(Rectangle((0.3, y - 0.33), 11.4, 0.72, fc='#f7f7f7' if i % 2 else 'white', ec='none'))
    ax.text(0.5, y, n, fontsize=15, weight='bold', va='center', color=RED)
    ax.text(1.1, y, q, fontsize=12, va='center')
    ax.text(11.6, y, src, fontsize=9.5, va='center', ha='right', color=GREY, style='italic')
save(fig, 'p4_wishlist.png')

# ---- B1: what we built for each question, and what it found
fig, ax = plt.subplots(figsize=(14, 7.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis('off')
ax.text(7, 7.3, 'How we answered: one measure per question, each checked on its own', ha='center', fontsize=14, weight='bold')
ax.text(0.3, 6.7, 'Question', fontsize=12, weight='bold'); ax.text(3.6, 6.7, 'What we built', fontsize=12, weight='bold'); ax.text(8.6, 6.7, 'What it found', fontsize=12, weight='bold')
rows = [('1  language vs meaning', 'LFS: the language share of the spread,\none number per floor (the depth profile)', 'Door 0.95, middle 0.61, exit 0.96 (Qwen3-8B).\nEvery machine dips in the middle: 36 of 36.\nDip depth 0.04 to 0.34, set by training recipe, not size.'),
        ('2  shared middle, English?', 'Hub test: predict each language\'s cards\nfrom English alone vs from a mix of others', 'The mix wins for 113 to 124 of 127 languages.\nThe shared room is not an English room.'),
        ('3  simple map?', 'Shift decomposition: how much of the gap\nis a shift, a stretch, a rotation, or worse', 'A shift plus a stretch explains 63 to 97 percent.\nRotation under 1 percent. Removing the shift on saved\ncards deletes 97 percent of the language part.'),
        ('4  shows up in behavior?', 'Hint test: does a foreign-language first\nsentence help predict the second one?', 'Machines with a deeper dip use the hint more:\nSpearman 0.51 across 19 machines, 13 families.\nBut LFS does not predict test scores (0.04): a limit.'),
        ('5  trust it? steer with it?', 'Chance mark 0.298; second dataset; blind\npredictions; injected faults; training against it', 'FLORES agrees 17 of 17; blind predictions 4 of 4;\nfaults caught. Training against LFS makes the machine\ncheat, so LFS is a thermometer, not a steering wheel.')]
for i, (q, b, f_) in enumerate(rows):
    y = 5.85 - i * 1.32
    ax.add_patch(Rectangle((0.2, y - 0.6), 13.6, 1.25, fc='#f7f7f7' if i % 2 else 'white', ec='none'))
    ax.text(0.3, y, q, fontsize=11.5, weight='bold', va='center')
    ax.text(3.6, y, b, fontsize=10.2, va='center')
    ax.text(8.6, y, f_, fontsize=10.2, va='center', color='#08306b')
save(fig, 'b1_answers_table.png')

# ---- B2: the verdict in one picture
fig, ax = plt.subplots(figsize=(12, 5)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')
ax.text(6, 4.7, 'The verdict on the proposal\'s three-stage picture', ha='center', fontsize=14, weight='bold')
box(ax, 0.3, 1.3, 3.6, 2.9, 'Door: language', 'Confirmed.\nLFS 0.95: cards group\nby language.', '#fff3cd', fs=11)
box(ax, 4.2, 1.3, 3.6, 2.9, 'Middle: shared concept', 'Confirmed, with a twist.\nLFS drops to 0.61 in every\nmachine, but never near 0.30:\nlanguage stays as a SHIFT,\nand the room is not English.', '#e2f0d9', fs=10.5)
box(ax, 8.1, 1.3, 3.6, 2.9, 'Exit: language', 'Confirmed.\nLFS 0.96: cards group\nby language again.', '#f8d7da', fs=11)
ax.text(6, 0.6, 'Answer to open question 1: the language component is large everywhere and smallest in the middle; how small depends on the training recipe.\nAnswer to open question 2: the map is simple, mostly a shift.', ha='center', fontsize=11)
save(fig, 'b2_verdict.png')
print('belebele used:', list(zip(names, [round(a, 3) for a in accs])))
print(sorted(os.listdir(OUT)))
