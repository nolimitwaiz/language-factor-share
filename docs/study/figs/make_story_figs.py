#!/usr/bin/env python3
"""Seven plain-language pictures that tell the Aim 1 story in order.
Real numbers come from results/grid/<model>/metrics.json. CPU only."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'docs', 'study', 'figs', 'story')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})

def lfs_curve(model):
    d = json.load(open(os.path.join(ROOT, 'results', 'grid', model, 'metrics.json')))
    L = d['n_layers']; ys = []
    for l in range(L):
        v = d['per_layer'][str(l)]['lfs']
        if isinstance(v, dict):
            v = v.get('lfs', v.get('value', next(x for x in v.values() if isinstance(x, (int, float)))))
        ys.append(float(v))
    return np.array(ys)

LANGS = ['English', 'German', 'French', 'Spanish', 'Russian', 'Arabic']
LCOL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
STORIES = ['"The cat sleeps"', '"Rain is coming"', '"I lost my keys"']
SMARK = ['o', 's', '^']

# ---------------------------------------------------------------- 1 problem
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
rng = np.random.default_rng(0)
ax = axes[0]
ax.add_patch(FancyBboxPatch((0.08, 0.12), 0.84, 0.76, boxstyle='round,pad=0.02', fc='#f3f6fa', ec='#444', lw=2))
for i, (lang, c) in enumerate(zip(LANGS, LCOL)):
    for j, m in enumerate(SMARK):
        cx, cy = [0.3, 0.5, 0.7][j], [0.6, 0.35, 0.6][j]
        ax.scatter(cx + rng.normal(0, 0.03), cy + rng.normal(0, 0.03), c=c, marker=m, s=110, edgecolor='k', zorder=3)
ax.text(0.5, 0.94, 'Picture A: one shared room', ha='center', fontsize=15, weight='bold')
ax.text(0.5, 0.03, 'Same story (same shape) lands in the same spot,\nwhatever the language (color).', ha='center', fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax = axes[1]
ax.add_patch(FancyBboxPatch((0.05, 0.3), 0.5, 0.58, boxstyle='round,pad=0.02', fc='#dbe9f6', ec='#1f77b4', lw=2.5))
ax.text(0.3, 0.83, 'big English room', ha='center', fontsize=12, color='#1f77b4', weight='bold')
for j, m in enumerate(SMARK):
    ax.scatter(0.15 + 0.15 * j, 0.55, c=LCOL[0], marker=m, s=130, edgecolor='k', zorder=3)
for i, (lang, c) in enumerate(zip(LANGS[1:], LCOL[1:])):
    x0 = 0.6 + 0.2 * (i % 2); y0 = 0.62 - 0.2 * (i // 2)
    ax.add_patch(Rectangle((x0, y0), 0.16, 0.14, fc='white', ec=c, lw=2))
    for j, m in enumerate(SMARK):
        ax.scatter(x0 + 0.03 + 0.05 * j, y0 + 0.07, c=c, marker=m, s=45, edgecolor='k', zorder=3)
    ax.text(x0 + 0.08, y0 - 0.035, lang, ha='center', fontsize=9, color=c)
ax.text(0.5, 0.94, 'Picture B: an English room with small copies', ha='center', fontsize=15, weight='bold')
ax.text(0.5, 0.03, 'Every other language gets its own small room.\nThe copies can never be as good as the original.', ha='center', fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
fig.suptitle("The question in Aim 1: which picture is true inside the machine?  (color = language, shape = story)", fontsize=14, y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'story1_problem.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 2 ruler pipeline
fig, ax = plt.subplots(figsize=(14, 4.8)); ax.set_xlim(0, 14); ax.set_ylim(0, 4.8); ax.axis('off')
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', fc=fc, ec='#333', lw=1.8))
    ax.text(x + w / 2, y + h - 0.45, title, ha='center', fontsize=13, weight='bold')
    ax.text(x + w / 2, y + h / 2 - 0.35, body, ha='center', va='center', fontsize=10.5)
def arrow(x1, x2, y=2.4, label=''):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=22, lw=2, color='#333'))
    if label: ax.text((x1 + x2) / 2, y + 0.3, label, ha='center', fontsize=10)
box(0.2, 0.9, 3.0, 3.1, '1. Same stories', '300 short stories\nwritten in 128 languages\n(same story, different words)\n= 38,400 texts', '#fff3cd')
arrow(3.3, 3.9)
box(4.0, 0.9, 2.6, 3.1, '2. The machine', 'Push every text through.\nThe machine has floors\n(layers). Each text lands\non a spot on every floor.', '#dbe9f6')
arrow(6.7, 7.3)
box(7.4, 0.9, 3.0, 3.1, '3. Look at one floor', 'Who lands together?\nSame language (color)\nor same story (shape)?', '#e2f0d9')
arrow(10.5, 11.1)
box(11.2, 0.9, 2.6, 3.1, '4. One number', 'LFS = the share of\n"landing together" that\nis due to language.\nNear 1: language rules.\nNear 0: meaning rules.', '#f8d7da')
ax.text(7, 0.35, 'Repeat step 3 and 4 on every floor. The numbers make a curve from the front door to the exit. That curve is the depth profile.', ha='center', fontsize=11.5, style='italic')
fig.savefig(os.path.join(OUT, 'story2_ruler.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 3 landing spots on three floors
q = lfs_curve('Qwen3-8B-Base'); dip = int(np.argmin(q))
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
rng = np.random.default_rng(1)
layouts = [('Front door (layer 0)', q[0], 'by_lang'), (f'Middle (layer {dip})', q[dip], 'by_story'), (f'Exit (layer {len(q)-1})', q[-1], 'by_lang')]
lang_centers = [(0.2, 0.75), (0.5, 0.8), (0.8, 0.75), (0.2, 0.3), (0.5, 0.25), (0.8, 0.3)]
story_centers = [(0.25, 0.55), (0.5, 0.5), (0.75, 0.55)]
for ax, (title, val, mode) in zip(axes, layouts):
    for i, (lang, c) in enumerate(zip(LANGS, LCOL)):
        for j, m in enumerate(SMARK):
            if mode == 'by_lang':
                cx, cy = lang_centers[i]; cx += 0.045 * (j - 1); cy += rng.normal(0, 0.02)
            else:
                cx, cy = story_centers[j]; cx += 0.06 * np.cos(i * 1.05) ; cy += 0.14 * np.sin(i * 1.05) + rng.normal(0, 0.015)
            ax.scatter(cx, cy, c=c, marker=m, s=120, edgecolor='k', zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=14, weight='bold')
    ax.text(0.5, -0.06, f'LFS = {val:.2f}', ha='center', fontsize=16, transform=ax.transAxes, weight='bold', color='#b30000')
    ax.text(0.5, -0.15, {'by_lang': 'stories group by LANGUAGE (color)', 'by_story': 'stories group by MEANING (shape),\nbut each language still sits a bit to one side'}[mode], ha='center', fontsize=10.5, transform=ax.transAxes)
handles = [plt.Line2D([], [], marker='o', color=c, ls='', markersize=9, label=l) for l, c in zip(LANGS, LCOL)] + \
          [plt.Line2D([], [], marker=m, color='grey', ls='', markersize=9, label=s) for m, s in zip(SMARK, STORIES)]
fig.legend(handles=handles, loc='upper center', ncol=9, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.suptitle('Where the same three stories land on three floors of Qwen3-8B (arrangement drawn to scale of the real LFS numbers)', fontsize=13, y=1.12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'story3_landing.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 4 real curves
fig, ax = plt.subplots(figsize=(11, 5.6))
models = [('Qwen3-8B-Base', 'Qwen3 8B', '#08306b', 3.0), ('Mistral-7B-v0.3', 'Mistral 7B', '#e6550d', 2), ('OLMo-2-1124-7B', 'OLMo-2 7B', '#a50f15', 2), ('EuroLLM-1.7B', 'EuroLLM 1.7B', '#31a354', 2), ('bloom-1b7', 'BLOOM 1.7B', '#756bb1', 2)]
for m, name, c, lw in models:
    y = lfs_curve(m); x = np.arange(len(y)) / (len(y) - 1)
    ax.plot(x, y, color=c, lw=lw, label=f'{name} (dip {y[0]-y.min():.2f})')
ax.axhline(0.298, color='grey', ls=':', lw=1.5); ax.text(0.99, 0.31, 'pure-chance level 0.298', ha='right', fontsize=10, color='grey')
x = np.arange(len(q)) / (len(q) - 1)
for xi, yi, txt, dx in [(x[0], q[0], f'front door {q[0]:.2f}', 0.02), (x[dip], q[dip], f'middle {q[dip]:.2f}', 0.02), (x[-1], q[-1], f'exit {q[-1]:.2f}', -0.02)]:
    ax.annotate(txt, (xi, yi), xytext=(xi + dx, yi - 0.08 if yi > 0.7 else yi + 0.08), fontsize=11, weight='bold', color='#08306b', ha='left' if dx > 0 else 'right', arrowprops=dict(arrowstyle='->', color='#08306b'))
ax.set_xlabel('floor, from front door (0) to exit (1)'); ax.set_ylabel('LFS  (share of the layout that is language)')
ax.set_ylim(0.2, 1.0); ax.legend(loc='lower left', fontsize=10, title='real curves, 300 stories x 128 languages'); ax.grid(alpha=0.25)
ax.set_title('The depth profile: language rules at the door and the exit, meaning gets strongest in the middle', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'story4_curve.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 5 the shift
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
rng = np.random.default_rng(2)
base = rng.normal(0, 1, (40, 2)) * np.array([1.0, 0.6])
for ax, shifted, title, note in [(axes[0], True, 'Before: English and Arabic clouds, same shape, moved apart', '63 to 97 percent of the difference between languages\nis exactly this sideways move (a shift)'),
                                  (axes[1], False, 'After: subtract the shift, the clouds overlap', 'Doing this on the saved states removed 97 percent\nof the language part; the meaning part did not change')]:
    off = np.array([3.2, 0.8]) if shifted else np.array([0, 0])
    ax.scatter(base[:, 0], base[:, 1], c=LCOL[0], s=60, edgecolor='k', label='English', zorder=3)
    ax.scatter(base[:, 0] + off[0], base[:, 1] + off[1], c=LCOL[5], s=60, edgecolor='k', marker='s', label='Arabic', zorder=3, alpha=0.85)
    if shifted:
        ax.add_patch(FancyArrowPatch((0, 0), (3.2, 0.8), arrowstyle='-|>', mutation_scale=25, lw=3, color='#b30000', zorder=4))
        ax.text(1.6, 0.85, 'the shift', color='#b30000', fontsize=14, weight='bold', ha='center')
    ax.set_xlim(-3.5, 6.5); ax.set_ylim(-3, 3.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12.5, weight='bold'); ax.legend(loc='upper left')
    ax.text(0.5, -0.08, note, transform=ax.transAxes, ha='center', fontsize=11)
fig.suptitle('What the difference between languages looks like in the middle floors', fontsize=14, y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'story5_shift.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 6 answers to Aim 1
rows = [('One shared room for all languages?', 'Yes, in the middle floors. Every machine has it.', '36 of 36 machines'),
        ('Is the shared room an English room?', 'No. A mix of other languages describes each\nlanguage better than English does.', '113 to 124 of 127 languages'),
        ('How different are languages inside?', 'Mostly a sideways shift, the simplest kind\nof difference.', '63 to 97 percent'),
        ('Does a deeper middle room\nchange behavior?', 'Yes. Machines with a deeper dip use a hint\ngiven in another language more.', 'Spearman 0.51 on 19 machines'),
        ('Size or teaching?', 'On teaching (the training recipe), not size.', 'dips from 0.04 to 0.34'),
        ('Can the ruler be trusted?', 'Predictions written before the runs came true;\ninjected faults were caught; a second dataset agreed.', '4 of 4 blind; 17 of 17 on FLORES'),
        ('Does the machine "think in English"?', 'The data say no; the middle room is a mix.', 'same hub test')]
fig, ax = plt.subplots(figsize=(14, 7.2)); ax.axis('off'); ax.set_xlim(0, 14); ax.set_ylim(0, 7.6)
ax.text(0.2, 7.25, 'What Aim 1 asked', fontsize=13, weight='bold'); ax.text(5.3, 7.25, 'What the ruler found', fontsize=13, weight='bold'); ax.text(10.9, 7.25, 'Number to remember', fontsize=13, weight='bold')
for i, (qq, a, n) in enumerate(rows):
    y = 6.5 - i * 0.98
    ax.add_patch(Rectangle((0.1, y - 0.42), 13.8, 0.9, fc='#f7f7f7' if i % 2 else 'white', ec='none'))
    ax.text(0.2, y, qq, fontsize=11.5, va='center', weight='bold')
    ax.text(5.3, y, a, fontsize=11, va='center')
    ax.text(10.9, y, n, fontsize=11.5, va='center', color='#b30000', weight='bold')
ax.set_title('How LFS answers Aim 1 ("Understanding Multilingual Representation Spaces")', fontsize=14, pad=10)
fig.savefig(os.path.join(OUT, 'story6_answers.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 7 limits
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
for k, (scale, x0, lab) in enumerate([(1.0, 0.27, 'healthy machine'), (0.22, 0.75, 'shrunk machine')]):
    ax.add_patch(Rectangle((x0 - 0.2 * scale, 0.35), 0.4 * scale, 0.5 * scale * 0.7, fc='#dbe9f6', ec='#1f77b4', lw=2))
    ax.add_patch(Rectangle((x0 - 0.2 * scale, 0.35), 0.4 * scale * 0.45, 0.5 * scale * 0.7, fc='#f8d7da', ec='#b30000', lw=2))
    ax.text(x0, 0.27, lab, ha='center', fontsize=12, weight='bold')
    ax.text(x0, 0.2, 'red share = 0.45 in both', ha='center', fontsize=10.5, color='#b30000')
ax.text(0.5, 0.85, 'A share cannot see shrinking', ha='center', fontsize=14, weight='bold')
ax.text(0.5, 0.05, 'Real case: after training, raw meaning spread fell from 13,791 to 699\nwhile LFS moved only 0.445 to 0.493. So print the pieces next to the number.', ha='center', fontsize=10.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax = axes[1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.5, 0.85, 'The ruler is for looking, not for training', ha='center', fontsize=14, weight='bold')
ax.text(0.5, 0.5, 'If you tell the machine "make LFS small",\nit finds a cheat: it squeezes everything\nor moves things in a way that lowers\nthe number without a better shared room.\n\nKoehn: fine, we do not need the ruler as a loss.\nUse it to choose the floor, to watch for\nshrinking, and to judge the fix afterwards.', ha='center', va='center', fontsize=12)
fig.suptitle('Two limits, found on purpose', fontsize=14, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'story7_limits.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
print('dip layer', dip, 'door', round(float(q[0]), 3), 'middle', round(float(q[dip]), 3), 'exit', round(float(q[-1]), 3))
print('written to', OUT, sorted(os.listdir(OUT)))
