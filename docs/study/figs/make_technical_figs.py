#!/usr/bin/env python3
"""Two technical diagrams in NLP/ML terms: how LFS was validated, and the family of measures."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'docs', 'study', 'figs', 'technical'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 11})
def box(ax, x, y, w, h, title, body, fc, tfs=11.5, fs=9.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06', fc=fc, ec='#333', lw=1.4))
    ax.text(x + w / 2, y + h - 0.32, title, ha='center', fontsize=tfs, weight='bold')
    ax.text(x + w / 2, y + h / 2 - 0.22, body, ha='center', va='center', fontsize=fs)

# ---- T5: validation
fig, ax = plt.subplots(figsize=(15, 13)); ax.set_xlim(0, 15); ax.set_ylim(0, 13); ax.axis('off')
ax.text(7.5, 12.7, 'How the LFS depth profile was validated (each box: method in ML/NLP terms, then the result)', ha='center', fontsize=14, weight='bold')
cells = [
 ('Null model', 'Analytic expectation of the ratio under\ni.i.d. noise with no language or concept\nstructure: (L-1)/(L+N-2).\nResult: 0.298 at (128, 300); every real\nreading compared to it.', '#fff3cd'),
 ('Sentence bootstrap', '2,000 resamples of 300 sentences from the\n1,500-sentence dumps, identical index sets\nfor every model; 95% intervals on dip depth.\nResult: half-widths 0.004 to 0.020 for 12 of\n14 models; ordering kept in 2,000 of 2,000.', '#dbe9f6'),
 ('Replication on a second corpus', 'Same pipeline on FLORES-200 (116 languages,\n300 sentences, Wikipedia domain) for 17 models.\nResult: interior minimum + endpoint recovery\n17 of 17; dip-depth Spearman 0.917 vs NTREX.', '#e2f0d9'),
 ('Prospective test on unseen families', 'Predictions for SmolLM2 and Falcon3 written\nand frozen before their first forward pass\n(profile shape, dip class, attribution criterion).\nResult: 4 of 4 predictions passed.', '#f8d7da'),
 ('Fault injection (synthetic test suite)', 'Inject known faults into saved states: per-language\noffset, isotropic scale, rotation, warp, collapse;\nscore against a frozen prediction matrix.\nResult: offsets recovered; rotation 0.06 to 0.2%\nof residual; uniform collapse invisible to any\nscale-invariant statistic (by theorem).', '#ede7f6'),
 ('Instruction tuning', 'Six base/instruct pairs run through the\nsame grid.\nResult: |change in dip depth| <= 0.008;\nthe profile is a property of pretraining.', '#fff3cd'),
 ('Estimator agreement', 'Direct sums of squares vs REML variance\ncomponents (LFS-VC) on 14 models.\nResult: dip-depth ranking Spearman 1.000;\nthe two estimators are reported separately.', '#dbe9f6'),
 ('Where the ratio fails (on purpose)', 'Train Qwen3-0.6B with word-alignment losses;\nread LFS with its raw components.\nResult: raw concept variance 13,791 -> 699\nwhile LFS 0.445 -> 0.493 and MEXA rose:\nthe share must be read with its components.', '#f8d7da')]
for i, (t, b, fc) in enumerate(cells):
    r, c = divmod(i, 2); x = 0.25 + c * 7.4; y = 9.5 - r * 3.1
    box(ax, x, y, 7.1, 2.85, t, b, fc, fs=10.2)
fig.savefig(os.path.join(OUT, 't5_validation.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---- T6: family of measures
fig, ax = plt.subplots(figsize=(15, 14)); ax.set_xlim(0, 15); ax.set_ylim(0, 14); ax.axis('off')
ax.text(7.5, 13.7, 'The family of measures on the same grid: what each computes and what it found', ha='center', fontsize=14, weight='bold')
cells = [
 ('LFS depth profile', 'Two-way ANOVA main-effect share per layer,\nSS_lang / (SS_lang + SS_con), after joint\nper-coordinate z-scoring; residual excluded.\nFound: interior minimum with endpoint recovery\nin all 36 measured profiles (33 models); depth 0.04 to 0.34, set\nby training recipe rather than parameter count.', '#fff3cd'),
 ('Raw components (CVP)', 'The same SS_lang, SS_con on unstandardized\nstates, mean vector norm, and concept variance\nrelative to a frozen reference (CVP).\nFound: the only measure that caught the\ntwenty-fold concept-variance collapse.', '#dbe9f6'),
 ('Misalignment decomposition', 'Map each language into a generalized-Procrustes\nconsensus with nested maps: M1 additive offset,\nM2 isotropic scale, M3 rotation, M4 general\nlinear, M5 kernel; credit = held-out error reduction.\nFound: offset + scale 63 to 97%; rotation under 1%.', '#e2f0d9'),
 ('Hub test', 'For each language, ridge regression predicting\nits states from English states vs from a latent\nmultilingual factor; compare held-out R^2.\nFound: the latent factor wins for 113 to 124\nof 127 languages; English is the best single\ndonor only for high-resource targets.', '#f8d7da'),
 ('Content transfer (the hint test)', 'Adjacent NTREX sentence pairs: log-likelihood of\nsentence 2 given the matched context vs a\nmismatched context, R_content = the difference;\nSpearman with 1 - LFS at the dip, confounder-\nadjusted, 2,000-replicate bootstrap.\nFound: 0.508 [0.19, 0.71] over 19 frozen models.', '#ede7f6'),
 ('Tail measure AaR', 'Per sentence, cross-lingual retrieval margin\n(true parallel pair vs 299 decoys); AaR@10 =\nmean margin of the worst decile per language.\nFound: mean-level agreement hides languages\nwhose worst decile fails; 0.443 with content\ntransfer on the same 19 models.', '#fff3cd'),
 ('Reference measure: MEXA', 'Published retrieval-alignment score against an\nEnglish pivot, reimplemented under the same\npreprocessing; reported beside the family and\nnever ranked against it.\nFound: 0.591 with content transfer; ranked a\ncollapsed model first (the components caught it).', '#dbe9f6'),
 ('Boundaries stated', 'Pooled benchmark accuracy (33 models, 2,001\nrows): confounder-adjusted Spearman 0.041, null.\nTwo benchmarks of the same cells agree on only\n0.315 of residual rank variation.\nTraining against LFS reverses its observational\nassociations: a measurement, not an objective.', '#e2f0d9')]
for i, (t, b, fc) in enumerate(cells):
    r, c = divmod(i, 2); x = 0.25 + c * 7.4; y = 10.2 - r * 3.35
    box(ax, x, y, 7.1, 3.1, t, b, fc, fs=10.2)
fig.savefig(os.path.join(OUT, 't6_family.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
print(sorted(os.listdir(OUT)))
