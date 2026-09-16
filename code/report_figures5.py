#!/usr/bin/env python3
"""Figures 17 to 21 of the technical report: the Aim 2 study.

  fig17_ratio_hides_scale.png   -> the ratio and the components, side by side
  fig18_aim2_arms.png           -> every Aim 2 arm on MEXA, AaR and CVP
  fig19_adversarial_probe.png   -> LFS curve against language-probe accuracy
  fig20_codeswitch_cost.png     -> displacement cost by switch rate and group
  fig21_guard_sweep.png         -> concept variance against guard weight

All read from results/ only; no model is loaded.

Usage: python code/report_figures5.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'paper', 'figs')
RES = os.path.join(ROOT, 'results')
SEEDS = ['0', '1', '2']


def load(p):
    f = os.path.join(RES, p)
    return json.load(open(f)) if os.path.exists(f) else None


def mean_over_seeds(ev, arm, key, layer='8', group='trained'):
    v = [ev['seeds'][s][arm][group][layer][key]
         for s in SEEDS if arm in ev['seeds'].get(s, {})]
    return float(np.mean(v)) if v else float('nan')


def cvp_of(ev, arm, frozen, layer='8'):
    v = [ev['seeds'][s][arm]['trained'][layer]['v_concept_raw'] /
         ev['seeds'][s][frozen]['trained'][layer]['v_concept_raw']
         for s in SEEDS if arm in ev['seeds'].get(s, {})]
    return float(np.mean(v)) if v else float('nan')


# ---------------------------------------------------------------- figure 17
def fig17():
    """The central result: a ratio cannot see what its own components can."""
    ev = load('aim2/evaluation_wordalign.json')
    if ev is None:
        return
    arms = ['WA-A', 'WA-B', 'WA-C', 'WA-E', 'WA-G']
    names = ['frozen', 'LM only\n(control)', 'word MSE\n(the spec)',
             'segment MSE', 'cosine\n(sound)']
    lfs = [mean_over_seeds(ev, a, 'lfs') for a in arms]
    raw = [mean_over_seeds(ev, a, 'v_concept_raw') for a in arms]
    col = ['#555555', '#4477aa', '#cc3311', '#ee7733', '#009988']

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(arms))
    ax[0].bar(x, lfs, color=col)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8)
    ax[0].set_ylabel('LFS (language share of variance)')
    ax[0].set_title('What the ratio reports', fontsize=11)
    ax[0].set_ylim(0, 0.62)
    for i, v in enumerate(lfs):
        ax[0].text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=8)
    ax[0].annotate('collapsed and sound models\nare indistinguishable',
                   xy=(2, lfs[2] + 0.004), xytext=(3.3, 0.28), fontsize=8,
                   ha='center', arrowprops=dict(arrowstyle='->', lw=0.8))

    ax[1].bar(x, raw, color=col)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_ylabel('raw concept variance (unstandardized)')
    ax[1].set_title('What the components report', fontsize=11)
    ax[1].set_yscale('log')
    for i, v in enumerate(raw):
        ax[1].text(i, v * 1.15, f'{v:,.0f}', ha='center', fontsize=8)
    ax[1].annotate('a factor of 20',
                   xy=(2, raw[2]), xytext=(3.1, raw[2] * 0.35), fontsize=8,
                   ha='center', arrowprops=dict(arrowstyle='->', lw=0.8))
    fig.suptitle('The ratio divides out exactly the quantity that failed',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig17_ratio_hides_scale.png'), dpi=160)
    plt.close(fig)
    print('  fig17_ratio_hides_scale.png')


# ---------------------------------------------------------------- figure 18
def fig18():
    """Every Aim 2 arm on the three instruments that matter."""
    specs = [('aim2/evaluation.json', 'A',
              [('B', 'LM only'), ('C', 'LM+LFS'), ('D', 'LM+align'),
               ('E', '+CVP'), ('F', 'all four')], 'sentence-level (InfoNCE)'),
             ('aim2/evaluation_wordalign.json', 'WA-A',
              [('WA-B', 'LM only'), ('WA-C', 'word MSE'),
               ('WA-D', '+langvec'), ('WA-E', 'segment MSE'),
               ('WA-G', 'cosine')], 'word-level (section 3.2.2)'),
             ('aim2/evaluation_sweep.json', 'A',
              [('B', 'LM only'), ('E_cvp10', 'guard 10'),
               ('E_cvp100', 'guard 100'), ('E_cvp1k', 'guard 1000'),
               ('G_tail', 'tail-directed')], 'guard sweep and tail condition')]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (path, frozen, arms, title) in zip(axes, specs):
        ev = load(path)
        if ev is None:
            ax.axis('off'); continue
        labels = [n for _, n in arms]
        mexa = [mean_over_seeds(ev, a, 'mexa_mean') for a, _ in arms]
        aar = [mean_over_seeds(ev, a, 'aar10_mean') for a, _ in arms]
        cvp = [cvp_of(ev, a, frozen) for a, _ in arms]
        x = np.arange(len(arms)); w = 0.27
        ax.bar(x - w, mexa, w, label='MEXA (retrieval)', color='#4477aa')
        ax.bar(x, [-v for v in aar], w, label='|AaR| (tail, lower better)',
               color='#ee7733')
        ax.bar(x + w, cvp, w, label='CVP (concept variance)', color='#009988')
        ax.axhline(mean_over_seeds(ev, frozen, 'mexa_mean'), ls=':', lw=1,
                   color='#4477aa')
        ax.axhline(1.0, ls=':', lw=1, color='#009988')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right',
                                             fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.25)
    axes[0].set_ylabel('value')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Aim 2 conditions: retrieval improves while concept variance '
                 'collapses', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig18_aim2_arms.png'), dpi=160)
    plt.close(fig)
    print('  fig18_aim2_arms.png')


# ---------------------------------------------------------------- figure 19
def fig19():
    """Section 3.2.3 premise: does LFS track what a discriminator can read?"""
    d = load('adversarial_probe/Qwen3-0.6B-Base.json')
    if d is None:
        return
    rows = d['per_layer']; s = d['summary']
    L = [r['lfs'] for r in rows]
    PL = [r['probe_linear'] for r in rows]
    PM = [r['probe_mlp'] for r in rows]
    layers = [r['layer'] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a0 = ax[0]
    a0.plot(layers, L, 'o-', color='#cc3311', ms=3, label='LFS')
    a0.set_xlabel('layer'); a0.set_ylabel('LFS', color='#cc3311')
    a0.tick_params(axis='y', labelcolor='#cc3311')
    a1 = a0.twinx()
    a1.plot(layers, PL, 's-', color='#4477aa', ms=3,
            label='linear probe accuracy')
    a1.plot(layers, PM, '^--', color='#009988', ms=3, label='MLP probe')
    lo = min(min(PL), min(PM)) - 0.004
    hi = max(max(PL), max(PM)) + 0.004
    a1.set_ylim(lo, hi)
    a1.set_ylabel('language-identification accuracy')
    a1.text(0.5, hi - 0.0015,
            f"axis zoomed: chance is {s['chance']:.3f}, far below. "
            f"Language stays ~88% identifiable at every layer.",
            fontsize=7, color='grey', va='top')
    a0.axvline(s['lfs_dip_layer'], ls='--', lw=0.8, color='#cc3311')
    a0.axvline(s['linear_min_layer'], ls='--', lw=0.8, color='#4477aa')
    a0.set_title(f"LFS dip L{s['lfs_dip_layer']}, probe minimum "
                 f"L{s['linear_min_layer']}", fontsize=10)
    h0, l0 = a0.get_legend_handles_labels(); h1, l1 = a1.get_legend_handles_labels()
    a0.legend(h0 + h1, l0 + l1, fontsize=7, loc='lower right')

    ax[1].scatter(L, PL, c=layers, cmap='viridis', s=26)
    ax[1].set_xlabel('LFS'); ax[1].set_ylabel('linear probe accuracy')
    ax[1].set_title(f"Pearson {s['lfs_vs_linear_pearson']:+.3f}, "
                    f"Spearman {s['lfs_vs_linear_spearman']:+.3f}", fontsize=10)
    cb = fig.colorbar(ax[1].collections[0], ax=ax[1]); cb.set_label('layer',
                                                                   fontsize=8)
    fig.suptitle('Section 3.2.3 premise: LFS predicts adversarial headroom '
                 'in direction, not in magnitude', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig19_adversarial_probe.png'), dpi=160)
    plt.close(fig)
    print('  fig19_adversarial_probe.png')


# ---------------------------------------------------------------- figure 20
def fig20():
    """Code-switch displacement cost, by language group and switch rate."""
    ev = load('aim2/evaluation_codeswitch.json')
    fz = load('aim2/evaluation_codeswitch_frozen.json')
    if ev is None or fz is None:
        return
    F = fz['seeds']['0']['CS-A']
    corp = {'ben', 'ces', 'deu', 'fra', 'hin', 'ind', 'kat', 'khm',
            'pol', 'rus', 'spa', 'swa', 'vie', 'zho-CN'}
    allg = ev['groups']['all']
    probed = [l for l in ev['groups']['trained'] if l != 'eng']
    groups = {'switch-exposed': probed,
              'replay-exposed': [l for l in allg if l in corp and l not in probed],
              'genuinely unseen': [l for l in allg if l not in corp and l != 'eng']}
    arms = [('CS-B', 0.0), ('CS-W10', 0.10), ('CS-W25', 0.25), ('CS-W50', 0.50)]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    col = {'switch-exposed': '#009988', 'replay-exposed': '#4477aa',
           'genuinely unseen': '#cc3311'}
    for g, ls in groups.items():
        ys = []
        for a, _ in arms:
            v = [np.mean([np.log(ev['seeds'][s][a]['perplexity'][l] /
                                 F['perplexity'][l]) for l in ls])
                 for s in SEEDS]
            ys.append(float(np.exp(np.mean(v))))
        ax.plot([r for _, r in arms], ys, 'o-', color=col[g],
                label=f'{g} (n={len(ls)})')
        for (a, r), y in zip(arms, ys):
            ax.annotate(f'{y:.2f}', (r, y), textcoords='offset points',
                        xytext=(0, 6), fontsize=7, ha='center', color=col[g])
    ax.axhline(1.0, ls=':', color='grey', lw=1)
    ax.set_xlabel('code-switch rate'); ax.set_ylabel('perplexity vs frozen model')
    ax.set_title('Replay protects the languages it covers, and nothing else',
                 fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig20_codeswitch_cost.png'), dpi=160)
    plt.close(fig)
    print('  fig20_codeswitch_cost.png')


# ---------------------------------------------------------------- figure 21
def fig21():
    """The guard saturates, and cannot restrain an unbounded term."""
    pts = []
    for tag, w in (('', 1.0), ('_cvp10', 10.0), ('_cvp100', 100.0),
                   ('_cvp1k', 1000.0)):
        fs = glob.glob(os.path.join(
            RES, 'aim2', f'Qwen3-0.6B-Base_armE{tag}_seed*.json'))
        fs = [f for f in fs if ('_cvp' in f) == bool(tag)]
        if not fs:
            continue
        vals = [np.mean([r['cvp'] for r in json.load(open(f))['log']])
                for f in fs]
        pts.append((w, float(np.mean(vals)), float(np.std(vals, ddof=1))
                    if len(vals) > 1 else 0.0))
    wf = glob.glob(os.path.join(RES, 'aim2',
                                'Qwen3-0.6B-Base_armWA-F_seed*.json'))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    if pts:
        w = [p[0] for p in pts]; m = [p[1] for p in pts]; e = [p[2] for p in pts]
        ax.errorbar(w, m, yerr=e, fmt='o-', color='#4477aa', capsize=3,
                    label='sentence-level condition (InfoNCE)')
        for a, b in zip(w, m):
            ax.annotate(f'{b:.3f}', (a, b), textcoords='offset points',
                        xytext=(0, 8), fontsize=7, ha='center')
    if wf:
        v = [np.mean([r['cvp'] for r in json.load(open(f))['log']
                      if r.get('cvp') == r.get('cvp')]) for f in wf]
        ax.axhline(float(np.mean(v)), color='#cc3311', ls='--',
                   label='word-level condition, same guard active')
        ax.text(2, float(np.mean(v)) + 0.03,
                f'guard active, still collapses: {np.mean(v):.3f}',
                fontsize=8, color='#cc3311')
    ax.axhline(0.90, ls=':', color='grey', lw=1)
    ax.text(1.2, 0.915, 'floor the guard names', fontsize=7, color='grey')
    ax.set_xscale('log'); ax.set_xlabel('CVP hinge weight')
    ax.set_ylabel('mean concept-variance preservation')
    ax.set_ylim(-0.05, 1.15)
    ax.set_title('A bounded penalty saturates and cannot restrain an\n'
                 'unbounded alignment term', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig21_guard_sweep.png'), dpi=160)
    plt.close(fig)
    print('  fig21_guard_sweep.png')


if __name__ == '__main__':
    os.makedirs(FIGS, exist_ok=True)
    print('[figures]')
    for fn in (fig17, fig18, fig19, fig20, fig21):
        try:
            fn()
        except Exception as e:
            print(f'  {fn.__name__} FAILED: {e}')
    print('[done]')
