#!/usr/bin/env python3
"""Score the code-switch arms against PREREG_CODESWITCH.md section 5.

What distinguishes this from `scorecard.py`: every criterion carries a
pre-registered minimum detectable effect, and an observed effect smaller than
that is reported as **NOT DETECTABLE** rather than as a pass or a fail. The
LFS study's criterion 5 returned verdicts decided by the sign of a quantity
eight times smaller than its own standard error (D7); this makes that
outcome impossible to produce silently.

Language groups are derived from the training corpus manifest rather than
copied from the previous study, per D8. Five of the eight languages the
earlier evaluation called held-out are in the code-switch corpus and were
seen as full sentences through the replay mixture, so the group is split
three ways and only the genuinely unseen languages speak to generalization.

Runs on CPU against the result JSONs. Usage:
    python -m src.aim2.scorecard_codeswitch
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, 'results', 'aim2')
CS = os.path.join(ROOT, 'data', 'codeswitch')
ARMS = ['CS-B', 'CS-W10', 'CS-W25', 'CS-W50', 'CS-P25']
CONTROL = 'CS-B'
PIVOT = 'eng'
CVP_FLOOR = 0.90

# Minimum detectable effects, frozen in PREREG_CODESWITCH.md section 5.
MDE = {1: 0.018,   # MEXA, two across-seed sd of 0.009
       4: 0.010,   # AaR, two across-seed sd of 0.005
       5: 0.052}   # context use at 200 pairs, two standard errors of 0.026


def corpus_languages():
    p = os.path.join(CS, 'manifest.json')
    if os.path.exists(p):
        return set(json.load(open(p))['languages'])
    # manifest lives on the cluster; fall back to the frozen list
    return set(['ben', 'ces', 'deu', 'fra', 'hin', 'ind', 'kat', 'khm',
                'pol', 'rus', 'spa', 'swa', 'vie', 'zho-CN'])


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def train_logs():
    out = {}
    for f in glob.glob(os.path.join(RES, 'Qwen3-0.6B-Base_armCS-*.json')):
        d = json.load(open(f))
        out[(d['arm'], d['seed'])] = d
    return out


def verdict(observed, mde, better_is_positive=True, tol=0.0):
    """Returns ('PASS'|'FAIL'|'NOT DETECTABLE', detail)."""
    if abs(observed) < mde:
        return 'NOT DETECTABLE', f'|{observed:+.4f}| < MDE {mde:.4f}'
    ok = observed > tol if better_is_positive else observed > -mde
    return ('PASS' if ok else 'FAIL'), f'{observed:+.4f} vs MDE {mde:.4f}'


def main():
    ev = load('evaluation_codeswitch.json')
    cx = load('context_use_codeswitch.json')
    logs = train_logs()
    if not logs:
        print('no code-switch training logs found'); return
    seeds = sorted({s for (_, s) in logs})
    corp = corpus_languages()

    print('Code-switch scorecard, criteria frozen in PREREG_CODESWITCH.md '
          f'section 5\nSeeds {seeds}. Control arm {CONTROL}.\n')
    print('Minimum detectable effects, pre-registered: '
          f'MEXA {MDE[1]}, AaR {MDE[4]}, context use {MDE[5]}.')
    print('An observed effect below its MDE is reported NOT DETECTABLE and '
          'returns no verdict.\n')

    groups = None
    if ev:
        allg = ev['groups']['all']
        probed = [l for l in ev['groups']['trained'] if l != PIVOT]
        groups = {
            'switch-exposed (probed)': probed,
            'replay-exposed': [l for l in allg
                               if l in corp and l not in probed],
            'genuinely unseen': [l for l in allg
                                 if l not in corp and l != PIVOT]}
        print('Language groups, derived from the corpus manifest (D8):')
        for g, ls in groups.items():
            print(f'  {g:26} {len(ls):>2}: {ls}')
        print('  Only the last group speaks to generalization or forgetting.\n')

    # ---- criteria 2 and 3, from the training logs ----
    print('--- Criterion 2: concept variance on the FINAL checkpoint (D6 form) ---')
    fin = {}
    for a in ARMS:
        v = [logs[(a, s)]['log'][-1]['cvp'] for s in seeds if (a, s) in logs]
        fin[a] = v
        ok = all(x >= CVP_FLOOR for x in v)
        print(f'  {a:>7}: {[f"{x:.3f}" for x in v]}  -> '
              f'{"PASS" if ok else "FAIL"}')

    print('\n--- Criterion 3: trajectory, mean CVP vs control minus one sd ---')
    means = {a: [np.mean([r['cvp'] for r in logs[(a, s)]['log']])
                 for s in seeds if (a, s) in logs] for a in ARMS}
    thr = np.mean(means[CONTROL]) - np.std(means[CONTROL], ddof=1)
    print(f'  threshold = control mean {np.mean(means[CONTROL]):.4f} '
          f'- sd {np.std(means[CONTROL], ddof=1):.4f} = {thr:.4f}')
    for a in ARMS:
        m = np.mean(means[a])
        print(f'  {a:>7}: mean CVP {m:.4f}  -> '
              f'{"PASS" if m >= thr else "FAIL"}')

    if not ev:
        print('\n[downstream evaluation not present; criteria 1, 4, 6 '
              'unscored]')
        return

    tl = str(ev.get('target_layer', 8))

    def stat(arm, seed, group, key):
        langs = groups[group]
        r = ev['seeds'][str(seed)].get(arm)
        if not r:
            return None
        # the stored per-group stats are keyed by the evaluation's own groups;
        # recompute the group mean from the 'all' group where possible
        return r['all'][tl][key]

    print(f'\n--- Criterion 1: mean-case retrieval vs {CONTROL} (MDE '
          f'{MDE[1]}) ---')
    for a in ARMS:
        if a == CONTROL:
            print(f'  {a:>7}: control'); continue
        d = []
        for s in seeds:
            x = stat(a, s, 'switch-exposed (probed)', 'mexa_mean')
            b = stat(CONTROL, s, 'switch-exposed (probed)', 'mexa_mean')
            if x is not None and b is not None:
                d.append(x - b)
        if not d:
            print(f'  {a:>7}: no data'); continue
        v, det = verdict(float(np.mean(d)), MDE[1], True)
        print(f'  {a:>7}: dMEXA {np.mean(d):+.4f}  -> {v}  ({det})')

    print(f'\n--- Criterion 4: tail alignment does not degrade (MDE '
          f'{MDE[4]}) ---')
    for a in ARMS:
        if a == CONTROL:
            print(f'  {a:>7}: control'); continue
        d = []
        for s in seeds:
            x = stat(a, s, 'switch-exposed (probed)', 'aar10_mean')
            b = stat(CONTROL, s, 'switch-exposed (probed)', 'aar10_mean')
            if x is not None and b is not None:
                d.append(x - b)
        if not d:
            print(f'  {a:>7}: no data'); continue
        v, det = verdict(float(np.mean(d)), MDE[4], False)
        print(f'  {a:>7}: dAaR {np.mean(d):+.4f}  -> {v}  ({det})')

    # ---- P3: does tail alignment fall as LFS falls, as it did in the LFS study?
    print('\n--- P3: is the LFS-AaR coupling present when the intervention '
          'touches only data? ---')
    pts = []
    for s in seeds:
        for a in ['CS-A'] + ARMS:
            r = ev['seeds'][str(s)].get(a)
            if r:
                pts.append((r['all'][tl]['lfs'], r['all'][tl]['aar10_mean']))
    if len(pts) > 3:
        L = np.array([p[0] for p in pts]); A = np.array([p[1] for p in pts])
        from scipy.stats import pearsonr, spearmanr
        print(f'  across {len(pts)} arm-seed points: Pearson '
              f'{pearsonr(L, A).statistic:+.3f}, Spearman '
              f'{spearmanr(L, A).statistic:+.3f}')
        print('  LFS study, for comparison: Pearson +0.924, Spearman +0.787')

    print(f'\n--- Criterion 5: context use at 200 pairs (MDE {MDE[5]}) ---')
    if cx:
        tr = [l for l in cx.get('trained_languages', []) ]
        for a in ARMS:
            if a == CONTROL:
                print(f'  {a:>7}: control'); continue
            d = []
            for s in seeds:
                ca = cx['seeds'].get(str(s), {}).get(a)
                cb = cx['seeds'].get(str(s), {}).get(CONTROL)
                if ca and cb:
                    d += [ca[l]['context_gain_mean'] - cb[l]['context_gain_mean']
                          for l in tr if l in ca and l in cb]
            if not d:
                print(f'  {a:>7}: no data'); continue
            v, det = verdict(float(np.mean(d)), MDE[5], True)
            print(f'  {a:>7}: dgain {np.mean(d):+.4f}  -> {v}  ({det})')
    else:
        print('  context-use results not present')

    print('\n--- Criterion 6 / P5: did replay reduce forgetting? ---')
    print('  LFS study, no replay: held-out perplexity x2.10 to x2.31')
    for a in ['CS-A'] + ARMS:
        rows = []
        for s in seeds:
            r = ev['seeds'][str(s)].get(a)
            fr = ev['seeds'][str(s)].get('CS-A')
            if r and fr:
                v = [r['perplexity'][l] / fr['perplexity'][l]
                     for l in groups['genuinely unseen'] if l in r['perplexity']]
                if v:
                    rows.append(float(np.exp(np.mean(np.log(v)))))
        if rows:
            print(f'  {a:>7}: genuinely-unseen perplexity ratio '
                  f'x{np.mean(rows):.3f}')

    print('\nNote: the genuinely unseen group is three languages (D8), so '
          'every statement\nabout generalization or forgetting here rests on '
          'three points.')


if __name__ == '__main__':
    main()
