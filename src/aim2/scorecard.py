#!/usr/bin/env python3
"""Score the Aim 2 study arms against the seven criteria frozen in
AIM2_STUDY_PLAN.md section 7, before any arm ran.

This is deliberately mechanical. The criteria were written down in advance
precisely so that the verdict would not depend on which numbers looked
interesting after the fact, and a script that applies them without judgement
is the only way that guarantee is worth anything. Where the plan left a
threshold in units it did not specify, the operationalization is stated in
OPERATIONALIZED below and printed with the results rather than buried here.

Runs on CPU against the result JSONs; no model is loaded.

Usage:
    python -m src.aim2.scorecard
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, 'results', 'aim2')
ARMS = ['B', 'C', 'D', 'E', 'F']   # A is the frozen reference, not an arm
CVP_FLOOR = 0.90
PIVOT = 'eng'

OPERATIONALIZED = [
    'Criterion 3 baseline: the plan says residual share "does not rise by '
    'more than 5 points" without naming a baseline. Scored against the '
    'frozen model, since it is a preservation criterion.',
    'Criterion 4 noise band: "holds within noise" is scored as no worse than '
    'one across-seed standard deviation of arm B below arm B.',
    'Criterion 6 units: "1 point" is scored as one percentage point of '
    'relative perplexity increase over the frozen model.',
]


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def arm_logs(seed):
    out = {}
    for arm in ARMS:
        p = os.path.join(RES, f'Qwen3-0.6B-Base_arm{arm}_seed{seed}.json')
        if os.path.exists(p):
            out[arm] = json.load(open(p))
    return out


def score_seed(seed, ev, cx, logs, layer):
    """Returns {arm: {criterion: (bool|None, detail)}}."""
    tl = str(layer)
    e = (ev or {}).get('seeds', {}).get(str(seed), {})
    c = (cx or {}).get('seeds', {}).get(str(seed), {})
    out = {}
    have_eval = 'B' in e and 'A' in e

    # across-seed SD of arm B, for criterion 4's noise band
    b_mex, b_aar = [], []
    for s in (ev or {}).get('seeds', {}):
        r = ev['seeds'][s].get('B')
        if r:
            b_mex.append(r['trained'][tl]['mexa_mean'])
            b_aar.append(r['trained'][tl]['aar10_mean'])
    sd_mex = float(np.std(b_mex, ddof=1)) if len(b_mex) > 1 else 0.0
    sd_aar = float(np.std(b_aar, ddof=1)) if len(b_aar) > 1 else 0.0

    for arm in ARMS:
        r = {}
        # 1. LFS falls at the targeted layer relative to B
        if have_eval and arm in e:
            d = e[arm]['trained'][tl]['lfs'] - e['B']['trained'][tl]['lfs']
            r[1] = (d < 0 if arm != 'B' else None, f'dLFS vs B {d:+.4f}')
        else:
            r[1] = (None, 'no evaluation')

        # 2. CVP at or above the floor throughout training
        if arm in logs:
            cvps = [x['cvp'] for x in logs[arm]['log']]
            br = sum(1 for v in cvps if v < CVP_FLOOR)
            r[2] = (br == 0, f'{br}/{len(cvps)} steps below {CVP_FLOOR}, '
                             f'min {min(cvps):.3f}')
        else:
            r[2] = (None, 'no training log')

        # 3. residual share rises by at most 5 points vs the frozen model
        if have_eval and arm in e:
            d = (e[arm]['trained'][tl]['residual_share'] -
                 e['A']['trained'][tl]['residual_share']) * 100
            r[3] = (d <= 5.0, f'residual share {d:+.2f} pts vs frozen')
        else:
            r[3] = (None, 'no evaluation')

        # 4. MEXA or AaR improves vs B, or holds within noise
        if have_eval and arm in e:
            dm = e[arm]['trained'][tl]['mexa_mean'] - e['B']['trained'][tl]['mexa_mean']
            da = e[arm]['trained'][tl]['aar10_mean'] - e['B']['trained'][tl]['aar10_mean']
            ok = (dm > 0 or da > 0) or (dm >= -sd_mex and da >= -sd_aar)
            r[4] = (ok if arm != 'B' else None,
                    f'dMEXA {dm:+.4f} (sd {sd_mex:.4f}), '
                    f'dAaR {da:+.4f} (sd {sd_aar:.4f})')
        else:
            r[4] = (None, 'no evaluation')

        # 5. cross-lingual context use improves vs B
        if c and arm in c and 'B' in c:
            tr = (cx.get('trained_languages') or [])
            ds = [c[arm][lg]['context_gain_mean'] - c['B'][lg]['context_gain_mean']
                  for lg in tr if lg in c[arm] and lg in c['B']]
            if ds:
                d = float(np.mean(ds))
                r[5] = (d > 0 if arm != 'B' else None,
                        f'context gain vs B {d:+.4f}')
            else:
                r[5] = (None, 'no shared languages')
        else:
            r[5] = (None, 'not measured')

        # 6. capability degrades by at most B plus one percentage point
        if have_eval and arm in e:
            def deg(a, lg):
                return 100 * (e[a]['perplexity'][lg] / e['A']['perplexity'][lg] - 1)
            non_eng = [lg for lg in e[arm]['perplexity'] if lg != PIVOT]
            d_en = deg(arm, PIVOT) - deg('B', PIVOT)
            d_mono = (float(np.mean([deg(arm, lg) for lg in non_eng])) -
                      float(np.mean([deg('B', lg) for lg in non_eng])))
            r[6] = (d_en <= 1.0 and d_mono <= 1.0 if arm != 'B' else None,
                    f'eng {d_en:+.2f} pts vs B, non-eng {d_mono:+.2f} pts vs B')
        else:
            r[6] = (None, 'no evaluation')
        out[arm] = r
    return out


def main():
    ev, cx = load('evaluation.json'), load('context_use.json')
    layer = (ev or {}).get('target_layer', 8)
    seeds = sorted((ev or {}).get('seeds', {}) or {'0': None})
    seeds = [int(s) for s in seeds]
    if not ev:
        seeds = [s for s in (0, 1, 2) if arm_logs(s)]

    print(f'Aim 2 study scorecard, criteria frozen in AIM2_STUDY_PLAN.md '
          f'section 7\nTarget layer L{layer}. Seeds {seeds}.\n')
    for note in OPERATIONALIZED:
        print(f'  note: {note}')
    print()

    per_seed = {}
    for s in seeds:
        per_seed[s] = score_seed(s, ev, cx, arm_logs(s), layer)

    names = {'B': 'LM only', 'C': 'LM+LFS', 'D': 'LM+align',
             'E': 'LM+align+CVP', 'F': 'LM+align+CVP+LFS'}
    sym = {True: 'pass', False: 'FAIL', None: ' -- '}
    for arm in ARMS:
        print(f'--- arm {arm}: {names[arm]} ---')
        for s in seeds:
            row = per_seed[s].get(arm, {})
            cells = ' '.join(f'{k}:{sym[row.get(k,(None,""))[0]]}'
                             for k in range(1, 7))
            print(f'  seed {s}: {cells}')
            for k in range(1, 7):
                v, detail = row.get(k, (None, ''))
                print(f'      {k}. {detail}')
        # criterion 7: consistency of direction across seeds
        if len(seeds) > 1:
            consistent = True
            for k in range(1, 7):
                vals = [per_seed[s].get(arm, {}).get(k, (None, ''))[0]
                        for s in seeds]
                vals = [v for v in vals if v is not None]
                if vals and len(set(vals)) > 1:
                    consistent = False
            print(f'  criterion 7 (consistency across seeds): '
                  f'{"pass" if consistent else "FAIL"}')
        # overall
        allv = [per_seed[s].get(arm, {}).get(k, (None, ''))[0]
                for s in seeds for k in range(1, 7)]
        decided = [v for v in allv if v is not None]
        if not decided:
            verdict = 'NOT SCORED, evidence missing'
        elif all(decided) and len(decided) == len(allv):
            verdict = 'PROMISING, all criteria met'
        elif all(decided):
            verdict = 'INCOMPLETE, no failure yet but criteria unmeasured'
        else:
            verdict = 'FAILED'
        print(f'  verdict: {verdict}\n')

    p = os.path.join(RES, 'scorecard.json')
    payload = {'target_layer': layer, 'seeds': seeds,
               'operationalized': OPERATIONALIZED,
               'per_seed': {str(s): {a: {str(k): list(v)
                                         for k, v in r.items()}
                                     for a, r in per_seed[s].items()}
                            for s in per_seed}}
    with open(p, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'[done] -> {p}')


if __name__ == '__main__':
    main()
