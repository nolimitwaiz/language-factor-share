#!/usr/bin/env python3
"""Criterion 5 of the Aim 2 study: cross-lingual context use, per arm.

The study plan requires that a promising arm improve cross-lingual context
use relative to arm B. That criterion had no implementation when the arms
were first scored, which is recorded in AIM2_STUDY_SCORECARD.md section 3.
This closes it.

The measurement is behavioural rather than mechanistic, and deliberately so.
`code/attention_313.py` measures where attention mass lands, which is the
mechanism; this measures whether the model's predictions actually get better
when given a translated context, which is the effect the criterion names. An
arm could redistribute attention without changing predictions, and that would
not be an improvement in context use.

Per language, over matched and mismatched context conditions:

    context_gain = NLL(English target | mismatched context)
                 - NLL(English target | matched translated context)

A positive gain means the model extracted usable information from a context
written in another language. The mismatched condition is the control: it has
identical length, language, register and tokenizer fertility, and differs
only in whether the content is relevant. Comparing against it rather than
against no context at all removes the confound whereby simply having more
tokens in the window changes the target's likelihood.

Design is deliberately identical to the Aim 1 experiment, same pairs, same
languages, same derangement, so the study numbers are directly comparable to
the Aim 1 baseline rather than to a re-specified quantity.

Every contrast is reported against arm B, not against the frozen model, for
the reason given in plan section 8 and confirmed at seed 0: ordinary
continued training moves these quantities on its own.

Usage:
    python -m src.aim2.context_use --seeds 0 1 2 --n_pairs 60
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from transfer_313b import NTREX, load_pairs, nll_of_target  # noqa: E402
from src.common import ledger  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'aim2')
ARMS = ['A', 'B', 'C', 'D', 'E', 'F']   # overridable with --arms
# See evaluate.py: the frozen reference is named per study.
FROZEN_ARMS = {'A', 'CS-A', 'WA-A'}

# The three non-English languages the arms trained on, plus three held out.
# Held-out languages test whether any gain is a four-language special case.
TRAINED = ['deu', 'hin', 'swa']
HELD_OUT = ['fra', 'rus', 'zho-CN']


def score_model(path, tok_path, eng, pairs, mis, texts, langs, dev):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32).to(dev).eval()
    res = {}
    for lg in langs:
        gains = []
        for i, j in zip(pairs, mis):
            tgt = eng[i + 1]
            nm = nll_of_target(model, tok, texts[lg][i], tgt, dev)
            nx = nll_of_target(model, tok, texts[lg][j], tgt, dev)
            gains.append(nx - nm)
        g = np.array(gains, dtype=np.float64)
        # paired over the same target sentences, so the standard error of the
        # mean of the per-pair differences is the right uncertainty here
        res[lg] = {'context_gain_mean': float(g.mean()),
                   'context_gain_sem': float(g.std(ddof=1) / np.sqrt(len(g))),
                   'n_pairs': int(len(g))}
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--arms', nargs='+', default=None,
                    help='arm names to score; defaults to the LFS study arms')
    ap.add_argument('--n_pairs', type=int, default=60)
    args = ap.parse_args()
    global ARMS
    if args.arms:
        ARMS = args.arms

    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rec = ledger.open_run('aim2_context_use', vars(args))
    try:
        eng, pairs, mis = load_pairs(args.n_pairs)
        texts, langs, missing = {'eng': eng}, [], []
        for lg in TRAINED + HELD_OUT:
            p = os.path.join(NTREX, f'newstest2019-ref.{lg}.txt')
            if os.path.exists(p):
                with open(p) as f:
                    texts[lg] = [l.strip() for l in f]
                langs.append(lg)
            else:
                missing.append(lg)
        print(f'[data] {len(pairs)} sentence pairs, {len(langs)} languages',
              flush=True)
        if missing:
            print(f'[data] not available, dropped: {missing}', flush=True)

        out = {'model': args.model, 'n_pairs': len(pairs),
               'trained_languages': [lg for lg in TRAINED if lg in texts],
               'held_out_languages': [lg for lg in HELD_OUT if lg in texts],
               'contrast_note': 'all deltas are against arm B',
               'seeds': {}}

        for seed in args.seeds:
            per_arm = {}
            for arm in ARMS:
                is_frozen = arm in FROZEN_ARMS
                path = (args.model if is_frozen else
                        os.path.join(OUT, f'ckpt_arm{arm}_seed{seed}'))
                if not is_frozen and not os.path.isdir(path):
                    print(f'[skip] seed {seed} arm {arm}: no checkpoint',
                          flush=True)
                    continue
                print(f'[eval] seed {seed} arm {arm}', flush=True)
                per_arm[arm] = score_model(path, args.model, eng, pairs, mis,
                                           texts, langs, dev)
                for lg in langs:
                    r = per_arm[arm][lg]
                    print(f'   {lg}: gain {r["context_gain_mean"]:+.4f} '
                          f'+/- {r["context_gain_sem"]:.4f}', flush=True)
            out['seeds'][str(seed)] = per_arm

            if 'B' in per_arm:
                print(f'\n  [criterion 5, seed {seed}] context gain minus '
                      f'arm B', flush=True)
                tr = [lg for lg in TRAINED if lg in langs]
                ho = [lg for lg in HELD_OUT if lg in langs]
                for arm in ARMS:
                    if arm not in per_arm:
                        continue
                    dt = np.mean([per_arm[arm][lg]['context_gain_mean'] -
                                  per_arm['B'][lg]['context_gain_mean']
                                  for lg in tr]) if tr else float('nan')
                    dh = np.mean([per_arm[arm][lg]['context_gain_mean'] -
                                  per_arm['B'][lg]['context_gain_mean']
                                  for lg in ho]) if ho else float('nan')
                    verdict = 'passes' if dt > 0 else 'fails'
                    if arm == 'B':
                        verdict = 'reference'
                    print(f'   arm {arm}: trained {dt:+.4f}  '
                          f'held_out {dh:+.4f}   criterion 5 {verdict}',
                          flush=True)

        os.makedirs(OUT, exist_ok=True)
        suffix = ('_codeswitch_frozen' if list(ARMS) == ['CS-A']
                  else '_codeswitch' if any(a.startswith('CS') for a in ARMS)
                  else '_wordalign' if any(a.startswith('WA') for a in ARMS)
                  else '_sweep' if any('cvp' in a or 'tail' in a for a in ARMS)
                  else '')
        p = os.path.join(OUT, f'context_use{suffix}.json')
        with open(p, 'w') as f:
            json.dump(out, f, indent=1)
        ledger.close_run(rec, 'done', {'out': p})
        print(f'[done] -> {p}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
