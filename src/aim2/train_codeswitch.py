#!/usr/bin/env python3
"""Code-switched training, proposal section 3.2.1.

Design, predictions and decision rules are frozen in
`prereg/PREREG_CODESWITCH.md`, written before this ran. Read that first; this
file implements it and adds nothing to it.

WHY THIS IS A SEPARATE TRAINER. `train.py` builds a [languages, concepts]
grid of parallel sentences every step, because its objectives act on
representations. Code-switched training is plain language modelling on
different *data*: there is no alignment term, no LFS term and no guard. The
grid is needed here only for monitoring, never for the loss, and folding a
loss-free mode into the grid trainer would put an unused code path next to a
live one for no benefit.

The concept-variance and LFS quantities are computed and logged exactly as in
`train.py`, on the same probe set, so the two studies are directly comparable.
They are **monitoring only**: nothing in this file optimizes them. That is
prediction P4's whole point, that collapse should be unreachable when no term
pushes representations together.

Every arm draws a fixed fraction of each batch from unmodified multilingual
text (the replay mixture, prereg section 2.1). It is identical across arms so
it cannot confound the code-switch contrast, and it exists because the LFS
study established that continued training on a narrow language set degrades
held-out languages more than twice over in every arm including its control,
which made its capability criterion undiscriminating.

Usage:
    python -m src.aim2.train_codeswitch --arm CS-W25 --seed 0
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.common import ledger  # noqa: E402
from src.aim2.train import (NTREX, concept_variance, effective_rank,  # noqa: E402
                            batch_lfs, pooled)

OUT = os.path.join(ROOT, 'results', 'aim2')
CS = os.path.join(ROOT, 'data', 'codeswitch')
TRAIN_RANGE = (500, 1900)
REPLAY_FRACTION = 0.25
PROBE_LANGS = ['eng', 'deu', 'hin', 'swa']   # same probe set as the LFS study
CVP_FLOOR = 0.90                             # monitoring tripwire only

# Arm -> (corpus subdirectory, use the switched text or the original)
ARMS = {
    'CS-A': (None, None),                    # frozen, no training
    'CS-B': ('rate0.25_word', 'original'),   # control: same sentences, unswitched
    'CS-W10': ('rate0.1_word', 'text'),
    'CS-W25': ('rate0.25_word', 'text'),
    'CS-W50': ('rate0.5_word', 'text'),
    'CS-P25': ('rate0.25_phrase', 'text'),
}


def load_corpus(subdir, field):
    """Returns the training sentences for an arm.

    CS-B reads the 'original' field of the same rows the switch arms read
    their 'text' field from, so the control sees identical sentences,
    identical count and identical domain, differing only in whether the
    substitutions were applied."""
    p = os.path.join(CS, subdir, 'corpus.jsonl')
    rows = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if d[field].strip():
                rows.append(d[field])
    return rows


def load_replay():
    """Unmodified full sentences in the switch languages, for rehearsal.

    Read from NTREX rather than from the corpus, because the corpus stores
    English source and switched English; genuine rehearsal needs whole
    sentences in the other languages."""
    man = json.load(open(os.path.join(CS, 'manifest.json')))
    lo, hi = TRAIN_RANGE
    out = []
    for lg in sorted(man['languages']):
        p = os.path.join(NTREX, f'newstest2019-ref.{lg}.txt')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            lines = [l.strip() for l in f]
        out.extend([s for s in lines[lo:hi] if s])
    return out


def probe_grid(model, tok, probe, dev, idx, max_len):
    """[L, K, D] pooled representations at the monitored layer, for the
    logged CVP, LFS and effective-rank diagnostics."""
    import torch
    rows = []
    for lg in PROBE_LANGS:
        sents = [probe[lg][i] for i in idx]
        enc = tok(sents, return_tensors='pt', padding=True, truncation=True,
                  max_length=max_len).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        rows.append(pooled(hs[8], enc['attention_mask']))
    return torch.stack(rows, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--probe_k', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_len', type=int, default=48)
    ap.add_argument('--log_every', type=int, default=20)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    subdir, field = ARMS[args.arm]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = args.model.split('/')[-1]
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    rec = ledger.open_run('aim2_codeswitch', vars(args), seed=args.seed)
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32).to(dev)

        # probe set: parallel sentences for the monitored diagnostics
        probe = {}
        lo, hi = TRAIN_RANGE
        for lg in PROBE_LANGS:
            cands = ([f'newstest2019-src.eng.txt'] if lg == 'eng' else []) + \
                    [f'newstest2019-ref.{lg}.txt']
            for c in cands:
                p = os.path.join(NTREX, c)
                if os.path.exists(p):
                    with open(p) as f:
                        probe[lg] = [l.strip() for l in f][lo:hi]
                    break
        n_probe = min(len(v) for v in probe.values())
        keep = [i for i in range(n_probe) if all(probe[lg][i] for lg in probe)]

        if subdir is None:
            print('[arm CS-A] frozen reference, no training', flush=True)
            train_rows, replay = [], []
        else:
            train_rows = load_corpus(subdir, field)
            replay = load_replay()
            print(f'[data] {len(train_rows)} training sentences from '
                  f'{subdir}:{field}, {len(replay)} replay sentences, '
                  f'replay fraction {REPLAY_FRACTION}', flush=True)
        print(f'[probe] {len(keep)} parallel sentences, {len(PROBE_LANGS)} '
              f'languages, monitored at layer 8', flush=True)
        K = args.probe_k
        floor = (len(PROBE_LANGS) - 1) / ((len(PROBE_LANGS) - 1) + (K - 1))
        print(f'[floor] probe LFS pure-noise value at L={len(PROBE_LANGS)}, '
              f'K={K}: {floor:.4f}  (monitoring only, not optimized)',
              flush=True)

        # frozen reference concept variance, on the probe set
        model.eval()
        ref_pool = {}
        with torch.no_grad():
            for lg in PROBE_LANGS:
                chunks = []
                for i in range(0, len(keep), 32):
                    sents = [probe[lg][j] for j in keep[i:i + 32]]
                    enc = tok(sents, return_tensors='pt', padding=True,
                              truncation=True, max_length=args.max_len).to(dev)
                    hs = model(**enc, output_hidden_states=True).hidden_states
                    chunks.append(pooled(hs[8], enc['attention_mask']).cpu())
                ref_pool[lg] = torch.cat(chunks, 0)
        ref_grid = torch.stack([ref_pool[lg] for lg in PROBE_LANGS], 0)
        print(f'[reference] {tuple(ref_grid.shape)} cached on cpu', flush=True)

        log = []
        if subdir is not None:
            from transformers.optimization import Adafactor
            opt = Adafactor(model.parameters(), lr=args.lr,
                            scale_parameter=False, relative_step=False,
                            warmup_init=False)
            model.gradient_checkpointing_enable()
            model.train()
            n_replay = int(round(args.batch_size * REPLAY_FRACTION))
            n_main = args.batch_size - n_replay

            for step in range(args.steps):
                batch = ([rng.choice(train_rows) for _ in range(n_main)] +
                         [rng.choice(replay) for _ in range(n_replay)])
                enc = tok(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=args.max_len).to(dev)
                labels = enc['input_ids'].clone()
                labels[enc['attention_mask'] == 0] = -100
                loss = model(**enc, labels=labels).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

                if step % args.log_every == 0 or step == args.steps - 1:
                    model.eval()
                    with torch.no_grad():
                        idx = rng.sample(range(len(keep)), K)
                        gi = [keep[j] for j in idx]
                        grid = probe_grid(model, tok, probe, dev, gi,
                                          args.max_len)
                        ref_var = concept_variance(
                            ref_grid[:, idx, :].to(dev))
                        cvp = float(concept_variance(grid) / (ref_var + 1e-9))
                        lfs_v = float(batch_lfs(grid))
                        er = effective_rank(grid)
                    model.train()
                    row = {'step': step, 'lm_loss': float(loss),
                           'probe_lfs': lfs_v, 'cvp': cvp,
                           'effective_rank': er}
                    log.append(row)
                    flag = ('   <-- tripwire: concept variance below floor'
                            if cvp < CVP_FLOOR else '')
                    print(f"  step {step:>4} lm {row['lm_loss']:.4f} "
                          f"lfs {lfs_v:.4f} cvp {cvp:.4f} rank {er:.1f}"
                          f"{flag}", flush=True)

        os.makedirs(OUT, exist_ok=True)
        res = {'arm': args.arm, 'corpus': subdir, 'field': field,
               'model': args.model, 'seed': args.seed, 'steps': args.steps,
               'lr': args.lr, 'batch_size': args.batch_size,
               'replay_fraction': REPLAY_FRACTION,
               'probe_languages': PROBE_LANGS,
               'probe_lfs_pure_noise_value': floor,
               'cvp_floor_monitoring_only': CVP_FLOOR,
               'n_train_sentences': len(train_rows),
               'n_replay_sentences': len(replay),
               'wallclock_s': round(time.time() - t0, 1), 'log': log}
        sfx = '' if args.steps == 400 else f'_s{args.steps}'
        p = os.path.join(OUT, f'{tag}_arm{args.arm}{sfx}_seed{args.seed}.json')
        with open(p, 'w') as f:
            json.dump(res, f, indent=1)
        if subdir is not None:
            sd = os.path.join(OUT, f'ckpt_arm{args.arm}{sfx}_seed{args.seed}')
            model.save_pretrained(sd)
            tok.save_pretrained(sd)
            print(f'[ckpt] -> {sd}', flush=True)
        ledger.close_run(rec, 'done', {'out': p})
        print(f'[done] -> {p}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
