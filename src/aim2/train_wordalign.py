#!/usr/bin/env python3
"""Word-level alignment objectives, proposal section 3.2.2 as written.

WHY THIS EXISTS. The six-arm study in `train.py` used mean-pooled sentence
representations with an InfoNCE term. Section 3.2.2 specifies something else
as its primary form:

    "we will process a full sentence and its translation in another language
     independently and link the words that are translations of each other
     ... We will add a loss term to the training objective that measures the
     difference between word representations for the matched words."

and names a variant:

    "we may allow for a language identity embedding vector that is subtracted
     from the word embeddings before computing their difference."

and only then, secondarily:

    "In addition to word-level loss terms, we will explore also loss terms
     that operate of the segment level, for instance obtained by mean or max
     pooling."

The earlier study ran the "in addition" variant. Its conclusion is a result
about segment-level pooling, not about 3.2.2. This file runs the primary
form, the language-vector variant, and the segment form side by side so the
three can be compared at matched data, steps, seeds and optimizer.

THE LANGUAGE-IDENTITY VECTOR IS THE LFS DECOMPOSITION. LFS models a
representation as grand mean plus concept plus language plus residual.
Subtracting a per-language vector before comparing matched words removes
exactly the component LFS measures. The proposal arrives at this
independently, which makes arm WA-D the most direct test available of whether
Aim 1's instrument informs an Aim 2 method.

Word links come from the Aim 1 alignments, where two independent aligners
agreed. Per-language agreement varies from 0.75 down to 0.39 and is recorded
in every result file, because a language whose links are unreliable
contributes noise to the loss rather than signal.

Usage:
    python -m src.aim2.train_wordalign --arm WA-D --seed 0
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
from src.aim2.train import concept_variance, effective_rank, batch_lfs, pooled  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'aim2')
ALIGN = os.path.join(ROOT, 'results', 'wordlevel', 'alignments')
TRAIN_RANGE = (500, 1900)
CVP_FLOOR = 0.90
LAYER = 8

# Languages: reliable alignments, spanning script and tier. English is the
# pivot every link is measured against.
LANGS = ['deu', 'fra', 'ces', 'hin', 'swa', 'ben']
MIN_AGREEMENT = 0.50

# WEIGHTING. The section specifies a loss on "the difference between word
# representations for the matched words" and names no weight. A raw mean
# squared difference on unnormalized representations is minimized globally at
# h = 0: shrink every representation to zero and the term vanishes. The smoke
# run made this concrete, with the word term at 615 to 2488 against a language
# modelling loss near 10, so the specified objective would dominate by two
# orders of magnitude and drive straight to the degenerate solution.
#
# WORD_W balances the two terms at initialization so neither trivially wins.
# That is a choice the proposal does not make, and it is recorded here rather
# than buried: 0.02 puts the word term at roughly the same magnitude as the
# language modelling term on the first batch.
#
# The degeneracy is not engineered away, only slowed. Whether it still occurs,
# and whether concept-variance preservation detects it, is the point of arms
# WA-C and WA-F and is predicted in prereg/PREREG_WORDALIGN.md.
WORD_W = 0.02

ARMS = {
    'WA-A': dict(lm=0.0, word=0.0,    seg=0.0,    langvec=False, cvp=0.0, train=False),
    'WA-B': dict(lm=1.0, word=0.0,    seg=0.0,    langvec=False, cvp=0.0, train=True),
    'WA-C': dict(lm=1.0, word=WORD_W, seg=0.0,    langvec=False, cvp=0.0, train=True),
    'WA-D': dict(lm=1.0, word=WORD_W, seg=0.0,    langvec=True,  cvp=0.0, train=True),
    'WA-E': dict(lm=1.0, word=0.0,    seg=WORD_W, langvec=False, cvp=0.0, train=True),
    'WA-F': dict(lm=1.0, word=WORD_W, seg=0.0,    langvec=True,  cvp=1.0, train=True),
    # Cosine distance instead of squared difference. Scale-invariant, so it
    # has no h=0 minimum, and it is the obvious repair if WA-C degenerates.
    'WA-G': dict(lm=1.0, word=WORD_W, seg=0.0, langvec=True, cvp=0.0,
                 cosine=True, train=True),
}


def load_alignments():
    """Per-language agreed one-to-one word links, plus the tokenized
    sentences they index into."""
    data, agree = {}, {}
    for lg in LANGS:
        p = os.path.join(ALIGN, f'{lg}.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        f1 = d['agreement_simalign_inter_vs_eflomal']['f1']
        if f1 < MIN_AGREEMENT:
            print(f'  [skip] {lg}: aligner agreement {f1:.3f} below '
                  f'{MIN_AGREEMENT}', flush=True)
            continue
        data[lg] = {'links': d['links_agreed'], 'src': d['sentences_src'],
                    'tgt': d['sentences_tgt']}
        agree[lg] = f1
    return data, agree


def word_reps(model, tok, words, dev, max_len, layer=LAYER, need_lm=False):
    """Per-word representations at `layer`, by averaging the subword pieces
    the tokenizer assigns to each word.

    is_split_into_words plus word_ids gives an exact word-to-subword map, so
    no heuristic alignment between the aligner's tokens and the model's
    tokens is needed. Words truncated away return no row."""
    import torch
    enc = tok(words, is_split_into_words=True, return_tensors='pt',
              truncation=True, max_length=max_len).to(dev)
    kw = {}
    if need_lm:
        labels = enc['input_ids'].clone()
        labels[enc['attention_mask'] == 0] = -100
        kw['labels'] = labels
    out = model(**enc, output_hidden_states=True, **kw)
    h = out.hidden_states[layer][0]                     # [T, D]
    wids = enc.word_ids(0)
    idx = {}
    for pos, w in enumerate(wids):
        if w is not None:
            idx.setdefault(w, []).append(pos)
    if not idx:
        return None, None, out
    keys = sorted(idx)
    rows = torch.stack([h[idx[k]].mean(0) for k in keys])
    return rows, {k: i for i, k in enumerate(keys)}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--pairs_per_step', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_len', type=int, default=64)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--word_weight', type=float, default=None)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.optimization import Adafactor

    cfg = dict(ARMS[args.arm])
    if args.word_weight is not None:
        for k in ('word', 'seg'):
            if cfg[k]:
                cfg[k] = args.word_weight
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = args.model.split('/')[-1]
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    rec = ledger.open_run('aim2_wordalign', vars(args), seed=args.seed)
    t0 = time.time()
    try:
        data, agree = load_alignments()
        langs = sorted(data)
        lo, hi = TRAIN_RANGE
        hi = min(hi, min(len(data[lg]['links']) for lg in langs))
        usable = [(lg, si) for lg in langs for si in range(lo, hi)
                  if data[lg]['links'][si]]
        print(f'[data] {len(langs)} languages {langs}', flush=True)
        print(f'[data] aligner agreement: '
              f'{ {k: round(v, 3) for k, v in agree.items()} }', flush=True)
        print(f'[data] {len(usable)} aligned sentence pairs in range '
              f'({lo}, {hi})', flush=True)

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32).to(dev)
        hidden = model.config.hidden_size

        # The language-identity vector of section 3.2.2, learned jointly.
        # Initialized at zero so arm WA-D starts identical to WA-C and any
        # divergence is attributable to the vector rather than to its init.
        lang_ix = {lg: i for i, lg in enumerate(['eng'] + langs)}
        langvec = torch.zeros(len(lang_ix), hidden, device=dev,
                              requires_grad=cfg['langvec'])

        params = list(model.parameters()) + ([langvec] if cfg['langvec'] else [])
        opt = (Adafactor(params, lr=args.lr, scale_parameter=False,
                         relative_step=False, warmup_init=False)
               if cfg['train'] else None)
        if cfg['train']:
            model.gradient_checkpointing_enable()
            model.train()

        # Frozen reference concept variance, measured on a fixed probe of
        # pairs before any update, so the guard compares against the
        # untrained model rather than against itself.
        probe = rng.sample(usable, min(32, len(usable)))
        model.eval()
        with torch.no_grad():
            pv_s, pv_t = [], []
            for lg, si in probe:
                d = data[lg]
                sr, _, _ = word_reps(model, tok, d['src'][si], dev, args.max_len)
                tr, _, _ = word_reps(model, tok, d['tgt'][si], dev, args.max_len)
                if sr is None or tr is None:
                    continue
                pv_s.append(sr.mean(0)); pv_t.append(tr.mean(0))
            ref_cv = concept_variance(
                torch.stack([torch.stack(pv_s), torch.stack(pv_t)], 0)).detach()
        print(f'[reference] frozen concept variance {float(ref_cv):.2f} '
              f'on {len(pv_s)} probe pairs', flush=True)
        if cfg['train']:
            model.train()
        log = []
        steps = args.steps if cfg['train'] else 1
        for step in range(steps):
            batch = [usable[rng.randrange(len(usable))]
                     for _ in range(args.pairs_per_step)]
            lm_terms, word_terms, seg_terms, norms = [], [], [], []
            src_vecs, tgt_vecs = [], []
            for lg, si in batch:
                d = data[lg]
                sw, tw, links = d['src'][si], d['tgt'][si], d['links'][si]
                if not sw or not tw or not links:
                    continue
                sr, smap, so = word_reps(model, tok, sw, dev, args.max_len,
                                         need_lm=True)
                tr, tmap, to = word_reps(model, tok, tw, dev, args.max_len,
                                         need_lm=True)
                if sr is None or tr is None:
                    continue
                lm_terms.append(so.loss + to.loss)
                pairs = [(smap[i], tmap[j]) for i, j in links
                         if i in smap and j in tmap]
                if not pairs:
                    continue
                si_ = torch.tensor([p[0] for p in pairs], device=dev)
                ti_ = torch.tensor([p[1] for p in pairs], device=dev)
                a, b = sr[si_], tr[ti_]
                if cfg['langvec']:
                    a = a - langvec[lang_ix['eng']]
                    b = b - langvec[lang_ix[lg]]
                if cfg['word']:
                    if cfg.get('cosine'):
                        word_terms.append(
                            (1 - F.cosine_similarity(a, b, dim=-1)).mean())
                    else:
                        # the section's own wording: "the difference between
                        # word representations for the matched words"
                        word_terms.append(F.mse_loss(a, b))
                if cfg['seg']:
                    seg_terms.append(F.mse_loss(sr.mean(0), tr.mean(0)))
                src_vecs.append(sr.mean(0))
                tgt_vecs.append(tr.mean(0))
                norms.append(float(sr.norm(dim=-1).mean()))

            if not lm_terms:
                continue
            loss = cfg['lm'] * torch.stack(lm_terms).mean()
            if cfg['word'] and word_terms:
                loss = loss + cfg['word'] * torch.stack(word_terms).mean()
            if cfg['seg'] and seg_terms:
                loss = loss + cfg['seg'] * torch.stack(seg_terms).mean()

            cvp = float('nan')
            if src_vecs:
                grid = torch.stack([torch.stack(src_vecs),
                                    torch.stack(tgt_vecs)], 0)
                # The reference must come from the FROZEN model. An earlier
                # version used the current batch as its own reference, which
                # makes the ratio identically 1.0 and the hinge inert: arms
                # with and without the guard produced bit-identical losses.
                cv = concept_variance(grid)
                cvp = float(cv / (ref_cv + 1e-9))
                if cfg['cvp']:
                    loss = loss + cfg['cvp'] * torch.clamp(
                        CVP_FLOOR - cv / (ref_cv + 1e-9), min=0) ** 2

            if cfg['train']:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params if p.requires_grad], 1.0)
                opt.step()

            if step % args.log_every == 0 or step == steps - 1:
                with torch.no_grad():
                    lv = (float(langvec.norm(dim=-1).mean())
                          if cfg['langvec'] else 0.0)
                row = {'step': step, 'loss': float(loss),
                       'rep_norm': float(np.mean(norms)) if norms else None,
                       'lm_loss': float(torch.stack(lm_terms).mean()),
                       'word_loss': (float(torch.stack(word_terms).mean())
                                     if word_terms else None),
                       'seg_loss': (float(torch.stack(seg_terms).mean())
                                    if seg_terms else None),
                       'langvec_norm': lv, 'cvp': cvp}
                log.append(row)
                print(f"  step {step:>4} loss {row['loss']:.4f} "
                      f"lm {row['lm_loss']:.4f} "
                      f"word {row['word_loss'] if row['word_loss'] is None else round(row['word_loss'], 4)} "
                      f"seg {row['seg_loss'] if row['seg_loss'] is None else round(row['seg_loss'], 4)} "
                      f"langvec {lv:.4f} "
                      f"|h| {row['rep_norm']:.2f} cvp {cvp:.3f}", flush=True)

        os.makedirs(OUT, exist_ok=True)
        res = {'arm': args.arm, 'config': {k: v for k, v in cfg.items()},
               'model': args.model, 'seed': args.seed, 'steps': args.steps,
               'layer': LAYER, 'languages': langs,
               'aligner_agreement': agree,
               'n_aligned_pairs': len(usable),
               'train_range': list(TRAIN_RANGE),
               'wallclock_s': round(time.time() - t0, 1), 'log': log}
        p = os.path.join(
            OUT, f'{tag}_arm{args.arm}{args.tag}_seed{args.seed}.json')
        with open(p, 'w') as f:
            json.dump(res, f, indent=1)
        if cfg['train']:
            sd = os.path.join(
                OUT, f'ckpt_arm{args.arm}{args.tag}_seed{args.seed}')
            model.save_pretrained(sd)
            tok.save_pretrained(sd)
            if cfg['langvec']:
                np.save(os.path.join(sd, 'langvec.npy'),
                        langvec.detach().cpu().numpy())
            print(f'[ckpt] -> {sd}', flush=True)
        ledger.close_run(rec, 'done', {'out': p})
        print(f'[done] -> {p}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
