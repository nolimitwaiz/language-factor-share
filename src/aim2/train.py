#!/usr/bin/env python3
"""Aim 2 study training (proposal section 3.2.2).

The proposal: "We plan to directly optimize the alignment of word
representations across languages by adding additional loss terms... This
work will be informed by the findings of our Aim 1 where we establish
metrics that indicate the multilingual robustness of models. Here, we adapt
these metrics into training objectives."

Arms, per docs/AIM2_STUDY_PLAN.md:

  A  frozen baseline, no training
  B  L_LM                                    domain-adaptation control
  C  L_LM + g*LFS                            naive, expected to collapse
  D  L_LM + a*align                          contrastive alignment
  E  L_LM + a*align + b*CVP                  adds the anti-collapse constraint
  F  L_LM + a*align + b*CVP + g*LFS          does LFS add anything over E

Arm F answers the only question that justifies the LFS term existing as a
loss at all. If E and F are indistinguishable, LFS is a diagnostic and a
layer selector, not a training signal, and that is a publishable answer.

WHY THE TRIPWIRES RUN EVERY STEP. The first study minimized LFS directly,
doubled the dip in 400 steps, and was found afterwards to have reached that
number by collapsing concept representations. A later controlled injection
established that under collapse LFS improves while MEXA and AaR stay flat,
and only concept-variance preservation detects it. So CVP, effective rank
and batch LFS are logged throughout, and a collapse is visible while it is
happening rather than diagnosed from the wreckage.

Batches are built as (sentences x languages) grids so that every loss term
and every tripwire is computable on the same forward pass.

Usage:
    python -m src.aim2.train --arm E --steps 400 --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.common import ledger  # noqa: E402

NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')
OUT = os.path.join(ROOT, 'results', 'aim2')
TRAIN_RANGE = (500, 1900)          # disjoint from every Aim 1 evaluation set

# Four languages, not eight. This is a deliberate scope reduction forced by
# the batch LFS term rather than by memory alone. The estimator's pure-noise
# value is (L-1)/((L-1)+(K-1)) for L languages and K concepts in a batch, so
# at L=8 and the batch sizes that fit on one card (K=4) the floor is 0.700
# and the first run measured 0.585, below its own floor: the LFS term would
# have been optimizing noise. Dropping to L=4 with K=8 puts the floor at
# 0.300, comfortably under the observed value. The four span Latin and
# Devanagari script and high, mid and low resource tiers, and all four clear
# the Stage 0 alignment-reliability threshold.
LANGS = ['eng', 'deu', 'hin', 'swa']

ARMS = {
    'A': dict(lm=0.0, align=0.0, cvp=0.0, lfs=0.0, train=False),
    'B': dict(lm=1.0, align=0.0, cvp=0.0, lfs=0.0, train=True),
    'C': dict(lm=1.0, align=0.0, cvp=0.0, lfs=1.0, train=True),
    'D': dict(lm=1.0, align=1.0, cvp=0.0, lfs=0.0, train=True),
    'E': dict(lm=1.0, align=1.0, cvp=1.0, lfs=0.0, train=True),
    'F': dict(lm=1.0, align=1.0, cvp=1.0, lfs=1.0, train=True),
    # Follow-ups ranked after the LFS study. G targets the tail directly,
    # which is the untested direction of the association: that study drove
    # LFS down and watched AaR follow, but never tried driving AaR up.
    'G': dict(lm=1.0, align=1.0, cvp=1.0, lfs=0.0, tail=1.0, train=True),
}
CVP_FLOOR = 0.90                   # tau in the hinge; see the study plan


def load_parallel():
    lo, hi = TRAIN_RANGE
    data = {}
    for lg in LANGS:
        cands = [f'newstest2019-ref.{lg}.txt']
        if lg == 'eng':
            cands.insert(0, 'newstest2019-src.eng.txt')
        for c in cands:
            p = os.path.join(NTREX, c)
            if os.path.exists(p):
                with open(p) as f:
                    lines = [l.strip() for l in f]
                data[lg] = lines[lo:hi]
                break
    n = min(len(v) for v in data.values())
    keep = [i for i in range(n) if all(data[lg][i] for lg in data)]
    return {lg: [data[lg][i] for i in keep] for lg in data}, len(keep)


def pooled(hidden, mask):
    """fp32 masked mean, matching the measurement pipeline exactly."""
    m = mask.unsqueeze(-1).float()
    return (hidden.float() * m).sum(1) / m.sum(1)


def batch_lfs(grid):
    """Differentiable batch LFS on a [L, K, D] grid: the language share of
    systematic variance, computed exactly as the evaluation metric is."""
    import torch
    mu = grid.mean(dim=(0, 1))
    mu_c = grid.mean(0)
    mu_l = grid.mean(1)
    L, K, _ = grid.shape
    ss_c = L * ((mu_c - mu) ** 2).sum()
    ss_l = K * ((mu_l - mu) ** 2).sum()
    return ss_l / (ss_l + ss_c + 1e-9)


def concept_variance(grid):
    """Total spread across concepts, averaged over languages. The quantity
    CVP preserves."""
    return grid.var(dim=1, unbiased=False).sum(-1).mean()


def effective_rank(grid):
    import torch
    X = grid.reshape(-1, grid.shape[-1])
    X = X - X.mean(0)
    s = torch.linalg.svdvals(X.float())
    p = s / (s.sum() + 1e-9)
    p = p[p > 1e-12]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def tail_margin(grid, tail_frac=0.10):
    """Differentiable stand-in for AaR: the mean retrieval margin of the
    worst decile of concepts, pivot against every other language.

    The LFS study found that objectives which improve mean-case retrieval can
    leave the tail worse than the frozen model, and that the two measures
    decouple entirely under optimization (rank -0.067). InfoNCE optimizes a
    mean over all pairs, so nothing in it attends to the tail. This term does:
    it selects the hardest concepts in the batch and optimizes only those.

    Margin for concept k in language l is the cosine similarity to the pivot's
    same-concept vector minus the best similarity to any other concept. Higher
    is better, so the loss returns the negated mean of the worst decile."""
    import torch
    import torch.nn.functional as F
    z = F.normalize(grid, dim=-1)          # [L, K, D]
    pivot = z[0]                           # [K, D]
    margins = []
    for l in range(1, z.shape[0]):
        sim = z[l] @ pivot.t()             # [K, K]
        correct = sim.diagonal()
        off = sim - torch.eye(sim.shape[0], device=sim.device) * 1e4
        margins.append(correct - off.max(dim=1).values)
    m = torch.cat(margins)
    k = max(1, int(round(len(m) * tail_frac)))
    worst = torch.topk(m, k, largest=False).values
    return -worst.mean()


def info_nce(z, n_lang, n_conc, temperature=0.07):
    """Translation-equivalent pairs are positives, other concepts in the
    batch are negatives. z is [L*K, d] with concept index varying fastest."""
    import torch
    import torch.nn.functional as F
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / temperature
    idx = torch.arange(len(z), device=z.device)
    conc = idx % n_conc
    same_conc = conc.unsqueeze(0) == conc.unsqueeze(1)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = same_conc & ~eye
    if pos.sum() == 0:
        return torch.zeros((), device=z.device)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * pos).sum() / pos.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--concepts_per_batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_len', type=int, default=48)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--cvp_weight', type=float, default=None,
                    help='override the CVP hinge weight. The LFS study found '
                         'the hinge at 1.0 yields a penalty near 0.0025 '
                         'against an alignment loss of order 3.4, which is '
                         'too small to enforce the constraint it names.')
    ap.add_argument('--tail_weight', type=float, default=None)
    ap.add_argument('--tag', default='',
                    help='suffix for output files, so sweep points at the '
                         'same arm do not overwrite each other')
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = dict(ARMS[args.arm])
    if args.cvp_weight is not None:
        cfg['cvp'] = args.cvp_weight
    if args.tail_weight is not None:
        cfg['tail'] = args.tail_weight
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = args.model.split('/')[-1]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rec = ledger.open_run('aim2_train',
                          {'arm': args.arm, 'model': args.model,
                           'steps': args.steps, 'lr': args.lr,
                           'languages': LANGS, **cfg}, seed=args.seed)
    try:
        data, n_sent = load_parallel()
        langs = [l for l in LANGS if l in data]
        print(f'[data] {len(langs)} languages, {n_sent} parallel sentences '
              f'from range {TRAIN_RANGE}', flush=True)

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32).to(dev)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        grid_meta = json.load(open(os.path.join(
            ROOT, 'results', 'grid', tag, 'metrics.json')))
        curve = {int(k): v['lfs']['lfs']
                 for k, v in grid_meta['per_layer'].items()}
        layer = min(curve, key=curve.get)
        print(f'[layer] applying alignment pressure at the dip layer L{layer}',
              flush=True)
        # The batch quantity has a different null than the evaluation
        # quantity, and reporting it is required by the study plan: an LFS
        # loss curve is uninterpretable without knowing what value pure
        # noise would produce at this batch shape.
        K_ = args.concepts_per_batch
        batch_floor = (len(langs) - 1) / ((len(langs) - 1) + (K_ - 1))
        print(f'[floor] batch LFS pure-noise value at L={len(langs)}, '
              f'K={K_}: {batch_floor:.4f}', flush=True)

        # Frozen reference for CVP. Precomputed once over the whole training
        # set rather than held as a second live model: it removes about
        # 2.4 GB of weights that existed only to produce one scalar per step
        # (which is what made the first attempt run out of memory), it is
        # cheaper per step, and being computed before any update it cannot
        # drift by construction.
        print('[reference] precomputing frozen concept variance', flush=True)
        ref_pool = {}
        with torch.no_grad():
            for lg in langs:
                chunks = []
                for i in range(0, n_sent, 32):
                    e = tok(data[lg][i:i + 32], return_tensors='pt',
                            padding=True, truncation=True,
                            max_length=args.max_len).to(dev)
                    hs = model(**e, output_hidden_states=True).hidden_states
                    chunks.append(pooled(hs[layer], e['attention_mask']).cpu())
                ref_pool[lg] = torch.cat(chunks, 0)
        ref_grid = torch.stack([ref_pool[lg] for lg in langs], 0)   # [L,N,D]
        print(f'[reference] {tuple(ref_grid.shape)} cached on cpu', flush=True)

        head = nn.Sequential(nn.Linear(model.config.hidden_size, 512),
                             nn.GELU(), nn.Linear(512, 256)).to(dev)
        params = list(model.parameters()) + list(head.parameters())
        # Adafactor rather than AdamW. AdamW keeps two fp32 moments per
        # parameter, about 4.8 GB for this model, which together with weights
        # and gradients does not fit an 11 GB card; Adafactor factors the
        # second moment and stores almost none. The choice is identical
        # across all six arms, so the between-arm comparison, which is the
        # entire point of the experiment, is unaffected.
        try:
            from transformers.optimization import Adafactor
            opt = Adafactor(params, lr=args.lr, scale_parameter=False,
                            relative_step=False, warmup_init=False)
            opt_name = 'adafactor'
        except Exception:
            opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
            opt_name = 'sgd-momentum'
        print(f'[optimizer] {opt_name}', flush=True)

        rng = np.random.default_rng(args.seed)
        K = args.concepts_per_batch
        log = []
        model.train() if cfg['train'] else model.eval()
        t0 = time.time()

        for step in range(args.steps if cfg['train'] else 1):
            idx = rng.choice(n_sent, K, replace=False)
            texts = [data[lg][i] for lg in langs for i in idx]
            enc = tok(texts, return_tensors='pt', padding=True,
                      truncation=True, max_length=args.max_len).to(dev)
            labels = enc['input_ids'].clone()
            labels[enc['attention_mask'] == 0] = -100

            out = model(**enc, labels=labels, output_hidden_states=True)
            h = pooled(out.hidden_states[layer], enc['attention_mask'])
            grid = h.view(len(langs), K, -1)

            ref_var = concept_variance(
                ref_grid[:, idx, :].to(dev)).detach()

            lfs_v = batch_lfs(grid)
            cvp = concept_variance(grid) / (ref_var + 1e-9)
            loss = cfg['lm'] * out.loss
            if cfg['align']:
                loss = loss + cfg['align'] * info_nce(head(h), len(langs), K)
            if cfg['cvp']:
                loss = loss + cfg['cvp'] * torch.clamp(
                    CVP_FLOOR - cvp, min=0) ** 2
            if cfg['lfs']:
                loss = loss + cfg['lfs'] * lfs_v
            if cfg.get('tail'):
                loss = loss + cfg['tail'] * tail_margin(grid)

            if cfg['train']:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                row = {'step': step, 'loss': float(loss),
                       'lm_loss': float(out.loss),
                       'batch_lfs': float(lfs_v), 'cvp': float(cvp),
                       'effective_rank': effective_rank(grid.detach()),
                       'concept_var': float(concept_variance(grid.detach()))}
                log.append(row)
                flag = ''
                if row['cvp'] < CVP_FLOOR:
                    flag = '   <-- TRIPWIRE: concept variance below floor'
                print(f"  step {step:>4} loss {row['loss']:.4f} "
                      f"lm {row['lm_loss']:.4f} lfs {row['batch_lfs']:.4f} "
                      f"cvp {row['cvp']:.4f} rank {row['effective_rank']:.1f}"
                      f"{flag}", flush=True)

        os.makedirs(OUT, exist_ok=True)
        res = {'arm': args.arm, 'config': cfg, 'model': args.model,
               'optimizer': opt_name,
               'layer': layer, 'languages': langs, 'seed': args.seed,
               'steps': args.steps, 'lr': args.lr, 'cvp_floor': CVP_FLOOR,
               'batch_lfs_pure_noise_value': batch_floor,
               'concepts_per_batch': K_, 'max_len': args.max_len,
               'wallclock_s': round(time.time() - t0, 1), 'log': log}
        p = os.path.join(OUT,
                         f'{tag}_arm{args.arm}{args.tag}_seed{args.seed}.json')
        with open(p, 'w') as f:
            json.dump(res, f, indent=1)
        if cfg['train']:
            sd = os.path.join(OUT,
                              f'ckpt_arm{args.arm}{args.tag}_seed{args.seed}')
            model.save_pretrained(sd)
            tok.save_pretrained(sd)
            print(f'[ckpt] -> {sd}')
        ledger.close_run(rec, 'done', {'out': p})
        print(f'[done] -> {p}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
