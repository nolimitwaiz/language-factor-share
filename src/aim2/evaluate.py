#!/usr/bin/env python3
"""Downstream evaluation of the Aim 2 study arms (plan section 7).

WHY THIS EXISTS SEPARATELY FROM TRAINING. The study plan states that a
falling LFS is not success. Arm C exists to demonstrate exactly that: an
objective can drive the metric down by destroying the concept structure the
metric is supposed to hold constant. Nothing in the training loop can settle
whether an arm helped, because the training loop only sees the quantity the
arm was optimizing. This script computes the quantities the arms were not
optimizing.

Three deliberate choices:

1. EVALUATION SENTENCES ARE DISJOINT FROM TRAINING SENTENCES. Training used
   range (500, 1900). Evaluation uses the first 300, the same sentences every
   Aim 1 result is reported on. An arm that improved only on its own training
   text has not improved anything.

2. EVALUATION LANGUAGES INCLUDE LANGUAGES NO ARM EVER SAW. Training used
   four. If alignment pressure at one layer produces a general improvement
   it should appear, attenuated, in held-out languages; if it produces only
   a four-language special case, that is worth knowing and is invisible
   without this split.

3. EVERY COMPARISON IS AGAINST ARM B, NOT THE FROZEN MODEL. Plan section 8
   warned that if ordinary continued training moves LFS on its own then the
   metric is partly tracking data domain. It does: Arm B moved batch LFS from
   0.282 to 0.193 with no LFS term present. Differences against the frozen
   model would therefore credit the objective for an effect that plain
   training produces anyway.

The evaluation LFS is not the batch LFS the arms optimized. It is computed
at the Aim 1 protocol scale, where the pure-noise value is small and the
quantity is dominated by signal rather than by degrees of freedom.

Usage:
    python -m src.aim2.evaluate --seeds 0 1 2
Outputs: results/aim2/evaluation_seed<k>.json and a combined scorecard.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from pilot_metrics import aar, lfs, mexa_score  # noqa: E402
from cluster_grid import load_all_ntrex  # noqa: E402
from src.common import ledger  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'aim2')
PIVOT = 'eng'

# The four the arms trained on.
TRAINED = ['eng', 'deu', 'hin', 'swa']
# Held out from training entirely. Chosen to span script and resource tier
# rather than to be easy: Latin high resource, Cyrillic, Arabic, Han,
# Devanagari sibling, and two low resource African languages, so that a
# four-language special case and a general effect look different here.
HELD_OUT = ['fra', 'rus', 'arb', 'zho-CN', 'ben', 'yor', 'zul', 'ind']

ARMS = ['A', 'B', 'C', 'D', 'E', 'F']   # overridable with --arms
# Arms that are the untrained model rather than a checkpoint. Named per
# study, so this is a set rather than a literal: hardcoding 'A' silently
# skipped CS-A in the code-switch run and left criterion 6 without a
# baseline.
FROZEN_ARMS = {'A', 'CS-A', 'WA-A'}


def embed(model, tok, data, langs, dev, max_len=128, bs=16):
    """Masked-mean pooled hidden states, fp32 accumulation, [N, L+1, D]."""
    import torch
    out = {}
    for lg in langs:
        chunks = []
        sents = data[lg]
        for i in range(0, len(sents), bs):
            enc = tok(sents[i:i + bs], return_tensors='pt', padding=True,
                      truncation=True, max_length=max_len).to(dev)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            m = enc['attention_mask'].unsqueeze(-1).float()
            pooled = [(h.float() * m).sum(1) / m.sum(1) for h in hs]
            chunks.append(torch.stack(pooled, 1).float().cpu().numpy())
        out[lg] = np.concatenate(chunks, 0)
    return out


def perplexity(model, tok, data, langs, dev, max_len=128, bs=8):
    """Token-level perplexity per language on held-out sentences."""
    import torch
    res = {}
    for lg in langs:
        tot_nll, tot_tok = 0.0, 0
        sents = data[lg]
        for i in range(0, len(sents), bs):
            enc = tok(sents[i:i + bs], return_tensors='pt', padding=True,
                      truncation=True, max_length=max_len).to(dev)
            ids, am = enc['input_ids'], enc['attention_mask']
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=am).logits.float()
            sl = logits[:, :-1].log_softmax(-1)
            tgt, tm = ids[:, 1:], am[:, 1:].float()
            nll = -sl.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tot_nll += float((nll * tm).sum())
            tot_tok += int(tm.sum())
        res[lg] = float(np.exp(tot_nll / max(tot_tok, 1)))
    return res


def layer_stats(emb, langs):
    """Per-layer LFS, MEXA, AaR and the raw concept variance.

    Concept variance is returned unnormalized because CVP is a ratio of it
    against the frozen model, and normalizing here would divide out exactly
    the quantity the ratio is meant to detect."""
    n_layers = emb[langs[0]].shape[1]
    per_layer = {}
    for layer in range(n_layers):
        E = {lg: emb[lg][:, layer, :].astype(np.float32) for lg in langs}
        allE = np.concatenate([E[lg] for lg in langs], 0)
        mu, sd = allE.mean(0), allE.std(0) + 1e-9
        Ez = {lg: (E[lg] - mu) / sd for lg in langs}
        X = np.stack([Ez[lg] for lg in langs], 0)
        d = lfs(X)
        mex = [mexa_score(Ez[PIVOT], Ez[lg]) for lg in langs if lg != PIVOT]
        a10 = [aar(Ez[PIVOT], Ez[lg])[0] for lg in langs if lg != PIVOT]
        # raw concept variance, on unstandardized embeddings, for CVP
        Xr = np.stack([E[lg] for lg in langs], 0)
        concept_means = Xr.mean(0)
        v_concept_raw = float(((concept_means - Xr.mean((0, 1))) ** 2).sum(1).mean())
        per_layer[str(layer)] = {
            'lfs': d['lfs'], 'var_lang': d['var_lang'],
            'var_concept': d['var_concept'], 'residual_share': d['var_resid'],
            'mexa_mean': float(np.mean(mex)), 'aar10_mean': float(np.mean(a10)),
            'v_concept_raw': v_concept_raw}
    return per_layer


def evaluate_one(path, tok_path, data, groups, dev, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
    res = {}
    all_langs = sorted(set(sum(groups.values(), [])))
    emb = embed(model, tok, data, all_langs, dev)
    for gname, glangs in groups.items():
        res[gname] = layer_stats({lg: emb[lg] for lg in glangs}, glangs)
    res['perplexity'] = perplexity(model, tok, data, all_langs, dev)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--arms', nargs='+', default=None,
                    help='arm names to score; defaults to the LFS study arms')
    ap.add_argument('--n_sents', type=int, default=300)
    ap.add_argument('--layer', type=int, default=8,
                    help='the layer alignment pressure was applied at')
    args = ap.parse_args()
    global ARMS
    if args.arms:
        ARMS = args.arms

    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    rec = ledger.open_run('aim2_evaluate', vars(args))
    try:
        langs = TRAINED + HELD_OUT
        data = load_all_ntrex(args.n_sents)
        missing = [lg for lg in langs if lg not in data]
        langs = [lg for lg in langs if lg in data]
        groups = {'trained': [lg for lg in TRAINED if lg in data],
                  'held_out': [PIVOT] + [lg for lg in HELD_OUT if lg in data],
                  'all': langs}
        print(f'[data] {len(langs)} evaluation languages on the first '
              f'{args.n_sents} sentences, disjoint from training range '
              f'(500, 1900)', flush=True)
        if missing:
            print(f'[data] not available in NTREX, dropped: {missing}',
                  flush=True)
        print(f'[groups] trained={groups["trained"]} '
              f'held_out={groups["held_out"]}', flush=True)

        out = {'model': args.model, 'n_sents': args.n_sents,
               'target_layer': args.layer, 'groups': groups,
               'evaluation_note': 'all contrasts are against arm B',
               'seeds': {}}

        for seed in args.seeds:
            per_arm = {}
            for arm in ARMS:
                if arm in FROZEN_ARMS:
                    path = args.model
                else:
                    path = os.path.join(OUT, f'ckpt_arm{arm}_seed{seed}')
                    if not os.path.isdir(path):
                        print(f'[skip] seed {seed} arm {arm}: no checkpoint',
                              flush=True)
                        continue
                print(f'[eval] seed {seed} arm {arm}', flush=True)
                per_arm[arm] = evaluate_one(path, args.model, data, groups,
                                            dev, dtype)
                tl = str(args.layer)
                a = per_arm[arm]['trained'][tl]
                h = per_arm[arm]['held_out'][tl]
                print(f'   trained  L{args.layer}: lfs {a["lfs"]:.4f} '
                      f'mexa {a["mexa_mean"]:.4f} aar {a["aar10_mean"]:.4f}',
                      flush=True)
                print(f'   held_out L{args.layer}: lfs {h["lfs"]:.4f} '
                      f'mexa {h["mexa_mean"]:.4f} aar {h["aar10_mean"]:.4f}',
                      flush=True)
            out['seeds'][str(seed)] = per_arm

            # CVP against the frozen model, and every contrast against B
            ref = next((a for a in FROZEN_ARMS if a in per_arm), None)
            if ref and 'B' in per_arm:
                tl = str(args.layer)
                print(f'\n  [contrasts at L{args.layer}, seed {seed}] '
                      f'deltas are arm minus arm B', flush=True)
                for arm in ARMS:
                    if arm not in per_arm:
                        continue
                    r, b = per_arm[arm], per_arm['B']
                    cvp = (r['trained'][tl]['v_concept_raw'] /
                           max(per_arm[ref]['trained'][tl]['v_concept_raw'], 1e-9))
                    d_lfs = r['trained'][tl]['lfs'] - b['trained'][tl]['lfs']
                    d_mex = r['trained'][tl]['mexa_mean'] - b['trained'][tl]['mexa_mean']
                    d_aar = r['trained'][tl]['aar10_mean'] - b['trained'][tl]['aar10_mean']
                    d_ho = r['held_out'][tl]['mexa_mean'] - b['held_out'][tl]['mexa_mean']
                    ppl_en = r['perplexity'][PIVOT] / b['perplexity'][PIVOT]
                    print(f'   arm {arm}: dLFS {d_lfs:+.4f}  CVP {cvp:.3f}  '
                          f'dMEXA {d_mex:+.4f}  dAaR {d_aar:+.4f}  '
                          f'dMEXA_heldout {d_ho:+.4f}  ppl_eng x{ppl_en:.3f}',
                          flush=True)

        os.makedirs(OUT, exist_ok=True)
        suffix = ('_codeswitch_frozen' if list(ARMS) == ['CS-A']
                  else '_codeswitch' if any(a.startswith('CS') for a in ARMS)
                  else '_wordalign' if any(a.startswith('WA') for a in ARMS)
                  else '_sweep' if any('cvp' in a or 'tail' in a for a in ARMS)
                  else '')
        p = os.path.join(OUT, f'evaluation{suffix}.json')
        with open(p, 'w') as f:
            json.dump(out, f, indent=1)
        ledger.close_run(rec, 'done', {'out': p})
        print(f'[done] -> {p}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
