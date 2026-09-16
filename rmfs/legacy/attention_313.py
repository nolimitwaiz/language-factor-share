#!/usr/bin/env python3
"""Attention mechanism for the cross-lingual context result (§3.1.3).

Predictions frozen in prereg/PREREG_ATTENTION_313.md BEFORE any attention
weight was extracted.

The behavioral experiment (transfer_313b.py) showed that a translated
context sentence makes an English target more predictable, and that the size
of that effect tracks per-language alignment. This measures whether the
target tokens actually attend to the context tokens, and whether that
attention scales with alignment.

Design is deliberately identical to the behavioral experiment: same sentence
pairs, same languages, same matched and mismatched conditions, so the two
results are directly comparable.

Primary quantity, per layer:

    raw     = mean over target positions and heads of the attention mass
              landing on context positions
    uniform = the same share under evenly spread attention, which depends
              only on sequence geometry
    ratio   = raw / uniform

The ratio is primary because context length varies with tokenizer fertility
across languages, and the raw share alone would be confounded with it.

Usage:
    python code/attention_313.py --model Qwen/Qwen3-0.6B-Base --n_pairs 60
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from transfer_313b import CTX_LANGS, NTREX, load_pairs  # noqa: E402
from src.common import ledger  # noqa: E402


def attention_to_context(model, tok, context, target, device, n_layers):
    """Per-layer attention mass from target positions onto context positions,
    with the geometric baseline. Returns (raw[n_layers], uniform[n_layers]).

    Attention weights are sliced and cast to fp32 immediately; the full
    [heads, seq, seq] tensors are never accumulated."""
    import torch
    tgt = tok(' ' + target, return_tensors='pt', add_special_tokens=False)
    ctx = tok(context + '\n', return_tensors='pt')
    ids = torch.cat([ctx.input_ids, tgt.input_ids], 1).to(device)
    n_ctx = ctx.input_ids.shape[1]
    n_tot = ids.shape[1]
    if n_tot <= n_ctx:
        return None, None
    with torch.no_grad():
        out = model(ids, output_attentions=True)

    tgt_pos = np.arange(n_ctx, n_tot)
    raw = np.zeros(n_layers, dtype=np.float64)
    for li, A in enumerate(out.attentions):        # [1, heads, seq, seq]
        # rows = target positions, columns = context positions
        block = A[0, :, n_ctx:, :n_ctx].float()    # [heads, n_tgt, n_ctx]
        raw[li] = float(block.sum(-1).mean().cpu())
    # geometric baseline: under uniform attention over visible positions,
    # position t sends n_ctx / (t + 1) of its mass to the context
    uniform = float(np.mean(n_ctx / (tgt_pos + 1.0)))
    del out
    return raw, np.full(n_layers, uniform)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--n_pairs', type=int, default=60)
    ap.add_argument('--langs', type=int, default=len(CTX_LANGS))
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tag = args.model.split('/')[-1]
    rec = ledger.open_run('attention_313',
                          {'model': args.model, 'n_pairs': args.n_pairs})
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        # eager attention is required: the default kernel does not return
        # attention weights
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16,
            attn_implementation='eager').to(dev).eval()
        n_layers = model.config.num_hidden_layers
        print(f'[model] {tag}, {n_layers} layers, eager attention', flush=True)

        eng, pairs, mis = load_pairs(args.n_pairs)
        texts, langs = {'eng': eng}, []
        for lg in CTX_LANGS[:args.langs]:
            if lg == 'eng':
                langs.append(lg); continue
            p = os.path.join(NTREX, f'newstest2019-ref.{lg}.txt')
            if os.path.exists(p):
                with open(p) as f:
                    texts[lg] = [l.strip() for l in f]
                langs.append(lg)
        print(f'[data] {len(pairs)} pairs, {len(langs)} languages', flush=True)

        res = {}
        for lg in langs:
            acc = {'matched': [], 'mismatched': []}
            unis = {'matched': [], 'mismatched': []}
            for i, j in zip(pairs, mis):
                for cond, ctx_idx in (('matched', i), ('mismatched', j)):
                    r, u = attention_to_context(model, tok, texts[lg][ctx_idx],
                                                eng[i + 1], dev, n_layers)
                    if r is not None:
                        acc[cond].append(r); unis[cond].append(u)
            row = {}
            for cond in ('matched', 'mismatched'):
                raw = np.mean(acc[cond], 0)
                uni = np.mean(unis[cond], 0)
                row[f'raw_{cond}'] = raw.tolist()
                row[f'ratio_{cond}'] = (raw / uni).tolist()
            ca = np.array(row['ratio_matched']) - np.array(row['ratio_mismatched'])
            row['content_attention'] = ca.tolist()
            row['best_layer'] = int(np.argmax(ca))
            row['content_attention_best'] = float(ca.max())
            row['content_attention_mean'] = float(ca.mean())
            res[lg] = row
            print(f'  [{lg}] content attention: best {ca.max():+.4f} '
                  f'at L{int(np.argmax(ca))}, mean {ca.mean():+.4f}', flush=True)

        out_dir = os.path.join(ROOT, 'results', 'attention313')
        os.makedirs(out_dir, exist_ok=True)
        payload = {'model': args.model, 'n_layers': n_layers,
                   'n_pairs': len(pairs), 'per_language': res}
        with open(os.path.join(out_dir, f'{tag}.json'), 'w') as f:
            json.dump(payload, f, indent=1)
        ledger.close_run(rec, 'done', {'out': f'{out_dir}/{tag}.json'})
        print(f'[done] -> {out_dir}/{tag}.json')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
