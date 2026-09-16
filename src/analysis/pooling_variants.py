#!/usr/bin/env python3
"""Fertility and pooling robustness (plan frozen in
docs/FERTILITY_ROBUSTNESS_PLAN.md before this ran).

Recomputes the full layerwise fingerprint under four pooling schemes on the
same sentences and languages, so that any difference is attributable to
pooling alone.

  mean        masked mean over tokens, the current baseline
  final       the last non-padding position, no length denominator
  unitnorm    each token L2-normalized before averaging, removing per-token
              magnitude differences
  lenmatch    every language truncated to the per-sentence minimum token
              count across languages, so denominators match exactly.
              Reported as a BOUND, not a primary comparison: truncation
              changes which part of the meaning is encoded.

All pooling is accumulated in fp32 regardless of model dtype.

Usage: python -m src.analysis.pooling_variants --model Qwen/Qwen3-0.6B-Base
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

OUT = os.path.join(ROOT, 'results', 'pooling_variants')
SCHEMES = ('mean', 'final', 'unitnorm', 'lenmatch')
PIVOT = 'eng'


def pool(hs, mask, scheme, keep=None):
    """hs: tuple of [B,T,D]; mask: [B,T]. Returns [B, n_layers, D] fp32."""
    import torch
    m = mask.unsqueeze(-1).float()
    if scheme == 'final':
        idx = mask.sum(1) - 1
        out = [h.float()[torch.arange(h.shape[0]), idx] for h in hs]
    elif scheme == 'unitnorm':
        out = []
        for h in hs:
            hf = h.float()
            hn = hf / (hf.norm(dim=-1, keepdim=True) + 1e-9)
            out.append((hn * m).sum(1) / m.sum(1))
    elif scheme == 'lenmatch':
        # keep only the first `keep` real tokens of every sequence
        km = torch.zeros_like(m)
        for b in range(m.shape[0]):
            km[b, :int(keep[b])] = 1.0
        km = km * m
        out = [(h.float() * km).sum(1) / km.sum(1).clamp(min=1) for h in hs]
    else:
        out = [(h.float() * m).sum(1) / m.sum(1) for h in hs]
    return torch.stack(out, 1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n_sents', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--dtype', default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    tag = args.model.split('/')[-1]
    dt = args.dtype or ('fp32' if 'bloom' in args.model.lower()
                        or 'salamandra' in args.model.lower() else 'fp16')
    tdt = {'fp16': torch.float16, 'bf16': torch.bfloat16,
           'fp32': torch.float32}[dt]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rec = ledger.open_run('pooling_variants',
                          {'model': args.model, 'n_sents': args.n_sents,
                           'dtype': dt, 'schemes': list(SCHEMES)})
    try:
        data = load_all_ntrex(args.n_sents)
        langs = list(data)
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(args.model, dtype=tdt,
                                          output_hidden_states=True)
        model = model.to(dev).eval()

        # token counts per (language, sentence), needed for the length-matched
        # scheme and for the fertility diagnostic
        counts = {}
        for lg in langs:
            counts[lg] = np.array([
                len(tok(s, truncation=True, max_length=128)['input_ids'])
                for s in data[lg]], dtype=np.int32)
        stack = np.stack([counts[lg] for lg in langs], 0)
        min_per_sentence = stack.min(0)
        fert = {lg: float(counts[lg].mean() / counts[PIVOT].mean())
                for lg in langs}
        print(f'[{tag}] {len(langs)} languages, dtype {dt}; fertility '
              f'{min(fert.values()):.2f} to {max(fert.values()):.2f}',
              flush=True)

        embs = {s: {} for s in SCHEMES}
        for lg in langs:
            sents = data[lg]
            chunks = {s: [] for s in SCHEMES}
            for i in range(0, len(sents), args.batch_size):
                bt = sents[i:i + args.batch_size]
                enc = tok(bt, return_tensors='pt', padding=True,
                          truncation=True, max_length=128).to(dev)
                with torch.no_grad():
                    hs = model(**enc).hidden_states
                keep = torch.tensor(
                    min_per_sentence[i:i + len(bt)], device=dev).float()
                for s in SCHEMES:
                    chunks[s].append(pool(hs, enc['attention_mask'], s,
                                          keep=keep))
            for s in SCHEMES:
                embs[s][lg] = np.concatenate(chunks[s], 0)
            print(f'  [{lg}] done', flush=True)

        res = {'model': args.model, 'dtype': dt, 'n_sents': args.n_sents,
               'languages': langs, 'fertility': fert, 'schemes': {}}
        n_layers = embs['mean'][PIVOT].shape[1]
        for s in SCHEMES:
            per_layer, per_lang_dip = {}, {}
            for layer in range(n_layers):
                E = {lg: embs[s][lg][:, layer, :].astype(np.float32)
                     for lg in langs}
                allE = np.concatenate([E[lg] for lg in langs], 0)
                mu, sd = allE.mean(0), allE.std(0) + 1e-9
                Ez = {lg: (E[lg] - mu) / sd for lg in langs}
                X = np.stack([Ez[lg] for lg in langs], 0)
                d = lfs(X)
                mex = [mexa_score(Ez[PIVOT], Ez[lg])
                       for lg in langs if lg != PIVOT]
                a10 = [aar(Ez[PIVOT], Ez[lg])[0]
                       for lg in langs if lg != PIVOT]
                per_layer[str(layer)] = {
                    'lfs': d['lfs'], 'var_lang': d['var_lang'],
                    'var_concept': d['var_concept'],
                    'residual_share': d['var_resid'],
                    'mexa_mean': float(np.mean(mex)),
                    'aar10_mean': float(np.mean(a10))}
            curve = {int(k): v['lfs'] for k, v in per_layer.items()}
            dip = min(curve, key=curve.get)
            # per-language contribution at the dip layer: distance of the
            # language mean from the grand mean, the quantity that enters
            # V_lang, used for the fertility correlation
            E = {lg: embs[s][lg][:, dip, :].astype(np.float32) for lg in langs}
            allE = np.concatenate([E[lg] for lg in langs], 0)
            mu, sd = allE.mean(0), allE.std(0) + 1e-9
            Ez = {lg: (E[lg] - mu) / sd for lg in langs}
            gm = np.concatenate([Ez[lg] for lg in langs], 0).mean(0)
            contrib = {lg: float(((Ez[lg].mean(0) - gm) ** 2).sum())
                       for lg in langs}
            res['schemes'][s] = {
                'per_layer': per_layer, 'dip_layer': dip,
                'dip_lfs': curve[dip], 'l0_lfs': curve[0],
                'dip_depth': curve[0] - curve[dip],
                'lang_contribution_at_dip': contrib}
            print(f'  [{s:9s}] dip L{dip} lfs {curve[dip]:.4f} '
                  f'depth {curve[0]-curve[dip]:.4f} '
                  f'mexa {per_layer[str(dip)]["mexa_mean"]:.3f}', flush=True)

        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f'{tag}.json')
        with open(p, 'w') as f:
            json.dump(res, f)
        ledger.close_run(rec, 'done', {'out': p})
        print(f'[done] -> {p}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
