#!/usr/bin/env python3
"""Phase 0 / Phase 3: dump pooled PRE-z-score fp32 embeddings at selected
layers, one npz per (model, layer).

Layer selection per model: L0 (embedding), ~0.25 relative depth, the model's
LFS-dip layer (argmin of per-layer LFS from prior grid results — read, never
recomputed), ~0.75, final.

Pooling: fp32 masked mean — copied VERBATIM from pilot_metrics.embed_all
(massive-activation outliers overflow fp16 sums; incident #1). Equivalence is
enforced two ways: a unit test on the pooling helper, and the Phase-0 parity
gate (LFS/MEXA recomputed from dumps must match prior grid numbers).

Storage: fp16 on disk is quantization for storage only, applied AFTER fp32
pooling. The in-process parity check quantifies the quantization error before
anything is written off.

Usage:
  python src/campaign/dump_embeddings.py --model Qwen/Qwen3-1.7B-Base \
      --n_sents 300 [--device mps] [--dtype fp16] [--max_langs 8]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'code'))
sys.path.insert(0, ROOT)

from pilot_metrics import aar, lfs, mexa_score  # noqa: E402
from cluster_grid import load_all_ntrex  # noqa: E402
from src.common import ledger  # noqa: E402

DUMPS = os.path.join(ROOT, 'results', 'dumps')

# BLOOM never in fp16 (bf16-trained; fp16 = garbage). Incident #2.
FP16_FORBIDDEN = ('bloom',)


def pool_hidden_states(hidden_states, attention_mask, layer_indices):
    """fp32 masked mean over tokens for the selected layers.
    Pooling formula copied verbatim from pilot_metrics.embed_all (parity-
    tested in tests/test_pooling.py). Returns [B, n_sel, D] float32."""
    import torch
    mask = attention_mask.unsqueeze(-1).float()
    pooled = [(hidden_states[li].float() * mask).sum(1) / mask.sum(1)
              for li in layer_indices]
    return torch.stack(pooled, 1).float().cpu().numpy()


def select_layers(model_tag, n_layers):
    """L0, ~0.25, LFS-dip (from prior grid metrics.json), ~0.75, final.
    n_layers counts hidden_states entries (transformer blocks + embedding)."""
    grid = os.path.join(ROOT, 'results', 'grid', model_tag, 'metrics.json')
    dip = None
    if os.path.exists(grid):
        r = json.load(open(grid))
        curve = {int(k): v['lfs']['lfs'] for k, v in r['per_layer'].items()}
        dip = min(curve, key=curve.get)
    last = n_layers - 1
    sel = [0, round(0.25 * last), dip if dip is not None else round(0.5 * last),
           round(0.75 * last), last]
    return sorted(set(sel)), dip


def per_sentence_margins(X32, langs):
    """Per-sentence retrieval margins vs the English pivot at one layer,
    under the standard pipeline (per-dim z-score across all (lang, sent)
    pairs, cosine, margin = true-pair sim minus best off-diagonal in the
    row). Enables AaR@{1,5,10,20} tail curves downstream. Returns
    {lang: margins[N]} (pivot excluded)."""
    L, N, D = X32.shape
    flat = X32.reshape(L * N, D)
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    Xz = (X32 - mu) / sd
    i_en = langs.index('eng')
    En = Xz[i_en] / (np.linalg.norm(Xz[i_en], axis=1, keepdims=True) + 1e-9)
    out = {}
    eye = np.eye(N, dtype=bool)
    for i, lg in enumerate(langs):
        if lg == 'eng':
            continue
        Ln = Xz[i] / (np.linalg.norm(Xz[i], axis=1, keepdims=True) + 1e-9)
        S = Ln @ En.T
        d = np.diag(S).copy()
        off = np.where(eye, -np.inf, S).max(1)
        out[lg] = (d - off).astype(np.float32)
    return out


def dump_model(model_name, n_sents, device, dtype, batch_size=16, max_len=128,
               max_langs=None, out_root=DUMPS, save_margins=False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tag = model_name.split('/')[-1]
    if any(b in model_name.lower() for b in FP16_FORBIDDEN) and dtype == 'fp16':
        raise SystemExit(f'{model_name}: fp16 forbidden (bf16-trained model); '
                         f'use fp32 (2080Ti) or bf16 (A100).')

    rec = ledger.open_run('phase0_dump', {
        'model': model_name, 'n_sents': n_sents, 'device': device,
        'dtype': dtype, 'max_langs': max_langs})
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    try:
        data = load_all_ntrex(n_sents, max_langs)
        langs = list(data)

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16,
                       'fp32': torch.float32}[dtype]
        model = AutoModel.from_pretrained(
            model_name, dtype=torch_dtype,
            output_hidden_states=True).to(device).eval()

        # probe layer count with one forward, then select layers
        with torch.no_grad():
            enc = tok(data[langs[0]][:2], return_tensors='pt', padding=True,
                      truncation=True, max_length=max_len).to(device)
            n_layers = len(model(**enc).hidden_states)
        sel, dip = select_layers(tag, n_layers)
        print(f'[layers] {tag}: n={n_layers} selected={sel} (dip={dip})')

        store = {li: [] for li in sel}               # layer -> list of [N, D]
        for lang in langs:
            per_layer = {li: [] for li in sel}
            sents = data[lang]
            for i in range(0, len(sents), batch_size):
                enc = tok(sents[i:i + batch_size], return_tensors='pt',
                          padding=True, truncation=True,
                          max_length=max_len).to(device)
                with torch.no_grad():
                    hs = model(**enc).hidden_states
                pooled = pool_hidden_states(hs, enc['attention_mask'], sel)
                for j, li in enumerate(sel):
                    per_layer[li].append(pooled[:, j, :])
            for li in sel:
                store[li].append(np.concatenate(per_layer[li], 0))
            print(f'[embed] {lang}: done', flush=True)

        paths = {}
        for li in sel:
            X32 = np.stack(store[li], 0).astype(np.float32)  # [L, N, D]
            p = os.path.join(out_dir, f'layer{li:03d}.npz')
            np.savez_compressed(p, X=X32.astype(np.float16),
                                langs=np.array(langs))
            paths[li] = p
            if save_margins:
                mp = os.path.join(out_dir, f'margins{li:03d}.npz')
                np.savez_compressed(mp, **per_sentence_margins(X32, langs))
                print(f'[margins] L{li} -> {mp}', flush=True)
            # quantization check at this layer: fp32 vs fp16-roundtrip metrics
            if li == (dip if dip in sel else sel[len(sel) // 2]):
                q = quantization_check(X32, langs)
                print(f'[quantization@L{li}] {q}')
        meta = {'model': model_name, 'tag': tag, 'n_sents': n_sents,
                'langs': langs, 'n_layers': n_layers, 'layers': sel,
                'dip_layer': dip, 'device': device, 'dtype_compute': dtype,
                'storage': 'fp16 (post-fp32-pooling quantization only)'}
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        ledger.close_run(rec, 'done', {'out_dir': out_dir,
                                       'layers': {str(k): v for k, v in paths.items()}})
        return out_dir
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


def metrics_from_array(X32, langs):
    """Standard pipeline from a [L, N, D] fp32 array: per-dim z-score across
    all (lang, sent) pairs (legacy convention), then LFS + mean MEXA/AaR."""
    L, N, D = X32.shape
    flat = X32.reshape(L * N, D)
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    Xz = (X32 - mu) / sd
    i_en = langs.index('eng')
    res = {'lfs': lfs(Xz)['lfs'], 'mexa': {}, 'aar10': {}}
    for i, lg in enumerate(langs):
        if lg == 'eng':
            continue
        res['mexa'][lg] = mexa_score(Xz[i_en], Xz[i])
        a10, _ = aar(Xz[i_en], Xz[i])
        res['aar10'][lg] = a10
    return res


def quantization_check(X32, langs):
    """fp16 storage round-trip: how much do LFS / mean-MEXA move?"""
    a = metrics_from_array(X32, langs)
    b = metrics_from_array(X32.astype(np.float16).astype(np.float32), langs)
    mexa_a = np.mean(list(a['mexa'].values()))
    mexa_b = np.mean(list(b['mexa'].values()))
    return {'d_lfs': round(abs(a['lfs'] - b['lfs']), 6),
            'd_mean_mexa': round(abs(mexa_a - mexa_b), 6)}


def parity_against_grid(tag, dump_dir=None):
    """Golden-file parity: recompute LFS + per-language MEXA/AaR from the
    dump at each dumped layer and compare with results/grid/<tag>/metrics.json.
    Emits numbers; tolerances are judged against the prereg, not by code."""
    dump_dir = dump_dir or os.path.join(DUMPS, tag)
    meta = json.load(open(os.path.join(dump_dir, 'meta.json')))
    grid = json.load(open(os.path.join(ROOT, 'results', 'grid', tag,
                                       'metrics.json')))
    rows = []
    for li in meta['layers']:
        z = np.load(os.path.join(dump_dir, f'layer{li:03d}.npz'),
                    allow_pickle=False)
        X = z['X'].astype(np.float32)
        langs = [str(x) for x in z['langs']]
        got = metrics_from_array(X, langs)
        ref = grid['per_layer'][str(li)]
        ref_mexa = {lg: v['mexa'] for lg, v in ref['langs'].items()}
        common = [lg for lg in got['mexa'] if lg in ref_mexa]
        d_mexa = [abs(got['mexa'][lg] - ref_mexa[lg]) for lg in common]
        rows.append({
            'layer': li,
            'd_lfs': abs(got['lfs'] - ref['lfs']['lfs']),
            'd_mexa_mean': float(np.mean(d_mexa)) if d_mexa else None,
            'd_mexa_max': float(np.max(d_mexa)) if d_mexa else None,
            'n_common_langs': len(common),
        })
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n_sents', type=int, default=300)
    ap.add_argument('--device', default=None)
    ap.add_argument('--dtype', default=None, help='fp16|bf16|fp32')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_langs', type=int, default=None)
    ap.add_argument('--parity_only', action='store_true',
                    help='skip dumping; run the parity report on an existing dump')
    ap.add_argument('--save_margins', action='store_true',
                    help='also save per-sentence retrieval margins per layer')
    args = ap.parse_args()

    tag = args.model.split('/')[-1]
    if args.parity_only:
        for row in parity_against_grid(tag):
            print(row)
        sys.exit(0)

    if args.device is None:
        import torch
        args.device = ('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.dtype is None:
        args.dtype = ('fp32' if any(b in args.model.lower()
                                    for b in FP16_FORBIDDEN) else 'fp16')
    dump_model(args.model, args.n_sents, args.device, args.dtype,
               batch_size=args.batch_size, max_langs=args.max_langs,
               save_margins=args.save_margins)
