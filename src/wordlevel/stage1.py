#!/usr/bin/env python3
"""Word-level LFS, stage 1: two-way decomposition on aligned word occurrences.

Predictions and gates frozen in prereg/PREREG_WORDLEVEL.md before any
word-level representation was extracted.

Concept unit is a specific English word occurrence, the (sentence, English
token) pair, restricted to content words. Language axis is the aligned word
in each target language. That keeps the grid structurally identical to
sentence level, so the existing decomposition applies unchanged.

Order matters: alignment first, then extraction. Hidden states are retained
only at aligned positions and only at the selected layers.

Usage:
    python -m src.wordlevel.stage1 --model Qwen/Qwen3-1.7B-Base \
        --operating_point primary
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from pilot_metrics import lfs  # noqa: E402
from src.common import ledger  # noqa: E402

ALIGN = os.path.join(ROOT, 'results', 'wordlevel', 'alignments')
OUT = os.path.join(ROOT, 'results', 'wordlevel', 'stage1')

# Operating points frozen in the pre-registration.
POINTS = {
    'primary': ['spa', 'ind', 'deu', 'khm', 'fra', 'vie', 'ces', 'swa',
                'hin', 'pol', 'zho-CN'],
    'secondary': ['spa', 'ind', 'deu', 'khm'],
}
PIVOT = 'eng'


def df_floor(L, N):
    return (L - 1) / ((L - 1) + (N - 1))


def load_alignment(lang):
    with open(os.path.join(ALIGN, f'{lang}.json')) as f:
        d = json.load(f)
    return d['links_agreed'], d['sentences_src'], d['sentences_tgt']


def shared_concepts(langs, content_only=True):
    """English (sentence, token) positions aligned in EVERY language, plus
    each language's map from that position to its own token index.

    The alignment files hold every agreed link, which is the correct raw
    output. Deciding which links constitute concepts belongs to the
    experiment, and the pre-registration specifies content words defined on
    the English side. An earlier version of this function omitted that
    filter, so 35 to 39 percent of the grid was function words; see D5 in
    the discrepancy ledger."""
    from src.wordlevel.align import is_content
    maps, sets, src_ref = {}, [], None
    for lg in langs:
        links, src, tgt = load_alignment(lg)
        src_ref = src
        m = {}
        for si, ls in enumerate(links):
            for i, j in ls:
                m[(si, i)] = j
        maps[lg] = (m, src, tgt)
        sets.append(set(m))
    common = sorted(set.intersection(*sets))
    n_all = len(common)
    if content_only:
        common = [(si, i) for si, i in common
                  if si < len(src_ref) and i < len(src_ref[si])
                  and is_content(src_ref[si][i])]
    print(f'[concepts] {n_all} aligned in all languages, '
          f'{len(common)} retained after the content-word filter '
          f'({100 * len(common) / max(n_all, 1):.1f}%)', flush=True)
    return common, maps


def word_vectors(model, tok, sentences, want, layers, device, pooling='mean'):
    """Hidden states at requested word positions.

    `want` maps sentence index to the list of word indices needed. Returns
    {(sent, word): array[n_layers, D]}. The sentence is passed pre-tokenized
    so the tokenizer's word_ids give an exact word-to-subword map; no
    heuristic offset matching is involved."""
    import torch
    out = {}
    for si, widx in want.items():
        words = sentences[si]
        if not words or not widx:
            continue
        enc = tok(words, is_split_into_words=True, return_tensors='pt',
                  truncation=True, max_length=256)
        wids = enc.word_ids(0)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        sel = torch.stack([hs[li][0].float() for li in layers], 0)  # [nl, T, D]
        for w in widx:
            pos = [t for t, wi in enumerate(wids) if wi == w]
            if not pos:
                continue
            if pooling == 'first':
                v = sel[:, pos[0], :]
            elif pooling == 'last':
                v = sel[:, pos[-1], :]
            else:
                v = sel[:, pos, :].mean(1)
            out[(si, w)] = v.cpu().numpy()
    return out


def build_grid(langs, common, maps, model, tok, layers, device, pooling):
    """[L+1, N, n_layers, D] over the pivot plus each language."""
    cols = []
    for lg in [PIVOT] + langs:
        if lg == PIVOT:
            m = {c: c[1] for c in common}
            _, src, _ = maps[langs[0]]
            sents = src
        else:
            m, _, tgt = maps[lg]
            sents = tgt
        want = {}
        for (si, i) in common:
            want.setdefault(si, []).append(m[(si, i)] if lg != PIVOT else i)
        vecs = word_vectors(model, tok, sents, want, layers, device, pooling)
        col, missing = [], 0
        for (si, i) in common:
            key = (si, m[(si, i)] if lg != PIVOT else i)
            if key in vecs:
                col.append(vecs[key])
            else:
                col.append(None); missing += 1
        cols.append(col)
        print(f'    [{lg}] {len(common) - missing}/{len(common)} vectors',
              flush=True)
    keep = [k for k in range(len(common))
            if all(c[k] is not None for c in cols)]
    grid = np.stack([np.stack([c[k] for k in keep], 0) for c in cols], 0)
    return grid, [common[k] for k in keep]


def lfs_of(block):
    """Standard pipeline: per-dimension z-score over all (language, concept)
    rows, then the two-way decomposition."""
    L, N, D = block.shape
    flat = block.reshape(L * N, D)
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    return float(lfs((block - mu) / sd)['lfs'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B-Base')
    ap.add_argument('--operating_point', default='primary',
                    choices=list(POINTS))
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tag = args.model.split('/')[-1]
    langs = POINTS[args.operating_point]
    rec = ledger.open_run('wordlevel_stage1',
                          {'model': args.model, 'point': args.operating_point,
                           'languages': langs})
    try:
        grid_meta = json.load(open(os.path.join(
            ROOT, 'results', 'grid', tag, 'metrics.json')))
        curve = {int(k): v['lfs']['lfs']
                 for k, v in grid_meta['per_layer'].items()}
        dip = min(curve, key=curve.get)
        n_layers = grid_meta['n_layers']
        layers = [l for l in range(dip - 2, dip + 3) if 0 <= l < n_layers]
        print(f'[{tag}] dip L{dip}, layers {layers}, '
              f'{len(langs)} languages', flush=True)

        common, maps = shared_concepts(langs)
        print(f'[grid] {len(common)} shared concepts, '
              f'pure-noise value {df_floor(len(langs) + 1, len(common)):.4f}',
              flush=True)

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(
            args.model, dtype=torch.float16).to(dev).eval()

        res = {'model': args.model, 'operating_point': args.operating_point,
               'languages': langs, 'dip_layer': dip, 'layers': layers,
               'n_concepts_requested': len(common), 'pooling': {}}

        for pooling in ('mean', 'first', 'last'):
            print(f'  [pooling={pooling}]', flush=True)
            grid, kept = build_grid(langs, common, maps, model, tok, layers,
                                    dev, pooling)
            L, N = grid.shape[0], grid.shape[1]
            floor = df_floor(L, N)
            per_layer = {str(layers[li]): lfs_of(grid[:, :, li, :])
                         for li in range(len(layers))}
            entry = {'n_concepts': N, 'n_languages': L,
                     'pure_noise_value': floor, 'lfs_by_layer': per_layer}

            if pooling == 'mean':
                # G-WL.1 reconstruction: sentence vectors built by averaging
                # the SAME aligned content words, which is the matched
                # comparison the prereg specifies
                by_sent = {}
                for k, (si, _) in enumerate(kept):
                    by_sent.setdefault(si, []).append(k)
                sids = sorted(by_sent)
                sent_grid = np.stack(
                    [np.stack([grid[l, by_sent[s], :, :].mean(0)
                               for s in sids], 0) for l in range(L)], 0)
                entry['reconstruction'] = {
                    'n_sentences': len(sids),
                    'sentence_lfs_by_layer': {
                        str(layers[li]): lfs_of(sent_grid[:, :, li, :])
                        for li in range(len(layers))}}

                # G-WL.2 monolingual null: pivot words split into L
                # pseudo-languages, arbitrary concept pairing
                rng = np.random.default_rng(0)
                per = N // L
                nulls = []
                for li in range(len(layers)):
                    piv = grid[0, :, li, :]
                    if per >= 2:
                        idx = rng.permutation(N)[:per * L]
                        nulls.append(lfs_of(piv[idx].reshape(L, per, -1)))
                entry['monolingual_null'] = {
                    'mean': float(np.mean(nulls)) if nulls else None,
                    'expected_floor': df_floor(L, per) if per >= 2 else None}

                # G-WL.3 permutation null: shuffle each language's concept
                # assignment independently
                perm = []
                for li in range(len(layers)):
                    draws = []
                    for _ in range(50):
                        g = grid[:, :, li, :].copy()
                        for l in range(1, L):
                            g[l] = g[l][rng.permutation(N)]
                        draws.append(lfs_of(g))
                    perm.append(draws)
                flat = np.concatenate(perm)
                entry['permutation_null'] = {
                    'mean': float(flat.mean()),
                    'p95': float(np.percentile(flat, 95))}

            res['pooling'][pooling] = entry
            print(f'    N={N} L={L} floor={floor:.4f} | ' +
                  ' '.join(f'L{k}:{v:.4f}' for k, v in per_layer.items()),
                  flush=True)

        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f'{tag}_{args.operating_point}.json')
        with open(p, 'w') as f:
            json.dump(res, f, indent=1)
        ledger.close_run(rec, 'done', {'out': p})
        print(f'[done] -> {p}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
