#!/usr/bin/env python3
"""§3.1.2 — corrected lens probe. See paper/PREREG_312_313.md (predictions A1-A4).

Per layer: unembed last-token hidden states (final norm -> output embedding),
classify top-1 token by Unicode script, measure input-script vs Latin mass;
plus lens-validity = mean max-cosine of states to output-embedding rows.
Non-Latin-script languages only. Overlays each model's LFS curve.
"""
import argparse, json, os, sys, unicodedata
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')

LANGS = {'rus': 'CYRILLIC', 'zho-CN': 'CJK', 'jpn': 'JPN', 'hin': 'DEVANAGARI',
         'ben': 'BENGALI', 'tam': 'TAMIL', 'tel': 'TELUGU', 'amh': 'ETHIOPIC',
         'khm': 'KHMER', 'kat': 'GEORGIAN', 'mya': 'MYANMAR'}


def script_of(tok):
    for ch in tok.lstrip('Ġ▁_ '):
        if ch.isalpha():
            try:
                name = unicodedata.name(ch)
            except ValueError:
                continue
            for s in ('CYRILLIC', 'DEVANAGARI', 'BENGALI', 'TAMIL', 'TELUGU',
                      'ETHIOPIC', 'KHMER', 'GEORGIAN', 'MYANMAR', 'LATIN'):
                if name.startswith(s):
                    return s
            if name.startswith(('CJK', 'HIRAGANA', 'KATAKANA')):
                return 'CJK' if name.startswith('CJK') else 'JPN'
            return 'OTHER'
    return 'NONE'


def match(script, target):
    if target == 'JPN':
        return script in ('JPN', 'CJK')
    return script == target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--n_sents', type=int, default=100)
    ap.add_argument('--langs', type=int, default=len(LANGS))
    ap.add_argument('--device', default=None)
    ap.add_argument('--pool', default='last', choices=['last', 'mean'])
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEV = args.device or ('cuda' if torch.cuda.is_available() else 'mps')
    tag = args.model.split('/')[-1]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, output_hidden_states=True).to(DEV).eval()
    if DEV == 'mps':
        W = model.get_output_embeddings().weight.detach().float().cpu()  # [V, D]
    else:
        # CUDA: reuse the model's fp16 storage (no copy) — fp32 copies OOM 11GB cards
        W = model.get_output_embeddings().weight.detach()
    Wn = torch.nn.functional.normalize(W, dim=-1)
    norm = model.model.norm  # final RMSNorm (Qwen3 / OLMo-2 arch)

    curves = {}
    for lg in list(LANGS)[:args.langs]:
        with open(os.path.join(NTREX, f'newstest2019-ref.{lg}.txt')) as f:
            sents = [l.strip() for l in f][:args.n_sents]
        frac_in, frac_lat, validity = None, None, None
        for i in range(0, len(sents), 16):
            enc = tok(sents[i:i + 16], return_tensors='pt', padding=True,
                      truncation=True, max_length=64).to(DEV)
            with torch.no_grad():
                hs = model(**enc).hidden_states
            last = enc['attention_mask'].sum(1) - 1
            idx = torch.arange(len(last))
            amask = enc['attention_mask'].unsqueeze(-1).to(hs[0].dtype)
            L = len(hs)
            fi = np.zeros(L); fl = np.zeros(L); va = np.zeros(L)
            for l in range(L):
                if args.pool == 'mean':
                    h = (hs[l] * amask).sum(1) / amask.sum(1)
                else:
                    h = hs[l][idx, last]
                with torch.no_grad():
                    if DEV == 'mps':
                        hh = norm(h.to(DEV)).float().cpu()
                        hv = h.float().cpu()
                    else:
                        hh = norm(h).half()
                        hv = h.half()
                    logits = hh @ W.T
                    top1 = logits.float().argmax(-1).cpu()
                    hn = torch.nn.functional.normalize(hv.float(), dim=-1)
                    va[l] = (hn.to(W.dtype) @ Wn.T).max(-1).values.float().mean().item()
                scripts = [script_of(tok.decode([t])) for t in top1]
                fi[l] = np.mean([match(s, LANGS[lg]) for s in scripts])
                fl[l] = np.mean([s == 'LATIN' for s in scripts])
            if frac_in is None:
                frac_in, frac_lat, validity, nb = fi, fl, va, 1
            else:
                frac_in += fi; frac_lat += fl; validity += va; nb += 1
        curves[lg] = {'input_script': (frac_in / nb).tolist(),
                      'latin': (frac_lat / nb).tolist(),
                      'validity': (validity / nb).tolist()}
        print(f'[lens] {tag} {lg}: mid-layer input-script mass '
              f'{np.mean(curves[lg]["input_script"][len(frac_in)//3:2*len(frac_in)//3]):.3f}',
              flush=True)

    out = os.path.join(ROOT, 'results', 'lens312')
    os.makedirs(out, exist_ok=True)
    # attach LFS curve + best layer from the existing grid for overlay
    gridp = os.path.join(ROOT, 'results', 'grid', tag, 'metrics.json')
    lfs = None
    if os.path.exists(gridp):
        g = json.load(open(gridp))
        lfs = [g['per_layer'][str(l)]['lfs']['lfs']
               for l in sorted(int(k) for k in g['per_layer'])]
    suffix = '_meanpool' if args.pool == 'mean' else ''
    json.dump({'model': args.model, 'pool': args.pool, 'curves': curves,
               'lfs': lfs},
              open(os.path.join(out, f'{tag}{suffix}.json'), 'w'), indent=1)
    print(f'[done] -> {out}/{tag}{suffix}.json')


if __name__ == '__main__':
    main()
