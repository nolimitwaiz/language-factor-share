#!/usr/bin/env python3
"""§3.1.3 ROUND 2 — see PREREG_312_313.md (B2', B4', B5).

Upgrades over transfer_313.py (kept frozen):
  1. ~40 context languages (power for B2').
  2. MISMATCHED-context control: same language, wrong document -> separates
     content transfer from any-text/entity leakage.
     content benefit = delta(matched) - delta(mismatched);
     R_content_L = content_L / content_eng.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX_DIR = os.path.join(ROOT, 'data', 'NTREX')
NTREX = os.path.join(NTREX_DIR, 'NTREX-128')
CTX_LANGS = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'swe', 'pol',
             'ces', 'ron', 'rus', 'ukr', 'ell', 'tur', 'vie', 'ind', 'zho-CN',
             'jpn', 'kor', 'arb', 'heb', 'fas', 'hin', 'ben', 'tam', 'tel',
             'mar', 'urd', 'tha', 'khm', 'mya', 'swa', 'amh', 'som', 'hau',
             'zul', 'kat', 'hye', 'aze-Latn', 'uzb']


def load_pairs(n_pairs):
    with open(os.path.join(NTREX, 'newstest2019-src.eng.txt')) as f:
        eng = [l.strip() for l in f]
    docs = None
    docp = os.path.join(NTREX_DIR, 'DOCUMENT_IDS.tsv')
    if os.path.exists(docp):
        with open(docp) as f:
            docs = [l.strip().split('\t')[0] for l in f]
    pairs = []
    for i in range(len(eng) - 1):
        if docs and (i + 1 >= len(docs) or docs[i] != docs[i + 1]):
            continue
        if 8 <= len(eng[i + 1].split()) <= 45 and len(eng[i].split()) >= 5:
            pairs.append(i)
        if len(pairs) >= n_pairs:
            break
    # mismatched partner: context index from a pair half the list away
    # (deterministic derangement; different document by construction for
    #  half-list offsets in a ~2000-sentence multi-document corpus)
    mis = [pairs[(k + len(pairs) // 2) % len(pairs)] for k in range(len(pairs))]
    return eng, pairs, mis


def nll_of_target(model, tok, context, target, device):
    import torch
    tgt = tok(' ' + target, return_tensors='pt', add_special_tokens=False)
    if context:
        ctx = tok(context + '\n', return_tensors='pt')
        ids = torch.cat([ctx.input_ids, tgt.input_ids], 1)
        n_ctx = ctx.input_ids.shape[1]
    else:
        ids = torch.cat([torch.tensor([[tok.bos_token_id or tok.eos_token_id]]),
                         tgt.input_ids], 1)
        n_ctx = 1
    ids = ids.to(device)
    with torch.no_grad():
        logits = model(ids).logits.float()
    logp = torch.log_softmax(logits[0, :-1], -1)
    tgt_ids = ids[0, 1:]
    return float(-(logp[torch.arange(len(tgt_ids)), tgt_ids][n_ctx - 1:]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--n_pairs', type=int, default=100)
    ap.add_argument('--langs', type=int, default=len(CTX_LANGS))
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEV = args.device or ('cuda' if torch.cuda.is_available() else 'mps')
    tag = args.model.split('/')[-1]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16).to(DEV).eval()

    eng, pairs, mis = load_pairs(args.n_pairs)
    print(f'[data] {len(pairs)} pairs, mismatched partners assigned')
    texts = {'eng': eng}
    langs = []
    for lg in CTX_LANGS[:args.langs]:
        p = os.path.join(NTREX, f'newstest2019-ref.{lg}.txt')
        if lg == 'eng':
            langs.append(lg); continue
        if os.path.exists(p):
            with open(p) as f:
                texts[lg] = [l.strip() for l in f]
            langs.append(lg)
        else:
            print(f'[skip] {lg}: no NTREX file')

    base = np.array([nll_of_target(model, tok, None, eng[i + 1], DEV)
                     for i in pairs])
    res = {}
    for lg in langs:
        nm = np.array([nll_of_target(model, tok, texts[lg][i], eng[i + 1], DEV)
                       for i in pairs])
        nx = np.array([nll_of_target(model, tok, texts[lg][j], eng[i + 1], DEV)
                       for i, j in zip(pairs, mis)])
        d_m = float(base.mean() - nm.mean())
        d_x = float(base.mean() - nx.mean())
        res[lg] = {'delta_matched': d_m, 'delta_mismatched': d_x,
                   'content': d_m - d_x}
        print(f'[ctx] {tag} {lg}: matched={d_m:+.4f} mismatched={d_x:+.4f} '
              f'content={d_m - d_x:+.4f}', flush=True)
    c_eng = res['eng']['content']
    for lg in res:
        res[lg]['R_content'] = (res[lg]['content'] / c_eng
                                if c_eng > 0 else float('nan'))

    gridp = os.path.join(ROOT, 'results', 'grid', tag, 'metrics.json')
    corr = None
    if os.path.exists(gridp):
        g = json.load(open(gridp))
        L = g['per_layer'][str(g['best_layer'])]['langs']
        xs, ys = zip(*[(L[lg]['mexa'], res[lg]['R_content'])
                       for lg in res if lg != 'eng' and lg in L])
        from scipy import stats
        sp = stats.spearmanr(xs, ys)
        corr = {'spearman_Rcontent_vs_mexa': float(sp.statistic),
                'p': float(sp.pvalue), 'n_langs': len(xs)}
        print(f"[B2' test] Spearman(R_content, MEXA) = {sp.statistic:.3f} "
              f"(p={sp.pvalue:.4f}, n={len(xs)})")

    out = os.path.join(ROOT, 'results', 'transfer313')
    os.makedirs(out, exist_ok=True)
    json.dump({'model': args.model, 'base_nll': float(base.mean()),
               'results': res, 'B2prime': corr},
              open(os.path.join(out, f'{tag}_v2.json'), 'w'), indent=1)
    print(f'[done] -> {out}/{tag}_v2.json')


if __name__ == '__main__':
    main()
