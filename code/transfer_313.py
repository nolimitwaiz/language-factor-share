#!/usr/bin/env python3
"""§3.1.3 — cross-lingual context benefit. See paper/PREREG_312_313.md (B1-B4).

Adjacent same-document NTREX pairs (s1, s2); target s2 always ENGLISH; context
s1 varies by language. Benefit Delta_L = mean NLL(s2|nothing) - mean NLL(s2|s1_L);
ratio R_L = Delta_L / Delta_eng. Correlate R_L with the model's per-language
MEXA from the existing grid.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX_DIR = os.path.join(ROOT, 'data', 'NTREX')
NTREX = os.path.join(NTREX_DIR, 'NTREX-128')
CTX_LANGS = ['eng', 'deu', 'fra', 'spa', 'rus', 'zho-CN', 'hin', 'tur',
             'vie', 'ben', 'swa', 'amh', 'khm']


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
        w2 = len(eng[i + 1].split())
        if 8 <= w2 <= 45 and len(eng[i].split()) >= 5:
            pairs.append(i)
        if len(pairs) >= n_pairs:
            break
    return eng, pairs


def nll_of_target(model, tok, context, target, device):
    """Mean NLL over target tokens, context masked out."""
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
    tok_lp = logp[torch.arange(len(tgt_ids)), tgt_ids][n_ctx - 1:]
    return float(-tok_lp.mean())


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

    eng, pairs = load_pairs(args.n_pairs)
    print(f'[data] {len(pairs)} same-document pairs')
    texts = {'eng': eng}
    for lg in CTX_LANGS[1:args.langs]:
        with open(os.path.join(NTREX, f'newstest2019-ref.{lg}.txt')) as f:
            texts[lg] = [l.strip() for l in f]

    base = np.array([nll_of_target(model, tok, None, eng[i + 1], DEV) for i in pairs])
    res = {}
    for lg in CTX_LANGS[:args.langs]:
        nll = np.array([nll_of_target(model, tok, texts[lg][i], eng[i + 1], DEV)
                        for i in pairs])
        res[lg] = {'delta': float(base.mean() - nll.mean())}
        print(f'[ctx] {tag} {lg}: delta={res[lg]["delta"]:+.4f}', flush=True)
    d_eng = res['eng']['delta']
    for lg in res:
        res[lg]['R'] = res[lg]['delta'] / d_eng if d_eng > 0 else float('nan')

    # correlate R with per-language MEXA from the existing grid
    gridp = os.path.join(ROOT, 'results', 'grid', tag, 'metrics.json')
    corr = None
    if os.path.exists(gridp):
        g = json.load(open(gridp))
        L = g['per_layer'][str(g['best_layer'])]['langs']
        xs, ys = [], []
        for lg in res:
            if lg != 'eng' and lg in L:
                xs.append(L[lg]['mexa']); ys.append(res[lg]['R'])
        from scipy import stats
        corr = {'spearman_R_vs_mexa': float(stats.spearmanr(xs, ys).statistic),
                'n_langs': len(xs)}
        print(f'[B2 test] Spearman(R, MEXA) = {corr["spearman_R_vs_mexa"]:.3f} '
              f'(n={len(xs)})')

    out = os.path.join(ROOT, 'results', 'transfer313')
    os.makedirs(out, exist_ok=True)
    json.dump({'model': args.model, 'base_nll': float(base.mean()),
               'results': res, 'B2': corr},
              open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print(f'[done] -> {out}/{tag}.json')


if __name__ == '__main__':
    main()
