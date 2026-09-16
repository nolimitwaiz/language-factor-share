#!/usr/bin/env python3
"""AIM-2 FEASIBILITY PILOT (pre-registered): train Qwen3-0.6B against the
differentiable batch-LFS objective; lambda=0 arm = domain-adaptation control.

loss = LM_loss + lambda * LFS_batch(layer L*), on parallel NTREX batches
(train indices 500-1900, DISJOINT from eval 0-300).
"""
import argparse, os, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')
LANGS = ['deu', 'fra', 'spa', 'rus', 'zho-CN', 'hin', 'tur', 'vie', 'ben',
         'swa', 'amh', 'khm', 'kat', 'tam', 'tel', 'mya']


def load(code, lo=500, hi=1900):
    fn = 'newstest2019-src.eng.txt' if code == 'eng' else f'newstest2019-ref.{code}.txt'
    with open(os.path.join(NTREX, fn)) as f:
        return [l.strip() for l in f][lo:hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--lam', type=float, default=5.0)
    ap.add_argument('--layer', type=int, default=18)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed); random.seed(args.seed)

    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, output_hidden_states=True).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    texts = {c: load(c) for c in ['eng'] + LANGS}
    n = len(texts['eng'])

    def pooled(hs_layer, mask):
        m = mask.unsqueeze(-1).to(hs_layer.dtype)
        return (hs_layer * m).sum(1) / m.sum(1)

    for step in range(args.steps):
        langs = random.sample(LANGS, 3)
        idx = random.sample(range(n), 6)
        batch_texts, groups = [], []
        for c in ['eng'] + langs:
            batch_texts += [texts[c][i] for i in idx]
            groups.append(c)
        enc = tok(batch_texts, return_tensors='pt', padding=True,
                  truncation=True, max_length=64).to(dev)
        out = model(**enc, labels=enc['input_ids'])
        loss_lm = out.loss
        H = pooled(out.hidden_states[args.layer], enc['attention_mask'])
        X = H.view(len(groups), len(idx), -1).float()   # [L, B, D]
        mu = X.mean((0, 1)); mu_c = X.mean(0); mu_l = X.mean(1)
        ss_c = (len(groups) * (mu_c - mu).pow(2)).sum()
        ss_l = (len(idx) * (mu_l - mu).pow(2)).sum()
        lfs = ss_l / (ss_l + ss_c + 1e-8)
        loss = loss_lm + args.lam * lfs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 25 == 0:
            print(f'[{step:4d}] lm={loss_lm.item():.3f} lfs_batch={lfs.item():.3f}',
                  flush=True)

    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f'[saved] {args.out}')


if __name__ == '__main__':
    main()
