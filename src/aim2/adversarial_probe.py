#!/usr/bin/env python3
"""Does LFS predict what an adversarial discriminator can exploit?
Proposal section 3.2.3, premise test.

WHY THIS IS THE STRONGEST REMAINING CASE FOR THE METRIC. Section 3.2.3 says:

    "The most obvious choice for the discriminator is for it to predict the
     language identity from intermediate representations. By treating the
     discriminator's ability to correctly predict the language as a loss term
     during training, we force the model to learn representations that are
     harder to distinguish between languages."

A discriminator can only succeed to the extent that language-specific
variance exists in the representation. That variance is `V_language`, the
numerator of LFS. So the two quantities measure the same thing by
construction, and LFS should predict, per layer and before any adversarial
training is built, how much headroom the adversary has.

This is a real test rather than a restatement, because the two measure it
differently:

  * LFS is **additive and linear**. It decomposes the representation into a
    language mean plus a concept mean plus a residual, and reports a variance
    share.
  * A discriminator is **nonparametric**. An MLP probe can exploit language
    information that is nonlinear, interaction-based, or rotational, none of
    which LFS counts, because they land in the residual.

So the comparison has three possible outcomes and all three are informative:

  1. Probe accuracy tracks the LFS curve -> LFS predicts adversarial headroom,
     which is a direct and previously untested use of the Aim 1 metric in an
     Aim 2 method.
  2. A **linear** probe tracks LFS but an **MLP** probe finds much more ->
     LFS's additive construct boundary is real and quantified: the gap is
     exactly the language information LFS cannot see.
  3. Neither tracks it -> LFS is not measuring what the adversarial method
     targets, and section 3.2.3 should not be built on it.

Outcome 2 is the most likely and the most useful, because the gap between the
linear and MLP probes **measures the construct boundary in the units that
matter for Aim 2** rather than on synthetic perturbations.

No adversarial training is run here. This tests the premise the adversarial
arm would rest on, before that arm is built, which is the cheap ordering.

Usage:
    python -m src.aim2.adversarial_probe --model Qwen/Qwen3-0.6B-Base
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from pilot_metrics import lfs  # noqa: E402
from cluster_grid import load_all_ntrex  # noqa: E402
from src.common import ledger  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'adversarial_probe')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--n_sents', type=int, default=300)
    ap.add_argument('--n_langs', type=int, default=24)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = args.model.split('/')[-1]
    rec = ledger.open_run('adversarial_probe', vars(args), seed=args.seed)
    try:
        data = load_all_ntrex(args.n_sents)
        langs = sorted(data)[:args.n_langs]
        print(f'[data] {len(langs)} languages, {args.n_sents} sentences',
              flush=True)

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(
            args.model, dtype=torch.float16,
            output_hidden_states=True).to(dev).eval()

        emb = {}
        for lg in langs:
            chunks = []
            for i in range(0, len(data[lg]), args.batch_size):
                enc = tok(data[lg][i:i + args.batch_size], return_tensors='pt',
                          padding=True, truncation=True, max_length=128).to(dev)
                with torch.no_grad():
                    hs = model(**enc).hidden_states
                m = enc['attention_mask'].unsqueeze(-1).float()
                pooled = [(h.float() * m).sum(1) / m.sum(1) for h in hs]
                chunks.append(torch.stack(pooled, 1).float().cpu().numpy())
            emb[lg] = np.concatenate(chunks, 0)
        n_layers = emb[langs[0]].shape[1]
        print(f'[model] {tag}, {n_layers} layers embedded', flush=True)

        # Split by SENTENCE INDEX, not at random. Sentences are parallel
        # across languages, so a random split would put the same concept in
        # both train and test in different languages and let the probe
        # memorize concepts instead of learning language identity.
        n = emb[langs[0]].shape[0]
        tr_i, te_i = train_test_split(np.arange(n), test_size=0.3,
                                      random_state=args.seed)
        print(f'[split] by sentence index, {len(tr_i)} train / {len(te_i)} '
              f'test, disjoint concepts', flush=True)

        rows = []
        for layer in range(n_layers):
            E = {lg: emb[lg][:, layer, :].astype(np.float32) for lg in langs}
            allE = np.concatenate([E[lg] for lg in langs], 0)
            mu, sd = allE.mean(0), allE.std(0) + 1e-9
            Ez = {lg: (E[lg] - mu) / sd for lg in langs}
            X = np.stack([Ez[lg] for lg in langs], 0)
            d = lfs(X)

            Xtr = np.concatenate([Ez[lg][tr_i] for lg in langs], 0)
            ytr = np.concatenate([[i] * len(tr_i) for i in range(len(langs))])
            Xte = np.concatenate([Ez[lg][te_i] for lg in langs], 0)
            yte = np.concatenate([[i] * len(te_i) for i in range(len(langs))])

            # multi_class was removed in recent scikit-learn; multinomial is
            # the default for multiclass problems.
            lin = LogisticRegression(max_iter=1000,
                                     random_state=args.seed).fit(Xtr, ytr)
            mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                                random_state=args.seed).fit(Xtr, ytr)
            a_lin = float(lin.score(Xte, yte))
            a_mlp = float(mlp.score(Xte, yte))
            rows.append({'layer': layer, 'lfs': d['lfs'],
                         'var_lang': d['var_lang'],
                         'probe_linear': a_lin, 'probe_mlp': a_mlp,
                         'nonlinear_gap': a_mlp - a_lin})
            print(f'  L{layer:>2}: lfs {d["lfs"]:.4f}  linear {a_lin:.4f}  '
                  f'mlp {a_mlp:.4f}  gap {a_mlp - a_lin:+.4f}', flush=True)

        from scipy.stats import pearsonr, spearmanr
        L = np.array([r['lfs'] for r in rows])
        PL = np.array([r['probe_linear'] for r in rows])
        PM = np.array([r['probe_mlp'] for r in rows])
        chance = 1.0 / len(langs)
        summary = {
            'chance': chance,
            'lfs_vs_linear_pearson': float(pearsonr(L, PL).statistic),
            'lfs_vs_linear_spearman': float(spearmanr(L, PL).statistic),
            'lfs_vs_mlp_pearson': float(pearsonr(L, PM).statistic),
            'lfs_vs_mlp_spearman': float(spearmanr(L, PM).statistic),
            'mean_nonlinear_gap': float(np.mean(PM - PL)),
            'max_nonlinear_gap': float(np.max(PM - PL)),
            'lfs_dip_layer': int(np.argmin(L)),
            'linear_min_layer': int(np.argmin(PL)),
            'mlp_min_layer': int(np.argmin(PM))}
        print('\n=== does LFS predict adversarial headroom? ===')
        print(f'  chance accuracy: {chance:.4f}')
        print(f'  LFS vs linear probe: Pearson '
              f'{summary["lfs_vs_linear_pearson"]:+.3f}, Spearman '
              f'{summary["lfs_vs_linear_spearman"]:+.3f}')
        print(f'  LFS vs MLP probe   : Pearson '
              f'{summary["lfs_vs_mlp_pearson"]:+.3f}, Spearman '
              f'{summary["lfs_vs_mlp_spearman"]:+.3f}')
        print(f'  mean nonlinear gap : {summary["mean_nonlinear_gap"]:+.4f} '
              f'(language info LFS cannot see)')
        print(f'  LFS dip at L{summary["lfs_dip_layer"]}, linear probe minimum '
              f'at L{summary["linear_min_layer"]}, MLP minimum at '
              f'L{summary["mlp_min_layer"]}')

        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f'{tag}.json')
        with open(p, 'w') as f:
            json.dump({'model': args.model, 'languages': langs,
                       'n_sents': args.n_sents, 'per_layer': rows,
                       'summary': summary}, f, indent=1)
        ledger.close_run(rec, 'done', {'out': p})
        print(f'[done] -> {p}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
