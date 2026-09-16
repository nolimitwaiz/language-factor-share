#!/usr/bin/env python3
"""What is actually IN the worst decile? Pull the worst-margin sentences for
mid-resource languages (Qwen3-0.6B, best layer) and dump their English sources
for manual reading: semantically-hard vs named-entity-heavy."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_metrics import embed_all, _norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')
LANGS = ['hin', 'tur', 'ben', 'vie']   # mid-resource, AaR ~ 0 at this scale
BEST_LAYER = 18                        # frozen rule output for Qwen3-0.6B
N = 300


def load(code):
    fn = 'newstest2019-src.eng.txt' if code == 'eng' else f'newstest2019-ref.{code}.txt'
    with open(os.path.join(NTREX, fn)) as f:
        return [l.strip() for l in f][:N]


def main():
    texts = {c: load(c) for c in ['eng'] + LANGS}
    embs = embed_all(texts, 'Qwen/Qwen3-0.6B-Base', 'mps', batch_size=24)
    E = {c: embs[c][:, BEST_LAYER, :] for c in texts}
    # z-score like the pipeline
    allE = np.concatenate(list(E.values()), 0)
    mu, sd = allE.mean(0), allE.std(0) + 1e-9
    Ez = {c: (E[c] - mu) / sd for c in E}

    out_path = os.path.join(ROOT, 'results', 'deflation', 'tail_autopsy.txt')
    with open(out_path, 'w') as f:
        for lg in LANGS:
            S = _norm(Ez[lg]) @ _norm(Ez['eng']).T
            d = np.diag(S).copy()
            off = np.where(np.eye(N, dtype=bool), -np.inf, S).max(1)
            margins = d - off
            worst = np.argsort(margins)[:30]
            best = np.argsort(margins)[-5:]
            f.write(f"\n{'='*80}\nWORST DECILE — {lg} (margins {margins[worst[0]]:.3f}"
                    f" .. {margins[worst[-1]]:.3f})\n{'='*80}\n")
            for i in worst:
                f.write(f"[{i:3d}] m={margins[i]:+.3f}  EN: {texts['eng'][i][:200]}\n")
            f.write(f"\n--- for contrast, 5 BEST-aligned ({lg}) ---\n")
            for i in best:
                f.write(f"[{i:3d}] m={margins[i]:+.3f}  EN: {texts['eng'][i][:160]}\n")
    print(f"[done] -> {out_path}")


if __name__ == '__main__':
    main()
