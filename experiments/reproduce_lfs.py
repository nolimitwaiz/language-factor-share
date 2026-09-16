#!/usr/bin/env python3
"""The deterministic end-to-end reproduction path.

One command runs the whole pipeline on a pinned small configuration and
writes a machine-readable result with full provenance:

    python experiments/reproduce_lfs.py \
        --config configs/reproduction/qwen3_0.6b_ntrex_small.yaml \
        --out results/reproduction/run_a.json

Two runs of the same configuration must agree to the tolerance in the
config file. Compare them with:

    python experiments/compare_runs.py run_a.json run_b.json

This reuses the existing metric implementations from code/pilot_metrics.py
without modification. It is a driver, not a reimplementation: if it and the
grid script ever disagree, the grid script is authoritative and the
divergence is a discrepancy to record.

NOTE ON SCOPE. The configuration is deliberately small and is for
verification only. Its grid shape puts the estimator's pure-noise value at
0.048 (section 5.3 of the report), so numbers from it are not comparable to
campaign numbers.
"""
import os

# CUBLAS determinism must be requested before torch initializes CUDA.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from pilot_metrics import aar, lfs, mexa_score  # noqa: E402
from src.common import ledger  # noqa: E402


def load_config(path):
    try:
        import yaml
    except ImportError:
        raise SystemExit('pyyaml required: pip install pyyaml')
    with open(path) as f:
        return yaml.safe_load(f)


def file_digest(path, n_lines):
    """Content hash of exactly the lines this run consumes, so a changed
    data file is detected rather than silently used."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            h.update(line)
    return h.hexdigest()[:16]


def load_sentences(cfg):
    root = os.path.join(ROOT, cfg['data']['root'])
    start, end = cfg['data']['sentence_indices']
    pivot = cfg['data']['pivot']
    data, digests = {}, {}
    for lang in cfg['data']['languages']:
        cands = [f'newstest2019-ref.{lang}.txt']
        if lang == pivot:
            cands.insert(0, f'newstest2019-src.{lang}.txt')
        for c in cands:
            p = os.path.join(root, c)
            if os.path.exists(p):
                break
        else:
            raise SystemExit(f'no NTREX file for language {lang}')
        with open(p) as f:
            lines = [l.strip() for l in f]
        if len(lines) < end:
            raise SystemExit(f'{lang}: {len(lines)} lines, need {end}')
        data[lang] = lines[start:end]
        digests[lang] = {'file': os.path.basename(p),
                         'sha256_16': file_digest(p, end)}
    return data, digests


def set_determinism(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def embed(cfg, data):
    """Pooled representations per language, every layer. Pooling is the
    masked fp32 mean, matching code/pilot_metrics.embed_all exactly; it is
    inlined here only so the sentence order and batch composition are
    explicit and fixed."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    mc = cfg['model']
    dt = {'fp16': torch.float16, 'bf16': torch.bfloat16,
          'fp32': torch.float32}[mc['dtype']]
    if 'bloom' in mc['name'].lower() and mc['dtype'] == 'fp16':
        raise SystemExit('fp16 forbidden for bf16-trained BLOOM models')

    tok = AutoTokenizer.from_pretrained(mc['name'], revision=mc['revision'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(mc['name'], revision=mc['revision'],
                                      dtype=dt,
                                      output_hidden_states=True)
    model = model.to(mc['device']).eval()

    bs = cfg['compute']['batch_size']
    ml = cfg['pooling']['max_length']
    out = {}
    for lang in cfg['data']['languages']:
        sents, chunks = data[lang], []
        for i in range(0, len(sents), bs):
            enc = tok(sents[i:i + bs], return_tensors='pt', padding=True,
                      truncation=True, max_length=ml).to(mc['device'])
            with torch.no_grad():
                hs = model(**enc).hidden_states
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = [(h.float() * mask).sum(1) / mask.sum(1) for h in hs]
            chunks.append(torch.stack(pooled, 1).float().cpu().numpy())
        out[lang] = np.concatenate(chunks, 0)          # [N, n_layers, D]
        print(f'  [embed] {lang}: {out[lang].shape}', flush=True)
    return out, model.config.num_hidden_layers


def compute_metrics(cfg, embs):
    langs = cfg['data']['languages']
    pivot = cfg['data']['pivot']
    n_layers = embs[pivot].shape[1]
    per_layer = {}
    for layer in range(n_layers):
        E = {l: embs[l][:, layer, :].astype(np.float32) for l in langs}
        allE = np.concatenate([E[l] for l in langs], 0)
        mu, sd = allE.mean(0), allE.std(0) + 1e-9
        Ez = {l: (E[l] - mu) / sd for l in langs}
        X = np.stack([Ez[l] for l in langs], 0)
        d = lfs(X)
        row = {'lfs': d['lfs'], 'var_lang': d['var_lang'],
               'var_concept': d['var_concept'],
               'residual_share': d['var_resid'], 'langs': {}}
        # CVP reference is the pivot's own concept spread at this layer,
        # so the quantity is well defined without a second model.
        piv_var = float(E[pivot].var(0).sum())
        for l in langs:
            if l == pivot:
                continue
            a10, amean = aar(Ez[pivot], Ez[l])
            row['langs'][l] = {
                'mexa': mexa_score(Ez[pivot], Ez[l]),
                'aar10': a10, 'margin_mean': amean,
                'cvp_vs_pivot': float(E[l].var(0).sum() / (piv_var + 1e-12))}
        per_layer[str(layer)] = row
    dip = min(per_layer, key=lambda k: per_layer[k]['lfs'])
    return per_layer, int(dip), n_layers


def provenance(cfg, cfg_path, digests, n_layers):
    import torch
    import transformers
    return {
        'config_file': os.path.relpath(cfg_path, ROOT),
        'config_sha256_16': file_digest(cfg_path, 10 ** 9),
        'config': cfg,
        'data_digests': digests,
        'n_layers': n_layers,
        'src_hash': ledger._src_hash(),
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'cuda': getattr(torch.version, 'cuda', None),
            'device_name': (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else 'cpu'),
        },
        'env': {'CUBLAS_WORKSPACE_CONFIG':
                os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        'host': platform.node(),
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--figure', action='store_true',
                    help='also write the layerwise LFS figure')
    args = ap.parse_args()

    cfg = load_config(args.config)
    t0 = time.time()
    rec = ledger.open_run('reproduce_lfs',
                          {'config': args.config, 'out': args.out},
                          seed=cfg['compute']['seed'])
    try:
        if cfg['compute'].get('deterministic', True):
            set_determinism(cfg['compute']['seed'])
        data, digests = load_sentences(cfg)
        print(f"[data] {len(data)} languages x "
              f"{cfg['data']['n_sents']} sentences", flush=True)
        embs, _ = embed(cfg, data)
        per_layer, dip, n_layers = compute_metrics(cfg, embs)

        result = {
            'name': cfg['name'],
            'purpose': cfg.get('purpose'),
            'pure_noise_lfs': cfg.get('pure_noise_lfs'),
            'dip_layer': dip,
            'dip_lfs': per_layer[str(dip)]['lfs'],
            'per_layer': per_layer,
            'wallclock_s': round(time.time() - t0, 1),
            'provenance': provenance(cfg, args.config, digests, n_layers),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=1, sort_keys=True)
        print(f"[result] dip layer {dip}, LFS {result['dip_lfs']:.6f} "
              f"(pure-noise value at this grid: {cfg.get('pure_noise_lfs')})")
        print(f'[done] -> {args.out}')

        if args.figure:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            xs = sorted(int(k) for k in per_layer)
            ys = [per_layer[str(k)]['lfs'] for k in xs]
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            ax.plot([x / max(xs) for x in xs], ys, lw=2.2, color='#0891A8')
            ax.axhline(cfg.get('pure_noise_lfs', 0), ls=':', color='#888',
                       lw=1.0, label='pure-noise value at this grid')
            ax.scatter([dip / max(xs)], [per_layer[str(dip)]['lfs']],
                       color='#B42318', zorder=3, s=36, label=f'dip (L{dip})')
            ax.set_xlabel('relative depth')
            ax.set_ylabel('Language-Factor Share')
            ax.set_title(f"{cfg['name']} (reproduction configuration)",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            figp = os.path.splitext(args.out)[0] + '_lfs.png'
            plt.savefig(figp, dpi=140)
            print(f'[fig] -> {figp}')

        ledger.close_run(rec, 'done', {'out': args.out, 'dip': dip})
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
