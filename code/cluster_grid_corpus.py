#!/usr/bin/env python3
"""Grid run with a configurable parallel corpus and output root (E-B FLORES replication,
E-C instruct-versus-base). The estimator code is imported unchanged from pilot_metrics
and the per-layer loop is identical to cluster_grid.py. Additions: --data_dir/--src_file/
--ref_prefix, --out_root, and a manifest with snapshot hash and input hashes.
Protocol: prereg/addenda/AIM1_EABC_PROTOCOL_2026-09-05.md
"""
import argparse, hashlib, json, os, platform, socket, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_metrics import embed_all, mexa_score, aar, lfs  # noqa: E402
from cluster_grid import TIER, hub_with_controls  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_parallel(data_dir, src_file, ref_prefix, n_sents, max_langs=None):
    with open(os.path.join(data_dir, src_file)) as f:
        eng = [l.strip() for l in f][:n_sents]
    data, used = {"eng": eng}, [src_file]
    files = sorted(f for f in os.listdir(data_dir) if f.startswith(ref_prefix) and f.endswith(".txt"))
    for fn in files:
        code = fn[len(ref_prefix):-len(".txt")]
        if code == "eng":
            continue
        with open(os.path.join(data_dir, fn)) as f:
            sents = [l.strip() for l in f][:n_sents]
        if len(sents) == len(eng) and all(sents):
            data[code] = sents
            used.append(fn)
        if max_langs and len(data) - 1 >= max_langs:
            break
    print(f"[data] {len(data) - 1} languages + eng pivot, {len(eng)} sents from {data_dir}", flush=True)
    return data, used


def sha256_text(lines):
    h = hashlib.sha256()
    for l in lines:
        h.update(l.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


def snapshot_hash(model_id):
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    ref = os.path.join(cache, "hub", "models--" + model_id.replace("/", "--"), "refs", "main")
    try:
        return open(ref).read().strip()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "NTREX", "NTREX-128"))
    ap.add_argument("--src_file", default="newstest2019-src.eng.txt")
    ap.add_argument("--ref_prefix", default="newstest2019-ref.")
    ap.add_argument("--out_root", required=True, help="e.g. results/flores_grid (never results/grid)")
    ap.add_argument("--n_sents", type=int, default=300)
    ap.add_argument("--max_langs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args()
    assert os.path.normpath(args.out_root) != os.path.normpath(os.path.join(ROOT, "results", "grid")), "refusing to write into results/grid"
    t0 = time.time()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.model.split("/")[-1]
    out_dir = os.path.join(ROOT, args.out_root, tag) if not os.path.isabs(args.out_root) else os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run] {args.model} on {args.device} dtype={args.dtype} -> {out_dir}", flush=True)

    data, used_files = load_parallel(args.data_dir, args.src_file, args.ref_prefix, args.n_sents, args.max_langs)
    embs = embed_all(data, args.model, args.device, batch_size=args.batch_size, dtype=args.dtype)
    langs = list(embs.keys())
    n_layers = embs["eng"].shape[1]
    results = {"model": args.model, "corpus": os.path.relpath(args.data_dir, ROOT), "n_sents": args.n_sents,
               "n_layers": n_layers, "langs": langs, "tiers": {lg: TIER.get(lg, "unk") for lg in langs}, "per_layer": {}}

    for layer in range(n_layers):
        E = {lg: embs[lg][:, layer, :].astype(np.float32) for lg in langs}
        allE = np.concatenate(list(E.values()), 0)
        assert np.isfinite(allE).all(), f"non-finite embeddings at layer {layer}"
        mu, sd = allE.mean(0), allE.std(0) + 1e-9
        Ez = {lg: (E[lg] - mu) / sd for lg in langs}
        X = np.stack([Ez[lg] for lg in langs], 0)
        lr = {"lfs": lfs(X), "langs": {}}
        for lg in langs:
            if lg == "eng":
                continue
            a10, amean = aar(Ez["eng"], Ez[lg])
            lr["langs"][lg] = {"mexa": mexa_score(Ez["eng"], Ez[lg]), "aar10": a10, "margin_mean": amean}
        results["per_layer"][layer] = lr
        print(f"[layer {layer:2d}] LFS={lr['lfs']['lfs']:.3f} meanMEXA={np.mean([v['mexa'] for v in lr['langs'].values()]):.3f}", flush=True)

    mean_mexa = {l: np.mean([v["mexa"] for v in r["langs"].values()]) for l, r in results["per_layer"].items()}
    best_layer = max(mean_mexa, key=mean_mexa.get)
    results["best_layer"] = int(best_layer)
    prof = [results["per_layer"][l]["lfs"]["lfs"] for l in range(n_layers)]
    results["lfs_profile"] = prof
    results["lfs_min_layer"] = int(np.argmin(prof))
    results["dip_depth"] = float(prof[0] - min(prof))
    E = {lg: embs[lg][:, best_layer, :].astype(np.float32) for lg in langs}
    allE = np.concatenate(list(E.values()), 0)
    mu, sd = allE.mean(0), allE.std(0) + 1e-9
    X = np.stack([(E[lg] - mu) / sd for lg in langs], 0)
    print(f"[hub] at best layer {best_layer} ...", flush=True)
    results["hub_at_best_layer"] = hub_with_controls(X, langs)

    json.dump(results, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    import torch, transformers
    manifest = {"model": args.model, "snapshot_hash": snapshot_hash(args.model), "corpus_dir": os.path.relpath(args.data_dir, ROOT),
                "src_file": args.src_file, "ref_prefix": args.ref_prefix, "n_sents": args.n_sents, "n_langs": len(langs) - 1,
                "input_sha256": {fn: sha256_text(data[fn[len(args.ref_prefix):-4] if fn != args.src_file else "eng"]) for fn in used_files},
                "dtype": args.dtype, "device": args.device, "batch_size": args.batch_size, "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
                "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                "estimator_sha256": hashlib.sha256(open(os.path.join(os.path.dirname(__file__), "pilot_metrics.py"), "rb").read()).hexdigest(),
                "wall_seconds": time.time() - t0, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[done] dip depth {results['dip_depth']:.3f} at layer {results['lfs_min_layer']} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
