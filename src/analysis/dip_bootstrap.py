#!/usr/bin/env python3
"""E-A: bootstrap intervals on dip depth from the saved 1,500-sentence campaign dumps,
plus the parity canary against results/grid and the per-language contribution
to SS_lang at the dip layer (E-D). Protocol: prereg/addenda/AIM1_EABC_PROTOCOL_2026-09-05.md.

Usage: python src/analysis/dip_bootstrap.py --model bloom-1b7 [--n_boot 2000] [--n_sub 300]
Reads  results/dumps/<model>/{meta.json, layer000.npz, layer<dip>.npz}
Writes results/dip_bootstrap/<model>/{bootstrap.json, manifest.json}
Never writes into results/dumps or results/grid.
"""
import argparse, hashlib, json, os, platform, socket, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20260905


def lfs_components(X):
    """X: (L, N, D) float32. Joint z-score, then the direct-SS decomposition."""
    L, N, D = X.shape
    flat = X.reshape(L * N, D)
    mu_d = flat.mean(0)
    sd_d = flat.std(0) + 1e-9
    Z = (X - mu_d) / sd_d
    mu = Z.mean((0, 1))
    mu_l = Z.mean(1)
    mu_c = Z.mean(0)
    per_lang = N * ((mu_l - mu) ** 2).sum(1)          # per-language contribution
    ss_lang = float(per_lang.sum())
    ss_con = float(L * ((mu_c - mu) ** 2).sum())
    ss_res = float(((Z - mu_l[:, None] - mu_c[None] + mu) ** 2).sum())
    return {"lfs": ss_lang / (ss_lang + ss_con), "ss_lang": ss_lang, "ss_con": ss_con,
            "ss_res": ss_res, "per_lang_share": (per_lang / per_lang.sum()).tolist()}


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_sub", type=int, default=300)
    ap.add_argument("--dumps", default=os.path.join(ROOT, "results", "dumps"))
    ap.add_argument("--grid", default=os.path.join(ROOT, "results", "grid"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "dip_bootstrap"))
    args = ap.parse_args()
    t0 = time.time()

    ddir = os.path.join(args.dumps, args.model)
    meta = json.load(open(os.path.join(ddir, "meta.json")))
    dip = int(meta["dip_layer"])
    f0, fd = os.path.join(ddir, "layer000.npz"), os.path.join(ddir, f"layer{dip:03d}.npz")
    z0, zd = np.load(f0), np.load(fd)
    X0, Xd = z0["X"], zd["X"]
    langs = [str(x) for x in z0["langs"]]
    assert X0.shape == Xd.shape and X0.shape[0] == len(langs), (X0.shape, Xd.shape, len(langs))
    L, N, D = X0.shape
    print(f"[load] {args.model}: L={L} N={N} D={D} dip={dip} storage={X0.dtype}", flush=True)
    assert np.isfinite(X0.astype(np.float32)).all() and np.isfinite(Xd.astype(np.float32)).all()

    # --- parity canary: first n_sub sentences vs the headline grid ---
    grid_path = os.path.join(args.grid, args.model, "metrics.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else None
    first = slice(0, args.n_sub)
    c0 = lfs_components(X0[:, first, :].astype(np.float32))
    cd = lfs_components(Xd[:, first, :].astype(np.float32))
    parity = {"lfs0_dump_first300": c0["lfs"], "lfs_dip_dump_first300": cd["lfs"]}
    if grid is not None:
        g0 = grid["per_layer"]["0"]["lfs"]["lfs"]
        gd = grid["per_layer"][str(dip)]["lfs"]["lfs"]
        parity.update({"lfs0_grid": g0, "lfs_dip_grid": gd,
                       "abs_diff_0": abs(c0["lfs"] - g0), "abs_diff_dip": abs(cd["lfs"] - gd),
                       "grid_n_sents": grid["n_sents"], "grid_langs_match": grid["langs"] == langs})
    print("[parity]", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in parity.items()}, flush=True)

    # --- bootstrap: same index sets for every model via fixed seed ---
    rng = np.random.default_rng(SEED)
    idx_sets = [np.sort(rng.choice(N, args.n_sub, replace=False)) for _ in range(args.n_boot)]
    X0f, Xdf = X0.astype(np.float32), Xd.astype(np.float32)
    lfs0, lfsd = np.empty(args.n_boot), np.empty(args.n_boot)
    for b, idx in enumerate(idx_sets):
        lfs0[b] = lfs_components(X0f[:, idx, :])["lfs"]
        lfsd[b] = lfs_components(Xdf[:, idx, :])["lfs"]
        if b % 200 == 0:
            print(f"[boot] {b}/{args.n_boot}  lfs0={lfs0[b]:.4f} lfsdip={lfsd[b]:.4f}  {time.time()-t0:.0f}s", flush=True)
    depth = lfs0 - lfsd
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.median(a)), float(np.percentile(a, 97.5))]

    # --- E-D: per-language contribution at the dip on all 1500 ---
    full = lfs_components(Xdf)
    per_lang = dict(zip(langs, full["per_lang_share"]))

    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)
    result = {"model": args.model, "dip_layer": dip, "L": L, "N": N, "D": D, "n_sub": args.n_sub,
              "n_boot": args.n_boot, "seed": SEED, "parity": parity,
              "lfs0": {"q025_median_q975": q(lfs0)}, "lfs_dip": {"q025_median_q975": q(lfsd)},
              "dip_depth": {"q025_median_q975": q(depth), "half_width": (q(depth)[2] - q(depth)[0]) / 2},
              "replicates": {"lfs0": lfs0.tolist(), "lfs_dip": lfsd.tolist()},
              "full1500_dip": {k: full[k] for k in ("lfs", "ss_lang", "ss_con", "ss_res")},
              "per_lang_share_dip_full1500": per_lang}
    json.dump(result, open(os.path.join(out_dir, "bootstrap.json"), "w"), indent=1)
    manifest = {"script": os.path.relpath(__file__, ROOT), "script_sha256": sha256_file(__file__),
                "inputs": {os.path.relpath(f0, ROOT): sha256_file(f0), os.path.relpath(fd, ROOT): sha256_file(fd)},
                "grid_reference": os.path.relpath(grid_path, ROOT) if grid is not None else None,
                "host": socket.gethostname(), "python": platform.python_version(), "numpy": np.__version__,
                "wall_seconds": time.time() - t0, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[done] depth median {q(depth)[1]:.4f} CI [{q(depth)[0]:.4f}, {q(depth)[2]:.4f}] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
