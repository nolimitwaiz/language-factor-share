#!/usr/bin/env python3
"""W0: offline pre-test of additive offset and per-language scale removal on the
saved 1,500-sentence campaign dumps. Zero GPU. Protocol (frozen before the run):
prereg/addenda/AIM2_W0_OFFSET_OFFLINE_PROTOCOL_2026-09-07.md.

Usage: python src/aim2/offset_offline.py --model Qwen3-1.7B-Base [--n_boot 300]
Reads  results/dumps/<model>/{meta.json, layer*.npz}, results/grid/<model>/metrics.json
Writes results/aim2/offset_offline/<model>/{readings.json, manifest.json, offsets_layer<dip>.npz}
Never writes into results/dumps or results/grid.

Readings per layer and condition follow code/pilot_metrics.py (mexa_score, aar,
copied verbatim below) and src/analysis/dip_bootstrap.py (lfs_components).
"""
import argparse, glob, hashlib, json, os, platform, socket, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))
from dip_bootstrap import lfs_components  # joint z-score, direct SS

SEED = 20260907
EVAL = slice(0, 300)          # evaluation sentences (the headline grid)
CALIB = slice(300, 1500)      # calibration sentences for the offsets
CALIB300 = slice(300, 600)    # small calibration set for P5


# ---- copied verbatim from code/pilot_metrics.py ----
def _norm(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def mexa_score(E_en, E_l):
    """Fraction of sentences whose parallel pair strictly dominates row & col."""
    S = _norm(E_en) @ _norm(E_l).T
    n = len(S)
    d = np.diag(S)
    row_ok = d > (np.where(np.eye(n, dtype=bool), -np.inf, S)).max(1)
    col_ok = d > (np.where(np.eye(n, dtype=bool), -np.inf, S)).max(0)
    return float((row_ok & col_ok).mean())


def aar(E_en, E_l, q=0.10):
    """Alignment-at-Risk: mean retrieval margin of the worst q-quantile."""
    S = _norm(E_l) @ _norm(E_en).T
    n = len(S)
    d = np.diag(S).copy()
    off = np.where(np.eye(n, dtype=bool), -np.inf, S).max(1)
    margins = d - off
    k = max(1, int(np.floor(q * n)))
    worst = np.sort(margins)[:k]
    return float(worst.mean()), float(margins.mean())
# ----------------------------------------------------


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fit_offsets(Xc):
    """Xc: (L, Ncal, D) float32 calibration block. Returns mu_l, m, b_l, s_l."""
    mu_l = Xc.mean(1)                                   # (L, D)
    m = mu_l.mean(0)                                    # (D,)
    b_l = mu_l - m                                      # (L, D)
    rms_l = np.sqrt(((Xc - mu_l[:, None, :]) ** 2).mean((1, 2)))   # (L,)
    s_l = rms_l / rms_l.mean()
    return mu_l, m, b_l, s_l


def raw_concept_var(X):
    L, N, D = X.shape
    mu = X.mean((0, 1)); mu_c = X.mean(0)
    return float(L * ((mu_c - mu) ** 2).sum())


def effective_rank(X):
    flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
    flat = flat - flat.mean(0)
    cov = flat.T @ flat / (flat.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov); ev = np.clip(ev, 0, None)
    p = ev / ev.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def readings(X, langs, eng_idx):
    """All readings for one (L, 300, D) float32 block."""
    comp = lfs_components(X)
    E_en = X[eng_idx]
    mexa, a1, a5, a10, a20 = [], [], [], [], []
    for l in range(len(langs)):
        if l == eng_idx:
            continue
        E_l = X[l]
        mexa.append(mexa_score(E_en, E_l))
        a1.append(aar(E_en, E_l, 0.01)[0]); a5.append(aar(E_en, E_l, 0.05)[0])
        a10.append(aar(E_en, E_l, 0.10)[0]); a20.append(aar(E_en, E_l, 0.20)[0])
    return {"lfs": comp["lfs"], "ss_lang": comp["ss_lang"], "ss_con": comp["ss_con"], "ss_res": comp["ss_res"],
            "ss_res_share": comp["ss_res"] / (comp["ss_lang"] + comp["ss_con"] + comp["ss_res"]),
            "raw_concept_var": raw_concept_var(X), "mean_norm": float(np.linalg.norm(X, axis=2).mean()),
            "effective_rank": effective_rank(X),
            "mexa_mean": float(np.mean(mexa)), "mexa_per_lang": [float(v) for v in mexa],
            "aar1_mean": float(np.mean(a1)), "aar5_mean": float(np.mean(a5)),
            "aar10_mean": float(np.mean(a10)), "aar20_mean": float(np.mean(a20)),
            "aar10_per_lang": [float(v) for v in a10]}


def conditions(Xe, mu_l, m, b_l, s_l, eng_idx, rng_seeds):
    out = {"I0": Xe}
    out["I1"] = Xe - b_l[:, None, :]
    out["I2"] = m[None, None, :] + (Xe - mu_l[:, None, :]) / s_l[:, None, None]
    for k, seed in enumerate(rng_seeds):
        rng = np.random.default_rng(seed)
        R = rng.normal(size=b_l.shape).astype(np.float32)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        R *= np.linalg.norm(b_l, axis=1, keepdims=True)
        out[f"I3_s{k}"] = Xe - R[:, None, :]
    out["I5"] = Xe - b_l[:, None, :] + b_l[eng_idx][None, None, :]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dumps", default=os.path.join(ROOT, "results", "dumps"))
    ap.add_argument("--grid", default=os.path.join(ROOT, "results", "grid"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "aim2", "offset_offline"))
    ap.add_argument("--n_boot", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()
    ddir = os.path.join(args.dumps, args.model)
    meta = json.load(open(os.path.join(ddir, "meta.json"))); dip = int(meta["dip_layer"])
    layer_files = sorted(glob.glob(os.path.join(ddir, "layer*.npz")))
    layers = [int(os.path.basename(f)[5:8]) for f in layer_files]
    out_dir = os.path.join(args.out, args.model); os.makedirs(out_dir, exist_ok=True)
    grid_path = os.path.join(args.grid, args.model, "metrics.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else None
    result = {"model": args.model, "dip_layer": dip, "layers": layers, "protocol": "AIM2_W0_OFFSET_OFFLINE_PROTOCOL_2026-09-07", "per_layer": {}}
    hashes = {}
    for f, layer in zip(layer_files, layers):
        z = np.load(f); X = z["X"]; langs = [str(x) for x in z["langs"]]
        assert X.shape[1] >= 1500, X.shape
        eng_idx = langs.index("eng")
        Xe = X[:, EVAL, :].astype(np.float32)
        Xc = X[:, CALIB, :].astype(np.float32)
        mu_l, m, b_l, s_l = fit_offsets(Xc)
        del Xc
        conds = conditions(Xe, mu_l, m, b_l, s_l, eng_idx, [SEED, SEED + 1, SEED + 2])
        rec = {"n_langs": len(langs), "D": int(X.shape[2]), "conditions": {}}
        for name, Xk in conds.items():
            rec["conditions"][name] = readings(Xk, langs, eng_idx)
            print(f"[{args.model}] layer {layer} {name}: lfs={rec['conditions'][name]['lfs']:.4f} mexa={rec['conditions'][name]['mexa_mean']:.4f} rawCV={rec['conditions'][name]['raw_concept_var']:.1f}", flush=True)
        # P5: offsets from 300 calibration sentences
        Xc3 = X[:, CALIB300, :].astype(np.float32)
        mu3, m3, b3, s3 = fit_offsets(Xc3); del Xc3
        comp3 = lfs_components(Xe - b3[:, None, :])
        rec["I1_calib300_lfs"] = comp3["lfs"]
        rec["offset_norms"] = {"b_l_norm_mean": float(np.linalg.norm(b_l, axis=1).mean()), "s_l_min": float(s_l.min()), "s_l_max": float(s_l.max())}
        if layer == 0 and grid is not None:
            rec["parity_grid_layer0"] = {"grid": grid["per_layer"]["0"]["lfs"]["lfs"], "here": rec["conditions"]["I0"]["lfs"]}
        if layer == dip:
            if grid is not None:
                rec["parity_grid_dip"] = {"grid": grid["per_layer"][str(dip)]["lfs"]["lfs"], "here": rec["conditions"]["I0"]["lfs"]}
            # sentence bootstrap of the I1 - I0 LFS drop at the dip
            rng = np.random.default_rng(SEED); drops = []
            XI1 = conds["I1"]
            for b in range(args.n_boot):
                idx = rng.integers(0, Xe.shape[1], Xe.shape[1])
                drops.append(lfs_components(Xe[:, idx, :])["lfs"] - lfs_components(XI1[:, idx, :])["lfs"])
            q = np.quantile(drops, [0.025, 0.5, 0.975])
            rec["dip_lfs_drop_bootstrap"] = {"n_boot": args.n_boot, "q025_median_q975": [float(v) for v in q]}
            np.savez(os.path.join(out_dir, f"offsets_layer{dip:03d}.npz"), mu_l=mu_l, m=m, b_l=b_l, s_l=s_l, langs=np.array(langs),
                     calib_index=np.arange(CALIB.start, CALIB.stop), dump_sha256=sha256_file(f))
        hashes[os.path.basename(f)] = sha256_file(f)
        result["per_layer"][str(layer)] = rec
        del X, Xe, conds
    json.dump(result, open(os.path.join(out_dir, "readings.json"), "w"), indent=1)
    manifest = {"code": os.path.relpath(__file__, ROOT), "code_sha256": sha256_file(__file__), "inputs": hashes,
                "grid_metrics": grid_path if grid is not None else None, "seed": SEED, "n_boot": args.n_boot,
                "host": socket.gethostname(), "platform": platform.platform(), "numpy": np.__version__,
                "wall_s": round(time.time() - t0, 1), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[done] {args.model} in {manifest['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
