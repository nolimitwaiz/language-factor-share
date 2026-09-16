#!/usr/bin/env python3
"""Five-layer anisotropy threat audit and per-coordinate language shares."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import beta, spearmanr
from sklearn.utils.extmath import randomized_svd


RMFS_ROOT = Path.home() / "rmfs"
PRIOR_ROOT = Path.home() / "multilingual-metrics"
LANGUAGES = [
    "eng", "deu", "rus", "arb", "zho-CN", "tur", "vie",
    "hin", "ben", "swa", "amh", "zul", "khm",
]
EPS = 1e-12
FDR_ALPHA = 0.05
TOP_VARIANCE_FRACTION = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_grid(grid: np.ndarray) -> np.ndarray:
    x = np.asarray(grid, dtype=np.float64)
    mean = x.mean(axis=(0, 1), keepdims=True)
    scale = x.std(axis=(0, 1), keepdims=True)
    return (x - mean) / np.where(scale > EPS, scale, 1.0)


def coordinate_shares(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(grid, dtype=np.float64)
    n_lang, n_concept, _ = x.shape
    grand = x.mean(axis=(0, 1))
    language_mean = x.mean(axis=1)
    concept_mean = x.mean(axis=0)
    ss_language = n_concept * np.square(language_mean - grand).sum(axis=0)
    ss_concept = n_lang * np.square(concept_mean - grand).sum(axis=0)
    denominator = ss_language + ss_concept
    share = np.divide(
        ss_language,
        denominator,
        out=np.full_like(ss_language, np.nan),
        where=denominator > EPS,
    )
    return share, ss_language, ss_concept


def by_rejections(pvalues: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    valid = np.isfinite(p)
    result = np.zeros(len(p), dtype=bool)
    if not valid.any():
        return result
    values = p[valid]
    order = np.argsort(values)
    harmonic = np.sum(1.0 / np.arange(1, len(values) + 1))
    thresholds = alpha * np.arange(1, len(values) + 1) / (len(values) * harmonic)
    passing = np.flatnonzero(values[order] <= thresholds)
    if passing.size:
        cutoff = values[order[passing[-1]]]
        valid_indices = np.flatnonzero(valid)
        result[valid_indices] = values <= cutoff
    return result


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[np.isfinite(x)]
    if not len(x) or x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def mean_pairwise_cosine(grid: np.ndarray) -> float:
    flat = np.asarray(grid, dtype=np.float64).reshape(-1, grid.shape[-1])
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normalized = flat / np.maximum(norms, EPS)
    summed = normalized.sum(axis=0)
    n = len(normalized)
    self_similarity = float(np.square(normalized).sum())
    return float((np.dot(summed, summed) - self_similarity) / (n * (n - 1)))


def top_eigen_share(grid: np.ndarray) -> float:
    flat = np.asarray(grid, dtype=np.float64).reshape(-1, grid.shape[-1])
    centered = flat - flat.mean(axis=0, keepdims=True)
    total = float(np.square(centered).sum())
    if total <= EPS:
        return 0.0
    _, singular, _ = randomized_svd(
        centered,
        n_components=1,
        n_iter=7,
        random_state=13,
        power_iteration_normalizer="QR",
    )
    return float(np.square(singular[0]) / total)


def lfs_direct(grid: np.ndarray) -> float:
    x = standardize_grid(grid)
    n_lang, n_concept, _ = x.shape
    grand = x.mean(axis=(0, 1))
    language_mean = x.mean(axis=1)
    concept_mean = x.mean(axis=0)
    ss_language = n_concept * np.square(language_mean - grand).sum()
    ss_concept = n_lang * np.square(concept_mean - grand).sum()
    return float(ss_language / (ss_language + ss_concept))


def load_layer(path: Path, n_concepts: int) -> np.ndarray:
    with np.load(path) as archive:
        available = [str(value) for value in archive["langs"]]
        missing = sorted(set(LANGUAGES) - set(available))
        if missing:
            raise ValueError(f"missing fixed languages: {missing}")
        indices = [available.index(language) for language in LANGUAGES]
        return archive["X"][indices, :n_concepts].astype(np.float64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_saved_layers(values: list[int]) -> list[int]:
    """Validate the frozen layer list without assuming one architecture's count."""
    layers = [int(value) for value in values]
    if len(layers) < 2:
        raise ValueError(f"coarse audit requires at least two saved layers, found {layers}")
    if len(set(layers)) != len(layers):
        raise ValueError(f"coarse audit requires unique saved layers, found {layers}")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dumps", type=Path, default=PRIOR_ROOT / "results" / "dumps")
    parser.add_argument("--n-concepts", type=int, default=300)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RMFS_ROOT / "results" / "round3_anisotropy_axes" / "parts",
    )
    args = parser.parse_args()

    started = time.time()
    model_root = args.dumps / args.model
    meta_path = model_root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layers = validate_saved_layers(meta["layers"])

    layer_rows = []
    axis_rows = []
    source_hashes = {}
    for layer in layers:
        path = model_root / f"layer{layer:03d}.npz"
        source_hashes[str(layer)] = sha256(path)
        grid = load_layer(path, args.n_concepts)
        shares, ss_language, ss_concept = coordinate_shares(grid)
        p_upper = beta.sf(
            shares,
            (grid.shape[0] - 1) / 2,
            (grid.shape[1] - 1) / 2,
        )
        significant = by_rejections(p_upper)
        variances = grid.var(axis=(0, 1))
        top_count = max(1, int(np.ceil(TOP_VARIANCE_FRACTION * grid.shape[-1])))
        top_variance = np.zeros(grid.shape[-1], dtype=bool)
        top_variance[np.argsort(variances)[-top_count:]] = True
        sig_count = int(significant.sum())
        matched_top = np.zeros(grid.shape[-1], dtype=bool)
        if sig_count:
            matched_top[np.argsort(variances)[-sig_count:]] = True
        overlap_top5 = int((significant & top_variance).sum())
        overlap_matched = int((significant & matched_top).sum())

        layer_rows.append(
            {
                "model": args.model,
                "layer": layer,
                "n_languages": grid.shape[0],
                "n_concepts": grid.shape[1],
                "dimension": grid.shape[2],
                "lfs_direct_standardized": lfs_direct(grid),
                "mean_pairwise_cosine_raw": mean_pairwise_cosine(grid),
                "top_eigen_share_raw": top_eigen_share(grid),
                "coordinate_share_mean": float(np.nanmean(shares)),
                "coordinate_share_median": float(np.nanmedian(shares)),
                "coordinate_share_gini": gini(shares),
                "gaussian_null_mean": (grid.shape[0] - 1) / (grid.shape[0] + grid.shape[1] - 2),
                "by_alpha": FDR_ALPHA,
                "n_language_heavy_by": sig_count,
                "fraction_language_heavy_by": sig_count / grid.shape[-1],
                "top_variance_fraction": TOP_VARIANCE_FRACTION,
                "overlap_with_top5_variance": overlap_top5,
                "fraction_significant_in_top5_variance": overlap_top5 / max(sig_count, 1),
                "overlap_with_matched_top_variance": overlap_matched,
                "fraction_significant_in_matched_top_variance": overlap_matched / max(sig_count, 1),
            }
        )
        for dimension in range(grid.shape[-1]):
            axis_rows.append(
                {
                    "model": args.model,
                    "layer": layer,
                    "dimension": dimension,
                    "language_share": float(shares[dimension]),
                    "ss_language": float(ss_language[dimension]),
                    "ss_concept": float(ss_concept[dimension]),
                    "raw_variance": float(variances[dimension]),
                    "gaussian_beta_upper_p": float(p_upper[dimension]),
                    "language_heavy_by": bool(significant[dimension]),
                    "top5_variance": bool(top_variance[dimension]),
                    "matched_top_variance": bool(matched_top[dimension]),
                }
            )
        print(f"[{args.model}] layer {layer}", flush=True)

    lfs_values = [row["lfs_direct_standardized"] for row in layer_rows]
    cosine_values = [row["mean_pairwise_cosine_raw"] for row in layer_rows]
    eigen_values = [row["top_eigen_share_raw"] for row in layer_rows]
    rho_cos, p_cos = spearmanr(lfs_values, cosine_values)
    rho_eig, p_eig = spearmanr(lfs_values, eigen_values)
    summary = {
        "model": args.model,
        "n_saved_layers": len(layers),
        "layers": layers,
        "spearman_lfs_vs_mean_cosine": float(rho_cos),
        "pvalue_lfs_vs_mean_cosine_descriptive": float(p_cos),
        "spearman_lfs_vs_top_eigen_share": float(rho_eig),
        "pvalue_lfs_vs_top_eigen_share_descriptive": float(p_eig),
        "warning": "coarse five-layer audit; correlations are descriptive",
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    prefix = args.output_root / args.model
    layers_csv = Path(f"{prefix}.layers.csv")
    axes_csv = Path(f"{prefix}.axes.csv")
    summary_json = Path(f"{prefix}.summary.json")
    layers_parquet = Path(f"{prefix}.layers.parquet")
    axes_parquet = Path(f"{prefix}.axes.parquet")
    manifest_json = Path(f"{prefix}.manifest.json")
    if layers_csv.exists():
        raise FileExistsError(f"refusing to overwrite result for {args.model}")
    write_csv(layers_csv, layer_rows)
    write_csv(axes_csv, axis_rows)
    summary_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    try:
        import pandas as pd

        pd.DataFrame(layer_rows).to_parquet(layers_parquet, index=False)
        pd.DataFrame(axis_rows).to_parquet(axes_parquet, index=False)
    except (ImportError, ModuleNotFoundError) as error:
        print(f"[warning] parquet unavailable: {error}", file=sys.stderr, flush=True)
    manifest = {
        "run_id": f"round3_anisotropy_axes_{args.model}",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": args.model,
        "source_sha256": source_hashes,
        "meta_sha256": sha256(meta_path),
        "n_concepts": args.n_concepts,
        "languages": LANGUAGES,
        "gaussian_reference": True,
        "causal_axis_claim": False,
        "wall_seconds": round(time.time() - started, 1),
    }
    manifest_json.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**summary, "wall_seconds": manifest["wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
