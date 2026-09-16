#!/usr/bin/env python3
"""Exact orthogonal-basis sensitivity for coordinate language-share summaries."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np
from scipy.stats import beta


LANGUAGES = [
    "eng", "deu", "rus", "arb", "zho-CN", "tur", "vie",
    "hin", "ben", "swa", "amh", "zul", "khm",
]
SEEDS = [1729, 2718, 3141, 5772, 8111]
SWEEPS = [0, 1, 4, 16]
EPS = 1e-12
FDR_ALPHA = 0.05
TOP_FRACTION = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def load_grid(path: Path, n_concepts: int) -> np.ndarray:
    with np.load(path) as archive:
        available = [str(value) for value in archive["langs"]]
        missing = sorted(set(LANGUAGES) - set(available))
        if missing:
            raise ValueError(f"missing fixed languages: {missing}")
        indices = [available.index(language) for language in LANGUAGES]
        return archive["X"][indices, :n_concepts].astype(np.float64)


def effect_matrices(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(grid, dtype=np.float64)
    n_lang, n_concept, _ = x.shape
    grand = x.mean(axis=(0, 1), keepdims=False)
    language = math.sqrt(n_concept) * (x.mean(axis=1) - grand)
    concept = math.sqrt(n_lang) * (x.mean(axis=0) - grand)
    return language, concept


def givens_sweep(matrices: list[np.ndarray], rng: np.random.Generator) -> None:
    dimension = matrices[0].shape[1]
    permutation = rng.permutation(dimension)
    left = permutation[0 : 2 * (dimension // 2) : 2]
    right = permutation[1 : 2 * (dimension // 2) : 2]
    angle = rng.uniform(0.0, 2.0 * np.pi, size=len(left))
    cosine = np.cos(angle)
    sine = np.sin(angle)
    for matrix in matrices:
        old_left = matrix[:, left].copy()
        old_right = matrix[:, right].copy()
        matrix[:, left] = old_left * cosine - old_right * sine
        matrix[:, right] = old_left * sine + old_right * cosine


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
        result[np.flatnonzero(valid)] = values <= cutoff
    return result


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[np.isfinite(x)]
    if not len(x) or x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def summarize_basis(
    language: np.ndarray,
    concept: np.ndarray,
    native_language_total: float,
    native_concept_total: float,
) -> dict[str, float | int]:
    ss_language = np.square(language).sum(axis=0)
    ss_concept = np.square(concept).sum(axis=0)
    denominator = ss_language + ss_concept
    share = np.divide(
        ss_language,
        denominator,
        out=np.full_like(ss_language, np.nan),
        where=denominator > EPS,
    )
    p_upper = beta.sf(share, (language.shape[0] - 1) / 2, (concept.shape[0] - 1) / 2)
    significant = by_rejections(p_upper)
    top_count = max(1, int(np.ceil(TOP_FRACTION * len(ss_language))))
    top = np.argsort(ss_language)[-top_count:]
    language_total = float(ss_language.sum())
    concept_total = float(ss_concept.sum())
    quantiles = np.nanquantile(share, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "dimension": int(len(share)),
        "share_q05": float(quantiles[0]),
        "share_q25": float(quantiles[1]),
        "share_q50": float(quantiles[2]),
        "share_q75": float(quantiles[3]),
        "share_q95": float(quantiles[4]),
        "fraction_share_gt_0_50": float(np.nanmean(share > 0.50)),
        "fraction_share_gt_0_75": float(np.nanmean(share > 0.75)),
        "fraction_share_gt_0_90": float(np.nanmean(share > 0.90)),
        "fraction_language_heavy_by": float(significant.mean()),
        "gini_ss_language": gini(ss_language),
        "top5_ss_language_fraction": float(ss_language[top].sum() / language_total),
        "raw_global_lfs": float(language_total / (language_total + concept_total)),
        "ss_language_relative_error": float(abs(language_total - native_language_total) / max(native_language_total, EPS)),
        "ss_concept_relative_error": float(abs(concept_total - native_concept_total) / max(native_concept_total, EPS)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dumps", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--n-concepts", type=int, default=300)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    model_root = args.dumps / args.model
    meta_path = model_root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest["model"] != args.model:
        raise ValueError("source manifest model mismatch")
    layers = [int(value) for value in meta["layers"]]
    rows: list[dict[str, Any]] = []
    verified_hashes: dict[str, str] = {}

    for layer in layers:
        path = model_root / f"layer{layer:03d}.npz"
        observed_hash = sha256(path)
        expected_hash = source_manifest["source_sha256"][str(layer)]
        if observed_hash != expected_hash:
            raise ValueError(f"source hash mismatch for {args.model} layer {layer}")
        verified_hashes[str(layer)] = observed_hash
        grid = load_grid(path, args.n_concepts)
        language_native, concept_native = effect_matrices(grid)
        native_language_total = float(np.square(language_native).sum())
        native_concept_total = float(np.square(concept_native).sum())

        for seed in SEEDS:
            language = language_native.copy()
            concept = concept_native.copy()
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, stable_int(args.model), layer])
            )
            completed = 0
            for target in SWEEPS:
                while completed < target:
                    givens_sweep([language, concept], rng)
                    completed += 1
                summary = summarize_basis(
                    language, concept, native_language_total, native_concept_total
                )
                rows.append({
                    "model": args.model,
                    "layer": layer,
                    "seed": seed,
                    "sweeps": target,
                    **summary,
                })
        print(f"[{args.model}] layer {layer}", flush=True)

    maximum_error = max(
        max(row["ss_language_relative_error"], row["ss_concept_relative_error"])
        for row in rows
    )
    if maximum_error > 1e-10:
        raise ValueError(f"orthogonal conservation failure: {maximum_error}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / f"{args.model}.csv"
    manifest_path = args.output_root / f"{args.model}.manifest.json"
    write_csv(csv_path, rows)
    manifest = {
        "run_id": f"aim1_axis_basis_sensitivity_{args.model}",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": args.model,
        "layers": layers,
        "seeds": SEEDS,
        "sweeps": SWEEPS,
        "n_concepts": args.n_concepts,
        "languages": LANGUAGES,
        "source_manifest_sha256": sha256(args.source_manifest),
        "source_sha256": verified_hashes,
        "rows": len(rows),
        "maximum_conservation_relative_error": maximum_error,
        "wall_seconds": round(time.time() - started, 3),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
