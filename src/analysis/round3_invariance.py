#!/usr/bin/env python3
"""Round 3 saved-embedding invariance and degradation battery.

This script reads one campaign model's pre-normalization dip-layer dump and
evaluates the project-defined metric family on raw and jointly standardized
representations. It does not train or load a language model and emits no
composite score.

The script intentionally keeps two preservation quantities separate:

* ``cvp_within`` matches the current RMFS Profile 0.2 primary preservation
  reading: within-language concept spread relative to the unperturbed grid.
* ``cpres_shared`` matches the RMFS campaign gate: additive shared-concept
  energy relative to the unperturbed grid.

Both are raw and scale-sensitive. Neither is computed after standardization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Callable

import numpy as np


RMFS_ROOT = Path.home() / "rmfs"
PRIOR_ROOT = Path.home() / "multilingual-metrics"
DEFAULT_DUMPS = PRIOR_ROOT / "results" / "dumps"
DEFAULT_OUTPUT = RMFS_ROOT / "results" / "round3_invariance" / "parts"
LANGUAGES = [
    "eng",
    "deu",
    "rus",
    "arb",
    "zho-CN",
    "tur",
    "vie",
    "hin",
    "ben",
    "swa",
    "amh",
    "zul",
    "khm",
]
BOTTOM_TIER = {"swa", "amh", "zul", "khm"}
UNIFORM_MAGNITUDES = (1.0, 0.9, 0.5, 0.1, 0.01)
DIAGONAL_FRACTIONS = (0.25, 0.50, 0.90)
EPS = 1e-12

sys.path.insert(0, str(RMFS_ROOT / "src"))

from rmfs.components.variance import variance_components  # noqa: E402
from rmfs.metrics.baselines import aar_shared, mexa_shared  # noqa: E402


@dataclass(frozen=True)
class MetricRow:
    model: str
    layer: int
    n_languages: int
    n_concepts: int
    dimension: int
    family: str
    perturbation: str
    magnitude: str
    preprocessing: str
    metric: str
    value: float
    reference_value: float | None
    expected_value: float | None
    analytic_expectation: str


@dataclass(frozen=True)
class Perturbation:
    family: str
    name: str
    magnitude: str
    transform: Callable[[np.ndarray], np.ndarray]


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
    scale = np.where(scale > EPS, scale, 1.0)
    return (x - mean) / scale


def direct_decomposition(grid: np.ndarray) -> dict[str, float]:
    x = np.asarray(grid, dtype=np.float64)
    n_lang, n_concept, _ = x.shape
    grand = x.mean(axis=(0, 1))
    language_mean = x.mean(axis=1)
    concept_mean = x.mean(axis=0)
    residual = x - language_mean[:, None, :] - concept_mean[None, :, :] + grand
    ss_language = float(n_concept * np.square(language_mean - grand).sum())
    ss_concept = float(n_lang * np.square(concept_mean - grand).sum())
    ss_residual = float(np.square(residual).sum())
    denominator = ss_language + ss_concept
    total = denominator + ss_residual
    return {
        "lfs_direct": ss_language / denominator if denominator > EPS else np.nan,
        "ss_language": ss_language,
        "ss_concept": ss_concept,
        "ss_residual": ss_residual,
        "language_share_total": ss_language / total if total > EPS else np.nan,
        "concept_share_total": ss_concept / total if total > EPS else np.nan,
        "residual_share_total": ss_residual / total if total > EPS else np.nan,
    }


def raw_geometry(grid: np.ndarray) -> dict[str, float]:
    x = np.asarray(grid, dtype=np.float64)
    grand = x.mean(axis=(0, 1))
    language_mean = x.mean(axis=1)
    concept_mean = x.mean(axis=0)
    within = x - language_mean[:, None, :]
    concept_centered = concept_mean - grand
    # N is much smaller than D in the campaign. The eigenvalues of C C^T are
    # the squared singular values of C and avoid forming a D by D covariance.
    gram = concept_centered @ concept_centered.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]
    singular = np.sqrt(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > EPS * eigenvalues.max()] if eigenvalues.size and eigenvalues.max() > 0 else eigenvalues
    if eigenvalues.size:
        eigen_prob = eigenvalues / eigenvalues.sum()
        covariance_entropy_rank = float(
            np.exp(-(eigen_prob * np.log(eigen_prob)).sum())
        )
        participation_ratio = float(
            np.square(eigenvalues.sum()) / np.square(eigenvalues).sum()
        )
    else:
        covariance_entropy_rank = 0.0
        participation_ratio = 0.0

    singular_positive = singular[singular > EPS * singular.max()] if singular.size and singular.max() > 0 else singular
    if singular_positive.size:
        singular_prob = singular_positive / singular_positive.sum()
        singular_entropy_rank = float(
            np.exp(-(singular_prob * np.log(singular_prob)).sum())
        )
    else:
        singular_entropy_rank = 0.0

    return {
        "cvp_within_raw": float(np.square(within).sum(axis=-1).mean()),
        "cpres_shared_raw": float(np.square(concept_centered).sum(axis=-1).mean()),
        "mean_norm": float(np.linalg.norm(x, axis=-1).mean()),
        "concept_rank_covariance_entropy": covariance_entropy_rank,
        "concept_rank_singular_entropy": singular_entropy_rank,
        "concept_participation_ratio": participation_ratio,
    }


def retrieval_profile(grid: np.ndarray) -> dict[str, float]:
    x = np.asarray(grid, dtype=np.float64)
    pivot = x[0]
    mexa_values = []
    aar_values = []
    margin_values = []
    cka_values = []
    for target in x[1:]:
        mexa = mexa_shared(pivot, target)
        aar = aar_shared(pivot, target, q=0.10)
        pivot_centered = pivot - pivot.mean(axis=0, keepdims=True)
        target_centered = target - target.mean(axis=0, keepdims=True)
        pivot_gram = pivot_centered @ pivot_centered.T
        target_gram = target_centered @ target_centered.T
        cka_denominator = np.sqrt(
            np.square(pivot_gram).sum() * np.square(target_gram).sum()
        )
        cka = (
            float((pivot_gram * target_gram).sum() / cka_denominator)
            if cka_denominator > EPS
            else 0.0
        )
        mexa_values.append(mexa.value)
        aar_values.append(aar.tail_mean)
        margin_values.append(aar.mean)
        cka_values.append(cka)
    return {
        "mexa_external": float(np.mean(mexa_values)),
        "aar10": float(np.mean(aar_values)),
        "mean_alignment_margin": float(np.mean(margin_values)),
        "linear_cka": float(np.mean(cka_values)),
    }


def evaluate_metrics(grid: np.ndarray, preprocessing: str) -> dict[str, float]:
    x = standardize_grid(grid) if preprocessing == "standardized" else np.asarray(grid, dtype=np.float64)
    direct = direct_decomposition(x)
    vc = variance_components(x)
    return {
        **direct,
        "lfs_vc_reml": float(vc.lfs_vc),
        **retrieval_profile(x),
        **raw_geometry(x),
    }


def diagonal_transform(dim: int, fraction: float, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    count = max(1, int(np.floor(fraction * dim)))
    selected = np.sort(rng.choice(dim, size=count, replace=False))
    scale = np.ones(dim, dtype=np.float64)
    scale[selected] = 0.10

    def transform(grid: np.ndarray) -> np.ndarray:
        return np.asarray(grid, dtype=np.float64) * scale[None, None, :]

    return transform


def language_transform(languages: list[str], factor: float) -> Callable[[np.ndarray], np.ndarray]:
    indices = [index for index, language in enumerate(languages) if language in BOTTOM_TIER]
    if not indices:
        raise ValueError("the fixed bottom-tier language panel is absent")

    def transform(grid: np.ndarray) -> np.ndarray:
        result = np.asarray(grid, dtype=np.float64).copy()
        result[indices] *= factor
        return result

    return transform


def perturbations(grid: np.ndarray, languages: list[str], seed: int) -> list[Perturbation]:
    dim = grid.shape[-1]
    available_rank = min(grid.shape[1] - 1, dim)
    requested_ranks = [
        available_rank,
        max(1, available_rank // 2),
        max(1, available_rank // 8),
        max(1, available_rank // 32),
        min(16, available_rank),
    ]
    ranks = list(dict.fromkeys(requested_ranks))
    x = np.asarray(grid, dtype=np.float64)
    grand = x.mean(axis=(0, 1))
    concept_effect = x.mean(axis=0) - grand
    concept_u, concept_singular, concept_vt = np.linalg.svd(
        concept_effect, full_matrices=False
    )

    def make_rank_transform(rank: int) -> Callable[[np.ndarray], np.ndarray]:
        retained = min(rank, len(concept_singular))
        reduced = (
            concept_u[:, :retained] * concept_singular[:retained]
        ) @ concept_vt[:retained]

        def transform(value: np.ndarray) -> np.ndarray:
            source = np.asarray(value, dtype=np.float64)
            return source - concept_effect[None, :, :] + reduced[None, :, :]

        return transform

    result = [
        Perturbation(
            family="uniform",
            name="uniform_shrink",
            magnitude=f"{magnitude:g}",
            transform=lambda x, magnitude=magnitude: np.asarray(x, dtype=np.float64) * magnitude,
        )
        for magnitude in UNIFORM_MAGNITUDES
    ]
    result.extend(
        Perturbation(
            family="diagonal",
            name="diagonal_shrink_0.1",
            magnitude=f"fraction={fraction:g}",
            transform=diagonal_transform(dim, fraction, seed),
        )
        for fraction in DIAGONAL_FRACTIONS
    )
    result.extend(
        Perturbation(
            family="rank",
            name="concept_main_effect_projection",
            magnitude=f"rank={rank}",
            transform=make_rank_transform(rank),
        )
        for rank in ranks
    )
    result.append(
        Perturbation(
            family="language",
            name="bottom_tier_shrink_0.1",
            magnitude="swa,amh,zul,khm",
            transform=language_transform(languages, 0.10),
        )
    )
    return result


def load_grid(dumps: Path, model: str, n_concepts: int) -> tuple[np.ndarray, int, Path]:
    meta_path = dumps / model / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layer = int(meta["dip_layer"])
    path = dumps / model / f"layer{layer:03d}.npz"
    with np.load(path) as archive:
        available = [str(item) for item in archive["langs"]]
        missing = sorted(set(LANGUAGES) - set(available))
        if missing:
            raise ValueError(f"{model} is missing fixed languages: {missing}")
        indices = [available.index(language) for language in LANGUAGES]
        grid = archive["X"][indices, :n_concepts].astype(np.float64)
    return grid, layer, path


def expectation(
    family: str,
    magnitude: str,
    preprocessing: str,
    metric: str,
    reference: float,
) -> tuple[float | None, str]:
    invariant = {
        "lfs_direct",
        "lfs_vc_reml",
        "mexa_external",
        "aar10",
        "mean_alignment_margin",
        "linear_cka",
        "concept_rank_covariance_entropy",
        "concept_rank_singular_entropy",
        "concept_participation_ratio",
    }
    quadratic = {
        "ss_language",
        "ss_concept",
        "ss_residual",
        "cvp_within_raw",
        "cpres_shared_raw",
    }
    linear = {"mean_norm"}
    shares = {"language_share_total", "concept_share_total", "residual_share_total"}
    if family == "diagonal" and preprocessing == "standardized":
        return reference, "positive-diagonal invariant after coordinate standardization"
    if family != "uniform":
        return None, "empirical"
    m = float(magnitude)
    if preprocessing == "standardized":
        return reference, "uniform-scale invariant after coordinate standardization"
    if metric in invariant or metric in shares:
        return reference, "uniform-scale invariant"
    if metric in quadratic:
        return reference * m * m, "quadratic in uniform scale"
    if metric in linear:
        return reference * m, "linear in uniform scale"
    return None, "not frozen"


def metric_rows(model: str, layer: int, base: np.ndarray) -> list[MetricRow]:
    n_lang, n_concept, dim = base.shape
    baseline = {
        preprocessing: evaluate_metrics(base, preprocessing)
        for preprocessing in ("raw", "standardized")
    }
    base_raw_geometry = raw_geometry(base)
    rows: list[MetricRow] = []
    seed = 13
    for perturbation in perturbations(base, LANGUAGES, seed):
        transformed = perturbation.transform(base)
        for preprocessing in ("raw", "standardized"):
            metrics = evaluate_metrics(transformed, preprocessing)
            if preprocessing == "raw":
                metrics["cvp_within_ratio"] = (
                    metrics["cvp_within_raw"]
                    / max(base_raw_geometry["cvp_within_raw"], EPS)
                )
                metrics["cpres_shared_ratio"] = (
                    metrics["cpres_shared_raw"]
                    / max(base_raw_geometry["cpres_shared_raw"], EPS)
                )
            for metric, value in metrics.items():
                if metric in {"cvp_within_ratio", "cpres_shared_ratio"}:
                    reference_value = 1.0
                    if perturbation.family == "uniform":
                        expected_value = float(perturbation.magnitude) ** 2
                        expectation_text = "quadratic preservation ratio"
                    else:
                        expected_value = None
                        expectation_text = "empirical"
                else:
                    reference_value = baseline[preprocessing][metric]
                    expected_value, expectation_text = expectation(
                        perturbation.family,
                        perturbation.magnitude,
                        preprocessing,
                        metric,
                        reference_value,
                    )
                rows.append(
                    MetricRow(
                        model=model,
                        layer=layer,
                        n_languages=n_lang,
                        n_concepts=n_concept,
                        dimension=dim,
                        family=perturbation.family,
                        perturbation=perturbation.name,
                        magnitude=perturbation.magnitude,
                        preprocessing=preprocessing,
                        metric=metric,
                        value=float(value),
                        reference_value=float(reference_value),
                        expected_value=None if expected_value is None else float(expected_value),
                        analytic_expectation=expectation_text,
                    )
                )
        del transformed
    return rows


def write_rows(path: Path, rows: list[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    try:
        import pandas as pd

        pd.DataFrame([asdict(row) for row in rows]).to_parquet(
            path.with_suffix(".parquet"), index=False
        )
    except (ImportError, ModuleNotFoundError) as error:
        print(f"[warning] parquet unavailable: {error}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", type=Path, default=DEFAULT_DUMPS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-concepts", type=int, default=300)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.time()
    grid, layer, source = load_grid(args.dumps, args.model, args.n_concepts)
    rows = metric_rows(args.model, layer, grid)
    output = args.output_root / f"{args.model}.csv"
    if output.exists() or output.with_suffix(".parquet").exists():
        raise FileExistsError(f"refusing to overwrite Round 3 result: {output}")
    write_rows(output, rows)
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=RMFS_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    manifest = {
        "run_id": f"round3_invariance_{args.model}",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": args.model,
        "layer": layer,
        "languages": LANGUAGES,
        "n_concepts": args.n_concepts,
        "source_dump": str(source),
        "source_dump_sha256": sha256(source),
        "script": "scripts/round3_invariance.py",
        "git": git_hash,
        "csv_sha256": sha256(output),
        "wall_seconds": round(time.time() - started, 1),
        "external_comparators": ["MEXA", "linear CKA"],
        "composite_score": False,
        "deferred_pending_contract": [
            "mapping ladder K",
            "Davies-Bouldin",
            "Dunn",
            "Xie-Beni",
            "Calinski-Harabasz",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "layer": layer,
                "rows": len(rows),
                "output": str(output),
                "wall_seconds": manifest["wall_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
