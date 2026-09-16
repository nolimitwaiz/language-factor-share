#!/usr/bin/env python3
"""RMFS diagnostic profile, version 0.2.

RMFS is returned as a profile, not a scalar. The four primary readings answer
different questions and must not be averaged in a way that can hide a failed
preservation or tail condition.

Input grids have shape [languages, concepts, dimensions]. Concept index c
must identify the same meaning/parallel item in every language.

See also: ``rmfs_profile_campaign.py`` for the observational 14-model table
on campaign dumps (selection-fit z-score, held-out blocks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-12
PROFILE_VERSION = "RMFS-profile-0.2-draft"


def _as_grid(value: Any, name: str) -> np.ndarray:
    grid = np.asarray(value, dtype=np.float64)
    if grid.ndim != 3:
        raise ValueError(f"{name} must have shape [languages, concepts, dimensions]")
    if min(grid.shape) < 1:
        raise ValueError(f"{name} cannot contain an empty axis")
    if not np.isfinite(grid).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return grid


def standardize_grid(grid: np.ndarray) -> np.ndarray:
    """Coordinate standardization fitted over the complete balanced grid."""
    mean = grid.mean(axis=(0, 1), keepdims=True)
    scale = grid.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale > EPS, scale, 1.0)
    return (grid - mean) / scale


def factor_decomposition(grid: np.ndarray, *, standardize: bool = True) -> dict[str, float]:
    """Direct balanced-grid decomposition used by the committed LFS code.

    This is the sums-of-squares estimator, not a residual-corrected
    method-of-moments / REML estimator from later mathematical notes.
    """
    raw = _as_grid(grid, "grid")
    x = standardize_grid(raw) if standardize else raw.copy()
    n_lang, n_concept, _ = x.shape

    grand = x.mean(axis=(0, 1))
    language_means = x.mean(axis=1)
    concept_means = x.mean(axis=0)
    residual = x - language_means[:, None, :] - concept_means[None, :, :] + grand

    ss_language = float(n_concept * np.square(language_means - grand).sum())
    ss_concept = float(n_lang * np.square(concept_means - grand).sum())
    ss_residual = float(np.square(residual).sum())
    systematic = ss_language + ss_concept
    total = systematic + ss_residual

    lfs = ss_language / systematic if systematic > EPS else float("nan")
    return {
        "lfs": lfs,
        "concept_dominance": 1.0 - lfs if np.isfinite(lfs) else float("nan"),
        "ss_language": ss_language,
        "ss_concept": ss_concept,
        "ss_residual": ss_residual,
        "language_share_total": ss_language / total if total > EPS else float("nan"),
        "concept_share_total": ss_concept / total if total > EPS else float("nan"),
        "residual_share_total": ss_residual / total if total > EPS else float("nan"),
    }


def raw_geometry(grid: np.ndarray) -> dict[str, float]:
    """Raw, scale-sensitive geometry reported beside the LFS ratio.

    ``within_concept_spread`` reproduces the quantity called concept variance
    by the existing Aim 2 training code: mean within-language squared spread.
    ``shared_concept_energy`` is the raw additive concept main-effect energy.
    """
    x = _as_grid(grid, "grid")
    grand = x.mean(axis=(0, 1))
    language_means = x.mean(axis=1)
    concept_means = x.mean(axis=0)
    residual = x - language_means[:, None, :] - concept_means[None, :, :] + grand

    within = x - language_means[:, None, :]
    centered = x.reshape(-1, x.shape[-1])
    centered = centered - centered.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    weights = singular / (singular.sum() + EPS)
    weights = weights[weights > EPS]
    effective_rank = float(np.exp(-(weights * np.log(weights)).sum()))

    return {
        "within_concept_spread": float(np.square(within).sum(axis=-1).mean()),
        "shared_concept_energy": float(np.square(concept_means - grand).sum(axis=-1).mean()),
        "language_energy": float(np.square(language_means - grand).sum(axis=-1).mean()),
        "residual_energy": float(np.square(residual).sum(axis=-1).mean()),
        "mean_vector_norm": float(np.linalg.norm(x, axis=-1).mean()),
        "effective_rank": effective_rank,
    }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), EPS)


def retrieval_margins(pivot: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Target-to-pivot true-pair margin against the hardest wrong concept."""
    if pivot.shape != target.shape or pivot.ndim != 2:
        raise ValueError("pivot and target must have equal [concepts, dimensions] shape")
    n_concept = pivot.shape[0]
    if n_concept < 2:
        raise ValueError("retrieval margins require at least two concepts")
    similarity = _normalize_rows(target) @ _normalize_rows(pivot).T
    correct = np.diag(similarity)
    wrong = np.where(np.eye(n_concept, dtype=bool), -np.inf, similarity).max(axis=1)
    return correct - wrong


def alignment_profile(
    grid: np.ndarray,
    *,
    pivot: int = 0,
    sentence_tail_fraction: float = 0.10,
    language_tail_fraction: float = 0.25,
    standardize: bool = True,
) -> dict[str, Any]:
    """Mean and lower-tail parallel-retrieval margins by language."""
    raw = _as_grid(grid, "grid")
    if not 0 <= pivot < raw.shape[0]:
        raise ValueError("pivot is outside the language axis")
    if not 0 < sentence_tail_fraction <= 1 or not 0 < language_tail_fraction <= 1:
        raise ValueError("tail fractions must lie in (0, 1]")
    x = standardize_grid(raw) if standardize else raw

    per_language: dict[str, dict[str, float]] = {}
    for language in range(x.shape[0]):
        if language == pivot:
            continue
        margins = retrieval_margins(x[pivot], x[language])
        n_tail = max(1, int(np.floor(sentence_tail_fraction * len(margins))))
        per_language[str(language)] = {
            "mean_margin": float(margins.mean()),
            "aar": float(np.sort(margins)[:n_tail].mean()),
            "retrieval_success_rate": float((margins > 0).mean()),
            "n_sentences": int(len(margins)),
        }

    means = np.array([item["mean_margin"] for item in per_language.values()])
    tails = np.array([item["aar"] for item in per_language.values()])
    n_language_tail = max(1, int(np.floor(language_tail_fraction * len(tails))))
    return {
        "macro_mean_margin": float(means.mean()),
        "weak_language_tail": float(np.sort(tails)[:n_language_tail].mean()),
        "sentence_tail_fraction": sentence_tail_fraction,
        "language_tail_fraction": language_tail_fraction,
        "per_language": per_language,
    }


def preservation_profile(
    grid: np.ndarray,
    reference_grid: np.ndarray | None,
    *,
    floor: float = 0.90,
) -> dict[str, Any]:
    """Scale-sensitive checkpoint preservation; undefined without a reference."""
    current = raw_geometry(grid)
    if reference_grid is None:
        return {
            "available": False,
            "reason": "a frozen reference checkpoint is required",
            "current": current,
        }

    reference = _as_grid(reference_grid, "reference_grid")
    if reference.shape != np.asarray(grid).shape:
        raise ValueError("reference_grid must have the same shape as grid")
    baseline = raw_geometry(reference)
    within_ratio = current["within_concept_spread"] / max(
        baseline["within_concept_spread"], EPS
    )
    shared_ratio = current["shared_concept_energy"] / max(
        baseline["shared_concept_energy"], EPS
    )
    rank_ratio = current["effective_rank"] / max(baseline["effective_rank"], EPS)
    return {
        "available": True,
        "floor": floor,
        "within_concept_ratio": float(within_ratio),
        "shared_concept_ratio": float(shared_ratio),
        "effective_rank_ratio": float(rank_ratio),
        "passes_legacy_cvp_gate": bool(within_ratio >= floor),
        "current": current,
        "reference": baseline,
    }


def compute_rmfs_profile(
    grid: np.ndarray,
    *,
    reference_grid: np.ndarray | None = None,
    pivot: int = 0,
    sentence_tail_fraction: float = 0.10,
    language_tail_fraction: float = 0.25,
    preservation_floor: float = 0.90,
) -> dict[str, Any]:
    """Compute RMFS v0.2 as four separate readings plus supporting values."""
    x = _as_grid(grid, "grid")
    decomposition = factor_decomposition(x, standardize=True)
    alignment = alignment_profile(
        x,
        pivot=pivot,
        sentence_tail_fraction=sentence_tail_fraction,
        language_tail_fraction=language_tail_fraction,
        standardize=True,
    )
    preservation = preservation_profile(x, reference_grid, floor=preservation_floor)

    return {
        "version": PROFILE_VERSION,
        "unit": "model/checkpoint at one layer and one fixed evaluation grid",
        "readings": {
            "concept_dominance": decomposition["concept_dominance"],
            "mean_alignment_margin": alignment["macro_mean_margin"],
            "weak_language_tail": alignment["weak_language_tail"],
            "preservation": preservation,
        },
        "supporting": {
            "decomposition": decomposition,
            "alignment": alignment,
            "raw_geometry": raw_geometry(x),
        },
        "composite_score": None,
        "warning": (
            "Do not average the readings; preservation is a gate and each "
            "reading has a different target."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path, help="NPZ containing an array named 'grid'")
    parser.add_argument("--reference", type=Path, help="NPZ containing an array named 'grid'")
    parser.add_argument("--pivot", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.grid) as archive:
        grid = archive["grid"]
    reference = None
    if args.reference:
        with np.load(args.reference) as archive:
            reference = archive["grid"]
    result = compute_rmfs_profile(grid, reference_grid=reference, pivot=args.pivot)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
