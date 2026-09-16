#!/usr/bin/env python3
"""Reliability audit for RMFS Profile 0.2 on its saved A1 artifact.

This is measurement-only A1 analysis allowed while PREREG_FAMILY_V0 remains
draft.  It does not inspect downstream outcomes.  For each R1/R2/R3 reading,
it measures selection-to-held-out rank stability, agreement among the four
held-out blocks, a balanced one-way ICC, and maximum rank movement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


READINGS = {
    "R1_concept_dominance": ("concept_dominance", "concept_dominance"),
    "R2_mean_alignment_margin": ("macro_mean_margin", "macro_mean_margin"),
    "R3_weak_language_tail": ("weak_language_tail", "weak_language_tail"),
}


def icc_one_way(values: np.ndarray) -> float:
    """ICC(1,1) for a balanced model x held-out-block matrix."""
    n_models, n_blocks = values.shape
    model_means = values.mean(axis=1)
    grand = values.mean()
    between_ms = n_blocks * np.square(model_means - grand).sum() / (n_models - 1)
    within_ms = np.square(values - model_means[:, None]).sum() / (
        n_models * (n_blocks - 1)
    )
    denominator = between_ms + (n_blocks - 1) * within_ms
    return float((between_ms - within_ms) / denominator) if denominator else float("nan")


def rank_positions(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(values, method="average")


def reading_matrix(profile: dict, reading: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    decomposition_key, alignment_key = READINGS[reading]
    models = []
    selection = []
    blocks = []
    for row in profile["profiles"]:
        models.append(row["model_tag"])
        selection.append(float(row["selection_profile"][reading]))
        if reading == "R1_concept_dominance":
            values = [
                float(block[decomposition_key])
                for block in row["supporting"]["heldout_decomposition_blocks"]
            ]
        else:
            values = [
                float(block[alignment_key])
                for block in row["supporting"]["heldout_alignment"]["blocks"]
            ]
        blocks.append(values)
    return models, np.asarray(selection), np.asarray(blocks)


def audit(profile: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    model_rows = []
    for reading in READINGS:
        models, selection, blocks = reading_matrix(profile, reading)
        heldout = blocks.mean(axis=1)
        pairwise = [
            stats.spearmanr(blocks[:, a], blocks[:, b]).statistic
            for a, b in combinations(range(blocks.shape[1]), 2)
        ]
        ranks = np.column_stack([rank_positions(blocks[:, index])
                                 for index in range(blocks.shape[1])])
        rank_range = ranks.max(axis=1) - ranks.min(axis=1)
        signal_sd = float(np.std(heldout, ddof=1))
        within_sd = np.std(blocks, axis=1, ddof=1)
        summary.append(
            {
                "reading": reading,
                "n_models": len(models),
                "n_heldout_blocks": blocks.shape[1],
                "selection_vs_heldout_spearman": float(
                    stats.spearmanr(selection, heldout).statistic
                ),
                "pairwise_block_spearman_median": float(np.median(pairwise)),
                "pairwise_block_spearman_min": float(np.min(pairwise)),
                "icc_1_1": icc_one_way(blocks),
                "median_within_model_sd_over_between_model_sd": float(
                    np.median(within_sd) / signal_sd if signal_sd else np.nan
                ),
                "max_rank_range_across_blocks": float(rank_range.max()),
                "median_rank_range_across_blocks": float(np.median(rank_range)),
            }
        )
        for index, model in enumerate(models):
            model_rows.append(
                {
                    "reading": reading,
                    "model": model,
                    "selection": float(selection[index]),
                    "heldout_mean": float(heldout[index]),
                    "heldout_sd": float(within_sd[index]),
                    "rank_range_across_blocks": float(rank_range[index]),
                    **{
                        f"block_{block_index + 1}": float(value)
                        for block_index, value in enumerate(blocks[index])
                    },
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(model_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/rmfs_profile_v02_14models.json")
    parser.add_argument("--output-dir", default="results/rmfs_profile_stability")
    args = parser.parse_args()

    input_path = Path(args.input)
    profile = json.loads(input_path.read_text())
    summary, by_model = audit(profile)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False, float_format="%.6f")
    by_model.to_csv(output / "by_model.csv", index=False, float_format="%.6f")
    manifest = {
        "analysis_status": "measurement-only A1 reliability audit",
        "downstream_outcomes_inspected": False,
        "profile_version": profile.get("version"),
        "input": str(input_path.resolve()),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "n_models": int(len(profile["profiles"])),
        "n_heldout_blocks": 4,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
