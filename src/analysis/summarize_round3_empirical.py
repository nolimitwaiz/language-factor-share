#!/usr/bin/env python3
"""Descriptive summary of the nonuniform Round 3 perturbation cells.

This is explicitly post-result descriptive analysis. It does not introduce a
new pass threshold or alter the prospective analytic invariance scorecard.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_number(value: str) -> int:
    if not value.startswith("rank="):
        raise ValueError(f"not a rank magnitude: {value}")
    return int(value.split("=", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.sweep)
    uniform_magnitude = pd.to_numeric(frame["magnitude"], errors="coerce")
    identity = frame[
        (frame["family"] == "uniform") & np.isclose(uniform_magnitude, 1.0)
    ][["model", "preprocessing", "metric", "value"]].rename(
        columns={"value": "identity_value"}
    )
    if identity.duplicated(["model", "preprocessing", "metric"]).any():
        raise ValueError("identity rows are not unique")
    empirical = frame[frame["family"].isin(["diagonal", "rank", "language"])].merge(
        identity, on=["model", "preprocessing", "metric"], how="left", validate="many_to_one"
    )
    if empirical["identity_value"].isna().any():
        raise ValueError("an empirical row lacks an identity reference")
    empirical["delta_from_identity"] = empirical["value"] - empirical["identity_value"]
    empirical["fractional_change_over_abs_identity"] = (
        empirical["delta_from_identity"]
        / np.maximum(empirical["identity_value"].abs(), EPS)
    )

    rank = empirical[empirical["family"] == "rank"].copy()
    rank["retained_rank"] = rank["magnitude"].map(rank_number)
    minimum_rank = rank.groupby("model")["retained_rank"].transform("min")
    strongest = rank[rank["retained_rank"] == minimum_rank].copy()
    strongest_summary = (
        strongest.groupby(["preprocessing", "metric"], as_index=False)
        .agg(
            n_models=("model", "nunique"),
            median_delta=("delta_from_identity", "median"),
            min_delta=("delta_from_identity", "min"),
            max_delta=("delta_from_identity", "max"),
            median_fractional_change=("fractional_change_over_abs_identity", "median"),
            n_increase=("delta_from_identity", lambda values: int((values > 0).sum())),
            n_decrease=("delta_from_identity", lambda values: int((values < 0).sum())),
        )
    )

    monotonic_rows = []
    for (model, preprocessing, metric), group in rank.groupby(
        ["model", "preprocessing", "metric"]
    ):
        if group["retained_rank"].nunique() < 3:
            continue
        rho, pvalue = spearmanr(group["retained_rank"], group["value"])
        monotonic_rows.append(
            {
                "model": model,
                "preprocessing": preprocessing,
                "metric": metric,
                "n_rank_levels": group["retained_rank"].nunique(),
                "spearman_with_retained_rank": float(rho),
                "pvalue_descriptive": float(pvalue),
            }
        )
    monotonic = pd.DataFrame(monotonic_rows)
    monotonic_summary = (
        monotonic.groupby(["preprocessing", "metric"], as_index=False)
        .agg(
            n_models=("model", "nunique"),
            median_spearman=("spearman_with_retained_rank", "median"),
            min_spearman=("spearman_with_retained_rank", "min"),
            max_spearman=("spearman_with_retained_rank", "max"),
        )
    )

    family_summary = (
        empirical.groupby(["family", "preprocessing", "metric"], as_index=False)
        .agg(
            n_cells=("value", "size"),
            n_models=("model", "nunique"),
            median_delta=("delta_from_identity", "median"),
            median_abs_delta=("delta_from_identity", lambda values: float(values.abs().median())),
            min_delta=("delta_from_identity", "min"),
            max_delta=("delta_from_identity", "max"),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "empirical_cells": empirical,
        "strongest_rank_cells": strongest,
        "strongest_rank_summary": strongest_summary,
        "rank_monotonicity_by_model": monotonic,
        "rank_monotonicity_summary": monotonic_summary,
        "family_summary": family_summary,
    }
    for name, table in outputs.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
        table.to_parquet(args.output_dir / f"{name}.parquet", index=False)

    manifest = {
        "run_id": "round3_empirical_descriptive_summary",
        "analysis_status": "post-result descriptive; no new pass thresholds",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "source_sweep": str(args.sweep.resolve()),
        "source_sweep_sha256": sha256(args.sweep),
        "rows": {name: len(table) for name, table in outputs.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
