#!/usr/bin/env python3
"""Completeness join and paired summaries for Aim 1 basis sensitivity."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket

import pandas as pd


METRICS = [
    "share_q50",
    "fraction_language_heavy_by",
    "gini_ss_language",
    "top5_ss_language_fraction",
]
SEEDS = [1729, 2718, 3141, 5772, 8111]
SWEEPS = [0, 1, 4, 16]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_models(path: Path) -> list[str]:
    models = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(models) != 14 or len(models) != len(set(models)):
        raise ValueError("expected frozen unique 14-model manifest")
    return models


def descriptive(series: pd.Series) -> dict[str, float]:
    return {
        "median": float(series.median()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "minimum": float(series.min()),
        "maximum": float(series.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    models = read_models(args.models)
    tables = []
    manifests = {}
    for model in models:
        csv_path = args.parts / f"{model}.csv"
        manifest_path = args.parts / f"{model}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        table = pd.read_csv(csv_path)
        expected = len(manifest["layers"]) * len(SEEDS) * len(SWEEPS)
        if len(table) != expected:
            raise ValueError(f"{model}: expected {expected} rows, found {len(table)}")
        cells = table.groupby(["layer", "seed", "sweeps"]).size()
        if len(cells) != expected or not cells.eq(1).all():
            raise ValueError(f"{model}: incomplete or duplicate cells")
        if sorted(table["seed"].unique()) != SEEDS or sorted(table["sweeps"].unique()) != SWEEPS:
            raise ValueError(f"{model}: seed or sweep contract mismatch")
        if table[["ss_language_relative_error", "ss_concept_relative_error"]].max().max() > 1e-10:
            raise ValueError(f"{model}: conservation tolerance exceeded")
        tables.append(table)
        manifests[model] = {
            "csv_sha256": sha256(csv_path),
            "manifest_sha256": sha256(manifest_path),
            "maximum_conservation_relative_error": manifest["maximum_conservation_relative_error"],
        }

    all_rows = pd.concat(tables, ignore_index=True)
    seed_mean = all_rows.groupby(["model", "layer", "sweeps"], as_index=False)[METRICS].mean()
    native = seed_mean.loc[seed_mean["sweeps"].eq(0), ["model", "layer", *METRICS]].rename(
        columns={metric: f"native_{metric}" for metric in METRICS}
    )
    sweep16 = seed_mean.loc[seed_mean["sweeps"].eq(16), ["model", "layer", *METRICS]].rename(
        columns={metric: f"sweep16_{metric}" for metric in METRICS}
    )
    paired = native.merge(sweep16, on=["model", "layer"], validate="one_to_one")
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[f"sweep16_{metric}"] - paired[f"native_{metric}"]

    model_summary = paired.groupby("model", as_index=False)[
        [f"delta_{metric}" for metric in METRICS]
    ].median()
    aggregate = {
        "n_models": len(models),
        "n_model_layers": int(paired.shape[0]),
        "n_random_seeds": len(SEEDS),
        "sweeps": SWEEPS,
        "maximum_conservation_relative_error": float(
            all_rows[["ss_language_relative_error", "ss_concept_relative_error"]].max().max()
        ),
        "primary_native_to_sweep16_layer_deltas": {
            metric: descriptive(paired[f"delta_{metric}"]) for metric in METRICS
        },
        "primary_native_to_sweep16_model_median_deltas": {
            metric: descriptive(model_summary[f"delta_{metric}"]) for metric in METRICS
        },
        "interpretation_guard": "coordinate sensitivity only; no causal-neuron claim",
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    all_rows.to_csv(args.output_dir / "all_basis_rows.csv", index=False)
    seed_mean.to_csv(args.output_dir / "model_layer_sweep_seed_mean.csv", index=False)
    paired.to_csv(args.output_dir / "native_vs_sweep16_paired.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_median_deltas.csv", index=False)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": "aim1_axis_basis_sensitivity_14model",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "model_manifest_sha256": sha256(args.models),
        "part_hashes": manifests,
        "aggregate": aggregate,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
