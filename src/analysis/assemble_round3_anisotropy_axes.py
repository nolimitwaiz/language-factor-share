#!/usr/bin/env python3
"""Completeness join for the frozen coarse-layer anisotropy and axis audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def models(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != 14 or len(values) != len(set(values)):
        raise ValueError("expected a unique 14-model manifest")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frozen = models(args.models)
    layer_tables = []
    axis_tables = []
    summaries = []
    input_hashes = {}
    for model in frozen:
        paths = {
            "layers": Path(f"{args.parts / model}.layers.csv"),
            "axes": Path(f"{args.parts / model}.axes.csv"),
            "summary": Path(f"{args.parts / model}.summary.json"),
            "manifest": Path(f"{args.parts / model}.manifest.json"),
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing outputs for {model}: {missing}")
        model_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        expected_layers = int(model_summary["n_saved_layers"])
        layer = pd.read_csv(paths["layers"])
        axes = pd.read_csv(paths["axes"])
        if expected_layers < 2:
            raise ValueError(f"invalid saved-layer count for {model}: {expected_layers}")
        if set(layer["model"]) != {model} or len(layer) != expected_layers:
            raise ValueError(
                f"invalid coarse-layer table for {model}: "
                f"expected {expected_layers}, found {len(layer)}"
            )
        if set(axes["model"]) != {model}:
            raise ValueError(f"invalid axis table for {model}")
        layer_tables.append(layer)
        axis_tables.append(axes)
        summaries.append(model_summary)
        input_hashes[model] = {key: sha256(path) for key, path in paths.items()}

    layers = pd.concat(layer_tables, ignore_index=True)
    axes = pd.concat(axis_tables, ignore_index=True)
    summary = pd.DataFrame(summaries)
    aggregate = {
        "n_models": len(frozen),
        "n_model_layers": len(layers),
        "minimum_saved_layers_per_model": int(summary["n_saved_layers"].min()),
        "maximum_saved_layers_per_model": int(summary["n_saved_layers"].max()),
        "n_axes": len(axes),
        "median_spearman_lfs_vs_mean_cosine": float(
            summary["spearman_lfs_vs_mean_cosine"].median()
        ),
        "models_abs_rho_cosine_over_0_7": int(
            (summary["spearman_lfs_vs_mean_cosine"].abs() > 0.7).sum()
        ),
        "median_spearman_lfs_vs_top_eigen_share": float(
            summary["spearman_lfs_vs_top_eigen_share"].median()
        ),
        "models_abs_rho_eigen_over_0_7": int(
            (summary["spearman_lfs_vs_top_eigen_share"].abs() > 0.7).sum()
        ),
        "median_language_heavy_axis_fraction": float(
            layers["fraction_language_heavy_by"].median()
        ),
        "minimum_language_heavy_axis_fraction": float(
            layers["fraction_language_heavy_by"].min()
        ),
        "maximum_language_heavy_axis_fraction": float(
            layers["fraction_language_heavy_by"].max()
        ),
        "warning": "all correlations use only the frozen coarse saved layers and are descriptive",
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (("layers", layers), ("axes", axes), ("model_summary", summary)):
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
        table.to_parquet(args.output_dir / f"{name}.parquet", index=False)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": "round3_anisotropy_axes_14model_coarse",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "models": frozen,
        "model_manifest_sha256": sha256(args.models),
        "input_hashes": input_hashes,
        "aggregate": aggregate,
        "gaussian_reference": True,
        "causal_axis_claim": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
