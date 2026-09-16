#!/usr/bin/env python3
"""Build the corrected raw-LFS, dip-depth, and analytic-null tables."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import socket
from typing import Any


FLAG_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"language[- ]agnostic",
        r"language[- ]neutral core",
        r"language[- ]free",
        r"language still dominates systematic variance in every",
    )
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_models(path: Path) -> list[str]:
    models = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(models) != 14 or len(models) != len(set(models)):
        raise ValueError("expected a unique 14-model manifest")
    return models


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flag_claims(root: Path) -> list[dict[str, Any]]:
    flagged = []
    for suffix in ("*.tex", "*.md"):
        for path in sorted(root.rglob(suffix)):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                matched = [pattern.pattern for pattern in FLAG_PATTERNS if pattern.search(line)]
                if matched:
                    flagged.append(
                        {
                            "file": str(path.resolve()),
                            "line": line_number,
                            "matched_patterns": " | ".join(matched),
                            "text": line.strip(),
                        }
                    )
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-results", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    models = read_models(args.models)
    dip_rows = []
    layer_rows = []
    metric_hashes = {}
    for model in models:
        metrics_path = args.grid_results / model / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_hashes[model] = sha256(metrics_path)
        per_layer = metrics["per_layer"]
        n_languages = len(metrics["langs"])
        n_concepts = int(metrics["n_sents"])
        null = (n_languages - 1) / (n_languages + n_concepts - 2)
        dip_layer = int(metrics["best_layer"])
        dip_lfs = float(per_layer[str(dip_layer)]["lfs"]["lfs"])
        l0_lfs = float(per_layer["0"]["lfs"]["lfs"])
        dip_depth = l0_lfs - dip_lfs
        dip_rows.append(
            {
                "model": model,
                "n_languages": n_languages,
                "n_concepts": n_concepts,
                "dip_layer": dip_layer,
                "raw_lfs_at_dip": dip_lfs,
                "lfs_at_layer0": l0_lfs,
                "dip_depth": dip_depth,
                "analytic_null": null,
                "raw_lfs_excess_over_null": dip_lfs - null,
                "language_main_effect_dominates_at_dip": dip_lfs > 0.5,
                "interpretation": (
                    "language main effect exceeds concept main effect"
                    if dip_lfs > 0.5
                    else "concept main effect equals or exceeds language main effect"
                ),
            }
        )
        for layer_text, values in sorted(per_layer.items(), key=lambda item: int(item[0])):
            factors = values["lfs"]
            layer_rows.append(
                {
                    "model": model,
                    "layer": int(layer_text),
                    "n_languages": n_languages,
                    "n_concepts": n_concepts,
                    "raw_lfs": float(factors["lfs"]),
                    "depth_from_layer0": l0_lfs - float(factors["lfs"]),
                    "analytic_null": null,
                    "excess_over_null": float(factors["lfs"]) - null,
                    "var_language_share_total": float(factors["var_lang"]),
                    "var_concept_share_total": float(factors["var_concept"]),
                    "residual_share_total": float(factors["var_resid"]),
                }
            )

    flagged = flag_claims(args.report_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "lfs_raw_and_depth.csv", dip_rows)
    write_csv(args.output_dir / "lfs_layerwise_raw_and_depth.csv", layer_rows)
    if flagged:
        write_csv(args.output_dir / "claims_requiring_review.csv", flagged)
    else:
        (args.output_dir / "claims_requiring_review.csv").write_text(
            "file,line,matched_patterns,text\n", encoding="utf-8"
        )
    try:
        import pandas as pd

        pd.DataFrame(dip_rows).to_parquet(
            args.output_dir / "lfs_raw_and_depth.parquet", index=False
        )
        pd.DataFrame(layer_rows).to_parquet(
            args.output_dir / "lfs_layerwise_raw_and_depth.parquet", index=False
        )
    except (ImportError, ModuleNotFoundError) as error:
        print(f"[warning] parquet unavailable: {error}", flush=True)

    manifest = {
        "run_id": "round3_lfs_raw_depth_14model",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "models": models,
        "model_manifest_sha256": sha256(args.models),
        "grid_metrics_sha256": metric_hashes,
        "dip_rows": len(dip_rows),
        "layer_rows": len(layer_rows),
        "flagged_claim_lines": len(flagged),
        "paper_text_modified": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
