#!/usr/bin/env python3
"""Assemble and score the 14-model Round 3 invariance battery."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
from typing import Any


RTOL = 1e-6
ATOL = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("model manifest contains duplicates")
    return values


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric_int = {"layer", "n_languages", "n_concepts", "dimension"}
    numeric_float = {"value", "reference_value", "expected_value"}
    for row in rows:
        for field in numeric_int:
            row[field] = int(row[field])
        for field in numeric_float:
            row[field] = None if row[field] == "" else float(row[field])
    return rows


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = row["expected_value"]
    if expected is None:
        return {
            **row,
            "absolute_error": None,
            "relative_error": None,
            "tolerance": None,
            "analytic_status": "EMPIRICAL",
        }
    value = row["value"]
    error = abs(value - expected)
    tolerance = ATOL + RTOL * abs(expected)
    relative = error / max(abs(expected), ATOL)
    return {
        **row,
        "absolute_error": error,
        "relative_error": relative,
        "tolerance": tolerance,
        "analytic_status": "PASS" if error <= tolerance else "FAIL",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["family"], row["preprocessing"], row["metric"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, items in sorted(groups.items()):
        scored = [item for item in items if item["analytic_status"] != "EMPIRICAL"]
        failures = [item for item in scored if item["analytic_status"] == "FAIL"]
        summary.append(
            {
                "family": key[0],
                "preprocessing": key[1],
                "metric": key[2],
                "n_cells": len(items),
                "n_analytic": len(scored),
                "n_pass": len(scored) - len(failures),
                "n_fail": len(failures),
                "max_absolute_error": max(
                    (item["absolute_error"] for item in scored), default=None
                ),
                "max_relative_error": max(
                    (item["relative_error"] for item in scored), default=None
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    models = read_manifest(args.model_manifest)
    if len(models) != 14:
        raise ValueError(f"expected 14 frozen models, found {len(models)}")
    observed_csv = {path.stem: path for path in args.parts.glob("*.csv")}
    missing = sorted(set(models) - set(observed_csv))
    extra = sorted(set(observed_csv) - set(models))
    if missing or extra:
        raise RuntimeError(json.dumps({"missing_models": missing, "extra_models": extra}, indent=2))

    rows = []
    input_hashes = {}
    for model in models:
        path = observed_csv[model]
        model_rows = load_rows(path)
        if not model_rows or {row["model"] for row in model_rows} != {model}:
            raise ValueError(f"invalid or empty part for {model}")
        rows.extend(model_rows)
        input_hashes[model] = sha256(path)
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"per-model manifest missing: {manifest_path}")

    scored = [score_row(row) for row in rows]
    summary = summarize(scored)
    failures = [row for row in scored if row["analytic_status"] == "FAIL"]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sweep_csv = args.output_dir / "sweep.csv"
    summary_csv = args.output_dir / "analytic_scorecard.csv"
    failures_csv = args.output_dir / "analytic_failures.csv"
    write_csv(sweep_csv, scored)
    write_csv(summary_csv, summary)
    if failures:
        write_csv(failures_csv, failures)
    else:
        failures_csv.write_text("analytic_status\n", encoding="utf-8")

    try:
        import pandas as pd

        pd.DataFrame(scored).to_parquet(args.output_dir / "sweep.parquet", index=False)
        pd.DataFrame(summary).to_parquet(
            args.output_dir / "analytic_scorecard.parquet", index=False
        )
    except (ImportError, ModuleNotFoundError) as error:
        print(f"[warning] parquet unavailable: {error}", flush=True)

    manifest = {
        "run_id": "round3_invariance_14model_core",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "models": models,
        "model_manifest": str(args.model_manifest.resolve()),
        "model_manifest_sha256": sha256(args.model_manifest),
        "input_csv_sha256": input_hashes,
        "rows": len(scored),
        "analytic_cells": sum(row["analytic_status"] != "EMPIRICAL" for row in scored),
        "analytic_failures": len(failures),
        "rtol": RTOL,
        "atol": ATOL,
        "composite_score": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
