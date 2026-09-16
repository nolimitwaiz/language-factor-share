#!/usr/bin/env python
"""Retrospective artifact audit for headline LFS submission claims.

This is not a confirmatory experiment. It reads the frozen direct-SS grid,
the later A9 REML LFS-VC profiles, and the consolidated same-support frame.
It deliberately keeps estimator cohorts separate and reports continuous
profile diagnostics instead of inventing a post-hoc binary definition of a
"U shape".
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PRIOR = Path(os.environ.get(
    "MULTILINGUAL_METRICS_ROOT", Path.home() / "multilingual-metrics"
)).expanduser()
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"

# Original direct-SS paper grid. Aim-2 checkpoints and the local N=500
# sensitivity directory are not model-grid members.
DIRECT_SS_MODELS = (
    "EuroLLM-1.7B",
    "Falcon3-7B-Base",
    "Llama-3.1-8B",
    "Mistral-7B-v0.3",
    "OLMo-2-0425-1B",
    "OLMo-2-1124-7B",
    "Qwen3-0.6B-Base",
    "Qwen3-1.7B-Base",
    "Qwen3-4B-Base",
    "Qwen3-8B-Base",
    "SmolLM2-1.7B",
    "Yi-1.5-9B",
    "bloom-1b7",
    "bloom-7b1",
    "granite-3.1-8b-base",
    "salamandra-2b",
    "salamandra-7b",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _profile_row(model: str, estimator: str, values: dict[int, float],
                 source: Path) -> dict:
    layers = np.asarray(sorted(values), dtype=int)
    vals = np.asarray([values[int(k)] for k in layers], dtype=float)
    if len(vals) < 3 or not np.isfinite(vals).all():
        raise ValueError(f"invalid depth profile: {source}")
    j = int(np.argmin(vals))
    minimum = float(vals[j])
    left = float(vals[0] - minimum)
    right = float(vals[-1] - minimum)
    return {
        "model": model,
        "estimator": estimator,
        "n_layers_measured": int(len(vals)),
        "min_layer": int(layers[j]),
        "relative_min_depth": float(j / (len(vals) - 1)),
        "first": float(vals[0]),
        "minimum": minimum,
        "last": float(vals[-1]),
        "left_endpoint_lift": left,
        "right_endpoint_lift": right,
        # A non-endpoint minimum is a minimal descriptive diagnostic, not a
        # preregistered pass rule for the qualitative phrase "U shaped".
        "interior_minimum": bool(0 < j < len(vals) - 1),
        "both_endpoints_above_minimum": bool(left > 0 and right > 0),
        "source": str(source),
        "source_sha256": _sha256(source),
    }


def direct_ss_profiles() -> list[dict]:
    rows = []
    for model in DIRECT_SS_MODELS:
        path = PRIOR / "results" / "grid" / model / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        d = json.loads(path.read_text())
        values = {}
        for key, value in d["per_layer"].items():
            values[int(key)] = float(value["lfs"]["lfs"])
        rows.append(_profile_row(model, "direct_ss", values, path))
    return rows


def a9_reml_profiles() -> list[dict]:
    dirs = sorted(RUNS.glob("new_readings_*"))
    if len(dirs) != 19:
        raise RuntimeError(f"expected 19 successful A9 directories, got {len(dirs)}")
    rows = []
    for directory in dirs:
        path = directory / "result.json"
        d = json.loads(path.read_text())
        values = {int(k): float(v) for k, v in d["depth_profile"].items()}
        rows.append(_profile_row(d["tag"], "reml_lfs_vc", values, path))
    return rows


def hub_audit() -> list[dict]:
    rows = []
    for model in DIRECT_SS_MODELS:
        path = PRIOR / "results" / "grid" / model / "metrics.json"
        d = json.loads(path.read_text())
        hub = d.get("hub_at_best_layer", {})
        values = [float(v["latent_advantage"]) for v in hub.values()
                  if np.isfinite(float(v["latent_advantage"]))]
        rows.append({
            "model": model,
            "n_languages": len(values),
            "latent_beats_english": int(sum(v > 0 for v in values)),
            "fraction": float(np.mean(np.asarray(values) > 0)) if values else None,
            "median_advantage": float(np.median(values)) if values else None,
            "source": str(path),
        })
    return rows


def estimator_parity() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import tournament  # noqa: PLC0415

    tournament.PRIOR = str(PRIOR)
    frame = tournament.load_frame()
    per_model = (frame[["model", "legacy_lfs", "q_L"]]
                 .dropna().groupby("model", as_index=False).first())
    if len(per_model) != 14:
        raise RuntimeError(f"expected 14 same-support models, got {len(per_model)}")
    direct_concept = 1.0 - per_model["legacy_lfs"].to_numpy(float)
    reml_concept = per_model["q_L"].to_numpy(float)
    return {
        "n_models": int(len(per_model)),
        "spearman": float(stats.spearmanr(direct_concept, reml_concept).statistic),
        "pearson": float(stats.pearsonr(direct_concept, reml_concept).statistic),
        "max_absolute_value_difference": float(
            np.max(np.abs(direct_concept - reml_concept))
        ),
        "models": per_model["model"].tolist(),
    }


def _profile_summary(frame: pd.DataFrame, estimator: str) -> dict:
    d = frame[frame.estimator == estimator]
    return {
        "n_models": int(len(d)),
        "n_interior_minima": int(d.interior_minimum.sum()),
        "n_both_endpoints_above_minimum": int(
            d.both_endpoints_above_minimum.sum()
        ),
        "median_relative_min_depth": float(d.relative_min_depth.median()),
        "min_relative_min_depth": float(d.relative_min_depth.min()),
        "max_relative_min_depth": float(d.relative_min_depth.max()),
        "median_left_endpoint_lift": float(d.left_endpoint_lift.median()),
        "median_right_endpoint_lift": float(d.right_endpoint_lift.median()),
        "smallest_left_endpoint_lift": float(d.left_endpoint_lift.min()),
        "smallest_right_endpoint_lift": float(d.right_endpoint_lift.min()),
    }


def main() -> None:
    if not PRIOR.exists():
        raise FileNotFoundError(PRIOR)
    TABLES.mkdir(parents=True, exist_ok=True)
    profile = pd.DataFrame(direct_ss_profiles() + a9_reml_profiles())
    hub = pd.DataFrame(hub_audit())
    profile.to_csv(TABLES / "lfs_depth_profile_audit.csv", index=False,
                   float_format="%.8f")
    hub.to_csv(TABLES / "lfs_hub_audit.csv", index=False,
               float_format="%.8f")
    summary = {
        "analysis_status": "retrospective artifact audit; no new outcome claim",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "multilingual_metrics_root": str(PRIOR.resolve()),
        "important_interpretation": (
            "The 17-model direct-SS cohort and 19-model A9 REML LFS-VC cohort "
            "are distinct estimator cohorts and must not be combined into a "
            "single direct-SS pass count. Three models appear in both cohorts."
        ),
        "direct_ss": _profile_summary(profile, "direct_ss"),
        "a9_reml_lfs_vc": _profile_summary(profile, "reml_lfs_vc"),
        "overlap_models": sorted(set(
            profile.loc[profile.estimator == "direct_ss", "model"]
        ) & set(profile.loc[profile.estimator == "reml_lfs_vc", "model"])),
        "same_support_estimator_parity": estimator_parity(),
        "hub_rows": int(len(hub)),
        "outputs": {
            "depth_profiles": str(TABLES / "lfs_depth_profile_audit.csv"),
            "hub": str(TABLES / "lfs_hub_audit.csv"),
        },
    }
    out = TABLES / "lfs_headline_audit.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nLOWEST ENDPOINT RECOVERIES")
    cols = ["model", "estimator", "relative_min_depth",
            "left_endpoint_lift", "right_endpoint_lift"]
    print(profile.sort_values(
        ["estimator", "right_endpoint_lift"]
    )[cols].groupby("estimator").head(5).to_string(index=False))
    print("\nHUB COUNTS")
    print(hub[["model", "latent_beats_english", "n_languages",
               "median_advantage"]].to_string(index=False))


if __name__ == "__main__":
    main()
