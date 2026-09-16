"""Retrospective Aim 1 analysis of Profile 0.2 and coordinate-axis audits.

This script implements the frozen analysis contract in
docs/AIM1_RETROSPECTIVE_PROFILE_AXES_ANALYSIS_2026-08-24.md.  It is deliberately
descriptive: the outcomes were already visible before this analysis was frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_HASHES = {
    "profile_by_checkpoint.csv": "f3b59e34d27f56bdc68bfd8b972204acaeac5131062fc7817f7c846d5cd0ef0c",
    "profile_with_outcomes.csv": "87be72cdaea47a96c47cd22601c52ceb14423564303c5d71e506d6e36f37525c",
    "layers.csv": "f426560d8166de22cb0a2277fc0b7df58f371dc7ea0d9971fd56152bbfe921d4",
    "axes.csv": "f1890a9d95c80df0445a3e89b96e1a852e4b75b69d69bc78f713f6b402c34f2e",
}

PROFILE_METRICS = [
    "R1_concept_dominance",
    "R2_mean_alignment_margin",
    "R3_weak_language_tail",
    "R4_within_concept_ratio",
    "R4_shared_concept_ratio",
    "R4_effective_rank_ratio",
    "lfs",
    "effective_rank",
]

ASSOCIATION_PREDICTORS = {
    "R1_concept_dominance": "paired_delta",
    "R2_mean_alignment_margin": "paired_delta",
    "R3_weak_language_tail": "paired_delta",
    "R4_within_concept_ratio": "absolute",
    "R4_shared_concept_ratio": "absolute",
    "R4_effective_rank_ratio": "absolute",
    "lfs": "paired_delta",
}

OUTCOME_TARGETS = {
    "delta_mexa_vs_control": "convergent_representation",
    "delta_aar10_vs_control": "convergent_representation",
    "ppl_favorable_vs_control": "task_facing_perplexity",
    "delta_context_trained_vs_control": "behavioral_health_inversion_risk",
    "delta_context_heldout_vs_control": "behavioral_health_inversion_risk",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(profile_dir: Path, axis_dir: Path) -> dict[str, str]:
    resolved = {
        "profile_by_checkpoint.csv": profile_dir / "profile_by_checkpoint.csv",
        "profile_with_outcomes.csv": profile_dir / "profile_with_outcomes.csv",
        "layers.csv": axis_dir / "layers.csv",
        "axes.csv": axis_dir / "axes.csv",
    }
    observed = {name: sha256(path) for name, path in resolved.items()}
    mismatches = {
        name: {"expected": EXPECTED_HASHES[name], "observed": value}
        for name, value in observed.items()
        if value != EXPECTED_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"Frozen-input hash mismatch: {json.dumps(mismatches, indent=2)}")
    return observed


def add_paired_control_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["is_control"] = frame["is_control"].astype(int)
    controls = frame.loc[frame["is_control"].eq(1)]
    counts = controls.groupby(["study", "seed"]).size()
    if not counts.eq(1).all() or len(counts) != frame.groupby(["study", "seed"]).ngroups:
        raise ValueError("Every (study, seed) cell must have exactly one matched control")

    control_columns = ["study", "seed", "arm", *PROFILE_METRICS]
    controls = controls[control_columns].rename(
        columns={"arm": "control_arm", **{m: f"control_{m}" for m in PROFILE_METRICS}}
    )
    merged = frame.merge(controls, on=["study", "seed"], how="left", validate="many_to_one")
    for metric in PROFILE_METRICS:
        merged[f"delta_{metric}_vs_control"] = merged[metric] - merged[f"control_{metric}"]
    ratio = pd.to_numeric(merged["ppl_geomean_ratio_vs_control"], errors="coerce")
    if (ratio.dropna() <= 0).any():
        raise ValueError("Perplexity ratios must be positive")
    merged["ppl_favorable_vs_control"] = -np.log(ratio)
    return merged


def _t_summary(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else np.nan
    if n > 1:
        radius = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        low, high = mean - radius, mean + radius
    else:
        low = high = np.nan
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": low,
        "ci95_high": high,
    }


def summarize_paired_arms(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    delta_metrics = [f"delta_{m}_vs_control" for m in PROFILE_METRICS]
    outcome_metrics = list(OUTCOME_TARGETS)
    raw_health = [
        "R4_within_concept_ratio",
        "R4_shared_concept_ratio",
        "R4_effective_rank_ratio",
    ]
    for (study, arm), group in paired.groupby(["study", "arm"], sort=True):
        for metric in [*delta_metrics, *outcome_metrics, *raw_health]:
            summary = _t_summary(pd.to_numeric(group[metric], errors="coerce"))
            rows.append({
                "study": study,
                "arm": arm,
                "is_control": int(group["is_control"].iloc[0]),
                "control_arm": group["control_arm"].iloc[0],
                "metric": metric,
                **summary,
            })
    return pd.DataFrame(rows)


def make_arm_level(paired: pd.DataFrame) -> pd.DataFrame:
    numeric = paired.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"seed", "is_control"}
    aggregate = [column for column in numeric if column not in excluded]
    arm = paired.groupby(["study", "arm"], as_index=False)[aggregate].mean()
    metadata = paired.groupby(["study", "arm"], as_index=False).agg(
        is_control=("is_control", "first"),
        control_arm=("control_arm", "first"),
        n_seeds=("seed", "nunique"),
        source_checkpoint_count=("checkpoint", "nunique"),
    )
    checkpoint_sets = (
        paired.groupby(["study", "arm"])["checkpoint"]
        .agg(lambda values: "|".join(sorted(set(map(str, values)))))
        .rename("source_checkpoints")
        .reset_index()
    )
    result = metadata.merge(checkpoint_sets, on=["study", "arm"], validate="one_to_one")
    result = result.merge(arm, on=["study", "arm"], validate="one_to_one")
    result["analysis_unit_key"] = np.where(
        result["source_checkpoint_count"].eq(1),
        result["source_checkpoints"],
        result["study"].astype(str) + ":" + result["arm"].astype(str),
    )
    return result


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return len(valid), np.nan
    return len(valid), float(stats.spearmanr(valid["x"], valid["y"]).statistic)


def compute_associations(arm_level: pd.DataFrame) -> pd.DataFrame:
    experiments = arm_level.loc[arm_level["is_control"].eq(0)].copy()
    rows: list[dict[str, object]] = []
    studies = sorted(experiments["study"].unique())
    for predictor, mode in ASSOCIATION_PREDICTORS.items():
        predictor_column = (
            f"delta_{predictor}_vs_control" if mode == "paired_delta" else predictor
        )
        for outcome, evidence_class in OUTCOME_TARGETS.items():
            work = experiments[["study", predictor_column, outcome]].dropna().copy()
            work["analysis_unit_key"] = experiments.loc[work.index, "analysis_unit_key"]
            work["x_centered"] = work[predictor_column] - work.groupby("study")[predictor_column].transform("mean")
            work["y_centered"] = work[outcome] - work.groupby("study")[outcome].transform("mean")
            # Center within every complete study first. Repeated frozen-base rows
            # are then averaged into one pooled analysis unit rather than counted
            # as independent observations.
            pooled_units = work.groupby("analysis_unit_key", as_index=False)[
                ["x_centered", "y_centered"]
            ].mean()
            n, rho = _safe_spearman(pooled_units["x_centered"], pooled_units["y_centered"])
            rows.append({
                "predictor": predictor,
                "predictor_mode": mode,
                "outcome": outcome,
                "evidence_class": evidence_class,
                "scope": "pooled_within_study_centered",
                "held_out_study": "",
                "n_arms": n,
                "spearman_rho": rho,
            })
            for study in studies:
                subset = work.loc[work["study"].eq(study)]
                n, rho = _safe_spearman(subset[predictor_column], subset[outcome])
                rows.append({
                    "predictor": predictor,
                    "predictor_mode": mode,
                    "outcome": outcome,
                    "evidence_class": evidence_class,
                    "scope": f"study:{study}",
                    "held_out_study": "",
                    "n_arms": n,
                    "spearman_rho": rho,
                })
            for held_out in studies:
                subset = work.loc[work["study"].ne(held_out)].copy()
                subset["x_centered"] = subset[predictor_column] - subset.groupby("study")[predictor_column].transform("mean")
                subset["y_centered"] = subset[outcome] - subset.groupby("study")[outcome].transform("mean")
                subset_units = subset.groupby("analysis_unit_key", as_index=False)[
                    ["x_centered", "y_centered"]
                ].mean()
                n, rho = _safe_spearman(subset_units["x_centered"], subset_units["y_centered"])
                rows.append({
                    "predictor": predictor,
                    "predictor_mode": mode,
                    "outcome": outcome,
                    "evidence_class": evidence_class,
                    "scope": "leave_one_study_out",
                    "held_out_study": held_out,
                    "n_arms": n,
                    "spearman_rho": rho,
                })
    return pd.DataFrame(rows)


def gini_nonnegative(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total == 0:
        return 0.0
    arr.sort()
    n = arr.size
    return float((2.0 * np.dot(np.arange(1, n + 1), arr) / (n * total)) - (n + 1) / n)


def summarize_axis_layers(axes: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, layer), group in axes.groupby(["model", "layer"], sort=True):
        share = pd.to_numeric(group["language_share"], errors="coerce")
        raw_var = pd.to_numeric(group["raw_variance"], errors="coerce")
        ss_lang = pd.to_numeric(group["ss_language"], errors="coerce")
        top5 = group["top5_variance"].astype(str).str.lower().eq("true")
        _, share_variance_rho = _safe_spearman(share, raw_var)
        top5_ss_fraction = float(ss_lang[top5].sum() / ss_lang.sum()) if ss_lang.sum() > 0 else np.nan
        rows.append({
            "model": model,
            "layer": int(layer),
            "n_axes": int(len(group)),
            "share_q05": float(share.quantile(0.05)),
            "share_q25": float(share.quantile(0.25)),
            "share_q50": float(share.quantile(0.50)),
            "share_q75": float(share.quantile(0.75)),
            "share_q95": float(share.quantile(0.95)),
            "fraction_share_gt_0_50": float((share > 0.50).mean()),
            "fraction_share_gt_0_75": float((share > 0.75).mean()),
            "fraction_share_gt_0_90": float((share > 0.90).mean()),
            "fraction_language_heavy_by": float(group["language_heavy_by"].astype(str).str.lower().eq("true").mean()),
            "gini_ss_language": gini_nonnegative(ss_lang),
            "top5_raw_variance_ss_language_fraction": top5_ss_fraction,
            "spearman_language_share_vs_raw_variance": share_variance_rho,
            "median_share_top5_raw_variance": float(share[top5].median()),
            "median_share_rest": float(share[~top5].median()),
            "median_share_top5_minus_rest": float(share[top5].median() - share[~top5].median()),
        })
    result = pd.DataFrame(rows)
    layer_columns = [
        "model", "layer", "n_languages", "n_concepts", "dimension",
        "lfs_direct_standardized", "mean_pairwise_cosine_raw", "top_eigen_share_raw",
        "gaussian_null_mean",
    ]
    return result.merge(layers[layer_columns], on=["model", "layer"], validate="one_to_one")


def summarize_model_minima(layer_summary: pd.DataFrame) -> pd.DataFrame:
    indices = layer_summary.groupby("model")["lfs_direct_standardized"].idxmin()
    columns = [
        "model", "layer", "lfs_direct_standardized", "share_q50",
        "fraction_language_heavy_by", "gini_ss_language",
        "top5_raw_variance_ss_language_fraction",
        "spearman_language_share_vs_raw_variance",
    ]
    return layer_summary.loc[indices, columns].sort_values("model").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--axis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    hashes = verify_inputs(args.profile_dir, args.axis_dir)
    profiles = pd.read_csv(args.profile_dir / "profile_by_checkpoint.csv")
    outcomes = pd.read_csv(args.profile_dir / "profile_with_outcomes.csv")
    layers = pd.read_csv(args.axis_dir / "layers.csv")
    axes = pd.read_csv(args.axis_dir / "axes.csv")

    if len(profiles) != 81 or len(outcomes) != 72:
        raise ValueError(f"Expected 81 profiles and 72 joined rows; got {len(profiles)} and {len(outcomes)}")
    if len(layers) != 69 or len(axes) != 195072:
        raise ValueError(f"Expected 69 layers and 195072 axes; got {len(layers)} and {len(axes)}")

    paired = add_paired_control_deltas(outcomes)
    paired_summary = summarize_paired_arms(paired)
    arm_level = make_arm_level(paired)
    associations = compute_associations(arm_level)

    r4_columns = [
        "study", "seed", "arm", "is_control", "R4_within_concept_ratio",
        "R4_shared_concept_ratio", "R4_effective_rank_ratio",
        "R4_passes_legacy_cvp_gate", "lfs", "effective_rank",
    ]
    frozen_r4 = paired.loc[
        paired["study"].eq("wordalign") & paired["arm"].isin(["WA-B", "WA-C", "WA-G"]),
        r4_columns,
    ].sort_values(["arm", "seed"])

    joined_checkpoints = set(outcomes["checkpoint"].astype(str))
    missing = profiles.loc[~profiles["checkpoint"].astype(str).isin(joined_checkpoints)].copy()

    layer_summary = summarize_axis_layers(axes, layers)
    model_minimum = summarize_model_minima(layer_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_summary.to_csv(args.output_dir / "profile_paired_arm_summary.csv", index=False)
    arm_level.to_csv(args.output_dir / "profile_arm_level.csv", index=False)
    associations.to_csv(args.output_dir / "profile_associations.csv", index=False)
    frozen_r4.to_csv(args.output_dir / "profile_r4_frozen_contrast.csv", index=False)
    missing.to_csv(args.output_dir / "profile_missing_outcomes.csv", index=False)
    layer_summary.to_csv(args.output_dir / "axis_layer_summary.csv", index=False)
    model_minimum.to_csv(args.output_dir / "axis_model_minimum_summary.csv", index=False)

    summary = {
        "analysis_status": "retrospective_descriptive",
        "input_sha256": hashes,
        "counts": {
            "checkpoint_profiles": int(len(profiles)),
            "outcome_join_rows": int(len(outcomes)),
            "profiles_without_joined_outcomes": int(len(missing)),
            "studies": int(outcomes["study"].nunique()),
            "study_seed_cells": int(outcomes.groupby(["study", "seed"]).ngroups),
            "noncontrol_arm_rows_before_checkpoint_deduplication": int(arm_level["is_control"].eq(0).sum()),
            "independent_noncontrol_analysis_units": int(
                arm_level.loc[arm_level["is_control"].eq(0), "analysis_unit_key"].nunique()
            ),
            "axis_models": int(layers["model"].nunique()),
            "axis_layers": int(len(layers)),
            "axes": int(len(axes)),
        },
        "r4_frozen_contrast": {
            arm: {
                "n": int(len(group)),
                "pass_count": int(group["R4_passes_legacy_cvp_gate"].astype(str).str.lower().eq("true").sum()),
                "within_ratio_mean": float(group["R4_within_concept_ratio"].mean()),
                "shared_ratio_mean": float(group["R4_shared_concept_ratio"].mean()),
                "effective_rank_ratio_mean": float(group["R4_effective_rank_ratio"].mean()),
            }
            for arm, group in frozen_r4.groupby("arm")
        },
        "axis_descriptives": {
            "median_layer_fraction_language_heavy_by": float(layer_summary["fraction_language_heavy_by"].median()),
            "minimum_layer_fraction_language_heavy_by": float(layer_summary["fraction_language_heavy_by"].min()),
            "median_layer_language_share": float(layer_summary["share_q50"].median()),
            "median_layer_gini_ss_language": float(layer_summary["gini_ss_language"].median()),
            "median_layer_top5_raw_variance_ss_language_fraction": float(layer_summary["top5_raw_variance_ss_language_fraction"].median()),
            "median_layer_spearman_share_vs_raw_variance": float(layer_summary["spearman_language_share_vs_raw_variance"].median()),
            "layers_with_all_axes_by_significant": int(layer_summary["fraction_language_heavy_by"].eq(1.0).sum()),
        },
        "interpretive_guards": [
            "No composite score was formed.",
            "MEXA and AaR are convergent representation comparisons, not downstream-task outcomes.",
            "The joined table contains no primary Belebele accuracy, so this analysis cannot establish downstream validity.",
            "The coordinate-axis audit is basis dependent and does not identify causal neurons.",
            "The Gaussian Beta reference is exact only under the stated i.i.d. isotropic null.",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
