#!/usr/bin/env python3
"""Audit completeness and join RMFS Profile 0.2 arm results to saved outcomes.

The output is a transparent analysis table, not a composite score.  Existing
outcome rows are copied without modification and each profile field keeps its
own column.  Checkpoints lacking a downstream row remain in the completeness
manifest instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


CHECKPOINT_RE = re.compile(r"^ckpt_arm(?P<arm>.+)_seed(?P<seed>[0-9]+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_profile(
    path: Path,
    *,
    arm_override: str | None = None,
    seed_override: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = payload["model_or_checkpoint"]
    if arm_override is None:
        matched = CHECKPOINT_RE.match(checkpoint)
        if not matched:
            raise ValueError(f"unrecognized checkpoint name: {checkpoint}")
        arm = matched.group("arm")
        seed = matched.group("seed")
    else:
        if seed_override is None:
            raise ValueError("seed_override is required with arm_override")
        arm = arm_override
        seed = seed_override
    profile = payload["profile"]
    readings = profile["readings"]
    preservation = readings["preservation"]
    decomposition = profile["supporting"]["decomposition"]
    geometry = profile["supporting"]["raw_geometry"]
    return {
        "checkpoint": checkpoint,
        "arm": arm,
        "seed": seed,
        "profile_result_sha256": sha256(path),
        "reference_grid_sha256": payload["reference_grid_sha256"],
        "R1_concept_dominance": readings["concept_dominance"],
        "R2_mean_alignment_margin": readings["mean_alignment_margin"],
        "R3_weak_language_tail": readings["weak_language_tail"],
        "R4_within_concept_ratio": preservation["within_concept_ratio"],
        "R4_shared_concept_ratio": preservation["shared_concept_ratio"],
        "R4_effective_rank_ratio": preservation["effective_rank_ratio"],
        "R4_passes_legacy_cvp_gate": preservation["passes_legacy_cvp_gate"],
        "lfs": decomposition["lfs"],
        "ss_language": decomposition["ss_language"],
        "ss_concept": decomposition["ss_concept"],
        "ss_residual": decomposition["ss_residual"],
        "language_share_total": decomposition["language_share_total"],
        "concept_share_total": decomposition["concept_share_total"],
        "residual_share_total": decomposition["residual_share_total"],
        "within_concept_spread": geometry["within_concept_spread"],
        "shared_concept_energy": geometry["shared_concept_energy"],
        "language_energy": geometry["language_energy"],
        "residual_energy": geometry["residual_energy"],
        "mean_vector_norm": geometry["mean_vector_norm"],
        "effective_rank": geometry["effective_rank"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--frozen-arm", action="append", default=[])
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoints", type=int, default=75)
    args = parser.parse_args()

    result_paths = sorted(args.profiles_root.glob("*/result.json"))
    profiles = [flatten_profile(path) for path in result_paths]
    by_key = {(row["arm"], row["seed"]): row for row in profiles}
    if len(by_key) != len(profiles):
        raise ValueError("duplicate arm-seed profile keys")

    with args.outcomes.open(encoding="utf-8", newline="") as stream:
        outcomes = list(csv.DictReader(stream))
    outcome_fields = list(outcomes[0]) if outcomes else []

    if not args.reference_profile.is_file():
        raise FileNotFoundError(f"reference profile missing: {args.reference_profile}")
    frozen_arms = set(args.frozen_arm)
    frozen_keys = {
        (row["arm"], row["seed"])
        for row in outcomes
        if row["arm"] in frozen_arms
    }
    frozen_profiles = {
        key: flatten_profile(
            args.reference_profile,
            arm_override=key[0],
            seed_override=key[1],
        )
        for key in frozen_keys
    }

    joined: list[dict[str, Any]] = []
    missing_outcome_profiles = []
    used_profile_keys = set()
    for outcome in outcomes:
        key = (outcome["arm"], outcome["seed"])
        profile = by_key.get(key, frozen_profiles.get(key))
        if profile is None:
            missing_outcome_profiles.append({"arm": key[0], "seed": key[1]})
            continue
        used_profile_keys.add(key)
        joined.append({**outcome, **profile})

    unmatched_profiles = [
        row["checkpoint"]
        for key, row in sorted(by_key.items())
        if key not in used_profile_keys
    ]
    reference_hashes = sorted({row["reference_grid_sha256"] for row in profiles})
    completeness = {
        "analysis_status": "retrospective Aim 2 checkpoint audit",
        "expected_checkpoint_profiles": args.expected_checkpoints,
        "observed_checkpoint_profiles": len(profiles),
        "complete": len(profiles) == args.expected_checkpoints,
        "reference_grid_hashes": reference_hashes,
        "one_reference_grid": len(reference_hashes) == 1,
        "frozen_reference_profile": str(args.reference_profile.resolve()),
        "frozen_arms": sorted(frozen_arms),
        "frozen_outcome_rows_joined": len(frozen_keys),
        "outcome_rows": len(outcomes),
        "joined_rows": len(joined),
        "outcome_rows_missing_profile": missing_outcome_profiles,
        "profiles_without_outcome_row": unmatched_profiles,
        "outcome_file": str(args.outcomes.resolve()),
        "outcome_sha256": sha256(args.outcomes),
        "composite_score_computed": False,
    }
    if not completeness["complete"]:
        raise RuntimeError(json.dumps(completeness, indent=2))
    if not completeness["one_reference_grid"]:
        raise RuntimeError(json.dumps(completeness, indent=2))
    if missing_outcome_profiles:
        raise RuntimeError(json.dumps(completeness, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_fields = list(profiles[0]) if profiles else []
    joined_fields = outcome_fields + [field for field in profile_fields if field not in outcome_fields]
    write_csv(args.output_dir / "profile_by_checkpoint.csv", profiles, profile_fields)
    write_csv(args.output_dir / "profile_with_outcomes.csv", joined, joined_fields)
    (args.output_dir / "completeness.json").write_text(
        json.dumps(completeness, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completeness, indent=2))


if __name__ == "__main__":
    main()
