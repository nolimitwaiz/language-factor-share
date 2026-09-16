#!/usr/bin/env python3
"""Completion-gated assembly and preregistered analysis for Aim 1 R1.

This program must run only after every element of the frozen 19-model array
has terminated.  It validates the scheduler state and every sealed artifact
before reading scientific outcomes, recomputes the content-transfer readings
from the per-example NLLs, and implements the uncertainty analyses frozen in
the 2026-08-24 R1 addendum.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_MODEL_LOCK_SHA256 = "dcb3c1fbaffa8cca70a4090b3b62d269a5daddc43779e62a9e2cdede8679930c"
EXPECTED_PAIR_SHA256 = "986d4f3a8730b7e8087ebf0d558de72e8735aba3b4fb5493a3f1e04b42bc1112"
EXPECTED_DATA_MANIFEST_SHA256 = "7c62075cdda2cecd4dfb3c3e06664c6fc9eae17e487eb4ef9c01be5a41d05fcd"
EXPECTED_MODELS = 19
EXPECTED_PAIRS = 100
EXPECTED_LANGUAGES = 41
EXPECTED_ROWS_PER_MODEL = EXPECTED_PAIRS * (1 + 2 * EXPECTED_LANGUAGES)
MIN_RATIO_LANGUAGES = 32
SEED = 13
ACCOUNTING_START = "2026-08-24"

N2F = {
    "deu": "deu_Latn", "fra": "fra_Latn", "spa": "spa_Latn",
    "por": "por_Latn", "ita": "ita_Latn", "nld": "nld_Latn",
    "swe": "swe_Latn", "pol": "pol_Latn", "ces": "ces_Latn",
    "ron": "ron_Latn", "rus": "rus_Cyrl", "ukr": "ukr_Cyrl",
    "ell": "ell_Grek", "tur": "tur_Latn", "vie": "vie_Latn",
    "ind": "ind_Latn", "zho-CN": "zho_Hans", "jpn": "jpn_Jpan",
    "kor": "kor_Hang", "arb": "arb_Arab", "heb": "heb_Hebr",
    "fas": "pes_Arab", "hin": "hin_Deva", "ben": "ben_Beng",
    "tam": "tam_Taml", "tel": "tel_Telu", "mar": "mar_Deva",
    "urd": "urd_Arab", "tha": "tha_Thai", "khm": "khm_Khmr",
    "mya": "mya_Mymr", "swa": "swh_Latn", "amh": "amh_Ethi",
    "som": "som_Latn", "hau": "hau_Latn", "zul": "zul_Latn",
    "kat": "kat_Geor", "hye": "hye_Armn", "aze-Latn": "azj_Latn",
    "uzb": "uzn_Latn",
}

TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}
REPAIR_INDICES = frozenset({7, 8, 9, 10, 11, 12, 15, 16, 17, 18})
FINAL_REPAIR_INDICES = frozenset({10, 18})
FIRST_REPAIR_TAGS = frozenset({
    "mGPT", "OLMo-2-1124-13B", "granite-3.1-8b-base", "TowerBase-7B",
    "occiglot-7b-eu5", "Llama-3.1-8B", "SmolLM2-360M",
    "Falcon3-3B-Base",
})
FINAL_REPAIR_TAGS = frozenset({"Yi-1.5-9B", "Mistral-Nemo-Base-2407"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, observed {observed}")
    return observed


def load_lock(path: Path) -> list[dict[str, str]]:
    verify_hash(path, EXPECTED_MODEL_LOCK_SHA256, "model lock")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != EXPECTED_MODELS or len({row["tag"] for row in rows}) != EXPECTED_MODELS:
        raise ValueError("model lock must contain 19 unique tags")
    return rows


def read_array_states(job_id: str) -> dict[int, str]:
    output = subprocess.run(
        [
            "sacct", "-X", "-S", ACCOUNTING_START, "-j", job_id,
            "--format=JobID,State", "-n", "-P",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    states: dict[int, str] = {}
    pattern = re.compile(rf"^{re.escape(job_id)}_(\d+)$")
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 2:
            continue
        match = pattern.match(fields[0])
        if match:
            states[int(match.group(1))] = fields[1].split("+")[0]
    return states


def verify_array_terminal(
    job_id: str,
    expected_indices: set[int] | frozenset[int],
    allowed_failed_indices: set[int] | frozenset[int] = frozenset(),
    attempts: int = 30,
    retry_seconds: int = 10,
) -> dict[str, str]:
    expected = set(expected_indices)
    states: dict[int, str] = {}
    for attempt in range(attempts):
        states = read_array_states(job_id)
        present = expected.intersection(states)
        complete_view = present == expected and all(
            states[index] in TERMINAL_STATES for index in expected
        )
        if complete_view:
            break
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    missing = sorted(expected - set(states))
    if missing:
        raise RuntimeError(
            f"reporting gate closed for {job_id}: missing array states {missing}"
        )
    nonterminal = {
        index: states[index]
        for index in sorted(expected)
        if states[index] not in TERMINAL_STATES
    }
    if nonterminal:
        raise RuntimeError(
            f"reporting gate closed for {job_id}: nonterminal array tasks {nonterminal}"
        )
    unsuccessful = {
        index: states[index]
        for index in sorted(expected)
        if states[index] != "COMPLETED" and index not in allowed_failed_indices
    }
    if unsuccessful:
        raise RuntimeError(
            f"{job_id}: undocumented technical failures block assembly: {unsuccessful}"
        )
    return {str(index): states[index] for index in sorted(expected)}


def _require_manifest(manifest: dict[str, Any], lock: dict[str, str]) -> None:
    expected = {
        "tag": lock["tag"],
        "hf_id": lock["hf_id"],
        "revision": lock["revision"],
        "model_family": lock["model_family"],
        "smoke": False,
        "n_pairs": EXPECTED_PAIRS,
        "n_rows": EXPECTED_ROWS_PER_MODEL,
        "expected_rows": EXPECTED_ROWS_PER_MODEL,
        "all_nll_finite": True,
        "model_lock_sha256": EXPECTED_MODEL_LOCK_SHA256,
        "pairs_sha256": EXPECTED_PAIR_SHA256,
        "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "outcomes_printed_to_log": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{lock['tag']} manifest {key}: expected {value!r}, "
                f"found {manifest.get(key)!r}"
            )
    if len(manifest.get("languages", [])) != EXPECTED_LANGUAGES:
        raise ValueError(f"{lock['tag']}: expected 41 languages")
    expected_ceiling = (
        2048 if lock["tag"] in FINAL_REPAIR_TAGS
        else 1024 if lock["tag"] in FIRST_REPAIR_TAGS
        else 512
    )
    observed_ceiling = int(manifest.get("max_sequence_tokens", 512))
    if observed_ceiling != expected_ceiling:
        raise ValueError(
            f"{lock['tag']}: expected {expected_ceiling}-token provenance, "
            f"found {observed_ceiling}"
        )


def verify_stored_summary(
    frame: pd.DataFrame, stored: dict[str, Any], languages: list[str], tag: str
) -> None:
    """Cross-check the scorer's sealed summary against the raw NLL rows."""
    none = frame[frame.condition == "none"].mean_target_nll.to_numpy(float)
    if len(none) != EXPECTED_PAIRS:
        raise ValueError(f"{tag}: stored-summary check found {len(none)} no-context rows")
    base = float(none.mean())
    if not np.isclose(base, float(stored["base_nll"]), rtol=0, atol=1e-10):
        raise ValueError(f"{tag}: stored base NLL disagrees with per-example rows")
    for language in languages:
        matched = frame[
            (frame.language == language) & (frame.condition == "matched")
        ].mean_target_nll.to_numpy(float)
        mismatched = frame[
            (frame.language == language) & (frame.condition == "mismatched")
        ].mean_target_nll.to_numpy(float)
        if len(matched) != EXPECTED_PAIRS or len(mismatched) != EXPECTED_PAIRS:
            raise ValueError(f"{tag}/{language}: incomplete stored-summary cells")
        observed = {
            "delta_matched": float(base - matched.mean()),
            "delta_mismatched": float(base - mismatched.mean()),
            "content": float((mismatched - matched).mean()),
        }
        sealed = stored["per_language"][language]
        for key, value in observed.items():
            if not np.isclose(value, float(sealed[key]), rtol=0, atol=1e-10):
                raise ValueError(f"{tag}/{language}: sealed {key} disagrees with raw rows")


def load_parts(parts_root: Path, lock_rows: list[dict[str, str]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for lock in lock_rows:
        directory = parts_root / lock["tag"]
        paths = {
            "manifest": directory / "manifest.json",
            "examples": directory / "per_example.csv",
            "summary": directory / "summary.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{lock['tag']}: missing sealed artifacts {missing}")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        _require_manifest(manifest, lock)
        if sha256(paths["examples"]) != manifest["per_example_sha256"]:
            raise ValueError(f"{lock['tag']}: per-example hash mismatch")
        if sha256(paths["summary"]) != manifest["summary_sha256"]:
            raise ValueError(f"{lock['tag']}: summary hash mismatch")
        frame = pd.read_csv(paths["examples"])
        if len(frame) != EXPECTED_ROWS_PER_MODEL:
            raise ValueError(f"{lock['tag']}: unexpected per-example row count")
        if not np.isfinite(frame["mean_target_nll"].to_numpy(float)).all():
            raise ValueError(f"{lock['tag']}: non-finite per-example NLL")
        stored_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        verify_stored_summary(frame, stored_summary, manifest["languages"], lock["tag"])
        frame["model"] = lock["tag"]
        frame["model_family"] = lock["model_family"]
        frames.append(frame)
        manifests.append(manifest)
    return pd.concat(frames, ignore_index=True), manifests


def _paired_cells(model_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    none = (
        model_rows[model_rows.condition == "none"]
        .set_index("pair_id")["mean_target_nll"]
        .sort_index()
    )
    if len(none) != EXPECTED_PAIRS or not none.index.is_unique:
        raise ValueError("no-context cells are not one per frozen pair")
    conditioned = model_rows[model_rows.condition != "none"].pivot(
        index=["language", "pair_id"], columns="condition", values="mean_target_nll"
    )
    if conditioned.shape != (EXPECTED_LANGUAGES * EXPECTED_PAIRS, 2):
        raise ValueError("conditioned cells do not form the frozen balanced grid")
    if set(conditioned.columns) != {"matched", "mismatched"}:
        raise ValueError("conditioned grid is missing matched or mismatched rows")
    return conditioned.reset_index(), none


def content_estimates(
    examples: pd.DataFrame, n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    estimates: list[dict[str, Any]] = []
    model_checks: list[dict[str, Any]] = []
    for (model, family), model_rows in examples.groupby(["model", "model_family"], sort=True):
        cells, none = _paired_cells(model_rows)
        by_language: dict[str, np.ndarray] = {}
        means: dict[str, dict[str, float]] = {}
        for language, group in cells.groupby("language", sort=False):
            group = group.sort_values("pair_id")
            if not np.array_equal(group.pair_id.to_numpy(), none.index.to_numpy()):
                raise ValueError(f"{model}/{language}: pair alignment failure")
            matched = group.matched.to_numpy(float)
            mismatched = group.mismatched.to_numpy(float)
            differences = mismatched - matched
            by_language[language] = differences
            means[language] = {
                "content": float(differences.mean()),
                "delta_matched": float((none.to_numpy(float) - matched).mean()),
                "delta_mismatched": float((none.to_numpy(float) - mismatched).mean()),
            }
        if len(by_language) != EXPECTED_LANGUAGES or "eng" not in by_language:
            raise ValueError(f"{model}: expected 41 languages including English")
        english_content = means["eng"]["content"]
        draws = rng.integers(0, EXPECTED_PAIRS, size=(n_boot, EXPECTED_PAIRS))
        english_boot = by_language["eng"][draws].mean(axis=1)
        ratio_count = 0
        for language, differences in by_language.items():
            raw_boot = differences[draws].mean(axis=1)
            raw_lo, raw_hi = np.percentile(raw_boot, [2.5, 97.5])
            ratio = means[language]["content"] / english_content if english_content > 0 else math.nan
            valid_draw = np.isfinite(english_boot) & (english_boot > 0)
            ratio_boot = raw_boot[valid_draw] / english_boot[valid_draw]
            if ratio_boot.size:
                ratio_lo, ratio_hi = np.percentile(ratio_boot, [2.5, 97.5])
            else:
                ratio_lo, ratio_hi = math.nan, math.nan
            ratio_valid = bool(np.isfinite(ratio))
            if language != "eng" and ratio_valid:
                ratio_count += 1
            estimates.append({
                "model": model,
                "model_family": family,
                "language": language,
                "flores_code": N2F.get(language),
                **means[language],
                "content_ci_lo": float(raw_lo),
                "content_ci_hi": float(raw_hi),
                "R_content": float(ratio) if ratio_valid else math.nan,
                "R_content_ci_lo": float(ratio_lo),
                "R_content_ci_hi": float(ratio_hi),
                "R_content_boot_valid_fraction": float(valid_draw.mean()),
                "n_pairs": EXPECTED_PAIRS,
                "n_boot": n_boot,
            })
        model_checks.append({
            "model": model,
            "model_family": family,
            "english_content": english_content,
            "english_content_positive": bool(np.isfinite(english_content) and english_content > 0),
            "valid_nonenglish_ratios": ratio_count,
            "ratio_valid_model": bool(english_content > 0 and ratio_count >= MIN_RATIO_LANGUAGES),
        })
    return pd.DataFrame(estimates), pd.DataFrame(model_checks)


def load_panel(path: Path, lock_rows: list[dict[str, str]]) -> pd.DataFrame:
    models = {row["tag"] for row in lock_rows}
    panel = pd.read_csv(path)
    need = {
        "model", "flores_code", "q_L", "mexa_shared", "aar_shared",
        "log_tokens", "macro_family_g", "script_g", "fertility",
    }
    missing = sorted(need - set(panel.columns))
    if missing:
        raise ValueError(f"expanded panel missing columns {missing}")
    panel = panel[panel.model.isin(models) & panel.q_L.notna()].copy()
    counts = panel.groupby("model").size()
    if set(counts.index) != models or not (counts == 40).all():
        raise ValueError(f"A9 panel is not 40 rows for all 19 models: {counts.to_dict()}")
    if panel.duplicated(["model", "flores_code"]).any():
        raise ValueError("A9 panel has duplicate model-language rows")
    for model, group in panel.groupby("model"):
        if group.q_L.nunique() != 1:
            raise ValueError(f"{model}: q_L must be one frozen model reading")
    return panel[list(need)].copy()


def design(frame: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = [np.ones(len(frame))]
    for name in ("log_tokens", "fertility"):
        columns.append(frame[name].to_numpy(float))
    for name in ("macro_family_g", "script_g"):
        dummy = pd.get_dummies(frame[name].astype(str), drop_first=True)
        columns.extend(dummy[column].to_numpy(float) for column in dummy.columns)
    return np.column_stack(columns)


def residualize(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    return values - matrix @ beta


def residual_spearman(frame: pd.DataFrame, candidate: str, target: str) -> float:
    matrix = design(frame)
    left = residualize(frame[candidate].to_numpy(float), matrix)
    right = residualize(frame[target].to_numpy(float), matrix)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return math.nan
    return float(stats.spearmanr(left, right).statistic)


def raw_spearman(frame: pd.DataFrame, candidate: str, target: str) -> float:
    return float(stats.spearmanr(frame[candidate], frame[target]).statistic)


def sampled_frame(
    frame: pd.DataFrame, axis: str, rng: np.random.Generator
) -> pd.DataFrame:
    if axis == "model":
        units = np.sort(frame.model.unique())
        groups = {unit: frame[frame.model == unit] for unit in units}
        picked = rng.choice(units, size=len(units), replace=True)
        return pd.concat([groups[unit] for unit in picked], ignore_index=True)
    if axis == "family":
        units = np.sort(frame.model_family.unique())
        groups = {unit: frame[frame.model_family == unit] for unit in units}
        picked = rng.choice(units, size=len(units), replace=True)
        return pd.concat([groups[unit] for unit in picked], ignore_index=True)
    if axis != "crossed":
        raise ValueError(axis)
    models = np.sort(frame.model.unique())
    languages = np.sort(frame.flores_code.unique())
    picked_models = rng.choice(models, size=len(models), replace=True)
    picked_languages = rng.choice(languages, size=len(languages), replace=True)
    cells = {
        (model, language): group
        for (model, language), group in frame.groupby(["model", "flores_code"])
    }
    pieces = [
        cells[(model, language)]
        for model in picked_models
        for language in picked_languages
        if (model, language) in cells
    ]
    if not pieces:
        raise RuntimeError("empty crossed-bootstrap sample")
    return pd.concat(pieces, ignore_index=True)


def interval(values: list[float]) -> tuple[float, float, float, int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return math.nan, math.nan, math.nan, 0
    lo, hi = np.percentile(finite, [2.5, 97.5])
    p = 2 * min(float((finite <= 0).mean()), float((finite >= 0).mean()))
    return float(lo), float(hi), float(min(1.0, max(p, 1 / len(finite)))), int(len(finite))


def target_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "R_content":
        return frame[frame.ratio_valid_model].copy()
    return frame.copy()


def association_table(frame: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    candidates = {
        "q_L": "project LFS concept-direction reading",
        "mexa_shared": "external MEXA comparator",
        "aar_shared": "external AaR comparator",
    }
    targets = ("R_content", "content")
    rows: list[dict[str, Any]] = []
    for target in targets:
        eligible = target_frame(frame, target)
        for candidate, role in candidates.items():
            common = eligible.dropna(subset=[candidate, target, "log_tokens", "fertility", "macro_family_g", "script_g"]).copy()
            point = residual_spearman(common, candidate, target)
            raw = raw_spearman(common, candidate, target)
            for axis_index, axis in enumerate(("model", "crossed", "family")):
                rng = np.random.default_rng(SEED + axis_index)
                values: list[float] = []
                for _ in range(n_boot):
                    try:
                        sample = sampled_frame(common, axis, rng)
                        values.append(residual_spearman(sample, candidate, target))
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                lo, hi, p, effective = interval(values)
                rows.append({
                    "candidate": candidate,
                    "candidate_role": role,
                    "target": target,
                    "bootstrap_axis": axis,
                    "rho_deflated": point,
                    "rho_raw": raw,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p_two_sided": p,
                    "n_rows": len(common),
                    "n_models": common.model.nunique(),
                    "n_model_families": common.model_family.nunique(),
                    "n_languages": common.flores_code.nunique(),
                    "n_boot_requested": n_boot,
                    "n_boot_effective": effective,
                })
    return pd.DataFrame(rows)


def model_level_tables(frame: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        frame.groupby(["model", "model_family"], as_index=False)
        .agg(q_L=("q_L", "first"), median_R_content=("R_content", "median"),
             median_content=("content", "median"), n_languages=("flores_code", "nunique"),
             ratio_valid_model=("ratio_valid_model", "first"))
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for target in ("median_R_content", "median_content"):
        eligible = summary[summary.ratio_valid_model] if target == "median_R_content" else summary
        eligible = eligible.dropna(subset=["q_L", target])
        point = float(stats.spearmanr(eligible.q_L, eligible[target]).statistic)
        values = []
        for _ in range(n_boot):
            sample = eligible.iloc[rng.integers(0, len(eligible), size=len(eligible))]
            values.append(float(stats.spearmanr(sample.q_L, sample[target]).statistic))
        lo, hi, p, effective = interval(values)
        rows.append({
            "candidate": "q_L", "target": target, "rho": point,
            "ci_lo": lo, "ci_hi": hi, "p_two_sided": p,
            "n_models": len(eligible), "n_boot_requested": n_boot,
            "n_boot_effective": effective,
        })
    return summary, pd.DataFrame(rows)


def leave_one_family_out(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for held_out in np.sort(frame.model_family.unique()):
        subset = frame[frame.model_family != held_out]
        for target in ("R_content", "content"):
            eligible = target_frame(subset, target)
            for candidate in ("q_L", "mexa_shared", "aar_shared"):
                common = eligible.dropna(subset=[candidate, target, "log_tokens", "fertility", "macro_family_g", "script_g"])
                rows.append({
                    "held_out_model_family": held_out,
                    "candidate": candidate,
                    "target": target,
                    "rho_deflated": residual_spearman(common, candidate, target),
                    "n_rows": len(common),
                    "n_models": common.model.nunique(),
                    "n_model_families": common.model_family.nunique(),
                    "n_languages": common.flores_code.nunique(),
                })
    return pd.DataFrame(rows)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def output_hashes(output: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256(output / name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-array-job-id", required=True)
    parser.add_argument("--repair-array-job-id", required=True)
    parser.add_argument("--final-repair-array-job-id", required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--expanded-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    original_states = verify_array_terminal(
        args.original_array_job_id, set(range(EXPECTED_MODELS)), REPAIR_INDICES
    )
    repair_states = verify_array_terminal(
        args.repair_array_job_id, REPAIR_INDICES, FINAL_REPAIR_INDICES
    )
    final_repair_states = verify_array_terminal(
        args.final_repair_array_job_id, FINAL_REPAIR_INDICES
    )
    states = {
        "original": original_states,
        "repair_1024": repair_states,
        "final_repair_2048": final_repair_states,
    }
    verify_hash(args.pairs, EXPECTED_PAIR_SHA256, "pair file")
    verify_hash(args.data_manifest, EXPECTED_DATA_MANIFEST_SHA256, "data manifest")
    lock_rows = load_lock(args.model_lock)
    examples, part_manifests = load_parts(args.parts_root, lock_rows)

    per_language, model_checks = content_estimates(examples, args.n_boot, SEED)
    ratio_valid = set(model_checks.loc[model_checks.ratio_valid_model, "model"])
    panel = load_panel(args.expanded_panel, lock_rows)
    analysis = per_language.merge(panel, on=["model", "flores_code"], how="inner")
    analysis = analysis.merge(
        model_checks[["model", "ratio_valid_model"]], on="model", how="left"
    )
    if analysis.empty:
        raise ValueError("no common-support analysis rows")
    expected_common = analysis.groupby("model").flores_code.nunique()
    if len(expected_common) != EXPECTED_MODELS or expected_common.nunique() != 1:
        raise ValueError(f"common support is unbalanced: {expected_common.to_dict()}")

    associations = association_table(analysis, args.n_boot)
    model_summary, model_associations = model_level_tables(analysis, args.n_boot)
    family_out = leave_one_family_out(analysis)

    args.output.mkdir(parents=True)
    files = {
        "per_language_content.csv": per_language,
        "model_completeness.csv": model_checks,
        "analysis_frame.csv": analysis,
        "associations.csv": associations,
        "model_level_summary.csv": model_summary,
        "model_level_associations.csv": model_associations,
        "leave_one_family_out.csv": family_out,
    }
    for name, frame in files.items():
        write_csv(args.output / name, frame)

    primary = associations[
        (associations.candidate == "q_L")
        & (associations.target == "R_content")
        & (associations.bootstrap_axis == "model")
    ].iloc[0]
    summary = {
        "claim_scope": "fixed NTREX content-transfer harness; Aim 1 section 3.1.3",
        "primary_candidate": "q_L = 1 - LFS-VC",
        "primary_target": "R_content",
        "primary_bootstrap_axis": "model",
        "primary_rho_deflated": float(primary.rho_deflated),
        "primary_ci": [float(primary.ci_lo), float(primary.ci_hi)],
        "primary_p_two_sided": float(primary.p_two_sided),
        "n_frozen_models": EXPECTED_MODELS,
        "n_ratio_valid_models": len(ratio_valid),
        "n_model_families_ratio_valid": int(model_checks[model_checks.ratio_valid_model].model_family.nunique()),
        "n_common_support_languages_per_model": int(expected_common.iloc[0]),
        "n_models_in_raw_content_analysis": int(analysis.model.nunique()),
        "model_generalization_claim_allowed": bool(
            len(ratio_valid) >= 15
            and model_checks[model_checks.ratio_valid_model].model_family.nunique() >= 10
        ),
        "external_comparators": ["MEXA", "AaR"],
        "interpretation_guards": [
            "No universal downstream-validity claim.",
            "No causal geometry-to-performance claim.",
            "MEXA and AaR are external comparators, not original project metrics.",
            "RMFS is not tested in R1.",
        ],
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    output_names = [*files, "analysis_summary.json"]
    manifest = {
        "run_id": "aim1_r1_content_transfer_analysis_2026-08-24",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "original_array_job_id": args.original_array_job_id,
        "repair_array_job_id": args.repair_array_job_id,
        "final_repair_array_job_id": args.final_repair_array_job_id,
        "array_states": states,
        "n_part_manifests": len(part_manifests),
        "n_per_example_rows": len(examples),
        "expected_per_example_rows": EXPECTED_MODELS * EXPECTED_ROWS_PER_MODEL,
        "n_boot": args.n_boot,
        "model_lock_sha256": EXPECTED_MODEL_LOCK_SHA256,
        "pairs_sha256": EXPECTED_PAIR_SHA256,
        "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "expanded_panel_sha256": sha256(args.expanded_panel),
        "input_part_manifest_sha256": {
            manifest["tag"]: sha256(args.parts_root / manifest["tag"] / "manifest.json")
            for manifest in part_manifests
        },
        "output_sha256": output_hashes(args.output, output_names),
        "outcome_opened_only_after_all_array_tasks_terminated": True,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "run_id": manifest["run_id"],
        "n_models": len(model_checks),
        "n_ratio_valid_models": len(ratio_valid),
        "n_common_support_rows": len(analysis),
        "all_array_tasks_completed": True,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
