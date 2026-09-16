#!/usr/bin/env python
"""Retrospective, same-support LFS/RMFS comparison for submission triage.

This script does not create a new confirmatory test and does not change any
frozen preregistration. It reuses the already scored Aim-1 artifacts and puts
the candidate readings on exactly the same rows before comparing their
deflated Spearman correlations. MEXA and AaR are external comparators, not
components of the original metric.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tournament as tournament_module  # noqa: E402
from rmfs.validity.deflation import fit_attribution  # noqa: E402
from rmfs.validity.tournament import (  # noqa: E402
    CATEGORICAL,
    COVARIATES,
    _design,
    _residualize,
)

N_BOOT = 2_000
SEED = 13


@dataclass(frozen=True)
class Estimate:
    target: str
    candidate: str
    cluster: str
    rho: float
    ci_lo: float
    ci_hi: float
    p_two_sided: float
    n_rows: int
    n_models: int
    n_languages: int


@dataclass(frozen=True)
class Difference:
    target: str
    candidate_a: str
    candidate_b: str
    cluster: str
    rho_a_minus_b: float
    ci_lo: float
    ci_hi: float
    p_two_sided: float
    n_rows: int
    n_models: int
    n_languages: int


def _rho(frame: pd.DataFrame, candidate: str, target: str) -> float:
    X = _design(frame)
    # The local Accelerate/OpenBLAS combination emits spurious matmul
    # RuntimeWarnings while still returning finite values. Keep the analysis
    # output readable, then hard-fail below if a statistic is not finite.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rc = _residualize(frame[candidate].to_numpy(dtype=float), X)
        rt = _residualize(frame[target].to_numpy(dtype=float), X)
    if not np.isfinite(rc).all() or not np.isfinite(rt).all():
        raise FloatingPointError("non-finite residuals in comparison frame")
    return float(stats.spearmanr(rc, rt).statistic)


def _bootstrap_values(
    frame: pd.DataFrame,
    candidates: list[str],
    target: str,
    cluster: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    # A fixed bootstrap seed is only portable if group labels have a canonical
    # order. Encounter order can differ after parquet/dataframe merges across
    # environments, which changes the sampled four-model sequences even when
    # every input value is identical.
    groups = np.sort(frame[cluster].drop_duplicates().to_numpy())
    by_group = {g: frame[frame[cluster] == g] for g in groups}
    point = {c: _rho(frame, c, target) for c in candidates}
    rng = np.random.default_rng(SEED)
    values = {c: [] for c in candidates}
    for _ in range(N_BOOT):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([by_group[g] for g in selected], ignore_index=True)
        for c in candidates:
            value = _rho(sample, c, target)
            if np.isfinite(value):
                values[c].append(value)
    return point, {c: np.asarray(v, dtype=float) for c, v in values.items()}


def _interval_and_p(values: np.ndarray) -> tuple[float, float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    p = 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean()))
    p = min(1.0, max(p, 1.0 / len(values)))
    return float(lo), float(hi), p


def score_target(
    data: pd.DataFrame,
    target: str,
    candidates: list[str],
) -> tuple[list[Estimate], list[Difference], dict[str, int]]:
    required = [target, *candidates, *COVARIATES, *CATEGORICAL, "model", "flores_code"]
    frame = data.dropna(subset=required).copy()
    if frame.empty:
        raise RuntimeError(f"no common-support rows for {target}")

    estimates: list[Estimate] = []
    differences: list[Difference] = []
    for cluster in ("flores_code", "model"):
        point, boot = _bootstrap_values(frame, candidates, target, cluster)
        for candidate in candidates:
            lo, hi, p = _interval_and_p(boot[candidate])
            estimates.append(Estimate(
                target=target,
                candidate=candidate,
                cluster=cluster,
                rho=point[candidate],
                ci_lo=lo,
                ci_hi=hi,
                p_two_sided=p,
                n_rows=len(frame),
                n_models=frame.model.nunique(),
                n_languages=frame.flores_code.nunique(),
            ))

        comparisons = [
            ("rmfs_v1", "lfs_legacy_concept"),
            ("rmfs_v1", "lfs_vc_concept"),
            ("rmfs_no_t", "lfs_vc_concept"),
            ("lfs_vc_concept", "lfs_legacy_concept"),
            ("mexa_external", "lfs_vc_concept"),
            ("aar_external", "lfs_vc_concept"),
        ]
        for a, b in comparisons:
            n = min(len(boot[a]), len(boot[b]))
            delta = boot[a][:n] - boot[b][:n]
            lo, hi, p = _interval_and_p(delta)
            differences.append(Difference(
                target=target,
                candidate_a=a,
                candidate_b=b,
                cluster=cluster,
                rho_a_minus_b=point[a] - point[b],
                ci_lo=lo,
                ci_hi=hi,
                p_two_sided=p,
                n_rows=len(frame),
                n_models=frame.model.nunique(),
                n_languages=frame.flores_code.nunique(),
            ))

    support = {
        "n_rows": len(frame),
        "n_models": int(frame.model.nunique()),
        "n_languages": int(frame.flores_code.nunique()),
    }
    return estimates, differences, support


def main() -> None:
    # Avoid local BLAS oversubscription. This is a CPU-only table rebuild.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # The historical campaign lived beside the original project on CLSP, but
    # under ~/Desktop on the laptop.  Make the data root explicit so the same
    # analysis script can be verified on either host without symlinks or edits
    # to frozen result artifacts.
    prior_root = Path(os.environ.get("MULTILINGUAL_METRICS_ROOT", "")).expanduser()
    if not str(prior_root):
        prior_root = Path.home() / "Desktop" / "multilingual-metrics"
    if not prior_root.exists():
        prior_root = Path.home() / "multilingual-metrics"
    if not prior_root.exists():
        raise FileNotFoundError(
            "Set MULTILINGUAL_METRICS_ROOT to the original project checkout"
        )
    tournament_module.PRIOR = str(prior_root)

    raw = tournament_module.load_frame()
    scored = fit_attribution(raw).alphas.copy()

    # Put every score in the same direction: higher means more favorable.
    # The first line is the original committed SS estimator; the second is
    # the later REML variance-component estimator. They are not conflated.
    scored["lfs_legacy_concept"] = 1.0 - scored["legacy_lfs"]
    scored["lfs_vc_concept"] = scored["q_L"]
    scored["rmfs_v1"] = scored["rmfs_obs"]
    scored["rmfs_no_t"] = scored["texcl"]
    scored["mexa_external"] = scored["mexa_shared"]
    scored["aar_external"] = scored["aar_shared"]

    candidates = [
        "lfs_legacy_concept",
        "lfs_vc_concept",
        "rmfs_v1",
        "rmfs_no_t",
        "mexa_external",
        "aar_external",
    ]
    all_estimates: list[Estimate] = []
    all_differences: list[Difference] = []
    supports: dict[str, dict[str, int]] = {}
    for target in ("alpha", "r_content"):
        estimates, differences, support = score_target(scored, target, candidates)
        all_estimates.extend(estimates)
        all_differences.extend(differences)
        supports[target] = support

    out_dir = ROOT / "results" / "submission_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    estimates_frame = pd.DataFrame(asdict(x) for x in all_estimates)
    differences_frame = pd.DataFrame(asdict(x) for x in all_differences)
    estimates_frame.to_csv(out_dir / "same_support_estimates.csv", index=False,
                           float_format="%.6f")
    differences_frame.to_csv(out_dir / "paired_differences.csv", index=False,
                             float_format="%.6f")

    summary = {
        "analysis_status": "retrospective synthesis of frozen/scored artifacts",
        "n_boot": N_BOOT,
        "seed": SEED,
        "multilingual_metrics_root": str(prior_root.resolve()),
        "direction": "all candidates recoded so higher is more favorable",
        "supports": supports,
        "candidate_definitions": {
            "lfs_legacy_concept": "1 - committed direct-SS LFS",
            "lfs_vc_concept": "1 - later REML LFS-VC (q_L)",
            "rmfs_v1": "preregistered soft-min observational RMFS scalar",
            "rmfs_no_t": "soft-min of q_L and q_K; excludes behavioral T",
            "mexa_external": "published MEXA baseline under shared preprocessing",
            "aar_external": "project tail diagnostic under shared preprocessing",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nSAME-SUPPORT ESTIMATES")
    print(estimates_frame.to_string(index=False))
    print("\nPAIRED DIFFERENCES")
    print(differences_frame.to_string(index=False))


if __name__ == "__main__":
    main()
