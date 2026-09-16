#!/usr/bin/env python
"""Exploratory robustness audit of the already-frozen RMFS v1 scalar.

This analysis does not redefine RMFS, select a replacement scalar, or alter
any preregistered verdict.  It asks why the frozen observational soft-min did
or did not generalize by reporting:

* every original component and leave-one-component-out ablation;
* fixed temperature sensitivity around the preregistered tau=0.1;
* language-, model-, and crossed model-by-language bootstrap intervals;
* leave-one-model-out point-estimate stability; and
* which component controls the soft minimum.

All candidates are put on the same rows for each target.  Higher is better.
MEXA is intentionally absent: it is an external comparator, not RMFS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
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
from rmfs.validity.tournament import _design, _residualize  # noqa: E402


TAUS = (0.025, 0.05, 0.1, 0.2, 0.5, 1.0)
BASE_COMPONENTS = ("q_T", "q_L", "q_K")
SEED = 13


@dataclass(frozen=True)
class Estimate:
    target: str
    candidate: str
    bootstrap_axis: str
    rho: float
    ci_lo: float
    ci_hi: float
    p_two_sided: float
    n_rows: int
    n_models: int
    n_languages: int
    n_boot_requested: int
    n_boot_effective: int


def softmin(frame: pd.DataFrame, columns: tuple[str, ...], tau: float) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    minimum = values.min(axis=1, keepdims=True)
    # Stable form of -tau * log(mean(exp(-q/tau))).
    return (
        minimum[:, 0]
        - tau * np.log(np.exp(-(values - minimum) / tau).mean(axis=1))
    )


def add_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    candidates = list(BASE_COMPONENTS)
    for tau in TAUS:
        name = f"rmfs_tau_{tau:g}"
        out[name] = softmin(out, BASE_COMPONENTS, tau)
        candidates.append(name)
    ablations = {
        "rmfs_no_T": ("q_L", "q_K"),
        "rmfs_no_L": ("q_T", "q_K"),
        "rmfs_no_K": ("q_T", "q_L"),
    }
    for name, columns in ablations.items():
        out[name] = softmin(out, columns, 0.1)
        candidates.append(name)
    out["component_mean"] = out.loc[:, BASE_COMPONENTS].mean(axis=1)
    candidates.append("component_mean")
    return out, candidates


def rho_vector(frame: pd.DataFrame, candidates: list[str], target: str) -> np.ndarray:
    design = _design(frame)
    target_residual = _residualize(frame[target].to_numpy(dtype=float), design)
    result = []
    for candidate in candidates:
        candidate_residual = _residualize(
            frame[candidate].to_numpy(dtype=float), design
        )
        if (
            np.allclose(candidate_residual, candidate_residual[0], atol=1e-12, rtol=0)
            or np.allclose(target_residual, target_residual[0], atol=1e-12, rtol=0)
        ):
            result.append(float("nan"))
        else:
            result.append(
                stats.spearmanr(candidate_residual, target_residual).statistic
            )
    return np.asarray(result, dtype=float)


def sampled_frame(
    frame: pd.DataFrame,
    axis: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    models = np.sort(frame.model.unique())
    languages = np.sort(frame.flores_code.unique())
    by_model = {value: frame[frame.model == value] for value in models}
    by_language = {
        value: frame[frame.flores_code == value] for value in languages
    }
    if axis == "language":
        picked = rng.choice(languages, size=len(languages), replace=True)
        return pd.concat([by_language[value] for value in picked], ignore_index=True)
    if axis == "model":
        picked = rng.choice(models, size=len(models), replace=True)
        return pd.concat([by_model[value] for value in picked], ignore_index=True)
    if axis != "crossed":
        raise ValueError(axis)

    picked_models = rng.choice(models, size=len(models), replace=True)
    picked_languages = rng.choice(languages, size=len(languages), replace=True)
    cells = {
        (model, language): cell
        for (model, language), cell in frame.groupby(["model", "flores_code"])
    }
    pieces = []
    for model in picked_models:
        for language in picked_languages:
            cell = cells.get((model, language))
            if cell is not None:
                pieces.append(cell)
    if not pieces:
        raise RuntimeError("crossed bootstrap produced an empty frame")
    return pd.concat(pieces, ignore_index=True)


def interval_and_p(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    p = 2.0 * min(float((finite <= 0).mean()), float((finite >= 0).mean()))
    p = min(1.0, max(p, 1.0 / len(finite)))
    return float(lo), float(hi), p


def score(
    frame: pd.DataFrame,
    candidates: list[str],
    target: str,
    n_boot: int,
) -> tuple[list[Estimate], pd.DataFrame]:
    required = [target, *candidates, "model", "flores_code"]
    common = frame.dropna(subset=required).copy()
    point = rho_vector(common, candidates, target)
    estimates: list[Estimate] = []
    for axis_index, axis in enumerate(("language", "model", "crossed")):
        rng = np.random.default_rng(SEED + axis_index)
        boot = []
        for _ in range(n_boot):
            sample = sampled_frame(common, axis, rng)
            try:
                boot.append(rho_vector(sample, candidates, target))
            except (ValueError, np.linalg.LinAlgError):
                continue
        values = np.asarray(boot, dtype=float)
        for index, candidate in enumerate(candidates):
            lo, hi, p = interval_and_p(values[:, index])
            estimates.append(
                Estimate(
                    target=target,
                    candidate=candidate,
                    bootstrap_axis=axis,
                    rho=float(point[index]),
                    ci_lo=lo,
                    ci_hi=hi,
                    p_two_sided=p,
                    n_rows=len(common),
                    n_models=common.model.nunique(),
                    n_languages=common.flores_code.nunique(),
                    n_boot_requested=n_boot,
                    n_boot_effective=int(np.isfinite(values[:, index]).sum()),
                )
            )

    jackknife = []
    for held_out in np.sort(common.model.unique()):
        subset = common[common.model != held_out]
        values = rho_vector(subset, candidates, target)
        for candidate, value in zip(candidates, values, strict=True):
            jackknife.append(
                {
                    "target": target,
                    "held_out_model": held_out,
                    "candidate": candidate,
                    "rho": float(value),
                    "n_models": int(subset.model.nunique()),
                }
            )
    return estimates, pd.DataFrame(jackknife)


def control_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = frame.loc[:, BASE_COMPONENTS].to_numpy(dtype=float)
    minimum = np.argmin(q, axis=1)
    counts = []
    for index, component in enumerate(BASE_COMPONENTS):
        counts.append(
            {
                "component": component,
                "n_cells_controlling_min": int((minimum == index).sum()),
                "fraction_controlling_min": float((minimum == index).mean()),
                "mean_value": float(q[:, index].mean()),
                "median_value": float(np.median(q[:, index])),
            }
        )

    shifted = q - q.min(axis=1, keepdims=True)
    weights = np.exp(-shifted / 0.1)
    weights /= weights.sum(axis=1, keepdims=True)
    for index, component in enumerate(BASE_COMPONENTS):
        counts[index]["mean_softmin_weight_tau_0_1"] = float(weights[:, index].mean())

    primary = frame["rmfs_tau_0.1"]
    tau_rows = []
    for tau in TAUS:
        column = f"rmfs_tau_{tau:g}"
        tau_rows.append(
            {
                "tau": tau,
                "spearman_vs_tau_0_1": float(
                    stats.spearmanr(primary, frame[column]).statistic
                ),
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std(ddof=1)),
            }
        )
    return pd.DataFrame(counts), pd.DataFrame(tau_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results" / "rmfs_v1_robustness")
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    prior_env = os.environ.get("MULTILINGUAL_METRICS_ROOT")
    prior = (
        Path(prior_env).expanduser()
        if prior_env
        else Path.home() / "multilingual-metrics"
    )
    if not prior.exists():
        raise FileNotFoundError("Set MULTILINGUAL_METRICS_ROOT")
    tournament_module.PRIOR = str(prior)

    raw = tournament_module.load_frame()
    fit = fit_attribution(raw)
    scored, candidates = add_candidates(fit.alphas)
    estimates: list[Estimate] = []
    jackknife = []
    for target in ("alpha", "r_content"):
        target_estimates, target_jackknife = score(
            scored, candidates, target, args.n_boot
        )
        estimates.extend(target_estimates)
        jackknife.append(target_jackknife)

    complete = scored.dropna(subset=list(BASE_COMPONENTS)).copy()
    controls, tau_stability = control_diagnostics(complete)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates_frame = pd.DataFrame(asdict(row) for row in estimates)
    jackknife_frame = pd.concat(jackknife, ignore_index=True)
    estimates_frame.to_csv(output / "estimates.csv", index=False, float_format="%.6f")
    jackknife_frame.to_csv(output / "model_jackknife.csv", index=False, float_format="%.6f")
    controls.to_csv(output / "softmin_control.csv", index=False, float_format="%.6f")
    tau_stability.to_csv(output / "tau_stability.csv", index=False, float_format="%.6f")

    manifest = {
        "analysis_status": "exploratory retrospective audit of frozen RMFS v1",
        "changes_frozen_verdicts": False,
        "selects_new_scalar": False,
        "mexa_included": False,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "seed": SEED,
        "n_boot_requested": args.n_boot,
        "taus": TAUS,
        "primary_tau": 0.1,
        "tokens_gate_passed": fit.tokens_gate_passed,
        "candidates": candidates,
        "multilingual_metrics_root": str(prior.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    print("\nESTIMATES")
    print(estimates_frame.to_string(index=False))
    print("\nSOFTMIN CONTROL")
    print(controls.to_string(index=False))
    print("\nTAU STABILITY")
    print(tau_stability.to_string(index=False))


if __name__ == "__main__":
    main()
