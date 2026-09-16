import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


STAGED = Path(__file__).parents[1] / "src" / "analysis" / "analyze_aim1_profile_axes.py"
INSTALLED = Path(__file__).parents[1] / "scripts" / "analyze_aim1_profile_axes.py"
SCRIPT = STAGED if STAGED.is_file() else INSTALLED
SPEC = importlib.util.spec_from_file_location("analyze_aim1_profile_axes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PROFILE_METRICS = MODULE.PROFILE_METRICS
_t_summary = MODULE._t_summary
add_paired_control_deltas = MODULE.add_paired_control_deltas
gini_nonnegative = MODULE.gini_nonnegative
summarize_axis_layers = MODULE.summarize_axis_layers
compute_associations = MODULE.compute_associations


def _profile_row(study, seed, arm, control, offset, ppl=1.0):
    row = {
        "study": study,
        "seed": seed,
        "arm": arm,
        "is_control": control,
        "ppl_geomean_ratio_vs_control": ppl,
    }
    row.update({metric: float(i + offset) for i, metric in enumerate(PROFILE_METRICS)})
    return row


def test_add_paired_control_deltas_uses_seed_matched_control():
    frame = pd.DataFrame([
        _profile_row("s", 0, "B", 1, 10),
        _profile_row("s", 0, "C", 0, 12, np.e),
        _profile_row("s", 1, "B", 1, 20),
        _profile_row("s", 1, "C", 0, 17, np.exp(-2)),
    ])
    out = add_paired_control_deltas(frame)
    experimental = out.loc[out["arm"].eq("C")].sort_values("seed")
    assert experimental["delta_R1_concept_dominance_vs_control"].tolist() == [2.0, -3.0]
    assert np.allclose(experimental["ppl_favorable_vs_control"], [-1.0, 2.0])


def test_t_summary_three_replicates():
    result = _t_summary([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["sd"] == 1.0
    assert result["ci95_low"] < 0.0
    assert result["ci95_high"] > 4.0


def test_gini_nonnegative_extremes():
    assert gini_nonnegative([1, 1, 1, 1]) == 0.0
    assert np.isclose(gini_nonnegative([0, 0, 0, 1]), 0.75)


def test_axis_summary_top5_fraction_and_join():
    axes = pd.DataFrame({
        "model": ["m"] * 4,
        "layer": [0] * 4,
        "language_share": [0.1, 0.2, 0.8, 0.9],
        "ss_language": [1.0, 1.0, 3.0, 5.0],
        "raw_variance": [1.0, 2.0, 3.0, 4.0],
        "language_heavy_by": [False, False, True, True],
        "top5_variance": [False, False, False, True],
    })
    layers = pd.DataFrame({
        "model": ["m"], "layer": [0], "n_languages": [2], "n_concepts": [3],
        "dimension": [4], "lfs_direct_standardized": [0.5],
        "mean_pairwise_cosine_raw": [0.1], "top_eigen_share_raw": [0.2],
        "gaussian_null_mean": [0.25],
    })
    result = summarize_axis_layers(axes, layers).iloc[0]
    assert result["n_axes"] == 4
    assert np.isclose(result["top5_raw_variance_ss_language_fraction"], 0.5)
    assert np.isclose(result["fraction_language_heavy_by"], 0.5)
    assert result["spearman_language_share_vs_raw_variance"] == 1.0
    assert result["lfs_direct_standardized"] == 0.5


def test_associations_deduplicate_repeated_checkpoint_unit():
    rows = []
    for study, arm, key, x, y in [
        ("s1", "A", "base", 0.0, 0.0),
        ("s2", "A", "base", 0.0, 0.0),
        ("s1", "C", "s1:C", 1.0, 1.0),
        ("s1", "D", "s1:D", 2.0, 2.0),
    ]:
        row = {
            "study": study,
            "arm": arm,
            "is_control": 0,
            "analysis_unit_key": key,
            "delta_mexa_vs_control": y,
            "delta_aar10_vs_control": y,
            "ppl_favorable_vs_control": y,
            "delta_context_trained_vs_control": y,
            "delta_context_heldout_vs_control": y,
        }
        for predictor, mode in MODULE.ASSOCIATION_PREDICTORS.items():
            row[predictor] = x
            row[f"delta_{predictor}_vs_control"] = x
        rows.append(row)
    result = compute_associations(pd.DataFrame(rows))
    pooled = result[
        result["scope"].eq("pooled_within_study_centered")
        & result["predictor"].eq("R2_mean_alignment_margin")
        & result["outcome"].eq("delta_aar10_vs_control")
    ].iloc[0]
    assert pooled["n_arms"] == 3
