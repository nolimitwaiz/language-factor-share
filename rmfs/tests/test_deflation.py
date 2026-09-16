"""Battery for the deflation port: on synthetic data with KNOWN causal
structure, the attribution regression must strip observable-driven
correlation and keep genuine residual signal — that is the entire job."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rmfs.utils.seeding import rng
from rmfs.validity.deflation import (
    candidate_bracket,
    fit_attribution,
    shrink_alphas,
)

FAMS = ["indo", "sino", "afro", "turkic"]
SCRIPTS = ["latn", "cyrl", "arab"]


def synth_df(seed=13, n_lang=40, n_models=6, alpha_signal=0.0,
             confound_only_metric=True):
    """acc is driven by tokens + model quality + noise; optionally a genuine
    per-language alpha. Metrics: 'conf' rides tokens only; 'true' rides the
    alpha when alpha_signal > 0."""
    g = rng(seed)
    langs = [f"lg{i:02d}_Xx" for i in range(n_lang)]
    fam = {lg: FAMS[i % len(FAMS)] for i, lg in enumerate(langs)}
    scr = {lg: SCRIPTS[i % len(SCRIPTS)] for i, lg in enumerate(langs)}
    tokens = {lg: g.uniform(6, 11) for lg in langs}
    alpha_true = {lg: g.normal(0, 0.4) for lg in langs}
    rows = []
    for m in range(n_models):
        skill = g.normal(0, 0.5)
        for lg in langs:
            lin = (-4.0 + 0.45 * tokens[lg] + skill
                   + alpha_signal * alpha_true[lg] + g.normal(0, 0.15))
            p = 1 / (1 + np.exp(-lin))
            acc = 0.25 + 0.75 * p
            conf = tokens[lg] + g.normal(0, 0.3)
            true = alpha_true[lg] + g.normal(0, 0.15)
            rows.append(dict(model=f"m{m}", flores_code=lg, acc=acc,
                             log_tokens=tokens[lg],
                             macro_family_g=fam[lg], script_g=scr[lg],
                             fertility=g.uniform(1, 3),
                             conf=conf, true=true))
    return pd.DataFrame(rows)


def test_tokens_gate_passes_on_sane_data_and_fails_on_shuffled():
    df = synth_df()
    fit = fit_attribution(df)
    assert fit.tokens_gate_passed
    assert fit.tokens_beta > 0
    sh = df.copy()
    sh["log_tokens"] = rng(42).permutation(sh.log_tokens.values)
    fit2 = fit_attribution(sh)
    assert not fit2.tokens_gate_passed


def test_confound_metric_deflates_to_zero():
    # 'conf' is tokens + noise: raw correlation strong, alpha correlation ~0
    df = synth_df(alpha_signal=0.0)
    fit = fit_attribution(df)
    rows = {(r.target, r.scheme): r
            for r in candidate_bracket(fit, "conf", n_boot=300)}
    raw = rows[("raw_acc", "raw_residual")]
    alp = rows[("alpha", "raw_residual")]
    assert raw.spearman > 0.5
    assert abs(alp.spearman) < 0.15
    assert alp.ci_low < 0 < alp.ci_high      # CI covers zero


def test_genuine_signal_survives_deflation():
    df = synth_df(alpha_signal=1.2)
    fit = fit_attribution(df)
    rows = {(r.target, r.scheme): r
            for r in candidate_bracket(fit, "true", n_boot=300)}
    alp = rows[("alpha", "raw_residual")]
    assert alp.spearman > 0.35
    assert alp.ci_low > 0                     # bound above zero
    fam = rows[("alpha", "family_demeaned")]
    assert fam.spearman > 0.25                # survives the harsher scheme
    dis = rows[("alpha", "disattenuated")]
    assert dis.spearman >= alp.spearman - 1e-9  # correction never shrinks


def test_bracket_carries_gate_failure_flag():
    df = synth_df()
    df["log_tokens"] = rng(7).permutation(df.log_tokens.values)
    fit = fit_attribution(df)
    for r in candidate_bracket(fit, "conf", n_boot=50):
        assert not r.tokens_gate_passed
        assert ("TOKENS_GATE_FAILED_DEFLATION_INVALID" in r.flags
                or "RELIABILITY_TOO_LOW_FOR_DISATTENUATION" in r.flags)


def test_shrinkage_moves_toward_family_means():
    df = synth_df()
    fit = fit_attribution(df)
    d = shrink_alphas(fit.alphas)
    assert "alpha_shrunk" in d
    spread = d.groupby("macro_family_g").apply(
        lambda s: float(np.abs(s.alpha_shrunk - s.alpha_shrunk.mean()).mean()
                        - np.abs(s.alpha - s.alpha.mean()).mean()))
    assert (spread <= 1e-9).all()


def test_missing_columns_hard_fail():
    df = synth_df().drop(columns=["fertility"])
    with pytest.raises(ValueError, match="missing columns"):
        fit_attribution(df)
