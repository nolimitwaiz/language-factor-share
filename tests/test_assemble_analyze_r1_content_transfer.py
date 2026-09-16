from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "analysis"
    / "assemble_analyze_r1_content_transfer.py"
)
SPEC = importlib.util.spec_from_file_location("r1_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_array_gate_requires_all_completed(monkeypatch):
    good = "\n".join(f"123_{index}|COMPLETED" for index in range(19))
    calls = []
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(stdout=good),
    )
    states = MODULE.verify_array_terminal("123", set(range(19)), attempts=1)
    assert len(states) == 19
    assert set(states.values()) == {"COMPLETED"}
    assert calls[0][0][0] == [
        "sacct", "-X", "-S", MODULE.ACCOUNTING_START, "-j", "123",
        "--format=JobID,State", "-n", "-P",
    ]

    failed = good.replace("123_7|COMPLETED", "123_7|FAILED")
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=failed),
    )
    with pytest.raises(RuntimeError, match="technical failures"):
        MODULE.verify_array_terminal("123", set(range(19)), attempts=1)


def test_original_failures_are_allowed_only_at_frozen_repair_indices(monkeypatch):
    repaired = "\n".join(
        f"123_{index}|{'FAILED' if index in MODULE.REPAIR_INDICES else 'COMPLETED'}"
        for index in range(19)
    )
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=repaired),
    )
    states = MODULE.verify_array_terminal(
        "123", set(range(19)), MODULE.REPAIR_INDICES, attempts=1
    )
    assert states["7"] == "FAILED"
    with pytest.raises(RuntimeError, match="undocumented technical failures"):
        MODULE.verify_array_terminal("123", set(range(19)), attempts=1)


def test_repair_array_requires_the_exact_failed_indices(monkeypatch):
    repaired = "\n".join(
        f"456_{index}|COMPLETED" for index in sorted(MODULE.REPAIR_INDICES)
    )
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=repaired),
    )
    states = MODULE.verify_array_terminal(
        "456", MODULE.REPAIR_INDICES, attempts=1
    )
    assert set(map(int, states)) == MODULE.REPAIR_INDICES


def test_first_repair_may_fail_only_for_documented_final_retry(monkeypatch):
    repaired = "\n".join(
        f"456_{index}|{'FAILED' if index in MODULE.FINAL_REPAIR_INDICES else 'COMPLETED'}"
        for index in sorted(MODULE.REPAIR_INDICES)
    )
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=repaired),
    )
    states = MODULE.verify_array_terminal(
        "456", MODULE.REPAIR_INDICES, MODULE.FINAL_REPAIR_INDICES, attempts=1
    )
    assert states["10"] == "FAILED"
    assert states["18"] == "FAILED"


def test_manifest_ceiling_provenance_is_model_specific():
    base = {
        "hf_id": "hf", "revision": "rev", "model_family": "family",
        "smoke": False, "n_pairs": 100, "n_rows": 8300,
        "expected_rows": 8300, "all_nll_finite": True,
        "model_lock_sha256": MODULE.EXPECTED_MODEL_LOCK_SHA256,
        "pairs_sha256": MODULE.EXPECTED_PAIR_SHA256,
        "data_manifest_sha256": MODULE.EXPECTED_DATA_MANIFEST_SHA256,
        "outcomes_printed_to_log": False,
        "languages": [f"l{i}" for i in range(41)],
    }
    cases = [
        ("Qwen2.5-0.5B", None),
        ("mGPT", 1024),
        ("Yi-1.5-9B", 2048),
    ]
    for tag, ceiling in cases:
        manifest = {**base, "tag": tag}
        if ceiling is not None:
            manifest["max_sequence_tokens"] = ceiling
        MODULE._require_manifest(
            manifest,
            {"tag": tag, "hf_id": "hf", "revision": "rev", "model_family": "family"},
        )


def synthetic_examples() -> pd.DataFrame:
    rows = []
    languages = ["eng", *[f"l{index:02d}" for index in range(40)]]
    for pair_id in range(100):
        base = 4.0 + pair_id / 1000
        rows.append({
            "model": "model-a", "model_family": "family-a",
            "pair_id": pair_id, "language": "eng", "condition": "none",
            "mean_target_nll": base,
        })
        for language_index, language in enumerate(languages):
            matched = base - 0.2 - language_index / 1000
            mismatched = matched + 0.1 + language_index / 1000
            rows.extend([
                {
                    "model": "model-a", "model_family": "family-a",
                    "pair_id": pair_id, "language": language,
                    "condition": "matched", "mean_target_nll": matched,
                },
                {
                    "model": "model-a", "model_family": "family-a",
                    "pair_id": pair_id, "language": language,
                    "condition": "mismatched", "mean_target_nll": mismatched,
                },
            ])
    return pd.DataFrame(rows)


def test_content_estimates_use_paired_differences_and_english_ratio():
    per_language, checks = MODULE.content_estimates(synthetic_examples(), 100, 13)
    assert len(per_language) == 41
    english = per_language[per_language.language == "eng"].iloc[0]
    last = per_language[per_language.language == "l39"].iloc[0]
    assert english.content == pytest.approx(0.1)
    assert english.R_content == pytest.approx(1.0)
    assert last.content == pytest.approx(0.14)
    assert last.R_content == pytest.approx(1.4)
    assert checks.iloc[0].valid_nonenglish_ratios == 40
    assert bool(checks.iloc[0].ratio_valid_model)


def test_residual_spearman_removes_shared_linear_control():
    rng = np.random.default_rng(4)
    n = 300
    control = rng.normal(size=n)
    signal = rng.normal(size=n)
    frame = pd.DataFrame({
        "candidate": 4 * control + signal,
        "target": -3 * control + signal,
        "log_tokens": control,
        "fertility": rng.normal(size=n),
        "macro_family_g": np.where(np.arange(n) % 2, "a", "b"),
        "script_g": np.where(np.arange(n) % 3, "x", "y"),
    })
    assert MODULE.raw_spearman(frame, "candidate", "target") < -0.7
    assert MODULE.residual_spearman(frame, "candidate", "target") > 0.9


def test_ratio_gate_does_not_remove_models_from_raw_content():
    frame = pd.DataFrame({
        "model": ["valid", "invalid"],
        "ratio_valid_model": [True, False],
        "R_content": [0.5, np.nan],
        "content": [0.2, -0.1],
    })
    assert MODULE.target_frame(frame, "R_content").model.tolist() == ["valid"]
    assert MODULE.target_frame(frame, "content").model.tolist() == ["valid", "invalid"]


def test_load_panel_keeps_only_frozen_a9_rows(tmp_path):
    lock = []
    rows = []
    for model_index in range(19):
        model = f"m{model_index}"
        lock.append({"tag": model})
        for language_index in range(40):
            rows.append({
                "model": model, "flores_code": f"l{language_index}",
                "q_L": model_index / 20, "mexa_shared": 0.1,
                "aar_shared": 0.2, "log_tokens": 10,
                "macro_family_g": "f", "script_g": "s", "fertility": 1.0,
            })
    rows.append({
        "model": "m0", "flores_code": "legacy", "q_L": np.nan,
        "mexa_shared": np.nan, "aar_shared": np.nan, "log_tokens": 9,
        "macro_family_g": "f", "script_g": "s", "fertility": 1.2,
    })
    path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    panel = MODULE.load_panel(path, lock)
    assert len(panel) == 19 * 40
    assert panel.groupby("model").size().eq(40).all()
