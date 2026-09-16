#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


STAGED = Path(__file__).parents[1] / "src" / "analysis" / "r1_content_transfer.py"
INSTALLED = Path(__file__).parents[1] / "scripts" / "r1_content_transfer.py"
SCRIPT = STAGED if STAGED.is_file() else INSTALLED
SPEC = importlib.util.spec_from_file_location("r1_content_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_specs_has_one_base_and_two_conditions_per_language():
    pairs = [{"pair_id": 0, "matched_context_index": 0, "target_index": 1,
              "mismatched_context_index": 2}]
    texts = {
        "eng": ["eng matched", "target", "eng mismatch"],
        "deu": ["deu matched", "unused", "deu mismatch"],
    }
    specs = MODULE.build_specs(pairs, texts, ["eng", "deu"])
    assert len(specs) == 5
    assert sum(spec["condition"] == "none" for spec in specs) == 1
    assert {(spec["language"], spec["condition"]) for spec in specs[1:]} == {
        ("eng", "matched"), ("eng", "mismatched"),
        ("deu", "matched"), ("deu", "mismatched"),
    }


def test_target_nll_uses_only_target_predictions():
    # ids: prefix=0, targets=1,2. Logit positions 0 and 1 predict targets.
    logits = torch.zeros((3, 3), dtype=torch.float32)
    logits[0, 1] = 10.0
    logits[1, 2] = 10.0
    ids = torch.tensor([0, 1, 2])
    nll = MODULE.target_mean_nll(logits, ids, target_start=1)
    assert nll < 0.001


def test_summary_content_is_mismatch_minus_match():
    rows = [
        {"language": "eng", "condition": "none", "mean_target_nll": 3.0},
        {"language": "eng", "condition": "matched", "mean_target_nll": 1.0},
        {"language": "eng", "condition": "mismatched", "mean_target_nll": 2.0},
        {"language": "deu", "condition": "matched", "mean_target_nll": 1.5},
        {"language": "deu", "condition": "mismatched", "mean_target_nll": 2.0},
    ]
    result = MODULE.summarize(rows, ["eng", "deu"])
    assert result["per_language"]["eng"]["content"] == 1.0
    assert result["per_language"]["deu"]["content"] == 0.5
    assert result["per_language"]["deu"]["R_content"] == 0.5


class LengthTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, context_tokens, target_tokens):
        self.context_tokens = context_tokens
        self.target_tokens = target_tokens

    def encode(self, text, add_special_tokens=False):
        length = self.target_tokens if text.startswith(" target") else self.context_tokens
        return list(range(length))


def test_repaired_ceiling_accepts_2048_and_rejects_longer_sequences():
    spec = {
        "pair_id": 0, "language": "eng", "condition": "matched",
        "context": "context", "target": "target",
    }
    accepted = MODULE.encode_spec(LengthTokenizer(1900, 148), spec)
    assert accepted["total_tokens"] == 2048
    with pytest.raises(ValueError, match="2048-token ceiling"):
        MODULE.encode_spec(LengthTokenizer(1901, 148), spec)
