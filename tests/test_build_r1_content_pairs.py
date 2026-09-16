#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


STAGED = Path(__file__).parents[1] / "src" / "analysis" / "build_r1_content_pairs.py"
INSTALLED = Path(__file__).parents[1] / "scripts" / "build_r1_content_pairs.py"
SCRIPT = STAGED if STAGED.is_file() else INSTALLED
SPEC = importlib.util.spec_from_file_location("build_r1_content_pairs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selects_only_one_pair_per_document():
    english = [
        "one two three four five", "target has exactly eight useful words right here now",
        "one two three four five six", "another target has exactly eight useful words right now",
        "one two three four five", "third target has exactly eight useful words right here",
    ]
    docs = ["a", "a", "a", "a", "b", "b"]
    selected = MODULE.select_one_pair_per_document(english, docs, 2)
    assert selected == [0, 4]
    assert len({docs[index] for index in selected}) == 2


def test_derangement_is_one_to_one_and_deterministic():
    english = ["x " * 5, "x " * 8, "x " * 6, "x " * 9]
    indices = [0, 1, 2, 3]
    first = MODULE.minimum_cost_derangement(indices, english)
    second = MODULE.minimum_cost_derangement(indices, english)
    assert first == second
    assert sorted(first) == indices
    assert all(left != right for left, right in zip(indices, first))
