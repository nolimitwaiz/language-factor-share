#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


STAGED = Path(__file__).parents[1] / "src" / "analysis" / "axis_basis_sensitivity.py"
INSTALLED = Path(__file__).parents[1] / "scripts" / "axis_basis_sensitivity.py"
SCRIPT = STAGED if STAGED.is_file() else INSTALLED
SPEC = importlib.util.spec_from_file_location("axis_basis_sensitivity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_effect_matrices_reproduce_factor_sums_of_squares():
    rng = np.random.default_rng(11)
    grid = rng.normal(size=(5, 20, 12))
    language, concept = MODULE.effect_matrices(grid)
    grand = grid.mean(axis=(0, 1))
    expected_l = 20 * np.square(grid.mean(axis=1) - grand).sum()
    expected_c = 5 * np.square(grid.mean(axis=0) - grand).sum()
    assert np.isclose(np.square(language).sum(), expected_l)
    assert np.isclose(np.square(concept).sum(), expected_c)


def test_givens_sweep_preserves_each_matrix_energy():
    rng = np.random.default_rng(13)
    left = rng.normal(size=(7, 17))
    right = rng.normal(size=(19, 17))
    expected = [np.square(left).sum(), np.square(right).sum()]
    MODULE.givens_sweep([left, right], np.random.default_rng(29))
    assert np.isclose(np.square(left).sum(), expected[0], rtol=1e-13)
    assert np.isclose(np.square(right).sum(), expected[1], rtol=1e-13)


def test_rotation_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(31)
    original = rng.normal(size=(8, 16))
    one = original.copy()
    two = original.copy()
    MODULE.givens_sweep([one], np.random.default_rng(1729))
    MODULE.givens_sweep([two], np.random.default_rng(1729))
    np.testing.assert_array_equal(one, two)


def test_summary_conserves_global_lfs_under_rotation():
    rng = np.random.default_rng(41)
    language = rng.normal(size=(6, 32))
    concept = rng.normal(size=(30, 32))
    total_l = float(np.square(language).sum())
    total_c = float(np.square(concept).sum())
    native = MODULE.summarize_basis(language, concept, total_l, total_c)
    MODULE.givens_sweep([language, concept], np.random.default_rng(43))
    rotated = MODULE.summarize_basis(language, concept, total_l, total_c)
    assert np.isclose(native["raw_global_lfs"], rotated["raw_global_lfs"], rtol=1e-13)
    assert rotated["ss_language_relative_error"] < 1e-13
    assert rotated["ss_concept_relative_error"] < 1e-13
