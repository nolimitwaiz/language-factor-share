#!/usr/bin/env python3
"""Closed-form contract tests for the Round 3 invariance battery."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


STAGED = Path(__file__).parents[1] / "src" / "analysis" / "round3_invariance.py"
INSTALLED = Path(__file__).parents[1] / "scripts" / "round3_invariance.py"
SCRIPT = STAGED if STAGED.is_file() else INSTALLED
SPEC = importlib.util.spec_from_file_location("round3_invariance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Round3InvarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(13)
        concepts = rng.normal(size=(24, 12))
        languages = rng.normal(scale=0.3, size=(5, 1, 12))
        residual = rng.normal(scale=0.02, size=(5, 24, 12))
        self.grid = concepts[None, :, :] + languages + residual

    def test_uniform_shrink_preservation_ratios_are_quadratic(self) -> None:
        baseline = MODULE.raw_geometry(self.grid)
        shrunk = MODULE.raw_geometry(self.grid * 0.1)
        self.assertAlmostEqual(
            shrunk["cvp_within_raw"] / baseline["cvp_within_raw"], 0.01, places=11
        )
        self.assertAlmostEqual(
            shrunk["cpres_shared_raw"] / baseline["cpres_shared_raw"],
            0.01,
            places=11,
        )

    def test_positive_diagonal_scaling_is_removed_by_standardization(self) -> None:
        scale = np.linspace(0.01, 2.0, self.grid.shape[-1])
        baseline = MODULE.standardize_grid(self.grid)
        changed = MODULE.standardize_grid(self.grid * scale[None, None, :])
        np.testing.assert_allclose(changed, baseline, rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(
            MODULE.direct_decomposition(changed)["lfs_direct"],
            MODULE.direct_decomposition(baseline)["lfs_direct"],
            places=12,
        )

    def test_rank_projection_targets_concept_effect(self) -> None:
        perturbations = MODULE.perturbations(
            self.grid, ["eng", "deu", "swa", "amh", "zul"], 13
        )
        rank_one = next(
            item
            for item in perturbations
            if item.family == "rank" and item.magnitude == "rank=1"
        )
        baseline = MODULE.direct_decomposition(self.grid)
        projected = MODULE.direct_decomposition(rank_one.transform(self.grid))
        self.assertAlmostEqual(
            projected["ss_language"], baseline["ss_language"], places=8
        )
        self.assertAlmostEqual(
            projected["ss_residual"], baseline["ss_residual"], places=8
        )
        self.assertLess(projected["ss_concept"], baseline["ss_concept"])

    def test_standardized_uniform_energy_expectation_is_invariant(self) -> None:
        expected, label = MODULE.expectation(
            "uniform", "0.1", "standardized", "ss_concept", 42.0
        )
        self.assertEqual(expected, 42.0)
        self.assertIn("invariant", label)


if __name__ == "__main__":
    unittest.main()
