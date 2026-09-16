#!/usr/bin/env python3
"""Closed-form tests for Round 3 coordinate-share utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


STAGED = Path(__file__).parents[1] / "src" / "analysis" / "round3_anisotropy_axes.py"
INSTALLED = Path(__file__).parents[1] / "scripts" / "round3_anisotropy_axes.py"
SCRIPT = STAGED if STAGED.is_file() else INSTALLED
SPEC = importlib.util.spec_from_file_location("round3_anisotropy_axes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CoordinateShareTests(unittest.TestCase):
    def test_four_layer_architecture_is_valid(self) -> None:
        self.assertEqual(MODULE.validate_saved_layers([0, 6, 18, 24]), [0, 6, 18, 24])

    def test_single_layer_audit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            MODULE.validate_saved_layers([0])

    def test_positive_diagonal_scaling_does_not_change_coordinate_share(self) -> None:
        rng = np.random.default_rng(13)
        grid = rng.normal(size=(7, 40, 32))
        scale = np.geomspace(0.01, 10.0, 32)
        original = MODULE.coordinate_shares(grid)[0]
        changed = MODULE.coordinate_shares(grid * scale[None, None, :])[0]
        np.testing.assert_allclose(changed, original, rtol=1e-11, atol=1e-11)

    def test_beta_null_mean_matches_monte_carlo_coordinates(self) -> None:
        rng = np.random.default_rng(19)
        language, concept, dimension = 7, 40, 8000
        grid = rng.normal(size=(language, concept, dimension))
        observed = np.nanmean(MODULE.coordinate_shares(grid)[0])
        expected = (language - 1) / (language + concept - 2)
        self.assertLess(abs(observed - expected), 0.005)

    def test_by_is_more_conservative_than_bh_style_cutoff(self) -> None:
        pvalues = np.array([1e-8, 1e-4, 0.01, 0.04, 0.5])
        rejected = MODULE.by_rejections(pvalues, alpha=0.05)
        self.assertTrue(rejected[0])
        self.assertFalse(rejected[-1])
        self.assertLessEqual(rejected.sum(), 3)


if __name__ == "__main__":
    unittest.main()
