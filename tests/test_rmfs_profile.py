#!/usr/bin/env python3
"""Synthetic contract tests for RMFS profile 0.2."""

import unittest

import numpy as np

from src.analysis.rmfs_profile import compute_rmfs_profile, factor_decomposition


class RMFSProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        concepts = rng.normal(size=(40, 24))
        language_offsets = rng.normal(scale=0.15, size=(5, 1, 24))
        noise = rng.normal(scale=0.01, size=(5, 40, 24))
        self.grid = concepts[None, :, :] + language_offsets + noise

    def test_uniform_shrinkage_is_caught_only_by_preservation(self) -> None:
        baseline = compute_rmfs_profile(self.grid, reference_grid=self.grid)
        collapsed = compute_rmfs_profile(self.grid * 0.10, reference_grid=self.grid)

        self.assertAlmostEqual(
            baseline["readings"]["concept_dominance"],
            collapsed["readings"]["concept_dominance"],
            places=10,
        )
        self.assertAlmostEqual(
            baseline["readings"]["mean_alignment_margin"],
            collapsed["readings"]["mean_alignment_margin"],
            places=10,
        )
        self.assertAlmostEqual(
            collapsed["readings"]["preservation"]["within_concept_ratio"],
            0.01,
            places=10,
        )
        self.assertFalse(
            collapsed["readings"]["preservation"]["passes_legacy_cvp_gate"]
        )

    def test_language_offsets_raise_lfs(self) -> None:
        no_offsets = np.repeat(self.grid.mean(axis=0, keepdims=True), 5, axis=0)
        low = factor_decomposition(no_offsets)["lfs"]
        high = factor_decomposition(self.grid)["lfs"]
        self.assertGreater(high, low)

    def test_concept_permutation_damages_alignment_and_tail(self) -> None:
        baseline = compute_rmfs_profile(self.grid)
        damaged = self.grid.copy()
        damaged[2] = damaged[2, np.random.default_rng(11).permutation(damaged.shape[1])]
        permuted = compute_rmfs_profile(damaged)
        self.assertLess(
            permuted["readings"]["mean_alignment_margin"],
            baseline["readings"]["mean_alignment_margin"],
        )
        self.assertLess(
            permuted["readings"]["weak_language_tail"],
            baseline["readings"]["weak_language_tail"],
        )

    def test_no_reference_means_no_preservation_claim(self) -> None:
        result = compute_rmfs_profile(self.grid)
        self.assertFalse(result["readings"]["preservation"]["available"])
        self.assertIsNone(result["composite_score"])

    def test_invalid_grid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_rmfs_profile(np.zeros((4, 8)))


if __name__ == "__main__":
    unittest.main()
