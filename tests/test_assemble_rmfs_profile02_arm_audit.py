#!/usr/bin/env python3
"""Contract test for joining trained and frozen RMFS Profile 0.2 arms."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


STAGED_SCRIPT = (
    Path(__file__).parents[1]
    / "src"
    / "analysis"
    / "assemble_rmfs_profile02_arm_audit.py"
)
INSTALLED_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "assemble_rmfs_profile02_arm_audit.py"
)
SCRIPT = STAGED_SCRIPT if STAGED_SCRIPT.is_file() else INSTALLED_SCRIPT
SPEC = importlib.util.spec_from_file_location("assemble_rmfs_profile02", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def profile_payload(name: str) -> dict:
    return {
        "model_or_checkpoint": name,
        "reference_grid_sha256": "frozen-grid-hash",
        "profile": {
            "readings": {
                "concept_dominance": 0.6,
                "mean_alignment_margin": 0.2,
                "weak_language_tail": -0.1,
                "preservation": {
                    "within_concept_ratio": 1.0,
                    "shared_concept_ratio": 1.0,
                    "effective_rank_ratio": 1.0,
                    "passes_legacy_cvp_gate": True,
                },
            },
            "supporting": {
                "decomposition": {
                    "lfs": 0.4,
                    "ss_language": 4.0,
                    "ss_concept": 6.0,
                    "ss_residual": 2.0,
                    "language_share_total": 1.0 / 3.0,
                    "concept_share_total": 0.5,
                    "residual_share_total": 1.0 / 6.0,
                },
                "raw_geometry": {
                    "within_concept_spread": 5.0,
                    "shared_concept_energy": 3.0,
                    "language_energy": 2.0,
                    "residual_energy": 1.0,
                    "mean_vector_norm": 4.0,
                    "effective_rank": 8.0,
                },
            },
        },
    }


class AssembleProfileAuditTests(unittest.TestCase):
    def test_frozen_controls_use_the_reference_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles = root / "profiles"
            trained = profiles / "ckpt_armB_seed0"
            trained.mkdir(parents=True)
            (trained / "result.json").write_text(
                json.dumps(profile_payload("ckpt_armB_seed0")), encoding="utf-8"
            )
            reference = root / "reference.json"
            reference.write_text(
                json.dumps(profile_payload("Qwen3-0.6B-Base")), encoding="utf-8"
            )
            outcomes = root / "outcomes.csv"
            with outcomes.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["study", "seed", "arm"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"study": "original", "seed": "0", "arm": "A"},
                        {"study": "wordalign", "seed": "0", "arm": "WA-A"},
                        {"study": "original", "seed": "0", "arm": "B"},
                    ]
                )
            output = root / "out"
            argv = [
                str(SCRIPT),
                "--profiles-root",
                str(profiles),
                "--reference-profile",
                str(reference),
                "--frozen-arm",
                "A",
                "--frozen-arm",
                "WA-A",
                "--outcomes",
                str(outcomes),
                "--output-dir",
                str(output),
                "--expected-checkpoints",
                "1",
            ]
            with patch.object(sys, "argv", argv):
                MODULE.main()

            completeness = json.loads(
                (output / "completeness.json").read_text(encoding="utf-8")
            )
            self.assertTrue(completeness["complete"])
            self.assertEqual(completeness["joined_rows"], 3)
            self.assertEqual(completeness["frozen_outcome_rows_joined"], 2)
            self.assertEqual(completeness["outcome_rows_missing_profile"], [])


if __name__ == "__main__":
    unittest.main()
