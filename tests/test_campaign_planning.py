"""Campaign summaries must include every planned seed, including failures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.kaggle_campaign import planned_cases, summarize, write_json
from scripts.prepare_kaggle import prepare


class TestCampaignPlanning(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.plan = json.loads((self.root / "configs/kaggle_t4x2.json").read_text())

    def test_each_worker_gets_three_distinct_seeds(self):
        cases = planned_cases(self.plan)
        self.assertEqual(len(cases), 6)
        self.assertEqual(cases[::2], [("ferminet", x) for x in (11, 22, 33)])
        self.assertEqual(cases[1::2], [("psiformer", x) for x in (11, 22, 33)])
        changed = copy.deepcopy(self.plan)
        changed["training"]["restore_best"] = True
        with self.assertRaises(ValueError):
            planned_cases(changed)

    def test_missing_or_failed_seed_cannot_produce_success(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(
                Path(directory) / "ferminet_seed11/result.json",
                dict(
                    status="completed",
                    interaction="ferminet",
                    seed=11,
                    evaluation=dict(
                        energy_mean=-1.1744,
                        energy_sem=0.00001,
                        diagnostics_status="passed",
                        chemical_accuracy_met=True,
                    ),
                ),
            )
            write_json(
                Path(directory) / "ferminet_seed22/result.json",
                dict(status="failed", interaction="ferminet", seed=22),
            )
            result = summarize(self.plan, directory)
        self.assertEqual(len(result["cases"]), 6)
        self.assertFalse(result["all_completed"])
        group = result["seed_summary"]["ferminet"]
        self.assertEqual(group["completed"], 1)
        self.assertFalse(group["all_diagnostics_passed"])
        self.assertFalse(group["chemical_accuracy_proven_for_all_seeds"])
        self.assertNotIn("descriptive_seed_mean_ci95", group)

    def test_submission_is_private_and_immutable_and_requires_two_t4s(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = prepare("test-owner", "smoke", directory, revision="a" * 40)
            source = (Path(directory) / "run.py").read_text()
            compile(source, "run.py", "exec")
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertIn("'" + "a" * 40 + "'", source)
        self.assertIn("len(names) != 2", source)

    def test_resume_submission_mounts_original_output_and_keeps_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = prepare(
                "test-owner",
                "long",
                directory,
                revision="b" * 40,
                resume_kernel="test-owner/original-run",
            )
            source = (Path(directory) / "run.py").read_text()
        self.assertEqual(metadata["kernel_sources"], ["test-owner/original-run"])
        self.assertIn("RESUME_KERNEL = 'test-owner/original-run'", source)


if __name__ == "__main__":
    unittest.main()
