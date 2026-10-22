"""Progress must be observable before a worker or evaluation finishes."""

import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import unittest

from scripts import kaggle_campaign as campaign
from scripts.prepare_kaggle import prepare


class TestCampaignLogging(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.plan = json.loads((self.root / "configs/kaggle_t4x2.json").read_text())

    def test_summary_distinguishes_pending_training_and_evaluating(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign.write_json(
                Path(directory) / "ferminet_seed11/progress.json",
                dict(status="training", epoch=17, energy_mean=-1.1),
            )
            campaign.write_json(
                Path(directory) / "psiformer_seed11/progress.json",
                dict(status="evaluating", epoch=20000, time_records=32),
            )
            result = campaign.summarize(self.plan, directory)
        rows = result["cases"]
        self.assertEqual(
            [r["status"] for r in rows[:3]], ["training", "evaluating", "pending"]
        )
        self.assertEqual(rows[0]["epoch"], 17)
        self.assertEqual(rows[1]["time_records"], 32)
        self.assertFalse(result["all_completed"])

    def test_child_log_is_forwarded_before_child_exit(self):
        messages = queue.Queue()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            child = campaign.LoggedWorker(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; print('epoch=1', flush=True); time.sleep(20)",
                ],
                path,
                index=1,
                sink=messages.put,
            )
            try:
                self.assertIn("[worker 1] epoch=1", messages.get(timeout=5))
                self.assertIsNone(child.process.poll())
                self.assertIn("epoch=1", path.read_text())
            finally:
                child.stop(grace_seconds=1)
            self.assertFalse(child.thread.is_alive())

    def test_smoke_is_exactly_one_epoch_with_long_run_shapes(self):
        small = json.loads((self.root / "configs/kaggle_smoke.json").read_text())
        self.assertEqual(small["training"]["num_epochs"], 1)
        self.assertEqual(small["model"], self.plan["model"])
        self.assertEqual(
            small["training"]["num_walkers"], self.plan["training"]["num_walkers"]
        )

    def test_long_submission_requires_one_epoch_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "gate"):
                prepare("test-owner", "long", directory, revision="a" * 40)

    def test_worker_error_is_visible_and_stops_further_cases(self):
        plan = json.loads((self.root / "configs/kaggle_smoke.json").read_text())
        plan["model"]["num_layers"] = 0
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid.json"
            config.write_text(json.dumps(plan))
            output = Path(directory) / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.kaggle_campaign",
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                    "--allow-cpu",
                    "--workers",
                    "1",
                ],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('"event": "case_failed"', result.stdout)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(
                [row["status"] for row in summary["cases"]], ["failed", "pending"]
            )

    def test_gate_from_another_revision_cannot_unlock_long_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(
                json.dumps(
                    dict(
                        passed=True,
                        source_commit="a" * 40,
                        epochs_per_case=1,
                        real_gpu_count=2,
                        console_forwarding_verified=True,
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "gate"):
                prepare(
                    "test-owner", "long", directory, revision="b" * 40, gate_path=path
                )


if __name__ == "__main__":
    unittest.main()
