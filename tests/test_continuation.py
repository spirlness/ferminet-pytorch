"""Verified schedule migration and synchronization-free production transitions."""

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from ferminet_pytorch import TrainingConfig
from ferminet_pytorch.vmc_driver import build_training_scheduler
from scripts.kaggle_campaign import worker_cases
from scripts.prepare_kaggle import prepare
from scripts.sampling_performance import configure_driver_sampling, configure_sampling
from tests.test_campaign_runtime import CampaignRuntimeChecks


class ContinuationChecks:
    device = "cpu"
    make_driver = CampaignRuntimeChecks.make_driver

    def parent_config(self):
        return TrainingConfig(
            num_epochs=3,
            num_walkers=8,
            mcmc_steps_per_epoch=2,
            two_stage_jastrow=True,
            jastrow_stage1_epochs=1,
            jastrow_lr=1e-4,
            restore_best=False,
            dtype=torch.float64,
        )

    def tail_config(self, parent):
        return replace(
            parent,
            num_epochs=9,
            lr=1e-4,
            jastrow_lr=1e-5,
            lr_schedule="warmup_inverse",
            lr_warmup_epochs=2,
            lr_decay_epochs=4,
        )

    def test_extension_preserves_parent_adam_rng_and_exact_tail_resume(self):
        torch.manual_seed(123)
        config = self.parent_config()
        parent = self.make_driver(config)
        parent.run()
        before = copy.deepcopy(parent.optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if self.device == "cuda" else None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.pt"
            parent.save_checkpoint(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            tail = self.make_driver(self.tail_config(config))
            tail.extend_from_checkpoint(path, expected_sha256=digest)
            self.assertEqual(tail.epoch, 3)
            self.assertEqual(tail.continuation["parent_epoch"], 3)
            self.assertEqual(tail.scheduler.last_epoch, 0)
            for group, expected in zip(tail.optimizer.param_groups, (1e-5, 1e-6)):
                self.assertAlmostEqual(group["lr"], expected)
            for key, value in parent.model.state_dict().items():
                torch.testing.assert_close(
                    value, tail.model.state_dict()[key], atol=0, rtol=0
                )
            for key, state in before["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        value,
                        tail.optimizer.state_dict()["state"][key][name],
                        atol=0,
                        rtol=0,
                    )
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            if cuda_rng is not None:
                self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
            tail.run(2)
            resume_path = Path(tmp) / "resume.pt"
            tail.save_checkpoint(resume_path)
            tail.run(4)
            resumed = self.make_driver(self.tail_config(config))
            resumed.load_checkpoint(resume_path)
            resumed.run(4)
            self.assertEqual(tail.training_history, resumed.training_history)
            self.assertEqual(tail.continuation, resumed.continuation)
            self.assertEqual(tail.scheduler.last_epoch, 6)
            self.assertEqual(
                tail.optimizer.param_groups[0]["lr"],
                resumed.optimizer.param_groups[0]["lr"],
            )
            for key, value in tail.model.state_dict().items():
                torch.testing.assert_close(
                    value, resumed.model.state_dict()[key], atol=0, rtol=0
                )
            torch.testing.assert_close(
                tail.sampler.walkers, resumed.sampler.walkers, atol=0, rtol=0
            )

    def test_extension_rejects_tampering_or_scientific_configuration_changes(self):
        parent = self.make_driver(self.parent_config())
        parent.run()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.pt"
            parent.save_checkpoint(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            tail_config = self.tail_config(parent.config)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.make_driver(tail_config).extend_from_checkpoint(
                    path, expected_sha256="0" * 64
                )
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.make_driver(
                    replace(tail_config, mcmc_steps_per_epoch=3)
                ).extend_from_checkpoint(path, expected_sha256=digest)
            saved = torch.load(path, weights_only=True)
            # Old schema-2 files predate the four new schedule options.
            for key in (
                "lr_schedule",
                "lr_warmup_epochs",
                "lr_decay_epochs",
                "lr_min_ratio",
            ):
                saved["config"].pop(key)
            torch.save(saved, path)
            migrated = self.make_driver(tail_config)
            migrated.extend_from_checkpoint(
                path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertEqual(migrated.epoch, 3)

    def test_batched_production_matches_step_loop_rng_cache_and_counters(self):
        driver = self.make_driver(self.parent_config())
        for steps in (0, 1, 10):
            first, second = copy.deepcopy(driver.sampler), copy.deepcopy(driver.sampler)
            first.adapt_step_size = second.adapt_step_size = False
            torch.manual_seed(937)
            first.sample(steps)
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = (
                torch.cuda.get_rng_state().clone() if self.device == "cuda" else None
            )
            torch.manual_seed(937)
            for _ in range(steps):
                second.step()
            for key in (
                "total_steps",
                "accepted_steps",
                "proposed_moves",
                "accepted_moves",
                "last_acceptance_rate",
                "step_size",
            ):
                self.assertEqual(getattr(first, key), getattr(second, key))
            torch.testing.assert_close(first.walkers, second.walkers, atol=0, rtol=0)
            torch.testing.assert_close(
                first.current_log_abs_psi, second.current_log_abs_psi, atol=0, rtol=0
            )
            self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
            if cuda_rng is not None:
                self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))


class TestContinuationCPU(ContinuationChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestContinuationCUDA(ContinuationChecks, unittest.TestCase):
    device = "cuda"

    def test_execution_selection_preserves_rng_and_model_without_amortized_gain(self):
        driver = self.make_driver(self.parent_config())
        sampler = driver.sampler
        model_state = copy.deepcopy(driver.model.state_dict())
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        # Compiler kernels are exercised by the existing compile suite and the
        # real full-size T4 smoke; this checks selection without timing noise.
        with patch("torch.compile", side_effect=lambda fn, **kwargs: fn):
            report = configure_sampling(sampler, remaining_updates=0)
        self.assertEqual(report["selected"], "eager")
        self.assertTrue(report["signs_equal"])
        self.assertEqual(report["max_log_difference"], 0)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        for key, value in model_state.items():
            torch.testing.assert_close(
                value, driver.model.state_dict()[key], atol=0, rtol=0
            )

    def test_full_update_probe_restores_history_parameters_walkers_and_rng(self):
        driver = self.make_driver(self.parent_config())
        driver.run(2)
        before = copy.deepcopy(driver.model.state_dict())
        history = copy.deepcopy(driver.training_history)
        walkers = driver.sampler.walkers.clone()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            with patch("torch.compile", side_effect=lambda fn, **kwargs: fn):
                restored, report = configure_driver_sampling(
                    driver, directory, 100, mode="compiled"
                )
            self.assertFalse((Path(directory) / "sampling_probe.pt").exists())
        self.assertTrue(report["complete_training_state_restored"])
        self.assertEqual(restored.training_history, history)
        self.assertEqual(restored.epoch, 2)
        torch.testing.assert_close(walkers, restored.sampler.walkers, atol=0, rtol=0)
        for key, value in before.items():
            torch.testing.assert_close(
                value, restored.model.state_dict()[key], atol=0, rtol=0
            )
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))


class TestTailPlanning(unittest.TestCase):
    def test_schedule_has_warmup_and_slow_tail_without_end_of_run_collapse(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        config = TrainingConfig(lr_schedule="warmup_inverse")
        scheduler = build_training_scheduler(optimizer, 30000, config)
        factor = scheduler.lr_lambdas[0]
        self.assertAlmostEqual(factor(0), 0.1)
        self.assertAlmostEqual(factor(500), 1)
        self.assertAlmostEqual(factor(30000), 1 / 3.95)
        self.assertGreaterEqual(factor(1000000), 0.1)

    def test_continuation_smoke_is_one_real_update_and_mounts_parent(self):
        root = Path(__file__).resolve().parents[1]
        large = json.loads((root / "configs/kaggle_continue_t4x2.json").read_text())
        small = json.loads((root / "configs/kaggle_continue_smoke.json").read_text())
        self.assertEqual(
            small["training"]["num_epochs"] - small["continuation"]["source_epochs"], 1
        )
        self.assertEqual(large["model"], small["model"])
        self.assertEqual(large["continuation"], small["continuation"])
        self.assertEqual(small["training"]["mcmc_steps_per_epoch"], 10)
        self.assertEqual(
            worker_cases(large, 0, 2),
            [("ferminet", 11), ("psiformer", 22), ("ferminet", 33)],
        )
        self.assertEqual(
            worker_cases(large, 1, 2),
            [("psiformer", 11), ("ferminet", 22), ("psiformer", 33)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            metadata = prepare("hwbwhzbsh", "continue-smoke", tmp, revision="a" * 40)
            self.assertEqual(
                metadata["kernel_sources"], [large["continuation"]["source_kernel"]]
            )
            self.assertIn("continue-smoke", metadata["id"])
            compile((Path(tmp) / "run.py").read_text(), "run.py", "exec")
            with self.assertRaisesRegex(ValueError, "gate"):
                prepare("hwbwhzbsh", "continue-long", tmp, revision="a" * 40)
