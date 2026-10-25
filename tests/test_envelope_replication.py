"""State transitions and evidence contracts for new independent parent pairs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ferminet_pytorch.evaluation_observations import ObservationWriter
from scripts import envelope_replication as replication
from scripts.attribute_residual_energy import (
    block_bootstrap,
    direct_kinetic,
    geometry,
    observations,
    sufficient_statistics,
    summarize_sufficient,
    variance_decomposition,
)
from scripts.envelope_replication_evidence import project_budget
from scripts.envelope_replication_pipeline import tick
from scripts.factorial_campaign import schedule as old_schedule
from scripts.factorial_pipeline import read_plan as old_plan
from scripts.verify_kaggle_smoke import plan_digest


class ReplicationRuntimeChecks:
    device = "cpu"

    def tiny(self):
        plan = replication.read_plan()
        plan["model"].update(
            num_determinants=2, num_layers=1, hidden_dim_one=16, hidden_dim_two=4
        )
        plan["training"].update(num_walkers=8, mcmc_steps_per_epoch=1)
        plan["parent"].update(
            initial_epochs=4, final_epochs=6, jastrow_stage1_epochs=2, burn_in_steps=2
        )
        return plan

    def test_parent_stage_extension_and_exact_resume(self):
        torch.manual_seed(101)
        plan = self.tiny()
        initial = replication.parent_driver(plan, 101, self.device)
        initial.sampler.burn_in(2)
        initial.step()
        self.assertFalse(
            any(p.requires_grad for p in initial.model.jastrow.parameters())
        )
        initial.step()
        initial.step()
        self.assertTrue(
            all(p.requires_grad for p in initial.model.jastrow.parameters())
        )
        self.assertEqual(len(initial.optimizer.param_groups), 2)
        initial.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial.pt"
            initial.save_checkpoint(path)
            tail = replication.parent_driver(plan, 101, self.device, tail=True)
            tail.extend_from_checkpoint(
                path, expected_sha256=replication.file_sha(path)
            )
            self.assertEqual(tail.epoch, 4)
            torch.testing.assert_close(
                tail.sampler.walkers, initial.sampler.walkers, atol=0, rtol=0
            )
            before_steps = [float(v["step"]) for v in tail.optimizer.state.values()]
            self.assertEqual(set(before_steps), {2.0})
            tail.step()
            resume = Path(tmp) / "tail.pt"
            tail.save_checkpoint(resume)
            tail.step()
            expected = copy.deepcopy(tail.model.state_dict())
            walkers = tail.sampler.walkers.clone()
            restored = replication.parent_driver(plan, 101, self.device, tail=True)
            restored.load_checkpoint(resume)
            restored.step()
            for name, value in restored.model.state_dict().items():
                torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
            torch.testing.assert_close(
                restored.sampler.walkers, walkers, atol=0, rtol=0
            )
            cached, _ = restored.model.forward_log_psi(restored.sampler.walkers)
            torch.testing.assert_close(
                restored.sampler.current_log_abs_psi, cached.detach()
            )

    def test_initial_frozen_stage_resume(self):
        plan = self.tiny()
        torch.manual_seed(303)
        driver = replication.parent_driver(plan, 303, self.device)
        driver.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.pt"
            driver.save_checkpoint(path)
            driver.step()
            driver.step()
            expected = copy.deepcopy(driver.model.state_dict())
            walkers = driver.sampler.walkers.clone()
            restored = replication.parent_driver(plan, 303, self.device)
            restored.load_checkpoint(path)
            restored.step()
            restored.step()
            for name, value in restored.model.state_dict().items():
                torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
            torch.testing.assert_close(
                restored.sampler.walkers, walkers, atol=0, rtol=0
            )

    def test_ac_fork_and_independent_amplitude_derivatives(self):
        plan = self.tiny()
        torch.manual_seed(202)
        parent = replication.parent_driver(plan, 202, self.device)
        for _ in range(4):
            parent.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.pt"
            parent.save_checkpoint(path)
            starts = []
            for arm in replication.ARMS:
                options = copy.deepcopy(plan)
                options["training"].update(num_epochs=3, burn_in_steps=2)
                driver = replication.factorial.make_driver(
                    options, 202, arm, self.device
                )
                driver.fork_from_checkpoint(
                    path, expected_sha256=replication.file_sha(path)
                )
                starts.append(
                    (driver.sampler.walkers.clone(), torch.get_rng_state().clone())
                )
                self.assertFalse(driver.optimizer.state)
                self.assertEqual(driver.epoch, 0)
                self.assertFalse(driver.history)
                self.assertTrue(all(p.requires_grad for p in driver.model.parameters()))
                high = copy.deepcopy(driver.model).double()
                r = driver.sampler.walkers[:3].detach().double()
                import copy as copy_module

                from ferminet_pytorch.hamiltonian import compute_local_energy

                system = copy_module.deepcopy(driver.system).to(dtype=torch.float64)
                _, kinetic, _ = compute_local_energy(
                    high, system, r.clone().requires_grad_(True)
                )
                independent = direct_kinetic(high, r)
                torch.testing.assert_close(independent, kinetic, atol=1e-9, rtol=1e-9)
                geo = geometry(high, r)
                self.assertTrue(((geo["flags"] >= 0) & (geo["flags"] < 16)).all())
                old = geo["log10_cancellation"].clone()
                with torch.no_grad():
                    high.det_weights.mul_(100)
                torch.testing.assert_close(
                    geometry(high, r)["log10_cancellation"], old, atol=1e-12, rtol=1e-12
                )
            for a, b in zip(starts[0], starts[1]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


class TestReplicationCPU(ReplicationRuntimeChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestReplicationCUDA(ReplicationRuntimeChecks, unittest.TestCase):
    device = "cuda:0"


class TestReplicationEvidence(unittest.TestCase):
    def test_pending_submission_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            plan = replication.read_plan()
            (output / "pipeline.json").write_text(
                json.dumps(
                    {
                        "phase": "submission_pending",
                        "source_commit": "a" * 40,
                        "protocol_sha256": plan_digest(plan),
                        "downloads": [],
                    }
                )
            )
            args = SimpleNamespace(output=output, revision="a" * 40)
            with patch("scripts.envelope_replication_pipeline.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "refusing duplicate push"):
                    tick(args)
                run.assert_not_called()

    def test_running_tick_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            plan = replication.read_plan()
            journal = output / "pipeline.json"
            journal.write_text(
                json.dumps(
                    {
                        "phase": "batch_submitted",
                        "active_kernel": "owner/kernel",
                        "source_commit": "b" * 40,
                        "protocol_sha256": plan_digest(plan),
                        "downloads": [],
                    }
                )
            )
            before = journal.read_bytes()
            args = SimpleNamespace(output=output, revision="b" * 40, kaggle="kaggle")
            with patch(
                "scripts.envelope_replication_pipeline.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="RUNNING", stderr=""),
            ) as run:
                self.assertEqual(tick(args)["phase"], "batch_submitted")
                self.assertEqual(
                    run.call_args.args[0],
                    ["kaggle", "kernels", "status", "owner/kernel"],
                )
            self.assertEqual(before, journal.read_bytes())

    def test_registered_seeds_queues_and_legacy_lock(self):
        plan = replication.read_plan()
        self.assertEqual(
            replication.batches(plan), [[101, 202], [303, 404], [505, 606]]
        )
        self.assertEqual(len(replication.batches(plan, 1)), 6)
        for batch in replication.batches(plan):
            queues = replication.child_schedule(batch, 20261026)
            self.assertEqual(queues, replication.child_schedule(batch, 20261026))
            for seed in batch:
                self.assertEqual(
                    {i for i, q in enumerate(queues) for c in q if c["seed"] == seed},
                    {0, 1},
                )
        seeds = replication.check_seed_collisions(plan, [900110, 800111])
        self.assertEqual(len(seeds), 24)
        with self.assertRaises(ValueError):
            replication.check_seed_collisions(plan, [seeds[0]])
        prior = old_plan(False)
        prior["seeds"] = replication.SEEDS
        with self.assertRaises(ValueError):
            old_schedule(prior)

    def test_primary_requires_all_new_pairs_and_diagnostics(self):
        plan = replication.read_plan()
        rows = [
            {
                "seed": s,
                "arm": a,
                "status": "completed",
                "precision_after": {"passed": True},
                "evaluation": {
                    "energy_mean": -1.0 if a == "A" else -1.001,
                    "energy_sem": 0.00001,
                    "diagnostics_status": "passed",
                    "extreme_precision_audit": {"passed": True},
                },
            }
            for s in replication.SEEDS
            for a in replication.ARMS
        ]
        result = replication.paired_effect(rows, plan)
        self.assertTrue(result["effect"]["improvement_supported"])
        self.assertAlmostEqual(result["effect"]["mean_mha"], -1.0)
        self.assertEqual(result["effect"]["degrees_of_freedom"], 5)
        self.assertEqual(
            replication.paired_effect(rows[:-1], plan)["status"], "incomplete"
        )
        duplicate = rows[:-1] + [rows[0]]
        self.assertEqual(
            replication.paired_effect(duplicate, plan)["status"], "incomplete"
        )
        rows[0]["evaluation"]["diagnostics_status"] = "insufficient_evidence"
        self.assertEqual(
            replication.paired_effect(rows, plan)["status"], "insufficient_evidence"
        )

    def test_budget_has_parent_barrier_and_never_reads_energy(self):
        plan = replication.read_plan()
        parents = [
            {
                "phase_step_seconds": {
                    k: [0.2] * 20 for k in ["frozen_jastrow", "joint_initial", "tail"]
                },
                "burn_seconds": 10,
            }
        ]
        cases = [
            {
                "step_seconds": [0.2] * 20,
                "rethermalization_seconds": 20,
                "evaluation": {
                    "total_with_observations_seconds": 50,
                    "burn_in_seconds": 10,
                    "time_records": 64,
                },
            }
        ]
        two, one = (project_budget(parents, cases, plan, n) for n in [2, 1])
        self.assertGreater(two["projected_seconds"], one["projected_seconds"])
        self.assertTrue(one["within_budget"])
        parents[0]["phase_step_seconds"]["tail"] = [10.0] * 20
        self.assertFalse(project_budget(parents, cases, plan, 1)["within_budget"])

    def test_disjoint_attribution_adds_and_variance_cancels(self):
        generator = torch.Generator().manual_seed(8)
        e = torch.randn(1024, generator=generator, dtype=torch.float64) * 0.01 - 1.174
        t = torch.randn(1024, generator=generator, dtype=torch.float64) * 4
        v = e - t
        flags = torch.arange(1024) % 16
        stats = sufficient_statistics(e, t, v, [flags == i for i in range(16)])
        total = stats.sum(0)
        regions = summarize_sufficient(stats, total)
        self.assertEqual(sum(r["count"] for r in regions), 1024)
        self.assertAlmostEqual(
            sum(r["variance_share"] for r in regions), 1.0, places=12
        )
        self.assertAlmostEqual(
            sum(r["signed_error_contribution_mha"] for r in regions),
            float((e.mean() + 1.174475714220) * 1000),
            places=10,
        )
        self.assertLess(
            abs(variance_decomposition(total)["reconstruction_residual"]), 1e-12
        )
        blocks = torch.stack([stats / 16] * 16)
        self.assertEqual(
            block_bootstrap(blocks, 7, 100), block_bootstrap(blocks, 7, 100)
        )
        constant = torch.ones_like(e) * -1.174475714220
        zero = sufficient_statistics(
            constant, t, constant - t, [flags == i for i in range(16)]
        )
        self.assertTrue(
            all(
                r["variance_share"] == 0
                for r in summarize_sufficient(zero, zero.sum(0))
            )
        )

    def test_observation_axis_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            writer = ObservationWriter(folder, 2, 512, 123)
            for _ in range(2):
                energy = torch.zeros(512, dtype=torch.float64)
                writer.append(
                    positions=torch.zeros(512, 2, 3, dtype=torch.float64),
                    energy=energy,
                    kinetic=energy,
                    potential=energy,
                    log_abs=energy,
                    sign=torch.ones_like(energy),
                )
            writer.close(True)
            self.assertEqual(len(list(observations(folder, 2, 123))), 1)
            path = folder / "observations_0000.pt"
            data = torch.load(path, weights_only=True)
            data["walker_id"] = data["walker_id"].flip(0)
            torch.save(data, path)
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["shards"][0]["sha256"] = replication.file_sha(path)
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                list(observations(folder, 2, 123))


if __name__ == "__main__":
    unittest.main()
