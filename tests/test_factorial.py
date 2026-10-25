"""Physical, state and evidence contracts for the paired factorial experiment."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ferminet_pytorch import FermiNet, TrainingConfig, VMCDriver
from ferminet_pytorch.envelope import NuclearEnvelope
from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction
from ferminet_pytorch.evaluation_observations import summarize_observations
from ferminet_pytorch.hamiltonian import compute_local_energy
from ferminet_pytorch.loss import clip_local_energy, vmc_surrogate_loss
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.spin import SpinProjector
from ferminet_pytorch.systems import build_h2_system
from scripts.factorial_campaign import ARMS, paired_effects, schedule, seed_for
from scripts.factorial_pipeline import prepare, project_seconds, read_plan

ROOT = Path(__file__).resolve().parents[1]


class FactorialRuntimeChecks:
    device = "cpu"

    def make_driver(self, envelope="gaussian", method="tanh", parent=False):
        system = build_h2_system(device=self.device, dtype=torch.float64)
        model = FermiNet(
            system.nuclear_coords,
            system.nuclear_charges,
            1,
            1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=4,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
            envelope_type=envelope,
        ).to(device=self.device, dtype=torch.float64)
        config = TrainingConfig(
            num_epochs=3 if parent else 5,
            num_walkers=8,
            mcmc_steps_per_epoch=1,
            dtype=torch.float64,
            lr=1e-4,
            jastrow_lr=1e-5,
            lr_schedule="warmup_inverse",
            lr_warmup_epochs=2,
            lr_decay_epochs=10,
            envelope_type=envelope,
            clip_method=method,
            two_stage_jastrow=parent,
            jastrow_stage1_epochs=1 if parent else None,
            restore_best=False,
        )
        sampler = MCMCSampler(model, system, num_walkers=8)
        return VMCDriver(model, system, sampler, config=config)

    def test_mean_hard_formula_shift_constant_and_graph_isolation(self):
        e = torch.tensor(
            [-100.0, -2.0, 1.0, 4.0, 5.0, 100.0],
            device=self.device,
            dtype=torch.float64,
        )
        got, center, scale = clip_local_energy(e, clip_factor=0.5, method="mean_hard")
        self.assertEqual(center.item(), 2.5)
        expected_scale = (e - 2.5).abs().mean()
        torch.testing.assert_close(scale, expected_scale)
        torch.testing.assert_close(got, e.clamp(2.5 - 0.5 * scale, 2.5 + 0.5 * scale))
        shifted = clip_local_energy(e + 1000, clip_factor=0.5, method="mean_hard")[0]
        torch.testing.assert_close(shifted - 1000, got, atol=1e-12, rtol=0)
        constant = torch.full_like(e, -1.17)
        torch.testing.assert_close(
            clip_local_energy(constant, method="mean_hard")[0], constant
        )
        parameter = torch.tensor(
            0.3, device=self.device, dtype=torch.float64, requires_grad=True
        )
        energy_parameter = torch.tensor(
            0.4, device=self.device, dtype=torch.float64, requires_grad=True
        )
        loss, _ = vmc_surrogate_loss(
            parameter * e, e * energy_parameter, clip_method="mean_hard"
        )
        g, forbidden = torch.autograd.grad(
            loss, (parameter, energy_parameter), allow_unused=True
        )
        self.assertTrue(torch.isfinite(g))
        self.assertIsNone(forbidden)
        median = e.median()
        mad = (e - median).abs().median()
        width = max(5 * mad.item(), 0.001, 0.01 * abs(median.item()))
        torch.testing.assert_close(
            clip_local_energy(e)[0], median + width * torch.tanh((e - median) / width)
        )

    def test_soft_envelope_coordinate_derivatives_and_asymptote(self):
        env = (
            NuclearEnvelope(1, 1, 1, envelope_type="soft_exponential")
            .to(self.device)
            .double()
        )
        with torch.no_grad():
            env.weights.fill_(1)
        alpha = torch.nn.functional.softplus(env.decay_rates_raw).item()
        r = torch.tensor(
            [0.0, 1e-6, 0.2, 5.0, 1e4],
            device=self.device,
            dtype=torch.float64,
            requires_grad=True,
        )
        # Use the exponent to avoid intentional amplitude underflow at infinity.
        radial = 2 * r.square() / (torch.sqrt(1 + (r / 2).square()) + 1)
        slope = torch.autograd.grad((-alpha * radial).sum(), r, create_graph=True)[0]
        curvature = torch.autograd.grad(slope.sum(), r)[0]
        torch.testing.assert_close(
            slope, -alpha * 2 * r / torch.sqrt(1 + (r / 2).square())
        )
        torch.testing.assert_close(
            curvature, -2 * alpha / (1 + (r / 2).square()).pow(1.5)
        )
        self.assertEqual(slope[0].item(), 0)
        self.assertAlmostEqual(slope[-1].item(), -4 * alpha, delta=1e-6)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: env(torch.sqrt(x.square().sum(-1) + 1e-24).reshape(2, 1, 1)),
                (
                    torch.tensor(
                        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                        device=self.device,
                        dtype=torch.float64,
                        requires_grad=True,
                    ),
                ),
            )
        )
        self.assertTrue(
            torch.autograd.gradgradcheck(
                lambda x: env(torch.sqrt(x.square().sum(-1) + 1e-24).reshape(2, 1, 1)),
                (
                    torch.tensor(
                        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                        device=self.device,
                        dtype=torch.float64,
                        requires_grad=True,
                    ),
                ),
            )
        )

    def test_four_forks_reset_state_and_resume_exactly(self):
        torch.manual_seed(372)
        parent = self.make_driver(parent=True)
        parent.run()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.pt"
            parent.save_checkpoint(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected_rng = torch.get_rng_state().clone()
            expected_cuda = (
                torch.cuda.get_rng_state().clone() if self.device == "cuda" else None
            )
            for arm, options in ARMS.items():
                with self.subTest(arm=arm):
                    fork = self.make_driver(
                        options["envelope_type"], options["clip_method"]
                    )
                    fork.fork_from_checkpoint(path, expected_sha256=digest)
                    self.assertEqual(fork.epoch, 0)
                    self.assertEqual(fork.stage, 2)
                    self.assertEqual(fork.history, [])
                    self.assertFalse(fork.optimizer.state)
                    self.assertEqual(len(fork.optimizer.param_groups), 2)
                    self.assertEqual(fork.scheduler.last_epoch, 0)
                    self.assertAlmostEqual(fork.optimizer.param_groups[0]["lr"], 1e-5)
                    self.assertAlmostEqual(fork.optimizer.param_groups[1]["lr"], 1e-6)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    if expected_cuda is not None:
                        self.assertTrue(
                            torch.equal(torch.cuda.get_rng_state(), expected_cuda)
                        )
                    torch.testing.assert_close(
                        fork.sampler.walkers, parent.sampler.walkers, rtol=0, atol=0
                    )
                    for key, value in parent.model.state_dict().items():
                        torch.testing.assert_close(
                            fork.model.state_dict()[key], value, rtol=0, atol=0
                        )
                    with torch.no_grad():
                        torch.testing.assert_close(
                            fork.sampler.current_log_abs_psi,
                            fork.model.forward_log_psi(fork.sampler.walkers)[0],
                            rtol=0,
                            atol=0,
                        )
                    fork.sampler.burn_in(2)
                    fork.sampler.adapt_step_size = False
                    fork.run(2)
                    checkpoint = Path(tmp) / f"{arm}.pt"
                    fork.save_checkpoint(checkpoint)
                    fork.run()
                    resumed = self.make_driver(
                        options["envelope_type"], options["clip_method"]
                    )
                    resumed.load_checkpoint(checkpoint)
                    resumed.run()
                    self.assertEqual(fork.training_history, resumed.training_history)
                    self.assertEqual(fork.experiment_origin, resumed.experiment_origin)
                    torch.testing.assert_close(
                        fork.sampler.walkers, resumed.sampler.walkers, rtol=0, atol=0
                    )
                    for key, value in fork.model.state_dict().items():
                        torch.testing.assert_close(
                            resumed.model.state_dict()[key], value, rtol=0, atol=0
                        )
            wrong = self.make_driver()
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                wrong.fork_from_checkpoint(path, expected_sha256="0" * 64)
            wrong.system.nuclear_coords[0, 0] = 1
            wrong.model.nuclear_coords[0, 0] = 1
            with self.assertRaisesRegex(ValueError, "system"):
                wrong.fork_from_checkpoint(path, expected_sha256=digest)
            legacy = torch.load(path, weights_only=True)
            identity = legacy.pop("model_identity")
            legacy_path = Path(tmp) / "legacy.pt"
            torch.save(legacy, legacy_path)
            legacy_sha = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "identity"):
                self.make_driver().fork_from_checkpoint(
                    legacy_path, expected_sha256=legacy_sha
                )
            self.make_driver().fork_from_checkpoint(
                legacy_path, expected_sha256=legacy_sha, parent_model_identity=identity
            )
            incompatible = self.make_driver("soft_exponential")
            with self.assertRaisesRegex(ValueError, "identity"):
                incompatible.load_checkpoint(path)

    def test_soft_envelope_complete_wavefunction_physics(self):
        torch.manual_seed(71)
        driver = self.make_driver("soft_exponential")
        model = driver.model
        directions = torch.cat((torch.eye(3), -torch.eye(3))).to(
            device=self.device, dtype=torch.float64
        )
        # Nuclear and unlike-spin s-wave spherical-average cusps and finite limits.
        for pair in (False, True):
            energies = []
            for distance in (1e-4, 1e-5):
                radius = torch.tensor(
                    distance,
                    device=self.device,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                fixed = radius.new_tensor([0.5, 0.7, 0.9]).expand(6, 3)
                moving = (
                    fixed if pair else driver.system.nuclear_coords[0]
                ) + radius * directions
                points = torch.stack((moving, fixed), 1)
                log, sign = model.forward_log_psi(points)
                amplitude = (sign * torch.exp(log - log.max().detach())).mean()
                self.assertNotEqual(float(amplitude.detach()), 0.0)
                slope = torch.autograd.grad(amplitude.abs().log(), radius)[0]
                energies.append(
                    compute_local_energy(
                        model, driver.system, points.detach().requires_grad_()
                    )[0]
                )
            self.assertAlmostEqual(slope.item(), 0.5 if pair else -1.0, delta=1e-3)
            self.assertLess(
                float((energies[0] - energies[1]).detach().abs().max()), 0.15
            )
        points = torch.randn(4, 2, 3, device=self.device, dtype=torch.float64) + 0.3
        with torch.no_grad():
            log, sign = model.forward_log_psi(points)
            swapped, ss = model.forward_log_psi(points[:, [1, 0]])
        torch.testing.assert_close(log, swapped, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(sign, ss, atol=0, rtol=0)
        many = (
            FermiNet(
                driver.system.nuclear_coords,
                driver.system.nuclear_charges,
                2,
                2,
                num_determinants=2,
                hidden_dim_one=16,
                hidden_dim_two=4,
                num_layers=1,
                envelope_type="soft_exponential",
                use_cusp_jastrow=True,
                spin_adapted=True,
            )
            .to(self.device)
            .double()
        )
        points = torch.randn(3, 4, 3, device=self.device, dtype=torch.float64) + 0.3
        with torch.no_grad():
            log, sign = many.forward_log_psi(points)
            swapped, ss = many.forward_log_psi(points[:, [1, 0, 2, 3]])
            torch.testing.assert_close(log, swapped, atol=1e-9, rtol=1e-9)
            torch.testing.assert_close(-sign, ss, atol=0, rtol=0)
            residual = 2 * sign * torch.exp(log - log.max())
            for i in (0, 1):
                for j in (2, 3):
                    perm = list(range(4))
                    perm[i], perm[j] = perm[j], perm[i]
                    lp, sp = many.forward_log_psi(points[:, perm])
                    residual = residual - sp * torch.exp(lp - log.max())
            self.assertLess(float(residual.norm()), 1e-9)
        triplet = (
            FermiNet(
                driver.system.nuclear_coords,
                driver.system.nuclear_charges,
                1,
                1,
                num_determinants=2,
                hidden_dim_one=16,
                hidden_dim_two=4,
                num_layers=1,
                envelope_type="soft_exponential",
                use_cusp_jastrow=True,
                spin_projector=SpinProjector(1, 1, "triplet"),
            )
            .to(self.device)
            .double()
        )
        # Remove the p-wave zero before checking its regular radial coefficient.
        for distance in (1e-4, 1e-5):
            radius = torch.tensor(
                distance, device=self.device, dtype=torch.float64, requires_grad=True
            )
            center = radius.new_tensor([0.5, 0.7, 0.9]).expand_as(directions)
            points = torch.stack(
                (
                    center - 0.5 * radius * directions,
                    center + 0.5 * radius * directions,
                ),
                1,
            )
            log, sign = triplet.forward_log_psi(points)
            coefficient = (
                sign * torch.exp(log - log.max().detach()) * directions[:, 0]
            ).mean() / radius
            slope = torch.autograd.grad(coefficient.abs().log(), radius)[0]
        self.assertAlmostEqual(slope.item(), 0.25, delta=1e-3)

    def test_observation_shards_reconstruct_energy_and_coordinates(self):
        driver = self.make_driver("soft_exponential")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = evaluate_frozen_wavefunction(
                driver.model,
                driver.system,
                num_walkers=8,
                num_steps=5,
                burn_in_steps=2,
                decorrelation_steps=1,
                energy_batch_size=3,
                device=self.device,
                dtype=torch.float64,
                verbose=False,
                trajectory_path=str(output / "energy.pt"),
                observations_path=str(output / "obs"),
                records_per_shard=2,
            )
            manifest = json.loads((output / "obs/manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual([s["records"] for s in manifest["shards"]], [2, 2, 1])
            shards = [
                torch.load(output / "obs" / s["file"], weights_only=True)
                for s in manifest["shards"]
            ]
            torch.testing.assert_close(
                torch.cat([s["energy"] for s in shards]),
                torch.load(output / "energy.pt", weights_only=True),
                rtol=0,
                atol=0,
            )
            for shard in shards:
                points = (
                    shard["positions"]
                    .reshape(-1, 2, 3)
                    .to(self.device)
                    .requires_grad_()
                )
                energy, kinetic, potential = compute_local_energy(
                    driver.model, driver.system, points
                )
                for name, value in [
                    ("energy", energy),
                    ("kinetic", kinetic),
                    ("potential", potential),
                ]:
                    torch.testing.assert_close(
                        shard[name].reshape(-1),
                        value.detach().cpu(),
                        rtol=1e-9,
                        atol=1e-9,
                    )
            summary = summarize_observations(
                output / "obs", driver.system.nuclear_coords
            )
            self.assertEqual(summary["total_samples"], 40)
            self.assertAlmostEqual(summary["energy_mean"], result["energy_mean"])
            with (
                patch(
                    "ferminet_pytorch.evaluation.compute_local_energy",
                    side_effect=FloatingPointError("test"),
                ),
                self.assertRaises(FloatingPointError),
            ):
                evaluate_frozen_wavefunction(
                    driver.model,
                    driver.system,
                    num_walkers=8,
                    num_steps=1,
                    burn_in_steps=0,
                    device=self.device,
                    dtype=torch.float64,
                    observations_path=str(output / "failed"),
                    verbose=False,
                )
            self.assertFalse(
                json.loads((output / "failed/manifest.json").read_text())["complete"]
            )


class TestFactorialCPU(FactorialRuntimeChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFactorialCUDA(FactorialRuntimeChecks, unittest.TestCase):
    device = "cuda"


class TestFactorialPlanning(unittest.TestCase):
    def test_timing_gate_is_energy_independent_and_cannot_launch_over_budget(self):
        plan = read_plan(False)
        rows = [
            {
                "step_seconds": [0.2] * 20,
                "rethermalization_seconds": 10.0,
                "evaluation": {
                    "total_with_observations_seconds": 44.8,
                    "burn_in_seconds": 6.0,
                    "time_records": 64,
                },
            }
        ] * 4
        timing = project_seconds(rows, plan)
        self.assertTrue(timing["within_budget"])
        rows = copy.deepcopy(rows)
        rows[0]["step_seconds"] = [1.0] * 20
        self.assertFalse(project_seconds(rows, plan)["within_budget"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "gate"):
                prepare("owner", "a" * 40, Path(tmp), smoke=False, gate=None)
            metadata = prepare("owner", "a" * 40, Path(tmp))
            self.assertEqual(
                metadata["kernel_sources"], [plan["fork"]["source_kernel"]]
            )
            self.assertIn(
                "scripts.factorial_campaign", (Path(tmp) / "run.py").read_text()
            )

    def test_balanced_schedule_unique_heldout_seeds_and_budgets(self):
        plan = json.loads((ROOT / "configs/factorial_t4x2.json").read_text())
        queues = schedule(plan)
        self.assertEqual([len(q) for q in queues], [6, 6])
        self.assertEqual(queues, schedule(plan))
        seeds = [seed_for(plan, s, a) for s in plan["seeds"] for a in ARMS]
        seeds += [
            seed_for(plan, s, zero=True, envelope_type=e)
            for s in plan["seeds"]
            for e in ("gaussian", "soft_exponential")
        ]
        self.assertEqual(len(seeds), len(set(seeds)))
        for s in plan["seeds"]:
            self.assertEqual([sum(c["seed"] == s for c in q) for q in queues], [2, 2])
        plan["training"]["num_epochs"] = 9999
        with self.assertRaises(ValueError):
            schedule(plan)

    def test_paired_effects_and_incomplete_or_uncertain_results(self):
        cases = []
        for seed in [11, 22, 33]:
            for arm, shift in {"A": 0, "B": -0.001, "C": -0.002, "D": -0.004}.items():
                cases.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "status": "completed",
                        "precision_after": {"passed": True},
                        "evaluation": {
                            "energy_mean": -1.17 + shift,
                            "energy_sem": 1e-7,
                            "diagnostics_status": "passed",
                            "extreme_precision_audit": {"passed": True},
                        },
                    }
                )
        result = paired_effects(cases, [11, 22, 33])
        self.assertAlmostEqual(result["effects"]["clipping"]["mean_mha"], -1.5)
        self.assertAlmostEqual(result["effects"]["envelope"]["mean_mha"], -2.5)
        self.assertAlmostEqual(result["effects"]["interaction"]["mean_mha"], -1.0)
        self.assertTrue(result["effects"]["clipping"]["improvement_supported"])
        self.assertIsNone(paired_effects(cases[:-1], [11, 22, 33])["effects"])
        cases[0]["evaluation"]["diagnostics_status"] = "insufficient_evidence"
        self.assertEqual(
            paired_effects(cases, [11, 22, 33])["status"], "insufficient_evidence"
        )


if __name__ == "__main__":
    unittest.main()
