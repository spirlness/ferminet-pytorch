"""Remote-run contracts: exact continuation and independent batched evaluation."""

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from ferminet_pytorch import FermiNet, TrainingConfig, VMCDriver
from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.systems import build_h2_system


class CampaignRuntimeChecks:
    device = "cpu"

    def make_driver(self, config):
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
        ).to(device=self.device, dtype=torch.float64)
        sampler = MCMCSampler(model, system, num_walkers=8)
        return VMCDriver(model, system, sampler, config=config)

    def test_resume_preserves_updates_rng_stages_and_cache(self):
        config = TrainingConfig(
            num_epochs=5,
            num_walkers=8,
            mcmc_steps_per_epoch=1,
            two_stage_jastrow=True,
            jastrow_stage1_epochs=2,
            restore_best=True,
            dtype=torch.float64,
        )
        for split in (1, 3):
            torch.manual_seed(84)
            uninterrupted = self.make_driver(config)
            uninterrupted.run(split)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "resume.pt"
                uninterrupted.save_checkpoint(path)
                uninterrupted.run(5 - split)
                expected_rng = torch.get_rng_state().clone()
                expected_cuda = (
                    torch.cuda.get_rng_state() if self.device == "cuda" else None
                )
                resumed = self.make_driver(config)
                resumed.load_checkpoint(path)
                resumed.run(5 - split)
            self.assertEqual(uninterrupted.training_history, resumed.training_history)
            for name, value in uninterrupted.model.state_dict().items():
                torch.testing.assert_close(
                    value, resumed.model.state_dict()[name], atol=0, rtol=0
                )
            torch.testing.assert_close(
                uninterrupted.sampler.walkers, resumed.sampler.walkers, atol=0, rtol=0
            )
            torch.testing.assert_close(
                resumed.sampler.current_log_abs_psi,
                resumed.model.forward_log_psi(resumed.sampler.walkers)[0].detach(),
                atol=0,
                rtol=0,
            )
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            if expected_cuda is not None:
                self.assertTrue(torch.equal(expected_cuda, torch.cuda.get_rng_state()))
            self.assertEqual(uninterrupted.best.keys(), resumed.best.keys())
            self.assertEqual(
                uninterrupted.sampler.proposed_moves, resumed.sampler.proposed_moves
            )
            self.assertEqual(
                uninterrupted.sampler.accepted_moves, resumed.sampler.accepted_moves
            )

    def test_resume_rejects_changed_training_schedule(self):
        config = TrainingConfig(num_epochs=5, num_walkers=8)
        original = self.make_driver(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            original.save_checkpoint(path)
            changed = self.make_driver(replace(config, num_epochs=6))
            with self.assertRaisesRegex(ValueError, "configuration"):
                changed.load_checkpoint(path)

    def test_frozen_energy_chunking_preserves_chain_axis_and_model(self):
        driver = self.make_driver(TrainingConfig(num_walkers=8))
        before = copy.deepcopy(driver.model.state_dict())
        kwargs = dict(
            model=driver.model,
            system=driver.system,
            num_walkers=8,
            num_steps=4,
            burn_in_steps=2,
            decorrelation_steps=1,
            dtype=torch.float64,
            device=self.device,
            verbose=False,
            seed=610,
        )
        updates = []
        with tempfile.TemporaryDirectory() as directory:
            full_path, chunk_path = (
                Path(directory) / "full.pt",
                Path(directory) / "chunk.pt",
            )
            full = evaluate_frozen_wavefunction(
                **kwargs, trajectory_path=str(full_path)
            )
            chunked = evaluate_frozen_wavefunction(
                **kwargs,
                energy_batch_size=3,
                trajectory_path=str(chunk_path),
                progress_callback=updates.append,
            )
            first, second = (
                torch.load(full_path, weights_only=True),
                torch.load(chunk_path, weights_only=True),
            )
            self.assertEqual(tuple(second.shape), (4, 8))
            torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)
            self.assertEqual(updates[0]["status"], "evaluation_burn_in")
            self.assertEqual(
                [
                    row["time_records"]
                    for row in updates
                    if row["status"] == "evaluating"
                ],
                [1, 2, 3, 4],
            )
        self.assertAlmostEqual(full["energy_mean"], chunked["energy_mean"], places=10)
        self.assertEqual(chunked["diagnostics_status"], "insufficient_evidence")
        for key, value in before.items():
            torch.testing.assert_close(
                value, driver.model.state_dict()[key], atol=0, rtol=0
            )


class TestCampaignCPU(CampaignRuntimeChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCampaignCUDA(CampaignRuntimeChecks, unittest.TestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
