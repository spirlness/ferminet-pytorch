"""Behavior shared by high-level training and directly constructed drivers."""

import copy
import tempfile
from pathlib import Path
import unittest

import torch

from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.systems import build_h2_system
from ferminet_pytorch.vmc_driver import TrainingConfig, VMCDriver, train_vmc


class TestUnifiedDriver(unittest.TestCase):
    def model(self, system, **kwargs):
        return FermiNet(
            system.nuclear_coords,
            system.nuclear_charges,
            1,
            1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=4,
            num_layers=1,
            **kwargs,
        ).double()

    def test_direct_and_high_level_training_have_identical_updates(self):
        system = build_h2_system(dtype=torch.float64)
        first = self.model(system)
        second = copy.deepcopy(first)
        config = TrainingConfig(
            num_epochs=2,
            num_walkers=8,
            mcmc_steps_per_epoch=1,
            burn_in_steps=0,
            device="cpu",
            dtype=torch.float64,
            restore_best=False,
            verbose=False,
            seed=91,
        )
        result = train_vmc(system, first, config)
        torch.manual_seed(config.seed)
        sampler = MCMCSampler(second, system, num_walkers=8, step_size=config.step_size)
        driver = VMCDriver(second, system, sampler, config=config)
        driver.run()
        self.assertEqual(result["history"], driver.training_history)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(
            result["sampler"].walkers, sampler.walkers, atol=0, rtol=0
        )

    def test_two_stage_driver_preserves_frozen_parameters_and_saves_updated_state(self):
        system = build_h2_system(dtype=torch.float64)
        model = self.model(system, use_cusp_jastrow=True, interaction_type="psiformer")
        initial = [p.detach().clone() for p in model.jastrow.parameters()]
        sampler = MCMCSampler(model, system, num_walkers=8)
        with tempfile.TemporaryDirectory() as directory:
            config = TrainingConfig(
                num_epochs=3,
                num_walkers=8,
                two_stage_jastrow=True,
                jastrow_stage1_epochs=1,
                jastrow_lr=0.01,
                mcmc_steps_per_epoch=1,
                restore_best=False,
                checkpoint_path=str(Path(directory) / "state.pt"),
            )
            driver = VMCDriver(model, system, sampler, config=config)

            def inspect(epoch, row):
                self.assertEqual(row["stage"], 1 if epoch == 1 else 2)
                if epoch == 1:
                    for before, after in zip(initial, model.jastrow.parameters()):
                        torch.testing.assert_close(before, after, atol=0, rtol=0)
                        self.assertIsNone(after.grad)
                expected = model.forward_log_psi(sampler.walkers)[0].detach()
                torch.testing.assert_close(
                    sampler.current_log_abs_psi, expected, atol=0, rtol=0
                )

            driver.run(callback=inspect)
            saved = torch.load(config.checkpoint_path, weights_only=True)
            self.assertEqual(saved["stage"], 2)
            self.assertEqual(saved["epoch"], 3)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    saved["model_state_dict"][key], value, atol=0, rtol=0
                )
            self.assertTrue(
                any(
                    not torch.equal(a, b)
                    for a, b in zip(initial, model.jastrow.parameters())
                )
            )


if __name__ == "__main__":
    unittest.main()
