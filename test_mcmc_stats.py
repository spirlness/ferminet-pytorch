"""链相关能量统计的回归测试。"""

import unittest

import torch

from ferminet_pytorch.mcmc_stats import estimate_mcmc_statistics


class TestMCMCStatistics(unittest.TestCase):
    def test_iid_samples_have_near_full_effective_sample_size(self) -> None:
        generator = torch.Generator().manual_seed(7)
        energies = torch.randn(2048, 8, generator=generator, dtype=torch.float64)
        stats = estimate_mcmc_statistics(energies)
        total = energies.numel()
        self.assertGreater(stats["ess"], 0.75 * total)
        self.assertLessEqual(stats["ess"], total)
        expected_sem = energies.std(unbiased=True).item() / total ** 0.5
        self.assertAlmostEqual(stats["energy_sem"], expected_sem, delta=0.2 * expected_sem)

    def test_ar1_sem_includes_autocorrelation_inflation(self) -> None:
        rho = 0.9
        generator = torch.Generator().manual_seed(11)
        time, walkers = 12000, 4
        noise = torch.randn(time, walkers, generator=generator, dtype=torch.float64)
        chains = torch.zeros_like(noise)
        for t in range(1, time):
            chains[t] = rho * chains[t - 1] + noise[t]
        chains = chains[2000:]
        stats = estimate_mcmc_statistics(chains)
        iid_sem = chains.std(unbiased=True).item() / chains.numel() ** 0.5
        inflation = stats["energy_sem"] / iid_sem
        expected = ((1.0 + rho) / (1.0 - rho)) ** 0.5
        self.assertAlmostEqual(inflation, expected, delta=0.8)

    def test_constant_and_short_chains_are_handled_explicitly(self) -> None:
        constant = torch.full((16, 3), -1.0, dtype=torch.float64)
        stats = estimate_mcmc_statistics(constant)
        self.assertEqual(stats["energy_variance"], 0.0)
        self.assertEqual(stats["energy_sem"], 0.0)
        self.assertEqual(stats["ess"], float(constant.numel()))

        short = estimate_mcmc_statistics(torch.ones(3, 2, dtype=torch.float64))
        self.assertTrue(torch.isnan(torch.tensor(short["r_hat"])))
        self.assertEqual(short["r_hat_status"], "insufficient_samples")

    def test_split_rhat_detects_inconsistent_chains(self) -> None:
        generator = torch.Generator().manual_seed(19)
        energies = torch.randn(512, 4, generator=generator, dtype=torch.float64)
        energies[:, 0] += 3.0
        stats = estimate_mcmc_statistics(energies)
        self.assertGreater(stats["r_hat"], 1.1)


if __name__ == "__main__":
    unittest.main()
