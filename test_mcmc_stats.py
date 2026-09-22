"""链相关能量统计的回归测试。"""

import math
import os
import sys
import unittest
import warnings

# 确保包能够被直接运行和识别
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from ferminet_pytorch.mcmc_stats import estimate_mcmc_statistics


def _ar1_chains(time: int, walkers: int, rho: float, seed: int,
                burn_in: int = 2000) -> torch.Tensor:
    """生成 AR(1) 链 x_t = rho * x_{t-1} + eps_t，解析统计量已知。

    平稳后统计低效率 g = 1 + 2*tau_int = (1+rho)/(1-rho)，即 tau_int = 0.5*(1+rho)/(1-rho)。
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(time + burn_in, walkers, generator=generator, dtype=torch.float64)
    chains = torch.zeros_like(noise)
    for t in range(1, time + burn_in):
        chains[t] = rho * chains[t - 1] + noise[t]
    return chains[burn_in:]


def _true_tau_int(rho: float) -> float:
    """AR(1) 的解析积分自相关时间。"""
    return 0.5 * (1.0 + rho) / (1.0 - rho)


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

    # ------------------------------------------------------------------
    # 解析对照：AR(1) 链的 tau_int 已知，可直接检验估计器偏差
    # ------------------------------------------------------------------
    def test_ar1_tau_int_recovers_analytic_value_on_long_chains(self) -> None:
        """长链上 tau_int 应收敛到解析值 0.5*(1+rho)/(1-rho)，ESS 比值应同步收敛。"""
        rho = 0.9
        chains = _ar1_chains(time=8192, walkers=64, rho=rho, seed=0)
        stats = estimate_mcmc_statistics(chains)
        expected_tau = _true_tau_int(rho)
        self.assertAlmostEqual(stats["tau_int"], expected_tau, delta=0.5)
        self.assertAlmostEqual(
            stats["ess"] / chains.numel(),
            1.0 / (1.0 + 2.0 * expected_tau),
            delta=0.01,
        )

    def test_short_chain_is_unresolved_before_positive_rho_zero_pair_is_trusted(self) -> None:
        """R1: a positive (rho_0, rho_1) pair cannot certify a short chain."""
        rho = 0.95
        expected_tau = _true_tau_int(rho)
        for time, max_fraction in ((32, None), (128, 0.70)):
            with self.subTest(time=time):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    stats = estimate_mcmc_statistics(_ar1_chains(time=time, walkers=256, rho=rho, seed=0))
                if max_fraction is None:
                    self.assertEqual(stats["autocorrelation_status"], "unresolved")
                    self.assertTrue(math.isnan(stats["tau_int"]))
                    continue
                self.assertLess(stats["tau_int"], max_fraction * expected_tau)
                # ESS 相应被高估：远高于真实比值 1/(1+2*tau)
                self.assertGreater(stats["ess"] / (time * 256), 1.5 / (1.0 + 2.0 * expected_tau))

    def test_mixed_reversible_modes_use_rho_zero_first_geyer_pair(self) -> None:
        """R1 regression: mixed positive/negative modes have g=5.0556 analytically."""
        positive, negative = 0.8, -0.5
        target_g = 5.0556
        weight = (target_g - (1 + negative) / (1 - negative)) / (
            (1 + positive) / (1 - positive) - (1 + negative) / (1 - negative)
        )
        generator = torch.Generator().manual_seed(917)
        time, walkers, burn_in = 16000, 16, 1000
        def ar1(rho: float) -> torch.Tensor:
            noise = (1.0 - rho ** 2) ** 0.5 * torch.randn(
                time + burn_in, walkers, generator=generator, dtype=torch.float64
            )
            values = torch.zeros_like(noise)
            for index in range(1, time + burn_in):
                values[index] = rho * values[index - 1] + noise[index]
            return values[burn_in:]
        chains = (weight ** 0.5) * ar1(positive) + ((1 - weight) ** 0.5) * ar1(negative)
        stats = estimate_mcmc_statistics(chains)
        self.assertEqual(stats["autocorrelation_status"], "ok")
        self.assertAlmostEqual(stats["statistical_inefficiency"], target_g, delta=0.1 * target_g)

    def test_unresolvable_autocorrelation_is_reported_as_nan_not_full_efficiency(self) -> None:
        """D1 已修复：链长不足时必须显式报告无法解析，而不是静默返回 ESS = N。

        短链扣除时间均值会系统性地引入约 1/T 的负自相关偏置，使 Geyer 初始正序列
        在首个 lag 即截断。修复前该路径返回 tau_int = 0.5、ESS = 总样本数，
        对真实效率仅 2.5% 的链报告 100% 效率，且**不发出任何警告**。
        现在应返回 nan、置 autocorrelation_status 为 "unresolved" 并发出 UserWarning。
        """
        chains = _ar1_chains(time=4, walkers=32, rho=0.95, seed=0, burn_in=0)
        with self.assertWarns(UserWarning):
            stats = estimate_mcmc_statistics(chains)
        self.assertEqual(stats["autocorrelation_status"], "unresolved")
        for key in ("tau_int", "statistical_inefficiency", "ess", "energy_sem"):
            self.assertTrue(math.isnan(stats[key]), f"{key} 应为 nan，实为 {stats[key]}")
        # 均值与方差本身与自相关无关，必须仍然有效
        self.assertTrue(math.isfinite(stats["energy_mean"]))
        self.assertTrue(math.isfinite(stats["energy_variance"]))

    def test_too_few_time_steps_are_reported_as_insufficient_samples(self) -> None:
        """D1：时间步数少到无法构成一个 lag 配对时，同样必须显式上报而非假定独立。"""
        chains = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        with self.assertWarns(UserWarning):
            stats = estimate_mcmc_statistics(chains)
        self.assertEqual(stats["autocorrelation_status"], "insufficient_samples")
        self.assertTrue(math.isnan(stats["tau_int"]))
        self.assertTrue(math.isnan(stats["ess"]))

    def test_long_iid_chain_is_not_falsely_flagged_as_unresolved(self) -> None:
        """回归保护：长链上"确实无自相关"必须仍判为有效，不能被过度标记。

        独立样本的 Gamma_1 约有一半概率 <= 0，此时 g = 1 是**正确**结论；
        只有当链长不足以支撑该结论时才应判为 unresolved。
        """
        for seed in (7, 11, 19):
            with self.subTest(seed=seed):
                generator = torch.Generator().manual_seed(seed)
                chains = torch.randn(2048, 8, generator=generator, dtype=torch.float64)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    stats = estimate_mcmc_statistics(chains)
                self.assertEqual(stats["autocorrelation_status"], "ok")
                self.assertTrue(math.isfinite(stats["energy_sem"]))
                self.assertGreater(stats["ess"], 0.75 * chains.numel())

    def test_ar1_sem_inflation_tracks_analytic_ratio_on_long_chains(self) -> None:
        """长链上 SEM 相对 iid 的膨胀因子应等于 sqrt((1+rho)/(1-rho))。"""
        rho = 0.9
        chains = _ar1_chains(time=8192, walkers=64, rho=rho, seed=3)
        stats = estimate_mcmc_statistics(chains)
        iid_sem = chains.std(unbiased=True).item() / chains.numel() ** 0.5
        inflation = stats["energy_sem"] / iid_sem
        expected = ((1.0 + rho) / (1.0 - rho)) ** 0.5
        self.assertAlmostEqual(inflation, expected, delta=0.3)


if __name__ == "__main__":
    unittest.main()
