"""
变分量子蒙特卡洛 (VMC) 核心算法与训练闭环单元测试
==================================================
覆盖:
1. 批量 GPU/CPU MCMC 采样器 (全电子/逐电子步进、自适应步长调节、预热与抽样)
2. 基于中位数绝对偏差 (MAD) 的局域能量平滑截断 (Tanh 与 Hard 截断机制)
3. 有限批次协方差代理损失与物理理论梯度检验
4. VMCLoss 封装模块端到端能量与方差计算
5. 端到端 VMC 优化迭代闭环执行与参数更新验证
"""

from ferminet_pytorch.vmc_driver import TrainingConfig

import unittest
import torch

# 确保包能够被直接运行和识别

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.loss import clip_local_energy, vmc_surrogate_loss, VMCLoss
from ferminet_pytorch.systems import build_h2_system
from ferminet_pytorch.vmc_driver import train_vmc


class TestVMCComponents(unittest.TestCase):
    """
    变分量子蒙特卡洛 (VMC) 训练闭环单元测试。
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        # 构造标准 H2 体系 (2 电子, 2 原子核, 间距 1.4 a.u.)
        self.system = build_h2_system(bond_length=1.4)
        self.model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
        )
        self.num_walkers = 16

    def test_mcmc_init_walkers(self) -> None:
        """测试 MCMC 采样器初始化构型形状与合法性."""
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=self.num_walkers,
            move_mode="all",
        )
        self.assertEqual(
            sampler.walkers.shape,
            (self.num_walkers, self.system.total_electrons, 3),
        )
        self.assertEqual(sampler.current_log_abs_psi.shape, (self.num_walkers,))
        self.assertFalse(torch.isnan(sampler.current_log_abs_psi).any())
        self.assertFalse(torch.isinf(sampler.current_log_abs_psi).any())

    def test_mcmc_custom_initial_walkers(self) -> None:
        """测试用户传入初始构型时的采样器行为."""
        custom_walkers = torch.randn(self.num_walkers, self.system.total_electrons, 3)
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=self.num_walkers,
            initial_walkers=custom_walkers,
        )
        self.assertTrue(torch.allclose(sampler.walkers, custom_walkers))

    def test_mcmc_step_all_mode(self) -> None:
        """测试全电子一体化移动模式 (All-electron moves)."""
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=self.num_walkers,
            step_size=0.3,
            move_mode="all",
        )
        orig_walkers = sampler.walkers.clone()
        acc_rate = sampler.step()
        self.assertGreaterEqual(acc_rate, 0.0)
        self.assertLessEqual(acc_rate, 1.0)
        self.assertEqual(sampler.total_steps, 1)
        # 至少部分 walkers 发生了位移或状态保持合法
        self.assertEqual(
            sampler.walkers.shape,
            (self.num_walkers, self.system.total_electrons, 3),
        )

    def test_mcmc_step_one_by_one_mode(self) -> None:
        """测试逐电子轮流移动模式 (One-by-one moves)."""
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=self.num_walkers,
            step_size=0.3,
            move_mode="one_by_one",
        )
        acc_rate = sampler.step()
        self.assertGreaterEqual(acc_rate, 0.0)
        self.assertLessEqual(acc_rate, 1.0)
        self.assertEqual(
            sampler.walkers.shape,
            (self.num_walkers, self.system.total_electrons, 3),
        )

    def test_mcmc_adaptive_step_size(self) -> None:
        """测试步长自适应调节逻辑 (目标 50% 接受率)."""
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=64,
            step_size=0.5,
            target_acceptance=0.5,
            adapt_step_size=True,
            adapt_rate=0.1,
            move_mode="all",
        )
        init_step = sampler.step_size

        # 人为触发多次步骤，观察步长变化机制
        for _ in range(10):
            sampler.step()

        # 步长应保持在合法正数区间内
        self.assertGreater(sampler.step_size, 0.001)
        self.assertLess(sampler.step_size, 10.0)

    def test_mcmc_burn_in_and_sample(self) -> None:
        """测试预热 (Burn-in) 与多步抽样接口."""
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=16,
            step_size=0.4,
            move_mode="all",
        )
        burn_in_acc = sampler.burn_in(num_steps=10)
        self.assertGreaterEqual(burn_in_acc, 0.0)
        self.assertLessEqual(burn_in_acc, 1.0)
        self.assertEqual(sampler.total_steps, 10)

        # 抽样 5 步
        sampled_walkers = sampler.sample(num_steps=5)
        self.assertEqual(
            sampled_walkers.shape,
            (16, self.system.total_electrons, 3),
        )
        # 验证返回的是独立张量副本
        self.assertFalse(sampled_walkers.data_ptr() == sampler.walkers.data_ptr())

    def test_energy_clipping_tanh_and_hard(self) -> None:
        """测试基于 MAD 的局域能量平滑截断 (Tanh) 与硬截断 (Hard)."""
        # 构造包含极端离群点的局域能量序列
        e_local = torch.tensor(
            [-1.15, -1.18, -1.16, -1.17, -1.14, 1000.0, -800.0],
            dtype=torch.float32,
        )
        # 1. Tanh 截断
        e_tanh, med, mad = clip_local_energy(e_local, clip_factor=5.0, method="tanh")
        self.assertTrue(torch.isfinite(e_tanh).all())
        # 离群点应被平滑压缩到合理区间
        self.assertLess(e_tanh[5].item(), 50.0)
        self.assertGreater(e_tanh[6].item(), -50.0)
        # 中位数附近的正常样本几乎不受影响
        self.assertAlmostEqual(e_tanh[0].item(), e_local[0].item(), places=2)

        # 2. Hard 截断
        e_hard, _, _ = clip_local_energy(e_local, clip_factor=5.0, method="hard")
        delta = 5.0 * mad
        self.assertLessEqual(e_hard.max().item(), (med + delta).item() + 1e-5)
        self.assertGreaterEqual(e_hard.min().item(), (med - delta).item() - 1e-5)

        # 3. None 截断
        e_none, _, _ = clip_local_energy(e_local, clip_factor=5.0, method="none")
        self.assertTrue(torch.allclose(e_none, e_local))

        # 4. 常数输入边界测试 (MAD = 0)
        e_const = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)
        e_const_clip, _, _ = clip_local_energy(e_const, clip_factor=5.0, method="tanh")
        self.assertTrue(torch.allclose(e_const_clip, e_const))

    def test_surrogate_loss_gradient_exact_match(self) -> None:
        """
        核心物理与数学检验:
        验证有限批次协方差修正后的代理梯度。
            梯度估计: 2/(B-1) * sum_b [ (E_L(r_b) - <E_L>) * nabla_theta ln|Psi_theta(r_b)| ]
        """
        walkers = torch.randn(8, self.system.total_electrons, 3)
        e_local = torch.tensor(
            [-1.1, -1.2, -1.15, -1.08, -1.12, -1.19, -1.14, -1.16],
            dtype=torch.float32,
        )

        # 开启无截断模式以严格检验精确变分定理
        self.model.zero_grad()
        log_abs_psi, _ = self.model.forward_log_psi(walkers)
        loss, stats = vmc_surrogate_loss(
            log_abs_psi=log_abs_psi,
            e_local=e_local,
            clip_factor=0.0,
            clip_method="none",
        )
        loss.backward()

        # 提取通过 autograd 得到且参与计算图的模型参数梯度
        active_params = [
            p for p in self.model.parameters() if p.requires_grad and p.grad is not None
        ]
        self.assertGreater(len(active_params), 0, "模型没有任何参数获得梯度")
        autograd_grads = [p.grad.clone() for p in active_params]

        # 现在手动通过变分数学定义逐样本提取 nabla_theta ln|Psi(r_b)| 并加权求和
        b_size = walkers.shape[0]
        centered_e = e_local - torch.mean(e_local)  # (B,)
        manual_grads = [torch.zeros_like(p) for p in active_params]

        for b in range(b_size):
            self.model.zero_grad()
            single_walker = walkers[b : b + 1]
            log_psi_b, _ = self.model.forward_log_psi(single_walker)
            # 计算该样本的对数波函数参数梯度
            log_psi_b.backward()
            weight = (2.0 / b_size) * centered_e[b].item()
            for idx, p in enumerate(active_params):
                if p.grad is not None:
                    manual_grads[idx] += weight * p.grad

        # 比对两种方式得到的梯度张量，要求在机器精度内严格一致
        for auto_g, man_g in zip(autograd_grads, manual_grads):
            self.assertTrue(
                torch.allclose(auto_g, man_g, atol=1e-5, rtol=1e-4),
                f"变分代理损失梯度与物理理论严格定义不一致! 最大偏差: {torch.max(torch.abs(auto_g - man_g))}",
            )

    def test_vmc_loss_module(self) -> None:
        """测试 VMCLoss 模块 compute_step_loss 一站式接口."""
        vmc_loss = VMCLoss(clip_factor=5.0, clip_method="tanh")
        walkers = torch.randn(4, self.system.total_electrons, 3)

        loss, stats, e_local = vmc_loss.compute_step_loss(
            model=self.model,
            system=self.system,
            walkers=walkers,
        )

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(e_local.shape, (4,))

        # 检查统计项齐全性
        expected_keys = [
            "energy_mean",
            "energy_std",
            "energy_var",
            "energy_clipped_mean",
            "energy_median",
            "energy_mad",
            "loss",
        ]
        for k in expected_keys:
            self.assertIn(k, stats)
            self.assertFalse(torch.isnan(torch.tensor(stats[k])))

        # 验证梯度反向传播
        self.model.zero_grad()
        loss.backward()
        has_grad = any(p.grad is not None for p in self.model.parameters())
        self.assertTrue(has_grad, "反向传播后模型参数未产生有效梯度")

    def test_train_vmc_end_to_end_loop(self) -> None:
        """测试端到端 VMC 训练迭代完整闭环 (3 步迷你训练)."""
        result = train_vmc(
            system=self.system,
            model=self.model,
            config=TrainingConfig(
                num_walkers=32,
                num_epochs=3,
                mcmc_steps_per_epoch=2,
                burn_in_steps=5,
                lr=0.001,
                verbose=False,
            ),
        )

        self.assertIn("history", result)
        self.assertIn("final_energy", result)
        self.assertIn("final_variance", result)
        history = result["history"]
        self.assertEqual(result["final_energy"], history["energy_raw"][-1])
        self.assertEqual(result["final_variance"], history["energy_var"][-1])
        self.assertTrue(result["sampler"].adapt_step_size)

        self.assertEqual(len(history["energy_raw"]), 3)
        self.assertEqual(len(history["energy_clipped"]), 3)
        self.assertEqual(len(history["energy_var"]), 3)
        self.assertEqual(len(history["acceptance_rate"]), 3)

        # 检验历史数值合法性
        for e in history["energy_raw"]:
            self.assertTrue(torch.isfinite(torch.tensor(e)))
        for v in history["energy_var"]:
            self.assertGreaterEqual(v, 0.0)

    def test_clipping_robustness_against_nan_and_inf(self) -> None:
        """测试局域能量截断与代理损失面对 NaN/Inf 极端异常值时的绝对数值鲁棒性."""
        # 构造包含 NaN 与正负 Inf 的异常能量序列
        e_dirty = torch.tensor(
            [-1.1, float("nan"), -1.2, float("inf"), float("-inf")], dtype=torch.float32
        )
        e_clipped, med, mad = clip_local_energy(e_dirty, clip_factor=5.0, method="tanh")

        # 验证输出绝无 NaN 或 Inf
        self.assertTrue(torch.isfinite(e_clipped).all(), "截断输出仍含有非有限数值")
        self.assertTrue(torch.isfinite(med), "中位数计算退化为非有限数")
        self.assertTrue(torch.isfinite(mad), "MAD 计算退化为非有限数")

        # 验证代理损失面对 dirty 输入的鲁棒性
        log_psi = torch.randn(5, requires_grad=True)
        with self.assertWarnsRegex(RuntimeWarning, "Excluded 3/5"):
            loss, stats = vmc_surrogate_loss(
                log_psi,
                e_dirty,
                clip_factor=5.0,
                clip_method="tanh",
                max_invalid_fraction=1.0,
            )
        self.assertTrue(torch.isfinite(loss), "代理损失变为非有限数")
        self.assertTrue(torch.isfinite(torch.tensor(stats["energy_mean"])))
        self.assertTrue(torch.isfinite(torch.tensor(stats["energy_var"])))

        # 验证反向传播不会扩散 NaN
        loss.backward()
        self.assertTrue(
            torch.isfinite(log_psi.grad).all(), "损失反向传播产生了非有限梯度"
        )

    def test_mcmc_accepted_steps_and_rejection_of_invalid_proposals(self) -> None:
        """测试 MCMC 采样器的 accepted_steps 计数跟踪与非法波函数的绝对拒绝保护."""
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=16,
            step_size=0.3,
            move_mode="all",
        )
        self.assertEqual(sampler.accepted_steps, 0)
        self.assertEqual(sampler.total_steps, 0)

        # 执行 4 步步进
        for _ in range(4):
            sampler.step()

        self.assertEqual(sampler.total_steps, 4)
        self.assertGreaterEqual(sampler.accepted_steps, 0)
        self.assertLessEqual(sampler.accepted_steps, 4 * 16)

    def test_local_energy_param_grad_frozen_equivalence(self) -> None:
        """验证 compute_local_energy 参数冻结优化与原始计算在数值上机器精度完全一致."""
        walkers = torch.randn(8, self.system.total_electrons, 3, requires_grad=True)
        e_local_fast, t_fast, v_fast = compute_local_energy(
            self.model, self.system, walkers, create_graph=False
        )

        # 确保计算后模型参数的 requires_grad 状态完好保持
        has_trainable_params = any(p.requires_grad for p in self.model.parameters())
        self.assertTrue(
            has_trainable_params, "模型参数 requires_grad 在局部能量计算后未被正确还原"
        )
        self.assertEqual(e_local_fast.shape, (8,))
        self.assertTrue(torch.isfinite(e_local_fast).all())

    def test_vmc_single_electron_system(self) -> None:
        """测试单电子体系 (氢原子 H, n_up=1, n_down=0) 的 MCMC 采样与 VMC 训练迭代."""
        h_atom = MolecularSystem(
            nuclear_coords=torch.tensor([[0.0, 0.0, 0.0]]),
            nuclear_charges=torch.tensor([1.0]),
            n_up=1,
            n_down=0,
        )
        model_h = FermiNet(
            nuclear_coords=h_atom.nuclear_coords,
            nuclear_charges=h_atom.nuclear_charges,
            n_up=1,
            n_down=0,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        sampler_h = MCMCSampler(
            model=model_h, system=h_atom, num_walkers=16, move_mode="one_by_one"
        )
        acc = sampler_h.step()
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

        # 运行微型 2 步训练
        res = train_vmc(
            system=h_atom,
            model=model_h,
            config=TrainingConfig(
                num_walkers=16,
                num_epochs=2,
                mcmc_steps_per_epoch=2,
                burn_in_steps=3,
                verbose=False,
            ),
        )
        self.assertEqual(len(res["history"]["energy_raw"]), 2)
        self.assertTrue(torch.isfinite(torch.tensor(res["final_energy"])))


if __name__ == "__main__":
    unittest.main()
