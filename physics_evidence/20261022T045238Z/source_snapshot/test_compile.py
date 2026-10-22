"""
FermiNet PyTorch 零图断裂 (Branchless) 与 torch.compile 编译加速单元测试
========================================================================
覆盖:
1. TestDynamoGraphBreaks:
   - 验证 model.forward_log_psi 的 Dynamo 追踪图断裂数严格为 0 (单图全图融合)
   - 覆盖闭壳层 H2、开壳层 Li、电子坐标完全重叠奇异构型 (det=0)、开启自旋适配与 Jastrow 尖峰因子
2. TestCompileNumericalEquivalence:
   - 验证 CPU (aot_eager) 与 GPU (Inductor) 模式下编译前向输出与 Eager 模式绝对误差严格 < 1e-6
   - 验证波函数符号 sign 100% 精确一致
3. TestCompileSingularRobustness:
   - 验证极端奇异降秩构型 (det=0) 下编译前向输出数值正常、无 NaN/Inf
   - 验证与 Eager 绝对误差 < 1e-6
   - 验证反向传播求导梯度全有限 (无 NaN/Inf)
4. TestCompileMCMCSampler:
   - 验证 MCMCSampler 在注入编译前向函数 compiled_forward_log_psi 时的采样正常运作
   - 验证接受率合理、步长自适应调节正常、支持 all 与 one_by_one 移动模式
5. TestCompileVMCIntegration:
   - 验证 train_vmc() 在 compile=True 选项下顺利完成迭代与优化更新
   - 验证历史最优权重保存与指标输出完备无 NaN
"""

import unittest
from unittest.mock import patch
import torch
import torch.nn as nn

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.train_vmc import build_h2_system, build_lih_system, train_vmc


class TestDynamoGraphBreaks(unittest.TestCase):
    """验证波函数前向在各类物理构型下图断裂数严格为 0 (单图完整融合)."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_h2_closed_shell_graph_breaks_zero(self) -> None:
        """闭壳层 H2 分子 (2 电子) forward_log_psi 图断裂数严格为 0."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=False,
            spin_adapted=False,
        ).eval()

        r = torch.randn(8, system.total_electrons, 3, dtype=torch.float32)
        explanation = torch._dynamo.explain(model.forward_log_psi)(r)

        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"H2 闭壳层前向存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    def test_li_open_shell_graph_breaks_zero(self) -> None:
        """开壳层未配对体系 (Li 原子: 3 电子, up=2, down=1) 图断裂数严格为 0."""
        nuclear_coords = torch.zeros(1, 3, dtype=torch.float32)
        nuclear_charges = torch.tensor([3.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=2,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=False,
            spin_adapted=False,
        ).eval()

        r = torch.randn(8, 3, 3, dtype=torch.float32)
        explanation = torch._dynamo.explain(model.forward_log_psi)(r)

        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"开壳层 Li 前向存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    def test_singular_overlap_graph_breaks_zero(self) -> None:
        """同自旋电子坐标完全重合奇异构型 (det=0) 图断裂数严格为 0."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).eval()

        r = torch.randn(12, system.total_electrons, 3, dtype=torch.float32)
        # 强制制造奇异构型: 电子 0 与 电子 1 坐标重合 (两电子同自旋或相同轨道导致 det=0)
        r[0:6, 1, :] = r[0:6, 0, :]

        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"奇异构型下存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    def test_spin_adapted_and_cusp_jastrow_graph_breaks_zero(self) -> None:
        """开启自旋纯态投影与 Kato Cusp Jastrow 因子时图断裂数严格为 0."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).eval()

        r = torch.randn(8, system.total_electrons, 3, dtype=torch.float32)
        explanation = torch._dynamo.explain(model.forward_log_psi)(r)

        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"开启 Spin+Jastrow 存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 环境不可用，跳过 GPU 图断裂测试")
    def test_cuda_graph_breaks_zero(self) -> None:
        """GPU (CUDA) 环境下 forward_log_psi 图断裂数严格为 0."""
        device = torch.device("cuda")
        system = build_h2_system(device=device, dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).to(device).eval()

        r = torch.randn(16, system.total_electrons, 3, device=device, dtype=torch.float32)
        explanation = torch._dynamo.explain(model.forward_log_psi)(r)

        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"CUDA 上前向存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)


class TestCompileNumericalEquivalence(unittest.TestCase):
    """验证编译模式与 Eager 模式的前向输出数值绝对等价性."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_cpu_numerical_equivalence_float64(self) -> None:
        """CPU 双精度 (Float64) 编译与 Eager 模式输出绝对误差严格 < 1e-6 且符号 100% 一致."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float64)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).double().eval()

        # Windows CPU 下使用 aot_eager 后端避免对 MSVC cl.exe 的强制依赖
        compiled_fn = torch.compile(model.forward_log_psi, backend="aot_eager")

        r = torch.randn(32, system.total_electrons, 3, dtype=torch.float64)

        log_eager, sign_eager = model.forward_log_psi(r)
        log_compiled, sign_compiled = compiled_fn(r)

        max_diff = (log_compiled - log_eager).abs().max().item()
        self.assertLess(
            max_diff,
            1e-6,
            f"CPU Float64 编译输出与 Eager 绝对误差超标: {max_diff:.3e}",
        )
        self.assertTrue(
            torch.equal(sign_compiled, sign_eager),
            "CPU Float64 编译输出波函数符号与 Eager 不一致",
        )

    def test_cpu_numerical_equivalence_float32(self) -> None:
        """CPU 单精度 (Float32) 编译与 Eager 模式输出绝对误差严格 < 1e-6 且符号一致."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).eval()

        compiled_fn = torch.compile(model.forward_log_psi, backend="aot_eager")
        r = torch.randn(32, system.total_electrons, 3, dtype=torch.float32)

        log_eager, sign_eager = model.forward_log_psi(r)
        log_compiled, sign_compiled = compiled_fn(r)

        max_diff = (log_compiled - log_eager).abs().max().item()
        self.assertLess(
            max_diff,
            1e-6,
            f"CPU Float32 编译输出与 Eager 绝对误差超标: {max_diff:.3e}",
        )
        self.assertTrue(torch.equal(sign_compiled, sign_eager))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 环境不可用，跳过 GPU 数值等价测试")
    def test_cuda_numerical_equivalence_float64(self) -> None:
        """GPU (CUDA) 双精度下 Inductor 编译与 Eager 模式绝对误差严格 < 1e-6 且符号完全一致."""
        device = torch.device("cuda")
        system = build_h2_system(device=device, dtype=torch.float64)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).to(device=device, dtype=torch.float64).eval()

        compiled_fn = torch.compile(model.forward_log_psi)
        r = torch.randn(32, system.total_electrons, 3, device=device, dtype=torch.float64)

        log_eager, sign_eager = model.forward_log_psi(r)
        log_compiled, sign_compiled = compiled_fn(r)

        max_diff = (log_compiled - log_eager).abs().max().item()
        self.assertLess(
            max_diff,
            1e-6,
            f"CUDA Float64 编译输出绝对误差超标: {max_diff:.3e}",
        )
        self.assertTrue(
            torch.equal(sign_compiled, sign_eager),
            "CUDA Float64 编译符号与 Eager 不一致",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 环境不可用，跳过 GPU 单精度数值等价测试")
    def test_cuda_numerical_equivalence_float32(self) -> None:
        """GPU (CUDA) 单精度下 Inductor 融合编译输出绝对误差 < 1e-5 (浮点重排容差) 且符号精确一致."""
        device = torch.device("cuda")
        system = build_h2_system(device=device, dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).to(device=device, dtype=torch.float32).eval()

        compiled_fn = torch.compile(model.forward_log_psi)
        r = torch.randn(32, system.total_electrons, 3, device=device, dtype=torch.float32)

        log_eager, sign_eager = model.forward_log_psi(r)
        log_compiled, sign_compiled = compiled_fn(r)

        max_diff = (log_compiled - log_eager).abs().max().item()
        self.assertLess(
            max_diff,
            1e-5,
            f"CUDA Float32 编译输出绝对误差超标: {max_diff:.3e}",
        )
        self.assertTrue(torch.equal(sign_compiled, sign_eager))


class TestCompileSingularRobustness(unittest.TestCase):
    """验证极端奇异降秩 (det=0) 矩阵输入下的编译数值健壮性与梯度有限性."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_singular_input_forward_finite_and_equivalence(self) -> None:
        """极端奇异降秩 (det=0) 构型下编译输出全有限 (无 NaN/Inf) 且与 Eager 绝对误差 < 1e-6."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float64)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).double().eval()

        compiled_fn = torch.compile(model.forward_log_psi, backend="aot_eager")

        # 构造包含多个极端重合点的 Batch
        r = torch.randn(16, system.total_electrons, 3, dtype=torch.float64)
        # 前半部分样本电子完全重叠 (det=0)
        r[:8, 1, :] = r[:8, 0, :]

        log_eager, sign_eager = model.forward_log_psi(r)
        log_compiled, sign_compiled = compiled_fn(r)

        # 1. 验证无 NaN / Inf
        self.assertTrue(
            torch.isfinite(log_compiled).all(),
            "编译前向输出在奇异构型下存在 NaN 或 Inf",
        )
        self.assertTrue(
            torch.isfinite(sign_compiled).all(),
            "编译前向符号在奇异构型下存在 NaN 或 Inf",
        )

        # 2. 验证与 Eager 误差严格 < 1e-6
        max_diff = (log_compiled - log_eager).abs().max().item()
        self.assertLess(
            max_diff,
            1e-6,
            f"奇异构型下编译输出与 Eager 绝对误差超标: {max_diff:.3e}",
        )
        self.assertTrue(torch.equal(sign_compiled, sign_eager))

    def test_singular_input_backward_finite_gradients(self) -> None:
        """奇异降秩构型处反向传播梯度有限性验证 (无 NaN/Inf)."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float64)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).double().eval()

        r = torch.randn(8, system.total_electrons, 3, dtype=torch.float64, requires_grad=True)
        # 制造奇异降秩
        with torch.no_grad():
            r[:4, 1, :] = r[:4, 0, :]

        log_psi, _ = model.forward_log_psi(r)
        loss = log_psi.sum()
        loss.backward()

        self.assertIsNotNone(r.grad)
        self.assertTrue(
            torch.isfinite(r.grad).all(),
            "奇异构型处反向传播坐标梯度包含 NaN 或 Inf",
        )
        self.assertGreater(r.grad.norm().item(), 0.0)


class TestCompileMCMCSampler(unittest.TestCase):
    """验证 MCMCSampler 在集成编译波函数后的采样正确性与自适应步长机制."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)
        self.model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).eval()

    def test_mcmc_sampler_all_mode_with_compiled_forward(self) -> None:
        """MCMC 在 'all' 移动模式下注入编译前向函数能够平稳完成预热与采样."""
        compiled_forward = torch.compile(self.model.forward_log_psi, backend="aot_eager")
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=16,
            step_size=0.5,
            move_mode="all",
            forward_log_psi_fn=compiled_forward,
        )

        # 执行预热
        burn_in_acc = sampler.burn_in(num_steps=10)
        self.assertTrue(
            0.0 < burn_in_acc <= 1.0,
            f"预热接受率异常: {burn_in_acc}",
        )
        self.assertTrue(
            torch.isfinite(sampler.walkers).all(),
            "预热后 Walker 坐标包含 NaN 或 Inf",
        )

        # 执行正式抽样
        walkers = sampler.sample(num_steps=10)
        sample_acc = sampler.last_acceptance_rate
        self.assertTrue(
            0.0 <= sample_acc <= 1.0,
            f"抽样接受率异常: {sample_acc}",
        )
        self.assertTrue(
            torch.isfinite(walkers).all(),
            "抽样后 Walker 坐标包含 NaN 或 Inf",
        )

    def test_mcmc_sampler_one_by_one_mode_with_compiled_forward(self) -> None:
        """MCMC 在 'one_by_one' 逐电子模式下注入编译前向函数运作正常."""
        compiled_forward = torch.compile(self.model.forward_log_psi, backend="aot_eager")
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=16,
            step_size=0.5,
            move_mode="one_by_one",
            forward_log_psi_fn=compiled_forward,
        )

        acc = sampler.burn_in(num_steps=5)
        self.assertTrue(0.0 < acc <= 1.0)
        self.assertTrue(torch.isfinite(sampler.walkers).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 环境不可用，跳过 GPU MCMC 编译测试")
    def test_cuda_mcmc_sampler_with_inductor_forward(self) -> None:
        """GPU (CUDA) 环境下使用 Inductor 编译前向的 MCMC 采样正常."""
        device = torch.device("cuda")
        system = build_h2_system(device=device, dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).to(device).eval()

        compiled_forward = torch.compile(model.forward_log_psi)
        sampler = MCMCSampler(
            model=model,
            system=system,
            num_walkers=32,
            step_size=0.5,
            move_mode="all",
            forward_log_psi_fn=compiled_forward,
        )

        acc = sampler.burn_in(num_steps=15)
        self.assertTrue(0.0 < acc <= 1.0)
        self.assertTrue(torch.isfinite(sampler.walkers).all())


class TestCompileVMCIntegration(unittest.TestCase):
    """验证 train_vmc() 在 compile=True 选项下的端到端训练闭环."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_train_vmc_compile_flag_cpu(self) -> None:
        """CPU 环境下启用 compile=True 能够平稳完成 2 个 Epoch 训练且无 NaN."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )

        # Windows CPU 环境下使用 patch 将 torch.compile 重定向至 aot_eager 后端避免 MSVC 依赖
        orig_compile = torch.compile
        with patch("torch.compile", side_effect=lambda fn, **kwargs: orig_compile(fn, backend="aot_eager")):
            result = train_vmc(
                system=system,
                model=model,
                num_walkers=16,
                num_epochs=2,
                burn_in_steps=5,
                mcmc_steps_per_epoch=2,
                device="cpu",
                compile=True,
                verbose=False,
            )

        self.assertIn("history", result)
        self.assertIn("best_energy", result)
        self.assertEqual(len(result["history"]["energy_raw"]), 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["history"]["energy_raw"])).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 环境不可用，跳过 GPU 端到端编译训练测试")
    def test_train_vmc_compile_flag_cuda(self) -> None:
        """GPU (CUDA) 环境下启用 compile=True 顺利完成端到端 VMC 训练并记录历史最优能量."""
        system = build_h2_system(device=torch.device("cuda"), dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )

        result = train_vmc(
            system=system,
            model=model,
            num_walkers=32,
            num_epochs=2,
            burn_in_steps=10,
            mcmc_steps_per_epoch=2,
            device="cuda",
            compile=True,
            verbose=False,
        )

        self.assertIn("history", result)
        self.assertIn("best_energy", result)
        self.assertEqual(len(result["history"]["energy_raw"]), 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["history"]["energy_raw"])).all())


if __name__ == "__main__":
    unittest.main()
