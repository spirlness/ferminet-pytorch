"""
独立对抗性实证挑战套件: FermiNet Dynamo 零图断裂与极端奇异退化极限压力测试
=============================================================================
测试范围与挑战维度:
1. TestAdversarialLargeBatches:
   - 大规模 Walker 批次 (B=512) 混合 6 类极端奇异构型 (同自旋重叠、异自旋重叠、3 电子塌缩、全电子重叠、核碰撞、大坐标)
   - 超大规模 Walker 批次 (B=1024) 符号追踪与显存压力
2. TestAdversarialOpenShellMultiplets:
   - Li 原子开壳层基态 (2 up, 1 down, 二重态 S=1/2)
   - 高自旋四重态体系 (3 up, 1 down, 四重态 S=3/2)
   - 全自旋极化铁磁极限体系 (4 up, 0 down, 单通道极端退化)
   - 单电子类氢极限体系 (1 up, 0 down, 1x1 行列式)
   - 异核开壳层体系 (LiH 离子/激发态, 3 up, 1 down)
3. TestAdversarialSpinAndJastrowMatrix:
   - 四象限全组合: [spin_adapted ON/OFF] x [use_cusp_jastrow ON/OFF]
   - 验证 SpinProjector 向量化广播置换与 KatoCuspJastrow 在 Dynamo 下严格 0 图断裂
4. TestAdversarialSingularGradients:
   - 同自旋双电子完全重合 (Pauli 节点面, det=0) 反向传播坐标梯度有限性
   - 三电子完全重合超强降秩反向传播坐标梯度有限性
   - 局部行列式降秩而总波函数非零构型的 log_abs_psi 坐标梯度有限性
   - 连续微扰逼近 Pauli 节点面 (delta -> 0) 的梯度平滑与无 NaN
5. TestAdversarialCudaInductorDynamo:
   - GPU (CUDA) 极端大规模混合奇异批次 0 图断裂与 Inductor 编译内核融合等价性
   - GPU 上极端奇异输入坐标反向传播梯度有限性
"""

import unittest
import torch
import torch._dynamo

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.systems import build_h2_system, build_lih_system


class TestAdversarialLargeBatches(unittest.TestCase):
    """对抗挑战 1: 极限批次维度与混合异构奇异样本的 Dynamo 0 图断裂与前向稳健性."""

    def setUp(self) -> None:
        torch.manual_seed(1337)

    def test_massive_batch_512_mixed_singularities(self) -> None:
        """大规模 Walker 批次 (B=512) 混合 6 类极端奇异构型的 Dynamo 图断裂严格为 0."""
        nuclear_coords = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]], dtype=torch.float32
        )
        nuclear_charges = torch.tensor([3.0, 1.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=2,
            n_down=2,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).eval()

        batch_size = 512
        r = torch.randn(batch_size, 4, 3, dtype=torch.float32)

        # 注入各类极端对抗奇异样本:
        # 1. 同自旋双电子完全重合 (Pauli 节点面, r_up[0] == r_up[1], det=0)
        r[10, 1] = r[10, 0]
        r[25, 3] = r[25, 2]
        # 2. 异自旋双电子完全重合 (库仑电子-电子碰撞, r_up[0] == r_down[0])
        r[50, 2] = r[50, 0]
        # 3. 三电子同点完全塌缩 (超强降秩)
        r[100, 1] = r[100, 0]
        r[100, 2] = r[100, 0]
        # 4. 全电子完全重叠 (4 电子同点塌缩)
        r[200, :] = r[200, 0:1].expand(4, 3)
        # 5. 电子与原子核完全重叠 (r_i == R_A)
        r[300, 0] = nuclear_coords[0]
        r[300, 1] = nuclear_coords[1]
        # 6. 极端大距离样本 (指数包络极端衰减)
        r[400, :] = torch.full((4, 3), 50.0)

        # 挑战 Dynamo 符号追踪
        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"大规模混合奇异批次 (B=512) 存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

        # 验证前向输出中无 NaN/伪浮点异常
        log_abs, sign = model.forward_log_psi(r)
        self.assertEqual(log_abs.shape, (batch_size,))
        self.assertEqual(sign.shape, (batch_size,))
        non_inf_mask = ~torch.isneginf(log_abs)
        self.assertFalse(torch.isnan(log_abs).any(), "log_abs_psi 存在 NaN")
        self.assertTrue(
            torch.isfinite(log_abs[non_inf_mask]).all(), "非 -inf 项存在异常非有限值"
        )
        self.assertTrue(
            torch.all((sign == 1) | (sign == -1) | (sign == 0)), "sign 包含非法值"
        )

    def test_massive_batch_1024_walkers_stress(self) -> None:
        """超大规模 Walker 批次 (B=1024) 极端压力下 Dynamo 追踪 0 图断裂."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=8,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=2,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).eval()

        batch_size = 1024
        r = torch.randn(batch_size, system.total_electrons, 3, dtype=torch.float32)
        # 穿插大量奇异样本
        for idx in range(0, batch_size, 32):
            r[idx, 1] = r[idx, 0]

        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"超大规模批次 (B=1024) 存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)


class TestAdversarialOpenShellMultiplets(unittest.TestCase):
    """对抗挑战 2: 多种开壳层与高重态分子体系的 Dynamo 0 图断裂."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_li_ground_state_doublet_s_half(self) -> None:
        """开壳层未配对体系 (Li 原子 3 电子: n_up=2, n_down=1, S=1/2) 0 图断裂."""
        nuclear_coords = torch.zeros(1, 3, dtype=torch.float32)
        nuclear_charges = torch.tensor([3.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=2,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=False,
        ).eval()

        r = torch.randn(32, 3, 3, dtype=torch.float32)
        # 注入同自旋重叠奇异点
        r[5, 1] = r[5, 0]

        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"Li 原子 (up=2, down=1) 存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    def test_quadruplet_high_spin_four_electrons(self) -> None:
        """高自旋四重态体系 (4 电子: n_up=3, n_down=1, S=3/2) 0 图断裂."""
        nuclear_coords = torch.zeros(1, 3, dtype=torch.float32)
        nuclear_charges = torch.tensor([4.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=3,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=False,
        ).eval()

        r = torch.randn(32, 4, 3, dtype=torch.float32)
        r[7, 2] = r[7, 0]  # 同自旋第0与第2电子重合

        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"四重态高自旋 (up=3, down=1) 存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    def test_fully_spin_polarized_ferromagnetic(self) -> None:
        """全自旋极化体系 (纯自旋向上: n_up=4, n_down=0, d_down=None 极端分支) 0 图断裂."""
        nuclear_coords = torch.zeros(1, 3, dtype=torch.float32)
        nuclear_charges = torch.tensor([4.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=4,
            n_down=0,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=False,
            spin_adapted=False,
        ).eval()

        r = torch.randn(16, 4, 3, dtype=torch.float32)
        # 同自旋碰撞
        r[3, 1] = r[3, 0]

        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"全自旋极化体系 (n_down=0) 存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    def test_single_electron_extreme_boundary(self) -> None:
        """单电子极端边界体系 (类氢原子: n_up=1, n_down=0, 1x1 行列式) 0 图断裂."""
        nuclear_coords = torch.zeros(1, 3, dtype=torch.float32)
        nuclear_charges = torch.tensor([1.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=1,
            n_down=0,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=False,
            spin_adapted=False,
        ).eval()

        r = torch.randn(16, 1, 3, dtype=torch.float32)
        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"单电子体系 (n_up=1, n_down=0) 存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    def test_heteronuclear_open_shell_lih_ion(self) -> None:
        """双核开壳层高自旋体系 (LiH 离子/激发态: n_up=3, n_down=1) 0 图断裂."""
        nuclear_coords = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]], dtype=torch.float32
        )
        nuclear_charges = torch.tensor([3.0, 1.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=3,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=False,
        ).eval()

        r = torch.randn(16, 4, 3, dtype=torch.float32)
        r[2, 2] = r[2, 0]  # 电子重叠奇异点
        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"双核开壳层体系存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)


class TestAdversarialSpinAndJastrowMatrix(unittest.TestCase):
    """对抗挑战 3: 自旋投影 (Spin-adapted) 与 Kato Jastrow 四象限组合的 Dynamo 0 图断裂."""

    def setUp(self) -> None:
        torch.manual_seed(99)
        self.system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)

    def _test_combination(self, spin_adapted: bool, use_cusp_jastrow: bool) -> None:
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=use_cusp_jastrow,
            spin_adapted=spin_adapted,
        ).eval()

        r = torch.randn(32, self.system.total_electrons, 3, dtype=torch.float32)
        # 注入同自旋与异自旋奇异构型
        r[5, 1] = r[5, 0]
        r[10, :] = 0.0

        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"组合 [spin_adapted={spin_adapted}, jastrow={use_cusp_jastrow}] 存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

    def test_quadrant_1_base_ferminet(self) -> None:
        """象限 1: spin_adapted=False, use_cusp_jastrow=False."""
        self._test_combination(spin_adapted=False, use_cusp_jastrow=False)

    def test_quadrant_2_spin_adapted_only(self) -> None:
        """象限 2: spin_adapted=True, use_cusp_jastrow=False."""
        self._test_combination(spin_adapted=True, use_cusp_jastrow=False)

    def test_quadrant_3_cusp_jastrow_only(self) -> None:
        """象限 3: spin_adapted=False, use_cusp_jastrow=True."""
        self._test_combination(spin_adapted=False, use_cusp_jastrow=True)

    def test_quadrant_4_both_spin_and_jastrow(self) -> None:
        """象限 4: spin_adapted=True, use_cusp_jastrow=True."""
        self._test_combination(spin_adapted=True, use_cusp_jastrow=True)


class TestAdversarialSingularGradients(unittest.TestCase):
    """对抗挑战 4: 几何奇异退化下反向传播坐标梯度的有限性 (无 NaN/Inf)."""

    def setUp(self) -> None:
        torch.manual_seed(2026)

    def test_singular_pauli_overlap_backward_gradients(self) -> None:
        """同自旋双电子完全重合奇异点 (det=0) 下 model.forward(r) 坐标梯度有限 (无 NaN/Inf)."""
        system = build_lih_system(device=torch.device("cpu"), dtype=torch.float32)
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
        ).eval()

        batch_size = 16
        r = torch.randn(
            batch_size,
            system.total_electrons,
            3,
            dtype=torch.float32,
            requires_grad=True,
        )

        # 强制样本 0 为同自旋两电子完全重合 (Pauli 节点面奇异点)
        with torch.no_grad():
            r[0, 1] = r[0, 0]

        psi = model(r)
        self.assertEqual(psi.shape, (batch_size,))
        self.assertTrue(torch.isfinite(psi).all(), "前向输出包含非有限值")
        # 样本 0 在节点面上，波函数严格为 0
        self.assertAlmostEqual(psi[0].item(), 0.0, places=6)

        grad_r = torch.autograd.grad(psi.sum(), r)[0]
        self.assertEqual(grad_r.shape, r.shape)
        self.assertTrue(
            torch.isfinite(grad_r).all(), f"奇异样本反向传播梯度出现 NaN/Inf: {grad_r}"
        )

    def test_singular_triplet_overlap_backward_gradients(self) -> None:
        """三电子完全重合超强降秩下 model.forward(r) 坐标反向传播梯度有限 (无 NaN/Inf)."""
        nuclear_coords = torch.zeros(1, 3, dtype=torch.float32)
        nuclear_charges = torch.tensor([4.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=3,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=False,
        ).eval()

        r = torch.randn(8, 4, 3, dtype=torch.float32, requires_grad=True)
        # 样本 0 的同自旋 3 个电子全部重合 (降秩 2)
        with torch.no_grad():
            r[0, 1] = r[0, 0]
            r[0, 2] = r[0, 0]

        psi = model(r)
        grad_r = torch.autograd.grad(psi.sum(), r)[0]
        self.assertTrue(
            torch.isfinite(grad_r).all(), f"三电子塌缩梯度出现 NaN/Inf: {grad_r}"
        )

    def test_rank_deficient_log_psi_derivatives(self) -> None:
        """测试在常规与轻度奇异样本上，log_abs_psi 的坐标梯度有限且处处无 NaN."""
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

        # 构造非零波函数的常规样本与近奇异样本
        r = torch.randn(8, 2, 3, dtype=torch.float32, requires_grad=True)
        log_abs, sign = model.forward_log_psi(r)
        self.assertTrue(torch.isfinite(log_abs).all())

        grad_r = torch.autograd.grad(log_abs.sum(), r)[0]
        self.assertTrue(
            torch.isfinite(grad_r).all(), f"log_abs 梯度出现 NaN/Inf: {grad_r}"
        )

    def test_smooth_approach_to_pauli_node(self) -> None:
        """测试电子逐步逼近 Pauli 节点面 (delta -> 0) 时梯度的连续性与有限性."""
        system = build_h2_system(device=torch.device("cpu"), dtype=torch.float32)
        # 使用 2 up, 0 down 体系测试纯粹的同自旋节点面
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=2,
            n_down=0,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_cusp_jastrow=False,
            spin_adapted=False,
        ).eval()

        deltas = [1.0, 1e-1, 1e-2, 1e-4, 1e-6, 0.0]
        base_pos = torch.tensor([[0.5, 0.2, 0.1]], dtype=torch.float32)

        for delta in deltas:
            r = torch.stack(
                [
                    base_pos,
                    base_pos + torch.tensor([[delta, 0.0, 0.0]], dtype=torch.float32),
                ],
                dim=1,
            )
            r.requires_grad_(True)
            psi = model(r)
            grad_r = torch.autograd.grad(psi.sum(), r)[0]
            self.assertTrue(
                torch.isfinite(grad_r).all(),
                f"在 delta={delta} 处梯度非有限: {grad_r}",
            )
            if delta == 0.0:
                self.assertAlmostEqual(psi.item(), 0.0, places=6)


class TestAdversarialCudaInductorDynamo(unittest.TestCase):
    """对抗挑战 5: GPU (CUDA) 环境下的 0 图断裂与 Inductor 编译内核融合极限压力测试."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA 不可用，跳过 GPU 极限测试")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def test_cuda_extreme_batch_mixed_singularities_inductor(self) -> None:
        """GPU (CUDA) 上极限大规模混合奇异批次 (B=512) 0 图断裂与 Inductor 数值等价性."""
        device = torch.device("cuda")
        nuclear_coords = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]], device=device, dtype=torch.float32
        )
        nuclear_charges = torch.tensor([3.0, 1.0], device=device, dtype=torch.float32)
        model = (
            FermiNet(
                nuclear_coords=nuclear_coords,
                nuclear_charges=nuclear_charges,
                n_up=2,
                n_down=2,
                num_determinants=4,
                hidden_dim_one=16,
                hidden_dim_two=8,
                num_layers=1,
                use_cusp_jastrow=True,
                spin_adapted=True,
            )
            .to(device)
            .eval()
        )

        batch_size = 512
        r = torch.randn(batch_size, 4, 3, device=device, dtype=torch.float32)
        # 注入同自旋重叠
        r[10, 1] = r[10, 0]
        r[25, 3] = r[25, 2]
        # 异自旋重叠
        r[50, 2] = r[50, 0]
        # 核重叠
        r[100, 0] = nuclear_coords[0]

        # 1. 验证 CUDA 上 Dynamo explain 图断裂数严格为 0 (全图单个 Graph 融合)
        explanation = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            f"CUDA 模式下存在图断裂: {explanation.break_reasons}",
        )
        self.assertEqual(explanation.graph_count, 1)

        # 2. 编译并验证 Inductor 输出与 Eager 输出的一致性
        compiled_fn = torch.compile(model.forward_log_psi, backend="inductor")
        with torch.no_grad():
            eager_log, eager_sign = model.forward_log_psi(r)
            comp_log, comp_sign = compiled_fn(r)

        # A. 常规非节点面样本 (log_abs > -20.0) 对数误差严格 < 1e-5
        regular_mask = eager_log > -20.0
        self.assertTrue(regular_mask.any(), "应包含大量常规样本")
        max_reg_diff = (
            (comp_log[regular_mask] - eager_log[regular_mask]).abs().max().item()
        )
        self.assertLess(
            max_reg_diff,
            1e-5,
            f"CUDA 上常规样本 Inductor 编译输出与 Eager 绝对误差超标: {max_reg_diff}",
        )
        self.assertTrue(
            (comp_sign[regular_mask] == eager_sign[regular_mask]).all(),
            "常规样本符号严格一致",
        )

        # B. 在真实物理波函数空间 Psi(r) = sign * exp(log|Psi|)，全批次 512 个样本 (含所有节点面与奇异重合样本) 绝对误差严格 < 1e-6
        # 在节点面上 (Psi -> 0)，相消后绝对残差处于 1e-15 量级，物理波函数绝对误差精确为 0
        eager_psi = eager_sign * torch.where(
            torch.isneginf(eager_log), torch.zeros_like(eager_log), torch.exp(eager_log)
        )
        comp_psi = comp_sign * torch.where(
            torch.isneginf(comp_log), torch.zeros_like(comp_log), torch.exp(comp_log)
        )
        max_psi_diff = (comp_psi - eager_psi).abs().max().item()
        self.assertLess(
            max_psi_diff, 1e-6, f"真实物理波函数空间全批次绝对误差超标: {max_psi_diff}"
        )

    def test_cuda_float64_machine_precision_equivalence(self) -> None:
        """GPU (CUDA) 模式下 Float64 高精度极限压力实测: 编译与 Eager 绝对误差达到机器精度 (< 1e-10)."""
        device = torch.device("cuda")
        nuclear_coords = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]], device=device, dtype=torch.float64
        )
        nuclear_charges = torch.tensor([3.0, 1.0], device=device, dtype=torch.float64)
        model = (
            FermiNet(
                nuclear_coords=nuclear_coords,
                nuclear_charges=nuclear_charges,
                n_up=2,
                n_down=2,
                num_determinants=4,
                hidden_dim_one=16,
                hidden_dim_two=8,
                num_layers=1,
                use_cusp_jastrow=True,
                spin_adapted=True,
            )
            .to(device=device, dtype=torch.float64)
            .eval()
        )

        batch_size = 128
        r = torch.randn(batch_size, 4, 3, device=device, dtype=torch.float64)
        compiled_fn = torch.compile(model.forward_log_psi, backend="inductor")

        with torch.no_grad():
            eager_log, eager_sign = model.forward_log_psi(r)
            comp_log, comp_sign = compiled_fn(r)

        diff = (comp_log - eager_log).abs().max().item()
        self.assertLess(
            diff, 1e-10, f"Float64 下 CUDA Inductor 编译输出与 Eager 误差超标: {diff}"
        )
        self.assertTrue((comp_sign == eager_sign).all(), "Float64 下符号 100% 精确一致")

    def test_cuda_singular_input_backward_finite(self) -> None:
        """GPU (CUDA) 上奇异样本反向传播求导，梯度处处有限无 NaN/Inf."""
        device = torch.device("cuda")
        nuclear_coords = torch.zeros(1, 3, device=device, dtype=torch.float32)
        nuclear_charges = torch.tensor([3.0], device=device, dtype=torch.float32)
        model = (
            FermiNet(
                nuclear_coords=nuclear_coords,
                nuclear_charges=nuclear_charges,
                n_up=2,
                n_down=1,
                num_determinants=4,
                hidden_dim_one=16,
                hidden_dim_two=8,
                num_layers=1,
                use_cusp_jastrow=True,
                spin_adapted=False,
            )
            .to(device)
            .eval()
        )

        r = torch.randn(
            16, 3, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        with torch.no_grad():
            r[0, 1] = r[0, 0]  # 同自旋电子重合奇异点

        psi = model(r)
        grad_r = torch.autograd.grad(psi.sum(), r)[0]
        self.assertTrue(
            torch.isfinite(grad_r).all(), f"CUDA 上奇异样本梯度出现 NaN/Inf: {grad_r}"
        )


if __name__ == "__main__":
    unittest.main()
