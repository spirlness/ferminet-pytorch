"""
物理不变量与已知限制回归测试
==============================
本套件把 `audit_review/independent_physics_audit.md` 第四节的测试缺口固化为可执行断言。

设计原则：
1. 断言的是**物理不变量**（平移不变、同自旋反对称、尖点远场斜率、估计器无偏性），
   这些是真正的正确性要求；
2. 对**已确认的架构限制**（D2–D6），断言的是当前真实行为，并在 docstring 中显式标注
   "已知限制"。若将来修复，这些测试会失败，提示同步更新断言——这是刻意的。
3. 每个断言都附带该缺陷的编号，便于与审查报告对照。
"""

import os
import sys
import unittest

# 确保包能够被直接运行和识别
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.envelope import NuclearEnvelope
from ferminet_pytorch.loss import vmc_surrogate_loss


def _build_model(seed: int = 3, n_up: int = 1, n_down: int = 1,
                 num_layers: int = 2, hidden_dim_one: int = 16,
                 hidden_dim_two: int = 8, num_determinants: int = 3) -> FermiNet:
    """构造双原子 H2 几何下的双精度小模型，随机种子固定以保证可复现。"""
    torch.manual_seed(seed)
    coords = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return FermiNet(
        nuclear_coords=coords,
        nuclear_charges=charges,
        n_up=n_up,
        n_down=n_down,
        num_determinants=num_determinants,
        hidden_dim_one=hidden_dim_one,
        hidden_dim_two=hidden_dim_two,
        num_layers=num_layers,
    ).double()


def _clone_with_geometry(model: FermiNet, coords: torch.Tensor,
                         charges: torch.Tensor) -> FermiNet:
    """复制同架构模型：保留全部可训练参数，仅替换核几何缓冲。"""
    clone = FermiNet(
        nuclear_coords=coords,
        nuclear_charges=charges,
        n_up=model.n_up,
        n_down=model.n_down,
        num_determinants=model.num_determinants,
        hidden_dim_one=model.hidden_dim_one,
        hidden_dim_two=model.hidden_dim_two,
        num_layers=model.num_layers,
        eps=model.eps,
        use_nuclear_charges=model.use_nuclear_charges,
        anisotropic_envelope=model.anisotropic_envelope,
    ).to(dtype=model.nuclear_coords.dtype)
    source = model.state_dict()
    target = clone.state_dict()
    for name, value in source.items():
        if name in ("nuclear_coords", "nuclear_charges"):
            continue
        target[name].copy_(value)
    clone.eval()
    return clone


class TestWavefunctionSymmetry(unittest.TestCase):
    """波函数对称性：哪些成立，哪些不成立。"""

    def setUp(self) -> None:
        self.model = _build_model(seed=3)
        self.r = torch.randn(6, 2, 3, dtype=torch.float64)

    def test_translation_invariance_holds(self) -> None:
        """核与电子同时平移时 log|Psi| 必须不变（相对坐标特征保证的不变量）。"""
        shift = torch.tensor([2.0, -1.0, 0.4], dtype=torch.float64)
        translated = _clone_with_geometry(
            self.model,
            self.model.nuclear_coords + shift,
            self.model.nuclear_charges,
        )
        with torch.no_grad():
            base, _ = self.model.forward_log_psi(self.r)
            moved, _ = translated.forward_log_psi(self.r + shift)
        self.assertLess(float((base - moved).abs().max()), 1e-9)

    def test_rotation_invariance_is_absent_known_limitation(self) -> None:
        """D4 已知限制：架构不具备旋转不变性，旋转会显著改变 log|Psi|。

        原因：features.py 把笛卡尔位移向量送入普通 MLP，无 SO(3) 等变约束。
        若将来引入旋转等变层，本测试会失败，应改为断言 < 1e-9。
        """
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        rotated = _clone_with_geometry(
            self.model,
            self.model.nuclear_coords @ rotation.T,
            self.model.nuclear_charges,
        )
        with torch.no_grad():
            base, _ = self.model.forward_log_psi(self.r)
            turned, _ = rotated.forward_log_psi(torch.einsum("ij,bnj->bni", rotation, self.r))
        self.assertGreater(float((base - turned).abs().max()), 0.1)

    def test_nuclear_permutation_invariance_is_absent_known_limitation(self) -> None:
        """D5 已知限制：同核体系的核序置换会改变 log|Psi|。

        原因：NuclearEnvelope 的 weights / decay_rates_raw 按核索引参数化。
        物理上完全相同的输入（两个质子互换）给出不同的波函数值。
        变分上界仍成立，但有效对称群小于哈密顿量的对称群。
        """
        permuted = _clone_with_geometry(
            self.model,
            self.model.nuclear_coords.flip(0),
            self.model.nuclear_charges.flip(0),
        )
        with torch.no_grad():
            base, _ = self.model.forward_log_psi(self.r)
            swapped, _ = permuted.forward_log_psi(self.r)
        self.assertGreater(float((base - swapped).abs().max()), 0.1)

    def test_same_spin_exchange_flips_sign_exactly(self) -> None:
        """同自旋电子交换必须保持 log|Psi| 不变并精确翻转符号（Slater 反对称性）。"""
        model = _build_model(seed=5, n_up=2, n_down=0)
        r = torch.randn(6, 2, 3, dtype=torch.float64)
        with torch.no_grad():
            log_base, sign_base = model.forward_log_psi(r)
            log_swap, sign_swap = model.forward_log_psi(r.flip(1))
        self.assertLess(float((log_base - log_swap).abs().max()), 1e-9)
        self.assertTrue(torch.allclose(sign_swap, -sign_base))


class TestKatoCuspAndSoftening(unittest.TestCase):
    """核尖点条件与距离软化的定量影响。"""

    def test_envelope_radial_slope_matches_kato_far_from_nucleus(self) -> None:
        """远离核处 (r >> sqrt(eps)) 径向对数斜率应趋于 -Z，尖点结构正确。"""
        envelope = NuclearEnvelope(
            num_determinants=1, num_orbitals=1, num_nuclei=1, init_decay_rate=1.0
        ).double()
        with torch.no_grad():
            envelope.weights.fill_(1.0)
        origin = torch.zeros(1, 3, dtype=torch.float64)

        r = torch.tensor([[[1e-2, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
        value = envelope(r, origin).sum()
        grad = torch.autograd.grad(value, r)[0]
        slope = float(grad[0, 0, 0] / value.item())
        self.assertAlmostEqual(slope, -1.0, delta=0.01)

    def test_envelope_radial_slope_is_softened_near_nucleus_known_limitation(self) -> None:
        """D6 已知限制：sqrt(r^2+eps) 使核处径向斜率趋于 0，破坏 Kato 尖点。

        Kato 条件要求球平均波函数在核处的径向导数为 -Z。软化尺度为 sqrt(eps)=1e-4。
        若将来改用不改变非重合点哈密顿量的稳定计算，本测试会失败，应改为断言斜率≈-1。
        """
        envelope = NuclearEnvelope(
            num_determinants=1, num_orbitals=1, num_nuclei=1, init_decay_rate=1.0
        ).double()
        with torch.no_grad():
            envelope.weights.fill_(1.0)
        origin = torch.zeros(1, 3, dtype=torch.float64)

        r = torch.tensor([[[1e-6, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
        value = envelope(r, origin).sum()
        grad = torch.autograd.grad(value, r)[0]
        slope = float(grad[0, 0, 0] / value.item())
        self.assertGreater(slope, -0.1)

    def test_softened_potential_underestimates_nuclear_attraction(self) -> None:
        """D6：势能的 sqrt(r^2+eps) 软化在核附近显著低估库仑吸引。"""
        system = MolecularSystem(
            nuclear_coords=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([1.0], dtype=torch.float64),
            n_up=1,
            n_down=0,
        )
        r = torch.tensor([[[1e-6, 0.0, 0.0]]], dtype=torch.float64)
        v_en = float(system.compute_potential_energy(r))
        # 软化使吸引势比真实库仑值更浅（更接近 0），但仍是有限的大负数
        self.assertGreater(v_en, -1.0 / 1e-6)
        self.assertLess(v_en, -2.0e4)
        self.assertAlmostEqual(v_en, -1.0 / (1e-12 + 1e-10) ** 0.5, delta=1.0)


class TestParameterReachability(unittest.TestCase):
    """参数可达性：哪些参数实际参与训练。"""

    def test_all_parameters_receive_gradient_except_final_layer_two_electron_stream(self) -> None:
        """D3：末层双电子流更新网络不在计算图上；其余参数必须获得非零梯度。

        末层 h2 输出无人消费，故 layers[-1].linear_two（以及末层为真实线性映射时的
        res_proj_two）恒为空转参数。若将来裁剪该层或改变数据流，本测试会失败，
        应同步更新死参数集合。
        """
        for num_layers in (1, 2, 3):
            with self.subTest(num_layers=num_layers):
                model = _build_model(seed=2, num_layers=num_layers)
                r = torch.randn(8, 2, 3, dtype=torch.float64)
                model.zero_grad()
                log_psi, _ = model.forward_log_psi(r)
                loss, _ = vmc_surrogate_loss(
                    log_psi, torch.randn(8, dtype=torch.float64), clip_method="none"
                )
                loss.backward()

                last = num_layers - 1
                dead = {
                    f"layers.{last}.linear_two.weight",
                    f"layers.{last}.linear_two.bias",
                }
                if isinstance(model.layers[last].res_proj_two, torch.nn.Linear):
                    dead.add(f"layers.{last}.res_proj_two.weight")

                for name, param in model.named_parameters():
                    if name in dead:
                        self.assertTrue(
                            param.grad is None or float(param.grad.abs().max()) == 0.0,
                            f"{name} 预期为空转参数，却获得了梯度",
                        )
                    else:
                        self.assertIsNotNone(param.grad, f"{name} 未获得梯度")
                        self.assertGreater(
                            float(param.grad.abs().max()), 0.0, f"{name} 梯度恒为零"
                        )


class TestHutchinsonEstimator(unittest.TestCase):
    """Hutchinson 随机迹估计器的无偏性与方差。"""

    def _kinetic(self, model, system, r, method, **kwargs) -> float:
        r_grad = r.clone().detach().requires_grad_(True)
        _, t_local, _ = compute_local_energy(
            model, system, r_grad, laplacian_method=method, **kwargs
        )
        return float(t_local.item())

    def setUp(self) -> None:
        self.model = _build_model(seed=21)
        self.system = MolecularSystem(
            self.model.nuclear_coords.clone(),
            self.model.nuclear_charges.clone(),
            1,
            1,
        )
        self.r = torch.randn(1, 2, 3, dtype=torch.float64)
        self.exact = self._kinetic(self.model, self.system, self.r, "exact")

    def test_hutchinson_estimate_converges_to_exact_laplacian(self) -> None:
        """估计器必须无偏：多探针重复平均应收敛到精确拉普拉斯。"""
        repetitions = 16
        probes = 64
        estimates = [
            self._kinetic(
                self.model, self.system, self.r, "hutchinson",
                num_hutchinson_samples=probes,
            )
            for _ in range(repetitions)
        ]
        mean = sum(estimates) / len(estimates)
        self.assertLess(abs(mean - self.exact), 0.5)

    def test_single_probe_variance_is_large_known_limitation(self) -> None:
        """D2 已知限制：单探针估计的重复间标准差与信号同量级，不可用于能量报告。

        估计器无偏但方差 ∝ 1/n，且 ln|Psi| 的 Hessian 迹在节点附近重尾。
        若将来提高默认探针数或实现 Forward Laplacian，本测试会失败，应更新阈值。
        """
        estimates = [
            self._kinetic(
                self.model, self.system, self.r, "hutchinson",
                num_hutchinson_samples=1,
            )
            for _ in range(16)
        ]
        mean = sum(estimates) / len(estimates)
        spread = (sum((x - mean) ** 2 for x in estimates) / (len(estimates) - 1)) ** 0.5
        self.assertGreater(spread, 0.3 * abs(self.exact))


if __name__ == "__main__":
    unittest.main()
