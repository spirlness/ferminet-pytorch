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
                 hidden_dim_two: int = 8, num_determinants: int = 3,
                 rotation_invariant: bool = False) -> FermiNet:
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
        rotation_invariant=rotation_invariant,
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
        tie_nuclear_envelope=model.tie_nuclear_envelope,
        rotation_invariant=model.rotation_invariant,
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

    def test_rotation_invariance_holds(self) -> None:
        """D4 修复：核与电子同时旋转时 log|Psi| 必须不变。

        机制：显式 `rotation_invariant=True` 时，特征只由标量不变量构成
        (电子-核距离 d_{iI}、电子-电子距离 d_{ij}、核电荷 Z_I)，不含任何笛卡尔
        位移向量，因此整个前向计算在整体旋转下逐位相同——不是"近似不变"，是精确不变。
        对本仓库的线性分子，到各核的距离已完整确定电子在分子标架下的位置
        (轴向 + 垂直距离)，故该特征集不损失表达力 (H2 收敛对照见提交信息)。
        """
        invariant_model = _build_model(seed=3, rotation_invariant=True)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        rotated = _clone_with_geometry(
            invariant_model,
            invariant_model.nuclear_coords @ rotation.T,
            invariant_model.nuclear_charges,
        )
        with torch.no_grad():
            base, _ = invariant_model.forward_log_psi(self.r)
            turned, _ = rotated.forward_log_psi(torch.einsum("ij,bnj->bni", rotation, self.r))
        self.assertLess(float((base - turned).abs().max()), 1e-12)

    def test_legacy_cartesian_features_are_not_rotation_invariant(self) -> None:
        """回归保护：`rotation_invariant=False` 的笛卡尔特征模式**确实**不具备
        旋转不变性——该开关只为复现历史结果与消融对照保留，不得据此作任何
        "等变" 声明。若将来该模式也变得旋转不变，本测试会失败，应更新文档措辞。
        """
        model = _build_model(seed=3)
        model.rotation_invariant = False
        legacy = FermiNet(
            nuclear_coords=model.nuclear_coords,
            nuclear_charges=model.nuclear_charges,
            n_up=model.n_up,
            n_down=model.n_down,
            num_determinants=model.num_determinants,
            hidden_dim_one=model.hidden_dim_one,
            hidden_dim_two=model.hidden_dim_two,
            num_layers=model.num_layers,
            rotation_invariant=False,
        ).double()
        self.assertEqual(legacy.in_dim_one, 5 * 2)
        self.assertEqual(legacy.in_dim_two, 4)

        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        rotated = _clone_with_geometry(
            legacy,
            legacy.nuclear_coords @ rotation.T,
            legacy.nuclear_charges,
        )
        with torch.no_grad():
            base, _ = legacy.forward_log_psi(self.r)
            turned, _ = rotated.forward_log_psi(torch.einsum("ij,bnj->bni", rotation, self.r))
        self.assertGreater(float((base - turned).abs().max()), 0.1)

    def test_nuclear_permutation_invariance_is_absent_known_limitation(self) -> None:
        """D5 部分修复后仍存在的限制：完整波函数不具备核置换不变性。

        已修复的部分：`NuclearEnvelope` 按电荷组共享 `weights` / `decay_rates_raw`，
        包络本身在同核置换下严格不变 (见 test_geometry_invariants.py)。

        仍存在的部分：单电子特征按**原子核顺序**拼接成 5*M 维后送入普通密集层，
        而密集层对核块不对称，故交换同种核仍改变 log|Psi|。

        注：曾尝试"按距离排序规范化核轴"来补上这一项，但**该方案被否决**——
        排序在电子穿越两同核的垂直平分面时翻转核块顺序，而两块在该面上并不相等
        (r_{iI} 方向不同)，导致波函数在余维 1 曲面上出现不连续跳变 (实测约 0.044)，
        ∇²ln|Psi| 失去意义、局域动能被污染，训练能量从 -1.174 退化到 -1.162 并
        一度跌破变分下界。要做到既不变又连续必须改为对核做对称聚合 (架构改写)。
        连续性回归见 test_wavefunction_is_continuous_across_nuclear_bisector。
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

    def test_wavefunction_is_continuous_across_nuclear_bisector(self) -> None:
        """波函数必须在同核垂直平分面上连续 (禁止以排序实现核序规范化)。

        让一个电子沿两核连线穿过垂直平分面：若实现用排序重排核块，log|Psi| 会在
        面两侧发生 O(1) 跳变；正确实现 (固定核块顺序) 只应有光滑的一阶变化。
        """
        model = _build_model(seed=0)
        xs = torch.tensor([-1e-2, -1e-3, -1e-6, 0.0, 1e-6, 1e-3, 1e-2], dtype=torch.float64)
        r = torch.zeros(len(xs), 2, 3, dtype=torch.float64)
        r[:, 0, 0] = xs
        r[:, 1, :] = torch.tensor([0.3, 0.9, -0.4], dtype=torch.float64)
        with torch.no_grad():
            log_psi, _ = model.forward_log_psi(r)

        # 跨越面 (x=-1e-6 -> +1e-6, 步长 2e-6) 的一阶差分
        across = float((log_psi[4] - log_psi[2]).abs())
        # 远离面 (x=-1e-2 -> -1e-3, 步长 9e-3, 是跨越步长的 4500 倍) 的一阶差分
        away = float((log_psi[1] - log_psi[0]).abs())
        self.assertLess(
            across, 5.0 * away,
            f"跨越垂直平分面的一阶差分 ({across:.6f}) 远大于远离面的一阶差分 "
            f"({away:.6f})，波函数在该面上不连续",
        )

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
        """远离核处 (r >> sqrt(eps)) 径向对数斜率应趋于 -alpha，尖点结构正确。

        注：D6 尖点断言依赖 alpha == Z。init_decay_rate=1.0 恰好对 H (Z=1) 正确，
        对 He/Li 不正确，故显式断言斜率 == -alpha，不隐式假设 Z。
        """
        envelope = NuclearEnvelope(
            num_determinants=1, num_orbitals=1, num_nuclei=1, init_decay_rate=1.0
        ).double()
        with torch.no_grad():
            envelope.weights.fill_(1.0)
        origin = torch.zeros(1, 3, dtype=torch.float64)
        # alpha = softplus(decay_rates_raw)，由 init_decay_rate=1.0 经 softplus 反函数初始化
        alpha = float(torch.nn.functional.softplus(envelope.decay_rates_raw.detach()).item())

        r = torch.tensor([[[1e-2, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
        value = envelope(r, origin).sum()
        grad = torch.autograd.grad(value, r)[0]
        slope = float(grad[0, 0, 0] / value.item())
        self.assertAlmostEqual(slope, -alpha, delta=0.01)

    def test_envelope_radial_slope_satisfies_kato_at_nucleus(self) -> None:
        """D6 修复：clamp 核不软化距离，核处径向对数斜率满足 Kato 条件 slope == -alpha。

        eps 现为最小**平方**距离 (默认 1e-24，软化半径 sqrt(eps)=1e-12 Bohr)，
        在 r^2 >= eps 处 clamp 是恒等变换，距离与未软化时逐位相同，偏置精确为零；
        r=1e-6 处 r^2=1e-12 > eps，因此斜率精确等于 -alpha。
        断言斜率 == -alpha (alpha = softplus(decay_rates_raw) 由 init_decay_rate=1.0
        反解)，不隐式假设 alpha == Z (init_decay_rate=1.0 仅对 H 正确)。
        """
        envelope = NuclearEnvelope(
            num_determinants=1, num_orbitals=1, num_nuclei=1, init_decay_rate=1.0
        ).double()
        with torch.no_grad():
            envelope.weights.fill_(1.0)
        origin = torch.zeros(1, 3, dtype=torch.float64)
        alpha = float(torch.nn.functional.softplus(envelope.decay_rates_raw.detach()).item())

        r = torch.tensor([[[1e-6, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
        value = envelope(r, origin).sum()
        grad = torch.autograd.grad(value, r)[0]
        slope = float(grad[0, 0, 0] / value.item())
        self.assertAlmostEqual(slope, -alpha, delta=1e-6)

    def test_potential_matches_exact_coulomb_away_from_nucleus(self) -> None:
        """D6 修复后：非重合点势能必须精确等于库仑值 (clamp 核在 r^2 >= eps 处恒等)。

        旧实现 `sqrt(r^2 + 1e-10)` 在 r=1e-6 Bohr 处给 V_en = -99,503.7 Ha，
        而真实库仑值为 -1e6 Ha (偏差 9.0e5 Ha)。clamp 核在所有 r^2 >= eps 处
        与未软化距离逐位相同，偏置精确为零。
        """
        system = MolecularSystem(
            nuclear_coords=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([1.0], dtype=torch.float64),
            n_up=1,
            n_down=0,
        )
        r = torch.tensor([[[1e-6, 0.0, 0.0]]], dtype=torch.float64)
        v_en = float(system.compute_potential_energy(r))
        # 体系为单核单电子，无 V_ee / V_nn，总势能即 V_en = -Z/r = -1e6 Ha
        self.assertAlmostEqual(v_en, -1.0 / 1e-6, delta=1.0)

    def test_potential_is_finite_and_nan_free_at_coalescence(self) -> None:
        """D6: r=0 重合点处势能必须有限且无 NaN (clamp 核的下限保护)。

        r=0 时距离被限制为 sqrt(eps)=1e-12 Bohr，V_en = -Z/sqrt(eps) = -1e12 Ha：
        有限、无 NaN，且坐标梯度为零；eps 不能直接删除 (裸 sqrt 在 r=0 处
        梯度为 inf*0=NaN)。
        """
        system = MolecularSystem(
            nuclear_coords=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([1.0], dtype=torch.float64),
            n_up=1,
            n_down=0,
        )
        r = torch.zeros(1, 1, 3, dtype=torch.float64)
        v = system.compute_potential_energy(r)
        self.assertFalse(torch.isnan(v).any(), "重合点势能出现 NaN")
        self.assertTrue(torch.isfinite(v).all(), "重合点势能出现 Inf")
        self.assertLess(float(v.item()), -1.0e11)

        # clamp 核在重合点梯度为零 (而非 NaN/Inf)
        r_grad = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
        v_g = system.compute_potential_energy(r_grad)
        grad = torch.autograd.grad(v_g.sum(), r_grad)[0]
        self.assertTrue(torch.isfinite(grad).all(), "重合点势能梯度非有限")
        self.assertLess(float(grad.abs().max()), 1e-6)


class TestParameterReachability(unittest.TestCase):
    """参数可达性：哪些参数实际参与训练。"""

    def test_all_parameters_receive_gradient(self) -> None:
        """D3：全部可训练参数都必须获得非零梯度。

        FermiNet 的末层不构造双电子流更新网络 (其 h2 输出不被消费)，因此
        模型中不应存在任何恒收不到梯度的空转参数。
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

                for name, param in model.named_parameters():
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
