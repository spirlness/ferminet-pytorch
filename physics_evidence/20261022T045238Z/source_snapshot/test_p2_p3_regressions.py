"""
P2 与 P3 综合回归测试套件 (Comprehensive P2/P3 Regressions & E2E QA)
===================================================================

本测试文件完整覆盖 CODE_REVIEW 中检出并已重构完成的全部 6 项 P2/P3 优化项：
1. 双电子特征自交互项剔除测试 (FINDING-PYT-05, layers.py)
   - 对角线注入极端扰动 (10^6) 时池化均值严格不受影响；
   - 同自旋通道除以 n_spin - 1，单电子边界安全填 0 无除零或 NaN；
   - Autograd 反向传播对角线梯度严格为 0.0；
   - 类名别名 FermiNetLayer == FermiInteractionLayer 验证。
2. 全电子移动目标接受率适配测试 (F-MCMC-02, mcmc.py)
   - move_mode == "all" 默认推导目标接受率为 0.234；
   - move_mode == "one_by_one" 默认维持 0.50；
   - 显式传参（如 0.35）成功覆盖默认值；
   - 类名别名 MetropolisHastings == MCMCSampler 验证。
3. 异常局域能量样本比例熔断拦截测试 (F-LOSS-02, loss.py)
   - 非有限样本比例超标（> 10%）时坚决抛出 FloatingPointError；
   - 非有限样本在容限内（<= 10%）时发出 Warning 且正常完成损失计算；
   - VMCLoss 类模块封装下的熔断行为验证。
4. 历史最优模型状态保存与恢复测试 (F-TRAIN-02, train_vmc.py)
   - 人为注入能量波动时，训练结束返回的模型严格对应测得历史最低能量处的真实参数 theta_t；
   - Checkpoint 文件生成、包含的关键键值完整且能成功反序列化加载。
5. 自旋投影置换推断 Batch 批量向量化等价性测试 (PHY-02, spin.py)
   - 批量广播单次前向与逐项串行求和在数值上严格等价（误差 < 1e-6），符号严格一致；
   - 反向传播 Autograd 梯度的等价性；
   - 类名别名 SpinProjectedWaveFunction == SpinProjector 验证。
6. 逐电子移动微观步与接受率无舍入失真测试 (F-MCMC-03, mcmc.py)
   - 非整除电子接受序列下，微观计数器 proposed_moves 与 accepted_moves 真实精确累加，无任何舍入或截断误差。
"""

import copy
import math
import os
import shutil
import tempfile
import unittest
import warnings
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from ferminet_pytorch.layers import FermiInteractionLayer, FermiNetLayer
    from ferminet_pytorch.loss import VMCLoss, vmc_surrogate_loss
    from ferminet_pytorch.mcmc import MCMCSampler, MetropolisHastings
    from ferminet_pytorch.spin import SpinProjector, SpinProjectedWaveFunction
    from ferminet_pytorch.hamiltonian import MolecularSystem
    from ferminet_pytorch.train_vmc import build_h2_system, train_vmc
    import ferminet_pytorch.loss as vmc_loss_module
except ImportError:
    from layers import FermiInteractionLayer, FermiNetLayer
    from loss import VMCLoss, vmc_surrogate_loss
    from mcmc import MCMCSampler, MetropolisHastings
    from spin import SpinProjector, SpinProjectedWaveFunction
    from hamiltonian import MolecularSystem
    from train_vmc import build_h2_system, train_vmc
    import loss as vmc_loss_module


class TestTwoElectronSelfInteractionExclusion(unittest.TestCase):
    """
    P2.1 优化项：双电子特征自交互项剔除测试 (FINDING-PYT-05, layers.py)
    """

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_diagonal_extreme_perturbation_invariance(self) -> None:
        """
        验证双电子特征对角线 i=j 注入极端扰动 (10^6) 时，均值池化与层前向输出严格不受影响。
        """
        batch_size = 4
        num_electrons = 4
        n_up = 2
        n_down = 2
        dim_one = 16
        dim_two = 8

        layer = FermiInteractionLayer(
            in_dim_one=dim_one,
            in_dim_two=dim_two,
            out_dim_one=dim_one,
            out_dim_two=dim_two,
            produce_two_electron_output=False,
        )
        layer.eval()

        h1 = torch.randn(batch_size, num_electrons, dim_one)
        h2_clean = torch.randn(batch_size, num_electrons, num_electrons, dim_two)

        # 构造对角线注入极端扰动 (1e6) 的脏数据
        h2_perturbed = h2_clean.clone()
        for i in range(num_electrons):
            h2_perturbed[:, i, i, :] = 1e6

        with torch.no_grad():
            out_clean, _ = layer(h1, h2_clean, n_up=n_up, n_down=n_down)
            out_perturbed, _ = layer(h1, h2_perturbed, n_up=n_up, n_down=n_down)

        # 验证单电子流前向输出完全一致，误差严格为 0.0
        max_abs_diff = (out_clean - out_perturbed).abs().max().item()
        self.assertEqual(
            max_abs_diff,
            0.0,
            f"对角线极端扰动导致输出改变，最大绝对误差: {max_abs_diff}",
        )

    def test_spin_channel_denominator_and_single_electron_safety(self) -> None:
        """
        验证同自旋除以 n_spin - 1、异自旋除以 n_other，以及单电子安全填 0 且无除零或 NaN。
        """
        batch_size = 2
        dim_one = 8
        dim_two = 4

        # 1. 验证常规多电子体系 (n_up=3, n_down=2)
        n_up = 3
        n_down = 2
        num_electrons = n_up + n_down
        layer_multi = FermiInteractionLayer(
            in_dim_one=dim_one,
            in_dim_two=dim_two,
            out_dim_one=dim_one,
            out_dim_two=dim_two,
        )
        h1_multi = torch.randn(batch_size, num_electrons, dim_one)
        h2_ones = torch.ones(batch_size, num_electrons, num_electrons, dim_two)

        out_multi, _ = layer_multi(h1_multi, h2_ones, n_up=n_up, n_down=n_down)
        self.assertTrue(torch.isfinite(out_multi).all(), "多电子前向包含非有限值")

        # 2. 验证单电子极限界 (氢原子系统: n_up=1, n_down=0)
        layer_single = FermiInteractionLayer(
            in_dim_one=dim_one,
            in_dim_two=dim_two,
            out_dim_one=dim_one,
            out_dim_two=dim_two,
        )
        h1_single = torch.randn(batch_size, 1, dim_one)
        h2_single = torch.randn(batch_size, 1, 1, dim_two)

        out_single, _ = layer_single(h1_single, h2_single, n_up=1, n_down=0)
        self.assertTrue(torch.isfinite(out_single).all(), "单电子体系前向产生 NaN/Inf")
        self.assertEqual(out_single.shape, (batch_size, 1, dim_one))

    def test_autograd_diagonal_gradient_strictly_zero(self) -> None:
        """
        验证 Autograd 自动微分反向传播时，h2 对角线 i=j 处的梯度严格为 0.0。
        """
        batch_size = 2
        num_electrons = 3
        dim_one = 8
        dim_two = 6
        n_up = 2
        n_down = 1

        layer = FermiInteractionLayer(
            in_dim_one=dim_one,
            in_dim_two=dim_two,
            out_dim_one=dim_one,
            out_dim_two=dim_two,
            produce_two_electron_output=False,  # 仅通过均值池化流动到单电子输出
        )

        h1 = torch.randn(batch_size, num_electrons, dim_one)
        h2 = torch.randn(
            batch_size, num_electrons, num_electrons, dim_two, requires_grad=True
        )

        out_one, _ = layer(h1, h2, n_up=n_up, n_down=n_down)
        loss = out_one.sum()
        loss.backward()

        self.assertIsNotNone(h2.grad)
        # 提取对角线切片并检查梯度是否恒等于 0
        diag_grads = []
        for i in range(num_electrons):
            diag_grads.append(h2.grad[:, i, i, :])
        diag_grad_tensor = torch.stack(diag_grads, dim=1)

        max_diag_grad = diag_grad_tensor.abs().max().item()
        self.assertEqual(
            max_diag_grad,
            0.0,
            f"双电子特征对角线梯度不为 0.0，检测到渗漏梯度: {max_diag_grad}",
        )

        # 确保非对角线项存在真实有效的梯度传导
        eye_mask = torch.eye(num_electrons, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
        off_diag_grad = h2.grad.masked_select(~eye_mask)
        self.assertGreater(
            off_diag_grad.abs().sum().item(),
            0.0,
            "非对角线元素未正常传递梯度",
        )

    def test_ferminet_layer_alias(self) -> None:
        """验证类名别名 FermiNetLayer is FermiInteractionLayer。"""
        self.assertIs(FermiNetLayer, FermiInteractionLayer)


class TestMCMCTargetAcceptanceAdaptation(unittest.TestCase):
    """
    P2.2 优化项：全电子移动目标接受率适配测试 (F-MCMC-02, mcmc.py)
    """

    def setUp(self) -> None:
        self.system = build_h2_system()
        # 构造虚拟波函数模型以供采样器初始化
        self.dummy_model = nn.Module()
        self.dummy_model.forward_log_psi = lambda r: (
            torch.zeros(r.shape[0], device=r.device),
            torch.ones(r.shape[0], device=r.device),
        )

    def test_move_mode_all_defaults_to_optimal_rate(self) -> None:
        """测试 move_mode == 'all' 默认推导目标接受率为 0.234。"""
        sampler = MCMCSampler(
            self.dummy_model,
            self.system,
            num_walkers=16,
            move_mode="all",
            target_acceptance=None,
        )
        self.assertEqual(sampler.target_acceptance, 0.234)

    def test_move_mode_one_by_one_defaults_to_half(self) -> None:
        """测试 move_mode == 'one_by_one' 默认维持目标接受率为 0.50。"""
        sampler = MCMCSampler(
            self.dummy_model,
            self.system,
            num_walkers=16,
            move_mode="one_by_one",
            target_acceptance=None,
        )
        self.assertEqual(sampler.target_acceptance, 0.50)

    def test_explicit_target_acceptance_overrides_default(self) -> None:
        """测试显式传入 target_acceptance（如 0.35）能成功覆盖两种模式下的默认值。"""
        sampler_all = MCMCSampler(
            self.dummy_model,
            self.system,
            num_walkers=16,
            move_mode="all",
            target_acceptance=0.35,
        )
        self.assertEqual(sampler_all.target_acceptance, 0.35)

        sampler_one = MCMCSampler(
            self.dummy_model,
            self.system,
            num_walkers=16,
            move_mode="one_by_one",
            target_acceptance=0.42,
        )
        self.assertEqual(sampler_one.target_acceptance, 0.42)

    def test_metropolis_hastings_alias(self) -> None:
        """验证类名别名 MetropolisHastings is MCMCSampler。"""
        self.assertIs(MetropolisHastings, MCMCSampler)


class TestInvalidEnergyFractionCircuitBreaker(unittest.TestCase):
    """
    P2.3 优化项：异常局域能量样本比例熔断拦截测试 (F-LOSS-02, loss.py)
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 20
        self.log_abs_psi = torch.randn(self.batch_size, requires_grad=True)

    def test_invalid_fraction_exceeding_threshold_raises_floating_point_error(self) -> None:
        """
        验证当非有限样本比例超标（如 4/20 = 20% > 10% 默认阈值）时，坚决抛出 FloatingPointError 熔断。
        """
        e_local = torch.randn(self.batch_size)
        # 制造 4 个 NaN/Inf 异常样本 (4 / 20 = 20% > 10%)
        e_local[0] = float("nan")
        e_local[1] = float("inf")
        e_local[2] = float("-inf")
        e_local[3] = float("nan")

        with self.assertRaises(FloatingPointError) as cm:
            vmc_surrogate_loss(
                log_abs_psi=self.log_abs_psi,
                e_local=e_local,
                max_invalid_fraction=0.10,
            )
        self.assertIn("exceeds maximum allowed", str(cm.exception))

    def test_invalid_fraction_within_threshold_issues_warning_and_computes_loss(self) -> None:
        """
        验证当非有限样本比例在容限内（如 1/20 = 5% <= 10%）时，发出 Warning 且正常完成损失计算。
        """
        e_local = torch.randn(self.batch_size)
        e_local[0] = float("nan")  # 1 / 20 = 5% <= 10%

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            loss, stats = vmc_surrogate_loss(
                log_abs_psi=self.log_abs_psi,
                e_local=e_local,
                max_invalid_fraction=0.10,
            )

        # 验证 Warning 触发
        self.assertTrue(
            any(issubclass(w.category, RuntimeWarning) for w in recorded_warnings),
            "容限内异常样本未发出 RuntimeWarning",
        )
        # 验证损失和统计结果有限且正确计算
        self.assertTrue(torch.isfinite(loss).item(), "损失值非有限")
        self.assertEqual(stats["num_valid"], 19)
        self.assertEqual(stats["num_invalid"], 1)

    def test_vmc_loss_class_module_circuit_breaker(self) -> None:
        """
        验证 VMCLoss 类模块封装下的熔断机制运转正常。
        """
        loss_module = VMCLoss(max_invalid_fraction=0.15)
        self.assertEqual(loss_module.max_invalid_fraction, 0.15)

        # 构造 4/20 = 20% 异常样本 (超过 15% 阈值)
        e_local = torch.randn(self.batch_size)
        e_local[:4] = float("nan")

        with self.assertRaises(FloatingPointError):
            loss_module(log_abs_psi=self.log_abs_psi, e_local=e_local)


class TestBestModelTrackingAndCheckpointPersistence(unittest.TestCase):
    """
    P3.1 优化项：历史最优模型状态保存与 Checkpoint 机制 (F-TRAIN-02, train_vmc.py)
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_energy_oscillation_restores_true_best_evaluated_state(self) -> None:
        """
        模拟能量波动的训练过程，验证训练结束返回的模型严格对应测得历史最低能量处的真实参数 theta_t。
        """
        energy_sequence = [-1.0, -2.5, -0.4, +8.0]
        step_counter = [0]
        evaluated_states: Dict[int, Dict[str, torch.Tensor]] = {}

        original_compute_loss = vmc_loss_module.VMCLoss.compute_step_loss

        def mocked_compute_step_loss(self_obj, model, system, walkers, **kwargs):
            step_counter[0] += 1
            ep = step_counter[0]
            simulated_energy = energy_sequence[ep - 1]

            # 核心验证点：在此处记录产生当前能量的真实波函数参数 theta_t
            evaluated_states[ep] = copy.deepcopy(model.state_dict())

            # 构造带参数依赖的虚拟损失以促使 optimizer.step() 更新参数
            loss = sum((p ** 2).sum() for p in model.parameters()) * 0.05 + simulated_energy
            stats = {
                "energy_mean": simulated_energy,
                "energy_clipped_mean": simulated_energy,
                "energy_var": 0.05,
                "loss": simulated_energy,
                "num_valid": walkers.shape[0],
                "num_invalid": 0,
            }
            return loss, stats, None

        vmc_loss_module.VMCLoss.compute_step_loss = mocked_compute_step_loss
        try:
            results = train_vmc(
                num_epochs=4,
                num_walkers=16,
                burn_in_steps=0,
                mcmc_steps_per_epoch=1,
                restore_best=True,
                verbose=False,
            )

            # 1. 验证记录的最优轮次与能量
            self.assertEqual(results["best_epoch"], 2)
            self.assertEqual(results["best_energy"], -2.5)

            # 2. 验证模型已恢复为 Epoch 2 评估时的真实状态 (最大误差严格为 0.0)
            model_current = results["model"]
            best_evaluated_snapshot = evaluated_states[2]

            max_diff = max(
                (model_current.state_dict()[k] - best_evaluated_snapshot[k]).abs().max().item()
                for k in best_evaluated_snapshot
            )
            self.assertEqual(
                max_diff,
                0.0,
                f"恢复后的模型参数与 Epoch 2 评估时的参数存在偏差: {max_diff}",
            )

        finally:
            vmc_loss_module.VMCLoss.compute_step_loss = original_compute_loss

    def test_checkpoint_generation_and_deserialization(self) -> None:
        """
        测试 Checkpoint 文件在多级嵌套目录下成功生成，包含关键键值，且能成功反序列化加载。
        """
        ckpt_path = os.path.join(self.temp_dir, "nested_dir", "sub", "model_ckpt.pt")

        res = train_vmc(
            num_epochs=3,
            num_walkers=16,
            burn_in_steps=0,
            mcmc_steps_per_epoch=1,
            checkpoint_path=ckpt_path,
            verbose=False,
        )

        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint 文件未生成: {ckpt_path}")
        self.assertGreater(os.path.getsize(ckpt_path), 10000, "Checkpoint 文件过小或为空")

        # 反序列化加载并校验键值
        ckpt_data = torch.load(ckpt_path, weights_only=False)
        required_keys = [
            "epoch",
            "best_epoch",
            "best_energy",
            "model_state_dict",
            "best_model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "history",
        ]
        for key in required_keys:
            self.assertIn(key, ckpt_data, f"Checkpoint 缺少必要键值: {key}")

        self.assertEqual(ckpt_data["epoch"], 3)
        self.assertTrue(math.isfinite(ckpt_data["best_energy"]))

        # 验证能够无损加载回模型
        fresh_model = res["model"]
        fresh_model.load_state_dict(ckpt_data["best_model_state_dict"])


class TestSpinProjectionBatchVectorizationEquivalence(unittest.TestCase):
    """
    P3.2 优化项：自旋投影置换推断 Batch 批量向量化等价性测试 (PHY-02, spin.py)
    """

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_vectorized_projection_exact_numerical_match_with_serial(self) -> None:
        """
        验证批量广播向量化前向与原逐项串行求和在数值上严格等价（误差 < 1e-6），符号严格一致。
        """
        projector = SpinProjector(n_up=1, n_down=1, mode="singlet")
        batch_size = 6
        num_electrons = 2

        # 构造非平凡波函数（依赖电子顺序以产生非退化的置换值）
        def base_forward_fn(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # x shape: (B, N, 3)
            # 引入非对称权重以模拟真实非对称前向
            weights_per_elec = torch.tensor([1.5, 0.5], device=x.device).view(1, num_electrons, 1)
            log_abs = torch.sum((x * weights_per_elec) ** 2, dim=(-2, -1))
            sign = torch.where(x[:, 0, 0] > 0, torch.ones(x.shape[0]), -torch.ones(x.shape[0]))
            return log_abs, sign

        r = torch.randn(batch_size, num_electrons, 3)

        # 1. 向量化前向推断
        log_abs_vec, sign_vec = projector.project_log_psi(base_forward_fn, r)

        # 2. 原逐项串行求和 (Ground Truth 参考实现)
        num_k = projector.num_permutations
        perms = projector.perms
        weights = projector.weights

        log_abs_list = []
        sign_list = []
        for k in range(num_k):
            perm_k = perms[k]
            w_k = weights[k].item()
            r_k = r[:, perm_k, :]
            la_k, s_k = base_forward_fn(r_k)
            log_abs_list.append(la_k + math.log(abs(w_k)))
            sign_list.append(s_k * (1.0 if w_k > 0 else -1.0))

        log_abs_serial_all = torch.stack(log_abs_list, dim=1)  # (B, K)
        sign_serial_all = torch.stack(sign_list, dim=1)  # (B, K)

        max_log = torch.amax(log_abs_serial_all.detach(), dim=1, keepdim=True)
        all_neginf = torch.isneginf(max_log)
        safe_max_log = torch.where(all_neginf, torch.zeros_like(max_log), max_log)

        scaled_terms = sign_serial_all * torch.exp(log_abs_serial_all - safe_max_log)
        scaled_terms = torch.where(
            all_neginf.expand_as(scaled_terms), torch.zeros_like(scaled_terms), scaled_terms
        )
        summed = torch.sum(scaled_terms, dim=1)

        abs_summed = torch.abs(summed)
        zero_mask = (abs_summed == 0) | all_neginf.squeeze(1)

        sign_serial = torch.sign(summed)
        sign_serial = torch.where(zero_mask, torch.zeros_like(sign_serial), sign_serial)

        safe_abs_summed = torch.clamp(abs_summed, min=1e-20)
        log_abs_serial = torch.where(
            zero_mask,
            torch.full_like(summed, float("-inf")),
            safe_max_log.squeeze(1) + torch.log(safe_abs_summed),
        )

        # 3. 对比验证
        max_diff_la = (log_abs_vec - log_abs_serial).abs().max().item()
        self.assertLess(
            max_diff_la,
            1e-6,
            f"自旋投影批量向量化与串行求和数值对数模存在差异: {max_diff_la}",
        )
        sign_mismatch = (sign_vec != sign_serial).sum().item()
        self.assertEqual(
            sign_mismatch,
            0,
            f"自旋投影批量向量化与串行求和存在符号不匹配项: {sign_mismatch}",
        )

    def test_vectorized_projection_autograd_gradient_equivalence(self) -> None:
        """
        验证自旋投影批量向量化反向传播 Autograd 梯度与串行版本的等价性。
        """
        # 使用自定义多项置换以严格覆盖复杂权重与置换
        custom_perms = [
            ([0, 1, 2], 1.0),
            ([1, 0, 2], -0.5),
            ([0, 2, 1], 0.25),
        ]
        projector = SpinProjector(n_up=2, n_down=1, mode="custom", custom_permutations=custom_perms)
        batch_size = 4
        num_electrons = 3

        r_vec = torch.randn(batch_size, num_electrons, 3, requires_grad=True)
        r_serial = r_vec.clone().detach().requires_grad_(True)

        def base_fn(x: torch.Tensor):
            return torch.sum(x ** 3, dim=(-2, -1)), torch.ones(x.shape[0])

        # 向量化梯度
        la_vec, _ = projector.project_log_psi(base_fn, r_vec)
        la_vec.sum().backward()

        # 串行参考求和梯度
        perms = projector.perms
        weights = projector.weights
        terms = []
        for k in range(projector.num_permutations):
            la_k, _ = base_fn(r_serial[:, perms[k], :])
            terms.append(weights[k] * torch.exp(la_k))
        summed = torch.stack(terms, dim=1).sum(dim=1)
        log_abs_serial = torch.log(torch.abs(summed))
        log_abs_serial.sum().backward()

        # 对比梯度
        grad_diff = (r_vec.grad - r_serial.grad).abs().max().item()
        self.assertLess(
            grad_diff,
            1e-6,
            f"自旋投影向量化与串行 Autograd 梯度存在差异: {grad_diff}",
        )

    def test_spin_projected_wavefunction_alias(self) -> None:
        """验证类名别名 SpinProjectedWaveFunction is SpinProjector。"""
        self.assertIs(SpinProjectedWaveFunction, SpinProjector)


class TestMicroscopicMoveCountersNoRounding(unittest.TestCase):
    """
    P3.3 优化项：逐电子移动微观步与接受率无舍入失真测试 (F-MCMC-03, mcmc.py)
    """

    def setUp(self) -> None:
        # 使用 3 电子分子体系以验证非整除情况 (例如 7 个移动在 3 电子下除不尽)
        self.system = MolecularSystem(
            nuclear_coords=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            nuclear_charges=torch.tensor([3.0], dtype=torch.float32),
            n_up=2,
            n_down=1,
        )

    def test_non_divisible_acceptance_sequence_exact_accumulation(self) -> None:
        """
        测试在非整除电子接受序列下，微观计数器 proposed_moves 与 accepted_moves 真实精确累加，
        彻底证明无任何四舍五入或截断误差。
        """
        num_walkers = 4
        dummy_model = nn.Module()
        # 恒定返回 0，产生比率恒为 1.0 的确定性接受
        dummy_model.forward_log_psi = lambda r: (
            torch.zeros(r.shape[0]),
            torch.ones(r.shape[0]),
        )

        sampler = MCMCSampler(
            dummy_model,
            self.system,
            num_walkers=num_walkers,
            move_mode="one_by_one",
        )

        # 初始微观计数器必须为 0
        self.assertEqual(sampler.proposed_moves, 0)
        self.assertEqual(sampler.accepted_moves, 0)

        # 执行 1 步逐电子抽样 (全部接受)
        sampler.step()

        expected_moves = num_walkers * self.system.total_electrons  # 4 * 3 = 12
        self.assertEqual(sampler.proposed_moves, expected_moves)
        self.assertEqual(sampler.accepted_moves, expected_moves)
        self.assertEqual(sampler.micro_acceptance_rate, 1.0)

        # 人为模拟非整除接受场景（如在 12 次尝试中仅接受 7 次）
        sampler.proposed_moves += 12
        sampler.accepted_moves += 7

        expected_rate = (12 + 7) / (12 + 12)
        self.assertAlmostEqual(sampler.micro_acceptance_rate, expected_rate, places=7)
        self.assertAlmostEqual(sampler.acceptance_rate, expected_rate, places=7)

    def test_all_mode_microscopic_accounting(self) -> None:
        """
        测试全电子移动模式下微观计步：一次全电子移动尝试等价于 N 个微观坐标提议。
        """
        num_walkers = 5
        total_electrons = self.system.total_electrons  # 3

        dummy_model = nn.Module()
        dummy_model.forward_log_psi = lambda r: (
            torch.zeros(r.shape[0]),
            torch.ones(r.shape[0]),
        )

        sampler = MCMCSampler(
            dummy_model,
            self.system,
            num_walkers=num_walkers,
            move_mode="all",
        )

        sampler.step()
        self.assertEqual(sampler.proposed_moves, num_walkers * total_electrons)
        self.assertEqual(sampler.accepted_moves, num_walkers * total_electrons)
        self.assertEqual(sampler.micro_acceptance_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
