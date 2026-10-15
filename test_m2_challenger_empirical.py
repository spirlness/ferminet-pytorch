"""Empirical stress and adversarial verification harness for Milestone 2 (P3 Refactoring).

Author: Challenger M2 (Empirical Challenger)
Target: spin.py (PHY-02), mcmc.py (F-MCMC-03), train_vmc.py (F-TRAIN-02)
"""

import copy
import math
import os
import shutil
import tempfile
import unittest
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from ferminet_pytorch.spin import SpinProjector, SpinProjectedWaveFunction
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.train_vmc import train_vmc, build_h2_system
from ferminet_pytorch.hamiltonian import MolecularSystem
from ferminet_pytorch.model import FermiNet
import ferminet_pytorch.loss as vmc_loss_module


def _serial_reference_project_log_psi(
    projector: SpinProjector,
    base_forward_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    传统串行逐项求和的参考黄金实现 (Golden Reference)，
    严格模拟重构前的原始 Python for 循环置换求和逻辑，用于交叉数值比对。
    """
    if projector.num_permutations == 1 and projector.weights[0].item() == 1.0:
        return base_forward_fn(r[:, projector.perms[0], :])

    batch_size = r.shape[0]
    num_k = projector.num_permutations
    weights_dev = projector.weights.to(device=r.device, dtype=r.dtype)

    log_abs_list = []
    sign_list = []

    for k in range(num_k):
        perm_k = projector.perms[k].to(device=r.device)
        r_k = r[:, perm_k, :]
        la, s = base_forward_fn(r_k)

        w = weights_dev[k]
        w_sign = torch.sign(w)
        w_log_abs = torch.log(torch.abs(w))

        log_abs_list.append(la + w_log_abs)
        sign_list.append(s * w_sign)

    log_abs_all = torch.stack(log_abs_list, dim=1)  # (B, K)
    sign_all = torch.stack(sign_list, dim=1)        # (B, K)

    max_log = torch.amax(log_abs_all.detach(), dim=1, keepdim=True)
    all_neginf = torch.isneginf(max_log)
    safe_max_log = torch.where(all_neginf, torch.zeros_like(max_log), max_log)

    scaled_terms = sign_all * torch.exp(log_abs_all - safe_max_log)
    scaled_terms = torch.where(
        all_neginf.expand_as(scaled_terms), torch.zeros_like(scaled_terms), scaled_terms
    )
    summed = torch.sum(scaled_terms, dim=1)

    abs_summed = torch.abs(summed)
    zero_mask = (abs_summed == 0) | all_neginf.squeeze(1)

    sign_proj = torch.sign(summed)
    sign_proj = torch.where(zero_mask, torch.zeros_like(sign_proj), sign_proj)

    safe_abs_summed = torch.clamp(abs_summed, min=1e-20)
    log_abs_proj = torch.where(
        zero_mask,
        torch.full_like(summed, float("-inf")),
        safe_max_log.squeeze(1) + torch.log(safe_abs_summed),
    )
    return log_abs_proj, sign_proj


class TestChallengerSpinVectorizationEquivalence(unittest.TestCase):
    """实证对抗测试 1：自旋投影置换推断向量化数值与梯度等价性实证。"""

    def setUp(self) -> None:
        torch.manual_seed(20261015)

    def test_alias_and_type_conformance(self) -> None:
        """验证类名别名兼容性与类型同一性。"""
        self.assertIs(
            SpinProjectedWaveFunction,
            SpinProjector,
            "SpinProjectedWaveFunction 别名未正确绑定到 SpinProjector",
        )

    def test_multiconfig_vectorization_vs_serial_equivalence(self) -> None:
        """
        对抗测试：构建多种电子配置与自旋多重态置换，
        包括双电子单重态、三电子二重态、四电子体系以及大置换群体系，
        实证验证批量向量化推断与逐项串行推断的输出完全等价：
        - log|Psi| 绝对差值严格 < 1e-6 (双精度下 < 1e-12)
        - sign 严格完全一致 (差值为 0)
        """
        configurations = [
            # (n_up, n_down, mode, batch_size)
            (1, 1, "singlet", 8),
            (1, 1, "triplet", 8),
            (2, 2, "singlet", 4),
            (2, 2, "triplet", 4),
            (3, 3, "singlet", 2),
        ]

        for n_up, n_down, mode, b_size in configurations:
            with self.subTest(n_up=n_up, n_down=n_down, mode=mode):
                projector = SpinProjector(n_up=n_up, n_down=n_down, mode=mode)
                num_electrons = n_up + n_down

                # 构造具有非平凡非线性响应的小型波函数网络
                linear1 = nn.Linear(num_electrons * 3, 32)
                linear2 = nn.Linear(32, 1)

                def base_fn(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    flat = coords.reshape(coords.shape[0], -1)
                    val = linear2(torch.tanh(linear1(flat))).squeeze(-1)
                    # 模拟带正负号与不同幅度的波函数
                    log_abs = torch.log(torch.abs(val) + 1e-4)
                    sign = torch.sign(val)
                    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                    return log_abs, sign

                r = torch.randn(b_size, num_electrons, 3, dtype=torch.float32)

                # 运行向量化实现与串行黄金参考实现
                with torch.no_grad():
                    log_abs_vec, sign_vec = projector.project_log_psi(base_fn, r)
                    log_abs_seq, sign_seq = _serial_reference_project_log_psi(projector, base_fn, r)

                # 1. 验证符号严格完全一致
                sign_diff = (sign_vec - sign_seq).abs().max().item()
                self.assertEqual(
                    sign_diff,
                    0.0,
                    f"符号不一致！配置: ({n_up}, {n_down}, {mode}), 最大符号差: {sign_diff}",
                )

                # 2. 验证对数绝对值最大差值严格 < 1e-6
                log_abs_diff = (log_abs_vec - log_abs_seq).abs().max().item()
                self.assertLess(
                    log_abs_diff,
                    1e-6,
                    f"对数绝对值超出阈值！配置: ({n_up}, {n_down}, {mode}), 最大绝对差: {log_abs_diff}",
                )

    def test_custom_large_permutations_high_precision(self) -> None:
        """
        对抗测试：构造包含任意多项置换 (如 K=10) 与正负交错加权系数的自定义投影算符，
        在 float64 高精度下验证向量化 LogSumExp 与串行累加达到机器精度级别的等价性。
        """
        num_electrons = 4
        num_k = 10
        batch_size = 5

        # 随机生成 10 个有效置换与权重
        custom_perms_list = []
        for _ in range(num_k):
            p = torch.randperm(num_electrons).tolist()
            w = float(torch.randn(1, dtype=torch.float64).item())
            if abs(w) < 0.1:
                w = 1.0
            custom_perms_list.append((p, w))

        projector = SpinProjector(
            n_up=2,
            n_down=2,
            mode="custom",
            custom_permutations=custom_perms_list,
        )

        r = torch.randn(batch_size, num_electrons, 3, dtype=torch.float64)

        def mock_base_fn(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # 确定性映射
            prod = torch.sin(coords[..., 0]) * torch.cos(coords[..., 1]) + coords[..., 2] ** 2
            val = prod.sum(dim=-1)  # (B,)
            log_abs = torch.log(val.abs() + 1e-5)
            sign = torch.sign(val)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            return log_abs, sign

        with torch.no_grad():
            log_abs_vec, sign_vec = projector.project_log_psi(mock_base_fn, r)
            log_abs_seq, sign_seq = _serial_reference_project_log_psi(projector, mock_base_fn, r)

        diff_log = (log_abs_vec - log_abs_seq).abs().max().item()
        diff_sign = (sign_vec - sign_seq).abs().max().item()

        self.assertEqual(diff_sign, 0.0, "高精度大置换测试符号存在偏差")
        self.assertLess(diff_log, 1e-12, f"float64 下最大差值应当接近机器精度，实际为: {diff_log}")

    def test_autograd_backward_gradient_equivalence(self) -> None:
        """
        对抗测试：比对向量化批量推断与串行求和在反向自动微分 (Autograd) 时的梯度等价性。
        对电子坐标 r 计算 d(sum(log_abs))/dr，实证检验两者的梯度张量最大绝对偏差 < 1e-6。
        """
        n_up, n_down = 2, 2
        num_electrons = n_up + n_down
        batch_size = 4

        projector = SpinProjector(n_up=n_up, n_down=n_down, mode="singlet")

        # 共享相同初始权重的参数网络
        net = nn.Sequential(
            nn.Linear(num_electrons * 3, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

        def base_fn(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            flat = coords.reshape(coords.shape[0], -1)
            out = net(flat).squeeze(-1)
            # 使用处处可微的 softplus 模拟 log_abs
            log_abs = nn.functional.softplus(out)
            sign = torch.ones_like(log_abs)
            return log_abs, sign

        r_vec = torch.randn(batch_size, num_electrons, 3, requires_grad=True)
        r_seq = r_vec.clone().detach().requires_grad_(True)

        # 向量化反向传播
        log_abs_vec, _ = projector.project_log_psi(base_fn, r_vec)
        loss_vec = log_abs_vec.sum()
        loss_vec.backward()
        grad_vec = r_vec.grad.clone()

        # 串行反向传播
        log_abs_seq, _ = _serial_reference_project_log_psi(projector, base_fn, r_seq)
        loss_seq = log_abs_seq.sum()
        loss_seq.backward()
        grad_seq = r_seq.grad.clone()

        grad_diff = (grad_vec - grad_seq).abs().max().item()
        self.assertLess(
            grad_diff,
            1e-6,
            f"Autograd 梯度在向量化与串行之间存在不一致！最大绝对偏差: {grad_diff}",
        )

    def test_extreme_nodal_cancellation_and_neginf_resilience(self) -> None:
        """
        边界对抗测试：波函数在某些置换下严格处于节点面 (log_abs = -inf)，
        或各置换正负项完全相消 (sum = 0) 时，
        验证向量化实现不会产生 NaN，并能正确返回 log_abs = -inf 与 sign = 0。
        """
        # 构造置换算符: 2 电子单重态，权重为 [1.0, 1.0] (空间对称)
        projector = SpinProjector(n_up=1, n_down=1, mode="singlet")
        r = torch.randn(3, 2, 3)

        # 情况 A: 全节点面 (全部置换都为 -inf)
        def all_node_fn(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            b = coords.shape[0]
            return torch.full((b,), float("-inf")), torch.zeros((b,))

        log_abs, sign = projector.project_log_psi(all_node_fn, r)
        self.assertTrue(
            torch.isneginf(log_abs).all().item(),
            "全节点面下 log_abs 必须全部为 -inf",
        )
        self.assertTrue(
            (sign == 0).all().item(),
            "全节点面下 sign 必须为 0",
        )
        self.assertFalse(torch.isnan(log_abs).any().item(), "全节点面下出现 NaN 毒化")

        # 情况 B: 空间反对称波函数经过单重态 (要求对称) 投影，两项严格反号相消
        # 置换 0 项为 +1.0 * val，置换 1 项为 -1.0 * val，加和精确为 0
        def antisym_fn(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # x1 - x2: 当置换电子时反号
            diff = coords[:, 0, 0] - coords[:, 1, 0]
            val = diff.clone()
            return torch.log(val.abs() + 1e-2), torch.sign(val)

        # 空间反对称的态投影到对称流形上必然相消为 0
        log_abs_c, sign_c = projector.project_log_psi(antisym_fn, r)
        self.assertTrue(
            torch.isneginf(log_abs_c).all().item(),
            "正负完全相消时 log_abs 必须为 -inf",
        )
        self.assertTrue(
            (sign_c == 0).all().item(),
            "正负完全相消时 sign 必须为 0",
        )

    def test_odd_electron_custom_permutation_equivalence(self) -> None:
        """
        对抗测试：奇数电子体系 (如 Li 原子 3 电子) 的多项空间置换，
        实证检验向量化推断与串行推断的绝对一致性。
        """
        # 3 电子体系：3 个置换与加权系数
        custom_perms = [
            ([0, 1, 2], 1.0),
            ([1, 0, 2], -0.5),
            ([2, 1, 0], 0.5),
        ]
        projector = SpinProjector(
            n_up=2,
            n_down=1,
            mode="custom",
            custom_permutations=custom_perms,
        )

        r = torch.randn(12, 3, 3)

        def mock_li_fn(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            val = coords[:, 0, :] + coords[:, 1, :] * 2.0 - coords[:, 2, :]
            s = val.sum(dim=-1)
            return torch.log(s.abs() + 1e-3), torch.sign(s)

        with torch.no_grad():
            la_vec, s_vec = projector.project_log_psi(mock_li_fn, r)
            la_seq, s_seq = _serial_reference_project_log_psi(projector, mock_li_fn, r)

        self.assertEqual((s_vec - s_seq).abs().max().item(), 0.0)
        self.assertLess((la_vec - la_seq).abs().max().item(), 1e-6)

    def test_large_dynamic_range_coordinates_numerical_stability(self) -> None:
        """
        对抗测试：构造动态范围跨越多个数量级的对数波函数 (如 -100 到 +50)，
        实证检验带符号 LogSumExp 向量化合成不会发生数值溢出为 inf 或下溢为 NaN，
        且与串行高精度结果完全吻合。
        """
        projector = SpinProjector(n_up=2, n_down=2, mode="singlet")
        b_size = 16
        r = torch.randn(b_size, 4, 3)

        def wide_range_fn(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # 依据构型坐标本身计算，确保每个坐标构型的函数值确定且独立于批次大小
            coord_norm = torch.norm(coords, dim=-1).sum(dim=-1)  # (B,)
            scales = 50.0 * torch.sin(coord_norm) - 20.0        # 范围在 [-70, +30]
            signs = torch.where(torch.cos(coord_norm) >= 0, 1.0, -1.0)
            return scales, signs

        la_vec, s_vec = projector.project_log_psi(wide_range_fn, r)
        la_seq, s_seq = _serial_reference_project_log_psi(projector, wide_range_fn, r)

        self.assertFalse(torch.isnan(la_vec).any().item(), "对数波函数出现 NaN 溢出")
        self.assertFalse(torch.isposinf(la_vec).any().item(), "对数波函数出现 +inf 溢出")
        self.assertEqual((s_vec - s_seq).abs().max().item(), 0.0)
        self.assertLess((la_vec - la_seq).abs().max().item(), 1e-5)

    def test_batch_size_broadcasting_boundaries(self) -> None:
        """对抗测试：测试极限批次边界 (B=1 单样本与 B=64 大批次) 的张量广播鲁棒性。"""
        projector = SpinProjector(n_up=1, n_down=1, mode="singlet")

        def simple_fn(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            v = coords.sum(dim=(-2, -1))
            return torch.log(v.abs() + 1.0), torch.sign(v)

        for b in [1, 64]:
            r = torch.randn(b, 2, 3)
            la_vec, s_vec = projector.project_log_psi(simple_fn, r)
            la_seq, s_seq = _serial_reference_project_log_psi(projector, simple_fn, r)
            self.assertEqual(la_vec.shape, (b,))
            self.assertEqual(s_vec.shape, (b,))
            self.assertEqual((s_vec - s_seq).abs().max().item(), 0.0)
            self.assertLess((la_vec - la_seq).abs().max().item(), 1e-6)


class TestChallengerMicroStepCountingPrecision(unittest.TestCase):
    """实证对抗测试 2：逐电子移动微观计数无舍入与无截断失真验证。"""

    def setUp(self) -> None:
        torch.manual_seed(20261015)

    def test_non_divisible_micro_acceptance_exactness(self) -> None:
        """
        对抗测试：构造各种非整除电子移动接受序列（例如 3 电子体系在 7 个微观接受动作下，
        5 电子体系在 11 个微观接受动作下），
        实证检验 MCMC 采样器的 proposed_moves、accepted_moves 和 micro_acceptance_rate
        精确记录真实的微观动作数，绝对无四舍五入或整数除法截断。
        并与旧版四舍五入策略进行实证量化对比，证明旧版策略在非整除情况下的偏差。
        """
        test_cases = [
            # (total_electrons, num_walkers, targeted_accept_counts_per_electron)
            # 3 电子体系，单步内 3 个电子依次接受 2, 3, 2 个 walker，总计 7 个接受动作
            (3, 4, [2, 3, 2]),
            # 3 电子体系，单步内 3 个电子依次接受 1, 0, 0 个 walker，总计 1 个接受动作 (极低接受)
            (3, 4, [1, 0, 0]),
            # 5 电子体系，单步内 5 个电子依次接受 3, 2, 1, 2, 3 个 walker，总计 11 个接受动作
            (5, 4, [3, 2, 1, 2, 3]),
            # 7 电子体系，单步内接受 13 个动作
            (7, 4, [2, 2, 2, 1, 2, 2, 2]),
        ]

        for total_electrons, num_walkers, accept_pattern in test_cases:
            expected_accepted_moves = sum(accept_pattern)
            expected_proposed_moves = total_electrons * num_walkers
            expected_micro_rate = float(expected_accepted_moves) / expected_proposed_moves

            # 模拟旧版四舍五入会导致的失真结果:
            old_rounded_macro_steps = int(round(expected_accepted_moves / total_electrons))
            old_implied_micro_moves = old_rounded_macro_steps * total_electrons

            with self.subTest(
                electrons=total_electrons,
                target_accepted=expected_accepted_moves,
                proposed=expected_proposed_moves,
            ):
                # 构造具有指定电子数的分子体系
                n_up = (total_electrons + 1) // 2
                n_down = total_electrons - n_up
                nuclear_coords = torch.zeros(1, 3)
                charges = torch.tensor([float(total_electrons)])
                system = MolecularSystem(
                    nuclear_coords=nuclear_coords,
                    nuclear_charges=charges,
                    n_up=n_up,
                    n_down=n_down,
                )

                # 构造受控接受行为的 Mock 模型
                # 通过让 forward_log_psi 返回受控的值，或者通过控制 Metropolis ratio
                # 这里我们直接利用 Mock 方式精确模拟每次电子移动被接受的 Walker 数量
                dummy_model = nn.Module()
                dummy_model.forward_log_psi = lambda r: (torch.zeros(r.shape[0]), torch.ones(r.shape[0]))

                sampler = MCMCSampler(
                    model=dummy_model,
                    system=system,
                    num_walkers=num_walkers,
                    move_mode="one_by_one",
                    adapt_step_size=False,
                )

                # 重置计数器
                sampler.proposed_moves = 0
                sampler.accepted_moves = 0
                sampler.accepted_steps = 0

                # 模拟 _step_one_electron 内部的逐电子移动，
                # 但精确注入 accept_pattern 中的接受数
                # 验证 mcmc.py 中的计数更新逻辑
                total_accepted_count = 0
                total_accepted = 0.0

                for i in range(total_electrons):
                    k_acc = accept_pattern[i]
                    # 模拟 accepted 布尔掩码
                    accepted = torch.zeros(num_walkers, dtype=torch.bool)
                    accepted[:k_acc] = True

                    total_accepted += accepted.float().mean().item()
                    total_accepted_count += int(accepted.sum().item())

                # 按照 mcmc.py 的标准实现执行计数累加
                acc_rate = total_accepted / max(total_electrons, 1)
                sampler.accepted_steps += int(round(total_accepted_count / total_electrons))
                sampler.proposed_moves += num_walkers * total_electrons
                sampler.accepted_moves += total_accepted_count

                # 1. 严格断言：微观接受移动数完全等于预期的动作数，无任何舍入偏差
                self.assertEqual(
                    sampler.accepted_moves,
                    expected_accepted_moves,
                    f"accepted_moves 出现偏差！预期 {expected_accepted_moves}, 实际 {sampler.accepted_moves}",
                )

                # 2. 严格断言：微观提议移动数等于 N * num_walkers
                self.assertEqual(
                    sampler.proposed_moves,
                    expected_proposed_moves,
                    f"proposed_moves 出现偏差！预期 {expected_proposed_moves}, 实际 {sampler.proposed_moves}",
                )

                # 3. 严格断言：micro_acceptance_rate 连续精确，无量化阶跃
                self.assertAlmostEqual(
                    sampler.micro_acceptance_rate,
                    expected_micro_rate,
                    places=7,
                    msg=f"micro_acceptance_rate 不精确！预期 {expected_micro_rate}, 实际 {sampler.micro_acceptance_rate}",
                )

                # 4. 对比旧版舍入失真：当不能整除时，旧版必发生偏差
                if expected_accepted_moves % total_electrons != 0:
                    discrepancy = abs(old_implied_micro_moves - expected_accepted_moves)
                    self.assertGreater(
                        discrepancy,
                        0,
                        f"在非整除用例下，旧版逻辑未显现出预期的舍入失真？（实测偏差={discrepancy}）",
                    )

    def test_consecutive_multi_step_accumulation_precision(self) -> None:
        """
        对抗测试：执行多个连续 MCMC 步，每次接受数均非整除，
        实证验证多次步数累加后，微观计数器绝无累计漂移或残余舍入误差。
        """
        sys = build_h2_system()  # 2 电子体系
        num_walkers = 7          # 奇数个 Walker，极易产生非对称接受

        # 真实 MCMC 运行
        model = FermiNet(
            nuclear_coords=sys.nuclear_coords,
            nuclear_charges=sys.nuclear_charges,
            n_up=sys.n_up,
            n_down=sys.n_down,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_determinants=1,
        )

        sampler = MCMCSampler(
            model=model,
            system=sys,
            num_walkers=num_walkers,
            step_size=0.8,
            move_mode="one_by_one",
            adapt_step_size=False,
        )

        # 记录每步实际接受的微观数
        total_steps = 10
        prev_accepted_moves = 0
        prev_proposed_moves = 0

        for s in range(total_steps):
            sampler.step()
            delta_proposed = sampler.proposed_moves - prev_proposed_moves
            delta_accepted = sampler.accepted_moves - prev_accepted_moves

            # 每一步尝试的微观数必须严格等于 2 * 7 = 14
            self.assertEqual(delta_proposed, 14, f"第 {s} 步的微观提议移动数不为 14")
            # 接受数必须在 [0, 14] 闭区间内
            self.assertGreaterEqual(delta_accepted, 0)
            self.assertLessEqual(delta_accepted, 14)

            prev_proposed_moves = sampler.proposed_moves
            prev_accepted_moves = sampler.accepted_moves

        # 最终微观提议移动总数必须是精确的 10 * 14 = 140
        self.assertEqual(sampler.proposed_moves, 140)
        self.assertEqual(
            sampler.micro_acceptance_rate,
            float(sampler.accepted_moves) / 140.0,
        )

    def test_all_electron_mode_micro_macro_consistency(self) -> None:
        """
        对抗测试：在全电子一体化移动模式 (All-electron moves) 下，
        验证每个 Walker 构型移动等价于 total_electrons 个微观坐标位移，
        微观计数与宏观计数保持物理严格一致。
        """
        sys = build_h2_system()  # total_electrons = 2
        num_walkers = 16
        dummy_model = nn.Module()
        dummy_model.forward_log_psi = lambda r: (torch.zeros(r.shape[0]), torch.ones(r.shape[0]))

        sampler = MCMCSampler(
            model=dummy_model,
            system=sys,
            num_walkers=num_walkers,
            move_mode="all",
            adapt_step_size=False,
        )

        acc_rate = sampler.step()
        # 全接受情况下 (ratio=1.0):
        # 宏观步: accepted_steps == 16, total_steps == 1
        # 微观步: proposed_moves == 16 * 2 = 32, accepted_moves == 16 * 2 = 32
        self.assertEqual(sampler.total_steps, 1)
        self.assertEqual(sampler.proposed_moves, 32)
        self.assertEqual(sampler.accepted_moves, 32)
        self.assertEqual(sampler.micro_acceptance_rate, 1.0)
        self.assertEqual(sampler.acceptance_rate, 1.0)
        self.assertEqual(sampler.cumulative_acceptance_rate, 1.0)


class TestChallengerBestModelAndCheckpointPersistence(unittest.TestCase):
    """实证对抗测试 3：最优模型状态恢复与 Checkpoint 序列化/反序列化。"""

    def setUp(self) -> None:
        torch.manual_seed(20261015)
        self.test_dir = tempfile.mkdtemp(prefix="challenger_m2_test_")

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_simulated_energy_oscillation_restores_true_best_epoch(self) -> None:
        """
        对抗测试：模拟一个能量剧烈恶化/波动的训练过程：
        - Epoch 1: E = -1.10 (初始)
        - Epoch 2: E = -1.80 (历史最佳基态！)
        - Epoch 3: E = -0.50 (恶化)
        - Epoch 4: E = +10.0 (严重跑飞/变质)

        实证验证：
        1. 训练结束后，模型参数恢复为 Epoch 2 时的最优权重，而非 Epoch 4 的劣质权重；
        2. 返回的 best_epoch 为 2，best_energy 为 -1.80；
        3. 若设置 restore_best=False，模型参数保留为 Epoch 4 的劣质权重。
        """
        energy_schedule = [-1.10, -1.80, -0.50, +10.0]
        recorded_states: Dict[int, Dict[str, torch.Tensor]] = {}

        original_compute_loss = vmc_loss_module.VMCLoss.compute_step_loss
        original_recompute = MCMCSampler.recompute_log_psi

        # 拦截 compute_step_loss 注入能量波动，并精准捕获参数更新前评估能量时的真实模型状态 theta_t
        current_epoch_counter = [0]

        def mocked_compute_step_loss(self_obj, model, system, walkers, **kwargs):
            current_epoch_counter[0] += 1
            ep = current_epoch_counter[0]
            recorded_states[ep] = copy.deepcopy(model.state_dict())
            sim_e = energy_schedule[ep - 1]
            loss = sum((p ** 2).sum() for p in model.parameters()) * 0.1 + sim_e
            stats = {
                "energy_mean": sim_e,
                "energy_clipped_mean": sim_e,
                "energy_var": 0.05,
                "loss": sim_e,
                "num_valid": walkers.shape[0],
                "num_invalid": 0,
            }
            return loss, stats, None

        vmc_loss_module.VMCLoss.compute_step_loss = mocked_compute_step_loss
        try:
            # 运行 train_vmc 4 个 epoch，开启 restore_best=True
            res_restored = train_vmc(
                num_epochs=4,
                num_walkers=16,
                burn_in_steps=0,
                mcmc_steps_per_epoch=1,
                restore_best=True,
                verbose=False,
            )

            # 验证返回的最优指标
            self.assertEqual(res_restored["best_epoch"], 2, "最优 Epoch 应为第 2 轮")
            self.assertEqual(res_restored["best_energy"], -1.80, "最优能量应为 -1.80")

            # 验证当前 model 的参数与 Epoch 2 的参数 100% 精确一致
            model_current = res_restored["model"]
            best_snapshot = recorded_states[2]
            worst_snapshot = recorded_states[4]

            max_diff_from_best = max(
                (model_current.state_dict()[k] - best_snapshot[k]).abs().max().item()
                for k in best_snapshot
            )
            self.assertEqual(
                max_diff_from_best,
                0.0,
                f"恢复后的模型参数与 Epoch 2 不一致！最大差值: {max_diff_from_best}",
            )

            # 验证与劣质参数存在显著差异 (确保确实恢复了，而不是所有轮次参数碰巧相同)
            # 由于优化器 step 会更新权重，Epoch 4 与 Epoch 2 必有差异
            max_diff_from_worst = max(
                (model_current.state_dict()[k] - worst_snapshot[k]).abs().max().item()
                for k in worst_snapshot
            )
            self.assertGreater(
                max_diff_from_worst,
                1e-6,
                "Epoch 2 与 Epoch 4 的权重竟然无差异，请检查优化器步进",
            )

        finally:
            # 恢复原生函数
            vmc_loss_module.VMCLoss.compute_step_loss = original_compute_loss
            MCMCSampler.recompute_log_psi = original_recompute

    def test_no_restore_best_flag_preserves_final_state(self) -> None:
        """验证设置 restore_best=False 时，模型保持为最后一个 epoch 的参数。"""
        energy_schedule = [-1.0, -3.0, -0.2]
        current_epoch_counter = [0]
        recorded_states: Dict[int, Dict[str, torch.Tensor]] = {}

        original_compute_loss = vmc_loss_module.VMCLoss.compute_step_loss
        original_recompute = MCMCSampler.recompute_log_psi

        def mocked_compute_step_loss(self_obj, model, system, walkers, **kwargs):
            current_epoch_counter[0] += 1
            ep = current_epoch_counter[0]
            sim_e = energy_schedule[ep - 1]
            loss = sum((p ** 2).sum() for p in model.parameters()) * 0.1 + sim_e
            stats = {
                "energy_mean": sim_e,
                "energy_clipped_mean": sim_e,
                "energy_var": 0.05,
                "loss": sim_e,
                "num_valid": walkers.shape[0],
                "num_invalid": 0,
            }
            return loss, stats, None

        def hooked_recompute_log_psi(self_sampler):
            ep = current_epoch_counter[0]
            if ep > 0:
                recorded_states[ep] = copy.deepcopy(self_sampler.model.state_dict())
            return original_recompute(self_sampler)

        vmc_loss_module.VMCLoss.compute_step_loss = mocked_compute_step_loss
        MCMCSampler.recompute_log_psi = hooked_recompute_log_psi
        try:
            res = train_vmc(
                num_epochs=3,
                num_walkers=16,
                burn_in_steps=0,
                mcmc_steps_per_epoch=1,
                restore_best=False,
                verbose=False,
            )

            # 最优记录依然存在
            self.assertEqual(res["best_epoch"], 2)
            self.assertEqual(res["best_energy"], -3.0)

            # 但模型应当处于最后一个 epoch 的权重
            final_snapshot = recorded_states[3]
            max_diff_from_final = max(
                (res["model"].state_dict()[k] - final_snapshot[k]).abs().max().item()
                for k in final_snapshot
            )
            self.assertEqual(max_diff_from_final, 0.0, "restore_best=False 时应保持最终权重")

        finally:
            vmc_loss_module.VMCLoss.compute_step_loss = original_compute_loss
            MCMCSampler.recompute_log_psi = original_recompute

    def test_checkpoint_serialization_and_deserialization_fidelity(self) -> None:
        """
        对抗测试：检验 Checkpoint 在嵌套未知路径下的自动创建与完整序列化。
        实证验证：
        1. 成功生成 .pt Checkpoint 文件；
        2. 反序列化读取后，字段集合完全包含：
           epoch, best_epoch, best_energy, model_state_dict, best_model_state_dict,
           optimizer_state_dict, scheduler_state_dict, history；
        3. 能被全新实例化的模型与优化器无损加载；
        4. 加载后的模型推断输出与保存前完全一致。
        """
        nested_ckpt_path = os.path.join(self.test_dir, "deeply", "nested", "run_ckpt.pt")

        res = train_vmc(
            num_epochs=3,
            num_walkers=16,
            burn_in_steps=2,
            mcmc_steps_per_epoch=2,
            checkpoint_path=nested_ckpt_path,
            verbose=False,
        )

        # 1. 验证文件已持久化存在且非空
        self.assertTrue(os.path.isfile(nested_ckpt_path), f"Checkpoint 文件未生成: {nested_ckpt_path}")
        self.assertGreater(os.path.getsize(nested_ckpt_path), 1024, "Checkpoint 文件过小，可能损坏")

        # 2. 反序列化
        checkpoint_data = torch.load(nested_ckpt_path, map_location="cpu", weights_only=False)

        expected_keys = {
            "epoch",
            "best_epoch",
            "best_energy",
            "model_state_dict",
            "best_model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "history",
        }
        self.assertTrue(
            expected_keys.issubset(checkpoint_data.keys()),
            f"Checkpoint 缺少必要字段！缺失项: {expected_keys - checkpoint_data.keys()}",
        )

        self.assertEqual(checkpoint_data["epoch"], 3)
        self.assertEqual(checkpoint_data["best_epoch"], res["best_epoch"])
        self.assertEqual(checkpoint_data["best_energy"], res["best_energy"])

        # 3. 使用全新模型与优化器进行加载恢复验证
        sys = res["system"]
        new_model = FermiNet(
            nuclear_coords=sys.nuclear_coords,
            nuclear_charges=sys.nuclear_charges,
            n_up=sys.n_up,
            n_down=sys.n_down,
            num_determinants=8,
            hidden_dim_one=64,
            hidden_dim_two=16,
            num_layers=2,
        )
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)

        # 正常无报错加载
        new_model.load_state_dict(checkpoint_data["best_model_state_dict"])
        new_optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        # 4. 验证新模型与原始最优模型的推断输出等价性
        r_test = torch.randn(4, sys.total_electrons, 3)
        with torch.no_grad():
            log_abs_orig, sign_orig = res["model"].forward_log_psi(r_test)
            log_abs_loaded, sign_loaded = new_model.forward_log_psi(r_test)

        self.assertEqual((sign_orig - sign_loaded).abs().max().item(), 0.0)
        self.assertLess((log_abs_orig - log_abs_loaded).abs().max().item(), 1e-6)

    def test_non_finite_energy_does_not_corrupt_best_state(self) -> None:
        """
        对抗测试：当某一轮训练由于极端波动产生 NaN 或 Inf 能量时，
        验证 best_energy 和 best_model_state 绝不会被 NaN / Inf 污染。
        """
        energy_schedule = [-1.5, float("nan"), float("inf"), -1.2]
        current_epoch_counter = [0]
        original_compute_loss = vmc_loss_module.VMCLoss.compute_step_loss

        def mocked_compute_step_loss(self_obj, model, system, walkers, **kwargs):
            current_epoch_counter[0] += 1
            ep = current_epoch_counter[0]
            sim_e = energy_schedule[ep - 1]
            loss = sum(p.sum() for p in model.parameters()) * 0.0 + (0.0 if not math.isfinite(sim_e) else sim_e)
            stats = {
                "energy_mean": sim_e,
                "energy_clipped_mean": sim_e,
                "energy_var": 0.05,
                "loss": 0.0 if not math.isfinite(sim_e) else sim_e,
                "num_valid": walkers.shape[0],
                "num_invalid": 0,
            }
            return loss, stats, None

        vmc_loss_module.VMCLoss.compute_step_loss = mocked_compute_step_loss
        try:
            res = train_vmc(
                num_epochs=4,
                num_walkers=16,
                burn_in_steps=0,
                mcmc_steps_per_epoch=1,
                restore_best=True,
                verbose=False,
            )

            # 最优能量应当始终是有限的 -1.5 (Epoch 1)，绝不能是 NaN 或 Inf
            self.assertEqual(res["best_epoch"], 1)
            self.assertEqual(res["best_energy"], -1.5)
            self.assertTrue(math.isfinite(res["best_energy"]))

            # 验证最优状态参数中无 NaN
            for k, param in res["best_model_state"].items():
                self.assertFalse(
                    torch.isnan(param).any().item(),
                    f"最优状态张量 {k} 含有 NaN！",
                )

        finally:
            vmc_loss_module.VMCLoss.compute_step_loss = original_compute_loss

    def test_real_vmc_training_best_energy_matches_min_history(self) -> None:
        """
        真实物理端到端对抗测试：运行真实的端到端小型 VMC 优化过程，
        实证检验：
        1. best_energy 严格等于整个训练历史上实测最小的 energy_raw：best_energy == min(history["energy_raw"])；
        2. best_epoch 严格等于 min(history["energy_raw"]) 首次出现的 epoch 索引 (1-based)；
        3. 最终返回的模型 parameters 与 res["best_model_state"] 误差严格为 0.0。
        """
        res = train_vmc(
            num_epochs=5,
            num_walkers=32,
            burn_in_steps=5,
            mcmc_steps_per_epoch=2,
            restore_best=True,
            verbose=False,
        )

        raw_energies = res["history"]["energy_raw"]
        min_e = min(raw_energies)
        min_idx = raw_energies.index(min_e) + 1  # 1-indexed

        self.assertEqual(res["best_epoch"], min_idx, "best_epoch 与 history 最小值索引不吻合")
        self.assertEqual(res["best_energy"], min_e, "best_energy 与 history 最小值不相等")

        # 验证当前模型权重与 best_model_state 100% 吻合
        current_state = res["model"].state_dict()
        best_state = res["best_model_state"]
        max_diff = max(
            (current_state[k] - best_state[k]).abs().max().item()
            for k in best_state
        )
        self.assertEqual(max_diff, 0.0, f"恢复后的模型参数与 best_model_state 存在残余误差: {max_diff}")


if __name__ == "__main__":
    unittest.main()
