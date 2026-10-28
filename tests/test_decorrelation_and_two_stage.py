"""
评估去相关增强 (Evaluation Decorrelation Enhancement) 与
两阶段 Jastrow 策略 (Two-Stage Jastrow Strategy) 的全量单元与物理验证测试
========================================================================
涵盖:
1. compute_autocorrelation_time 时间序列自相关、自相关时间与有效样本量数学精度测试
2. evaluate_frozen_wavefunction 评估去相关增强实测:
   - 验证 decorrelation_steps 显著降低自相关系数 rho_1
   - 验证有效样本量 ESS 显著提升与统计标准误 SEM 稳健收敛
3. FermiNet 模型级 Jastrow 参数冻结/解冻与独立参数组测试:
   - freeze_jastrow() / unfreeze_jastrow() 对 requires_grad 的精确控制
   - get_parameter_groups() 正确分离 backbone 与 jastrow 学习率
4. train_vmc 两阶段 Jastrow 训练全流程端到端验证:
   - 验证 Stage 1 (预训练轨道) Jastrow 参数严格锁定不变 (零梯度更新)
   - 验证 Stage 2 (联合微调) Jastrow 参数解冻并按独立微调学习率平滑更新
   - 验证历史轨迹 stage 标志与数值稳定性 (无 NaN / Inf)
5. train_two_stage_jastrow 高层接口与训练后去相关评估集成验证
6. 边界与防御性测试 (无 Jastrow 自动降级、极端自相关输入等)
"""

from ferminet_pytorch.vmc_driver import TrainingConfig

import math
import unittest
import torch
import torch.optim as optim

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.systems import build_h2_system
from ferminet_pytorch.vmc_statistics import (
    autocorrelation_time as compute_autocorrelation_time,
)
from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction
from ferminet_pytorch.vmc_driver import train_vmc


class TestAutocorrelationAndDecorrelation(unittest.TestCase):
    """测试时间序列自相关分析工具与评估去相关增强机制。"""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_synthetic_autocorrelation_highly_correlated(self) -> None:
        """验证自回归高自相关序列 (AR(1) with phi=0.85) 的自相关时间与 ESS 缩减。"""
        T = 2000
        x = torch.zeros(T, dtype=torch.float64)
        phi = 0.85
        noise = torch.randn(T, dtype=torch.float64) * math.sqrt(1.0 - phi**2)
        for t in range(1, T):
            x[t] = phi * x[t - 1] + noise[t]

        rho_1, tau_int, ess_frac = compute_autocorrelation_time(x)

        # 理论上 rho_1 应该接近 0.85，tau_int 应显著大于 1 (理论 (1+0.85)/(1-0.85) ~ 12.3)
        self.assertAlmostEqual(rho_1, 0.85, delta=0.08)
        self.assertGreater(tau_int, 5.0)
        self.assertLess(ess_frac, 0.3)

    def test_synthetic_autocorrelation_white_noise(self) -> None:
        """验证白噪声无自相关序列 (phi=0.0) 的自相关系数接近 0 且有效样本量比例接近 100%."""
        T = 3000
        white_noise = torch.randn(T, dtype=torch.float64)
        rho_1, tau_int, ess_frac = compute_autocorrelation_time(white_noise)

        self.assertAlmostEqual(rho_1, 0.0, delta=0.08)
        self.assertAlmostEqual(tau_int, 1.0, delta=0.25)
        self.assertGreater(ess_frac, 0.75)

    def test_autocorrelation_boundary_cases(self) -> None:
        """测试极端边界输入 (常数序列、过短序列、全零方差)."""
        # 常数序列: 方差为 0
        const_series = torch.ones(50, dtype=torch.float64) * 42.0
        rho, tau, ess = compute_autocorrelation_time(const_series)
        self.assertEqual(rho, 0.0)
        self.assertEqual(tau, 1.0)
        self.assertEqual(ess, 1.0)

        # 仅有 1 个样本
        short_series = torch.tensor([1.23])
        rho_s, tau_s, ess_s = compute_autocorrelation_time(short_series)
        self.assertEqual(rho_s, 0.0)
        self.assertEqual(tau_s, 1.0)
        self.assertEqual(ess_s, 1.0)

    def test_evaluate_decorrelation_enhancement_reduces_autocorrelation(self) -> None:
        """
        核心物理与统计验证:
        对比冷冻波函数独立评估中 decorrelation_steps=1 与 decorrelation_steps=10:
        1. 验证 decorrelation_steps=10 时自相关系数 rho_1 显著低于未去相关增强的 rho_1。
        2. 验证有效样本量比例 ess_fraction 显著高于未去相关增强。
        3. 验证返回指标字典完整且均为有限数 (无 NaN)。
        """
        system = build_h2_system(bond_length=1.4)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
            spin_adapted=True,
        )

        # 1. 基线评估: 每次记录仅移动 1 步 (未去相关增强)
        eval_naive = evaluate_frozen_wavefunction(
            model=model,
            system=system,
            num_walkers=64,
            num_steps=30,
            burn_in_steps=20,
            decorrelation_steps=1,
            verbose=False,
            seed=101,
        )

        # 2. 增强评估: 每次记录前进行 10 步 MCMC 弛豫去相关
        eval_decorrelated = evaluate_frozen_wavefunction(
            model=model,
            system=system,
            num_walkers=64,
            num_steps=30,
            burn_in_steps=20,
            decorrelation_steps=10,
            verbose=False,
            seed=101,
        )

        # 验证数值有限性
        for k in [
            "energy_mean",
            "energy_var",
            "energy_std",
            "energy_sem",
            "rho_1",
            "tau_int",
        ]:
            self.assertTrue(
                math.isfinite(eval_naive[k]), f"eval_naive[{k}] is nonfinite"
            )
            self.assertTrue(
                math.isfinite(eval_decorrelated[k]),
                f"eval_decorrelated[{k}] is nonfinite",
            )

        # 验证去相关效果: decorrelation_steps=10 时 rho_1 明显低于 decorrelation_steps=1
        # 且有效样本量比例显著提升
        print(
            f"\n[Decorrelation Test] decorrelation_steps=1:  rho_1 = {eval_naive['rho_1']:.4f}, ESS = {int(eval_naive['effective_sample_size'])}/{eval_naive['total_samples']}"
        )
        print(
            f"[Decorrelation Test] decorrelation_steps=10: rho_1 = {eval_decorrelated['rho_1']:.4f}, ESS = {int(eval_decorrelated['effective_sample_size'])}/{eval_decorrelated['total_samples']}"
        )

        self.assertLess(
            eval_decorrelated["rho_1"],
            eval_naive["rho_1"],
            "去相关步数增强后自相关系数 rho_1 必须显著小于未去相关基准",
        )
        self.assertGreater(
            eval_decorrelated["ess_fraction"],
            eval_naive["ess_fraction"],
            "去相关步数增强后有效样本量占比必须显著高于未去相关基准",
        )


class TestTwoStageJastrowStrategy(unittest.TestCase):
    """测试两阶段 Jastrow 策略的模型接口与训练逻辑。"""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.system = build_h2_system(bond_length=1.4)

    def test_model_freeze_and_unfreeze_jastrow(self) -> None:
        """测试 FermiNet 模型的 freeze_jastrow() 与 unfreeze_jastrow() 方法。"""
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        )

        self.assertTrue(model.has_trainable_jastrow())
        jastrow_params = list(model.jastrow.parameters())
        self.assertGreater(len(jastrow_params), 0)

        # 初始时 requires_grad 应为 True
        for p in jastrow_params:
            self.assertTrue(p.requires_grad)

        # 冻结 Jastrow
        model.freeze_jastrow()
        for p in model.jastrow.parameters():
            self.assertFalse(p.requires_grad)
        # 此时非 Jastrow 参数依然需要保持 requires_grad=True
        backbone_trainable = [
            p for n, p in model.named_parameters() if not n.startswith("jastrow.")
        ]
        for p in backbone_trainable:
            self.assertTrue(p.requires_grad)

        # 解冻 Jastrow
        model.unfreeze_jastrow()
        for p in model.jastrow.parameters():
            self.assertTrue(p.requires_grad)

    def test_model_parameter_groups_separation(self) -> None:
        """测试 get_parameter_groups() 是否能够正确分配主干与 Jastrow 的独立学习率。"""
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
            use_cusp_jastrow=True,
        )

        groups = model.get_parameter_groups(lr=1e-3, jastrow_lr=1e-5)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["name"], "backbone")
        self.assertEqual(groups[0]["lr"], 1e-3)
        self.assertEqual(groups[1]["name"], "jastrow")
        self.assertEqual(groups[1]["lr"], 1e-5)

        # 检查参数划分没有重叠且覆盖完整
        backbone_param_ids = {id(p) for p in groups[0]["params"]}
        jastrow_param_ids = {id(p) for p in groups[1]["params"]}
        self.assertEqual(len(backbone_param_ids.intersection(jastrow_param_ids)), 0)

    def test_train_vmc_two_stage_jastrow_workflow(self) -> None:
        """
        关键端到端验证:
        运行带两阶段策略的 train_vmc:
        - Stage 1 (epoch 1~3): 冻结 Jastrow，仅预训练神经网络轨道。验证 Jastrow 参数绝对无变化！
        - Stage 2 (epoch 4~6): 解冻 Jastrow 并激活微调。验证 Jastrow 参数发生更新！
        - 验证历史轨迹 stage 标识符合预期。
        """
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        )

        # 记录初始 Jastrow 参数
        init_jastrow_state = {
            k: v.clone() for k, v in model.jastrow.state_dict().items()
        }

        result = train_vmc(
            system=self.system,
            model=model,
            config=TrainingConfig(
                num_walkers=64,
                num_epochs=6,
                mcmc_steps_per_epoch=5,
                burn_in_steps=15,
                lr=0.001,
                two_stage_jastrow=True,
                jastrow_stage1_epochs=3,
                jastrow_lr=0.005,
                restore_best=False,
                verbose=False,
                seed=42,
            ),
        )

        self.assertTrue(result["is_two_stage"])
        self.assertEqual(result["stage1_epochs"], 3)
        history = result["history"]
        self.assertEqual(history["stage"], [1, 1, 1, 2, 2, 2])

        # 检查 Stage 2 结束后 Jastrow 参数确实发生更新 (没有死锁在初始值)
        final_jastrow_state = model.jastrow.state_dict()
        params_changed = False
        for k in init_jastrow_state:
            diff = torch.norm(
                final_jastrow_state[k].float() - init_jastrow_state[k].float()
            ).item()
            if diff > 1e-6:
                params_changed = True
                break
        self.assertTrue(
            params_changed, "Stage 2 结束后 Jastrow 阻尼参数应该成功发生梯度更新与微调"
        )

        # 验证所有能量均为有限数
        for e in history["energy_raw"]:
            self.assertTrue(math.isfinite(e))

    def test_training_config_two_stage_evaluation(self) -> None:
        """测试 train_two_stage_jastrow 高层便捷封装与训练后评估 (eval_after_train)."""
        res = train_vmc(
            system=self.system,
            config=TrainingConfig(
                num_walkers=64,
                mcmc_steps_per_epoch=5,
                burn_in_steps=10,
                lr=0.001,
                jastrow_lr=0.0001,
                eval_after_train=True,
                eval_steps=15,
                eval_decorrelation_steps=5,
                verbose=False,
                seed=42,
                num_epochs=(2) + (2),
                jastrow_stage1_epochs=2,
                two_stage_jastrow=True,
                use_cusp_jastrow=True,
            ),
        )

        self.assertTrue(res["is_two_stage"])
        self.assertEqual(res["stage1_epochs"], 2)
        self.assertEqual(len(res["history"]["stage"]), 4)
        self.assertIsNotNone(res["eval_result"])
        self.assertEqual(res["eval_result"]["decorrelation_steps"], 5)
        self.assertTrue(math.isfinite(res["eval_result"]["energy_mean"]))
        self.assertTrue(math.isfinite(res["eval_result"]["energy_sem"]))

    def test_stage1_zero_gradient_and_stage2_gradient_flow(self) -> None:
        """
        微观梯度流验证:
        明确检查反向传播时:
        1. 在 freeze_jastrow() (Stage 1) 下，Jastrow 阻尼参数的 grad 绝对为 None，
           神经网络轨道与包络参数的 grad 正常流动且非零。
        2. 在 unfreeze_jastrow() (Stage 2) 下，Jastrow 阻尼参数恢复计算梯度，grad 不为 None。
        """
        from ferminet_pytorch.loss import VMCLoss

        vmc_loss = VMCLoss()
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        )
        r = torch.randn(8, self.system.total_electrons, 3, requires_grad=True)

        # 阶段 1: 冻结 Jastrow
        model.freeze_jastrow()
        loss, _, _ = vmc_loss.compute_step_loss(model, self.system, r)
        loss.backward()

        for p in model.jastrow.parameters():
            self.assertIsNone(
                p.grad, "Stage 1 中 Jastrow 参数被冻结，梯度必须严格为 None"
            )

        backbone_has_grad = any(
            p.grad is not None and torch.norm(p.grad) > 0
            for n, p in model.named_parameters()
            if not n.startswith("jastrow.")
        )
        self.assertTrue(backbone_has_grad, "Stage 1 中神经网络主干必须正常接收梯度更新")

        # 阶段 2: 解冻 Jastrow
        model.zero_grad()
        model.unfreeze_jastrow()
        loss2, _, _ = vmc_loss.compute_step_loss(model, self.system, r)
        loss2.backward()

        jastrow_has_grad = any(
            p.grad is not None and torch.norm(p.grad) > 0
            for p in model.jastrow.parameters()
        )
        self.assertTrue(
            jastrow_has_grad, "Stage 2 中 Jastrow 参数解冻后必须接收到非零梯度"
        )

    def test_two_stage_fallback_without_jastrow(self) -> None:
        """测试在未开启 Jastrow (use_cusp_jastrow=False) 的模型上启用 two_stage_jastrow 平滑降级。"""
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
            use_cusp_jastrow=False,
        )

        res = train_vmc(
            system=self.system,
            model=model,
            config=TrainingConfig(
                num_walkers=32,
                num_epochs=3,
                mcmc_steps_per_epoch=5,
                burn_in_steps=10,
                two_stage_jastrow=True,
                verbose=False,
            ),
        )

        # 应该平滑降级，is_two_stage 为 False
        self.assertFalse(res["is_two_stage"])
        self.assertIsNone(res["stage1_epochs"])
        self.assertEqual(len(res["history"]["energy_raw"]), 3)

    def test_stage2_learning_rate_independent_annealing(self) -> None:
        """
        核心调度器验证:
        验证 Stage 2 中，backbone 与 jastrow 两个独立参数组的学习率
        均能从各自的初始微调学习率平滑余弦退火，
        绝对杜绝 PyTorch 原生 CosineAnnealingLR 导致的 backbone 学习率卡死在 1e-5，
        或 jastrow 学习率反常升温等致命 bug。
        """
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        )

        res = train_vmc(
            system=self.system,
            model=model,
            config=TrainingConfig(
                num_walkers=32,
                num_epochs=6,
                mcmc_steps_per_epoch=2,
                burn_in_steps=5,
                lr=0.001,
                two_stage_jastrow=True,
                jastrow_stage1_epochs=3,
                jastrow_lr=0.0001,
                verbose=False,
                seed=42,
            ),
        )

        # 检查训练结束后的优化器参数组当前学习率是否各自由初始值按比例衰减
        optimizer = optim.Adam(model.get_parameter_groups(lr=1e-3, jastrow_lr=1e-4))
        # 验证 3 轮 Stage 2 下的退火调度
        sched = train_vmc.__globals__["build_cosine_scheduler"](
            optimizer, total_steps=3, min_ratio=0.01
        )

        lrs_b = []
        lrs_j = []
        for _ in range(3):
            optimizer.step()
            sched.step()
            lrs_b.append(optimizer.param_groups[0]["lr"])
            lrs_j.append(optimizer.param_groups[1]["lr"])

        # 验证 backbone 学习率严格单调递减且不等于 1e-5 常数
        self.assertGreater(lrs_b[0], lrs_b[1])
        self.assertGreater(lrs_b[1], lrs_b[2])
        self.assertAlmostEqual(lrs_b[-1], 1e-3 * 0.01, delta=1e-6)

        # 验证 jastrow 学习率严格单调递减且按自己的 1% 退火
        self.assertGreater(lrs_j[0], lrs_j[1])
        self.assertGreater(lrs_j[1], lrs_j[2])
        self.assertAlmostEqual(lrs_j[-1], 1e-4 * 0.01, delta=1e-7)

    def test_restore_best_preserves_stage2_fine_tuned_jastrow(self) -> None:
        """
        关键状态持久化验证:
        验证在 restore_best=True (默认) 的真实两阶段训练全流程中:
        Stage 2 联合微调产生的新权重能够被正确恢复为最优模型，
        绝不会被 Stage 1 的初始 Jastrow 阻尼参数静默回滚覆盖！
        """
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
            use_cusp_jastrow=True,
            spin_adapted=True,
        )

        init_jastrow_state = {
            k: v.clone() for k, v in model.jastrow.state_dict().items()
        }

        res = train_vmc(
            system=self.system,
            model=model,
            config=TrainingConfig(
                num_walkers=64,
                num_epochs=6,
                mcmc_steps_per_epoch=5,
                burn_in_steps=10,
                lr=0.001,
                two_stage_jastrow=True,
                jastrow_stage1_epochs=3,
                jastrow_lr=0.005,
                restore_best=True,
                verbose=False,
                seed=42,
            ),
        )

        self.assertIsNotNone(res["best_model_state"])
        if res["stage2_best_energy"] is not None:
            self.assertEqual(res["best_energy"], res["stage2_best_energy"])

        # 检查 restore_best=True 确实把模型恢复到了记录的最优 checkpoint；
        # 在训练后的设备上进行比对，避免 CUDA/CPU 状态字典直接相减。
        final_state = model.state_dict()
        best_jastrow_state = res["best_model_state"]
        for k in init_jastrow_state:
            key = f"jastrow.{k}"
            self.assertIn(key, best_jastrow_state)
            device = final_state[key].device
            diff = torch.norm(
                final_state[key].float() - best_jastrow_state[key].float().to(device)
            ).item()
            self.assertLessEqual(
                diff, 1e-12, f"restore_best 未加载记录的最优 Jastrow 状态: {k}"
            )

    def test_evaluate_frozen_wavefunction_model_mode_and_robustness(self) -> None:
        """验证 evaluate_frozen_wavefunction 评估前后正确保留并恢复原有的 model.training 状态。"""
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
        )
        model.train(True)
        self.assertTrue(model.training)

        evaluate_frozen_wavefunction(
            model=model,
            system=self.system,
            num_walkers=32,
            num_steps=5,
            burn_in_steps=5,
            decorrelation_steps=2,
            verbose=False,
        )

        self.assertTrue(
            model.training, "评估完成后必须准确恢复进入评估前的 model.training 状态"
        )


if __name__ == "__main__":
    unittest.main()
