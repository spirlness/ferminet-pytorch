"""
CLI 守卫单元测试 (D2: Hutchinson 禁用于对外报告能量)
======================================================
覆盖:
1. evaluate_trained_model 拒绝 'hutchinson' Laplacian 方法 (显式 ValueError)
2. train_vmc 在 evaluation_laplacian_method='hutchinson' 时快速失败
3. CLI --evaluation_laplacian_method 不再提供 'hutchinson' 选项
"""

import os
import sys
import unittest

import torch

# 确保包能够被直接运行和识别
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.train_vmc import (
    build_h2_system,
    evaluate_trained_model,
    parse_args,
    train_vmc,
)


class TestEvaluationHutchinsonGuard(unittest.TestCase):
    """对外报告能量不得来自 Hutchinson 随机估计器的守卫测试。"""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.system = build_h2_system(bond_length=1.4)
        self.model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=self.system.n_up,
            n_down=self.system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_nuclear_charges=True,
        )
        self.sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=8,
            step_size=0.5,
            move_mode="all",
        )

    def test_evaluate_rejects_hutchinson(self) -> None:
        """复评路径显式拒绝 hutchinson，且错误信息指明替代方法。"""
        with self.assertRaisesRegex(ValueError, "hutchinson"):
            evaluate_trained_model(
                model=self.model,
                system=self.system,
                sampler=self.sampler,
                evaluation_steps=2,
                evaluation_mcmc_steps=1,
                evaluation_burn_in_steps=1,
                laplacian_method="hutchinson",
            )

    def test_evaluate_accepts_exact_jvp(self) -> None:
        """守卫只针对 hutchinson；确定性方法 (exact_jvp) 仍然可用。"""
        stats = evaluate_trained_model(
            model=self.model,
            system=self.system,
            sampler=self.sampler,
            evaluation_steps=2,
            evaluation_mcmc_steps=1,
            evaluation_burn_in_steps=1,
            laplacian_method="exact_jvp",
        )
        self.assertEqual(stats["laplacian_method"], "exact_jvp")

    def test_train_vmc_rejects_hutchinson_evaluation(self) -> None:
        """端到端 API: 复评请求 hutchinson 时训练循环快速失败 (防御 evaluate 入口被绕过)。"""
        with self.assertRaisesRegex(ValueError, "hutchinson"):
            train_vmc(
                system=self.system,
                model=self.model,
                num_walkers=8,
                num_epochs=1,
                mcmc_steps_per_epoch=1,
                burn_in_steps=1,
                evaluation_steps=2,
                evaluation_mcmc_steps=1,
                evaluation_burn_in_steps=1,
                evaluation_laplacian_method="hutchinson",
                verbose=False,
            )

    def test_cli_evaluation_choices_exclude_hutchinson(self) -> None:
        """CLI 复评 Laplacian 选项不得包含 hutchinson (argparse 直接拒绝)。"""
        original_argv = sys.argv
        sys.argv = ["train_vmc.py", "--evaluation_laplacian_method", "hutchinson"]
        try:
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = original_argv

    def test_cli_exact_evaluation_method_parses(self) -> None:
        """合法的确定性选项 (exact / exact_jvp / jvp) 仍可正常解析。"""
        for method in ("exact", "exact_jvp", "jvp"):
            original_argv = sys.argv
            sys.argv = ["train_vmc.py", "--evaluation_laplacian_method", method]
            try:
                args = parse_args()
            finally:
                sys.argv = original_argv
            self.assertEqual(args.evaluation_laplacian_method, method)


if __name__ == "__main__":
    unittest.main()
