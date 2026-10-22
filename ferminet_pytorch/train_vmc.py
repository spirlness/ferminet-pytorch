"""Command-line entry point; optimization is implemented in vmc_driver."""

import argparse
from .systems import build_h2_system, build_lih_system
from .vmc_driver import TrainingConfig, train_vmc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FermiNet 变分量子蒙特卡洛 (VMC) 教学训练"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="h2",
        choices=["h2", "lih"],
        help="目标体系 (h2 或 lih)",
    )
    parser.add_argument(
        "--epochs",
        dest="num_epochs",
        type=int,
        default=50,
        help="训练迭代步数 (默认: 50)",
    )
    parser.add_argument(
        "--walkers",
        dest="num_walkers",
        type=int,
        default=512,
        help="MCMC Walker 数量 (默认: 512)",
    )
    parser.add_argument(
        "--mcmc_steps",
        dest="mcmc_steps_per_epoch",
        type=int,
        default=10,
        help="每步训练采样的 MCMC 步数 (默认: 10)",
    )
    parser.add_argument(
        "--burn_in",
        dest="burn_in_steps",
        type=int,
        default=50,
        help="预热步数 (默认: 50)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="优化器学习率 (默认: 1e-3)"
    )
    parser.add_argument(
        "--step_size", type=float, default=0.5, help="MCMC 初始移动步长 (默认: 0.5)"
    )
    parser.add_argument(
        "--move_mode",
        type=str,
        default="all",
        choices=["all", "one_by_one"],
        help="MCMC 移动模式",
    )
    parser.add_argument(
        "--clip_factor",
        type=float,
        default=5.0,
        help="局域能量 MAD 截断倍数 (默认: 5.0)",
    )
    parser.add_argument(
        "--clip_method",
        type=str,
        default="tanh",
        choices=["tanh", "hard", "none"],
        help="局域能量截断方法",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="梯度范数裁剪阈值 (默认: 1.0)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="运行设备 (cuda 或 cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机数种子 (默认: 42)")
    parser.add_argument(
        "--cusp_jastrow",
        dest="use_cusp_jastrow",
        action="store_true",
        help="启用显式 Kato 尖峰 Jastrow 因子",
    )
    parser.add_argument(
        "--two_stage_jastrow",
        action="store_true",
        help="启用两阶段 Jastrow 训练策略 (先冻结预训练后解冻微调)",
    )
    parser.add_argument(
        "--jastrow_stage1_epochs",
        type=int,
        default=None,
        help="两阶段中 Stage 1 冻结 Jastrow 的迭代轮数",
    )
    parser.add_argument(
        "--jastrow_lr",
        type=float,
        default=None,
        help="Stage 2 中 Jastrow 因子的独立微调学习率",
    )
    parser.add_argument(
        "--eval_after_train",
        action="store_true",
        help="训练完成后立即执行冷冻波函数独立统计评估",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="独立统计评估的采样记录步数 (默认: 50)",
    )
    parser.add_argument(
        "--eval_decorrelation_steps",
        type=int,
        default=10,
        help="评估采样前执行的 MCMC 弛豫去相关步数 (默认: 10)",
    )
    parser.add_argument(
        "--spin_adapted", action="store_true", help="启用自旋纯态投影网络"
    )
    parser.add_argument(
        "--envelope_type",
        type=str,
        default=None,
        choices=["gaussian", "exponential"],
        help="核包络类型",
    )
    parser.add_argument(
        "--smooth_cuspless",
        action="store_true",
        default=None,
        help="强制使用平滑无尖峰几何特征",
    )
    parser.add_argument(
        "--target_acceptance",
        type=float,
        default=None,
        help="MCMC 目标接受率 (全电子默认 0.234, 逐电子默认 0.50)",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="模型检查点保存路径 (.pt)"
    )
    parser.add_argument(
        "--no_restore_best",
        dest="restore_best",
        action="store_false",
        help="禁用训练结束时自动恢复历史最优模型状态",
    )
    parser.add_argument(
        "--compile",
        dest="compile_sampler",
        action="store_true",
        help="启用 torch.compile 对 MCMC 采样波函数进行内核融合编译加速",
    )
    parser.add_argument(
        "--interaction_type", choices=["ferminet", "psiformer"], default="ferminet"
    )
    return parser.parse_args()


def main():
    values = vars(parse_args())
    system_name = values.pop("system")
    values["use_cusp_jastrow"] = (
        values["use_cusp_jastrow"] or values["two_stage_jastrow"]
    )
    system = build_lih_system() if system_name == "lih" else build_h2_system()
    train_vmc(system=system, config=TrainingConfig(**values))


if __name__ == "__main__":
    main()
