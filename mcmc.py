"""
批量并行 GPU 马尔可夫链蒙特卡洛 (MCMC) 采样器
==============================================
基于 Metropolis-Hastings (MH) 拒绝采样算法，实现全向量化批量电子构型采样。
保证平衡态电子空间分布严格符合量子力学玻恩概率密度:
    p(r) = |Psi(r)|^2 / int |Psi(r')|^2 dr'
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    from .hamiltonian import MolecularSystem
except (ImportError, ValueError):
    from hamiltonian import MolecularSystem


class MCMCSampler:
    """
    批量 GPU/CPU 并行 Metropolis-Hastings MCMC 采样器。

    特性：
    1. 批量并行：支持成百上千个 Walker 在 GPU/CPU 上并行游走。
    2. 双步进模式：
       - 'all': 全电子一体化移动 (All-electron moves)，速度快。
       - 'one_by_one': 逐电子轮换移动 (One-electron moves)，大体系接受率高。
    3. 自适应步长：根据当前接受率动态调节提议方差，使平均接受率稳定在目标值 (默认 50%)。
    4. 显存安全：采样前向使用 no_grad 并与优化计算图完全解耦，彻底防范 CUDAGraphs 显存覆写。
    5. 波函数缓存：每次接受后缓存对数波函数值，下一次提议仅需计算新点波函数，节省一半前向开销。
    """

    def __init__(
        self,
        model: nn.Module,
        system: MolecularSystem,
        num_walkers: int = 1024,
        step_size: float = 0.5,
        target_acceptance: float = 0.5,
        adapt_step_size: bool = True,
        adapt_rate: float = 0.05,
        move_mode: str = "all",
        init_width: float = 0.8,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        initial_walkers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        初始化 MCMC 采样器。

        Args:
            model (nn.Module): FermiNet 波函数网络，具备 forward_log_psi 接口。
            system (MolecularSystem): 分子体系定义 (包含核坐标与电子配置)。
            num_walkers (int): 并行 MCMC Walker (构型) 数量。
            step_size (float): 高斯提议分布的标准差 sigma。
            target_acceptance (float): 目标接受率 (通常在 0.4 ~ 0.6 之间，默认 0.5)。
            adapt_step_size (bool): 是否动态自适应调节 step_size。
            adapt_rate (float): 步长自适应调节的学习率/步长因子。
            move_mode (str): 移动模式，可选 'all' (全电子) 或 'one_by_one' (逐电子)。
            init_width (float): 初始构型绕核高斯分布的标准差。
            device (Optional[torch.device]): 运行设备。若为 None 则跟随 system.nuclear_coords.device。
            dtype (torch.dtype): 数据精度，默认 torch.float32。
            initial_walkers (Optional[torch.Tensor]): 可选的用户指定初始构型，Shape: `(num_walkers, N, 3)`。
        """
        self.model = model
        self.system = system
        self.num_walkers = num_walkers
        self.step_size = float(step_size)
        self.target_acceptance = float(target_acceptance)
        self.adapt_step_size = adapt_step_size
        self.adapt_rate = float(adapt_rate)
        assert move_mode in ("all", "one_by_one"), f"不支持的移动模式: {move_mode}"
        self.move_mode = move_mode
        self.init_width = float(init_width)

        if device is None:
            self.device = self.system.nuclear_coords.device
        else:
            self.device = torch.device(device)
        self.dtype = dtype

        # 初始化或使用用户传入的 Walker 构型: Shape (num_walkers, total_electrons, 3)
        if initial_walkers is not None:
            assert initial_walkers.shape == (
                self.num_walkers,
                self.system.total_electrons,
                3,
            ), f"初始 walkers 形状不符合要求: {initial_walkers.shape}"
            self.walkers = initial_walkers.to(device=self.device, dtype=self.dtype).clone().detach()
        else:
            self.walkers = self.init_walkers(
                num_walkers=self.num_walkers,
                system=self.system,
                init_width=self.init_width,
                device=self.device,
                dtype=self.dtype,
            )

        # 缓存当前构型的对数波函数值 log|Psi(r)|: Shape (num_walkers,)
        self.current_log_abs_psi: torch.Tensor = torch.zeros(
            self.num_walkers, device=self.device, dtype=self.dtype
        )
        self.recompute_log_psi()

        # 统计指标跟踪
        self.total_steps = 0
        self.accepted_steps = 0
        self.last_acceptance_rate = 0.0

    @staticmethod
    def init_walkers(
        num_walkers: int,
        system: MolecularSystem,
        init_width: float = 0.8,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        为分子体系生成符合物理直觉的初始电子空间构型。
        将总电子随机分配并围绕各个原子核坐标高斯弥散，加速 MCMC 弛豫收敛。

        Args:
            num_walkers (int): Walker 数量。
            system (MolecularSystem): 分子体系。
            init_width (float): 电子围绕原子核分布的高斯标准差。
            device (Optional[torch.device]): 运行设备。
            dtype (torch.dtype): 浮点精度。

        Returns:
            torch.Tensor: 初始电子三维坐标，Shape: `(num_walkers, total_electrons, 3)`。
        """
        if device is None:
            device = system.nuclear_coords.device

        total_electrons = system.total_electrons
        num_nuclei = system.num_nuclei
        coords = system.nuclear_coords.to(device=device, dtype=dtype)  # (M, 3)

        # 随机分配电子到原子核索引: (num_walkers, total_electrons)
        nucleus_indices = torch.randint(
            low=0,
            high=num_nuclei,
            size=(num_walkers, total_electrons),
            device=device,
        )
        # 获取对应的原子核中心坐标: (num_walkers, total_electrons, 3)
        centers = coords[nucleus_indices]

        # 叠加高斯位移扰动
        noise = torch.randn(
            num_walkers, total_electrons, 3, device=device, dtype=dtype
        ) * init_width

        walkers = centers + noise
        return walkers

    def recompute_log_psi(self) -> torch.Tensor:
        """
        当模型参数经过优化器更新后，重新计算并刷新当前 walkers 的 log|Psi(r)| 缓存。
        严格处于 no_grad 模式下，避免保留过时计算图或产生梯度。

        Returns:
            torch.Tensor: 重新计算后的当前构型对数波函数值 log|Psi(r)|。
        """
        with torch.no_grad():
            log_abs_psi, _ = self.model.forward_log_psi(self.walkers)
            invalid = torch.isnan(log_abs_psi) | torch.isposinf(log_abs_psi)
            if invalid.any():
                raise FloatingPointError(
                    f"current walker log|Psi| contains {int(invalid.sum())} NaN/+inf values; "
                    "cache was not updated"
                )
            # -inf is a valid nodal value.  Retaining it lets a finite proposal
            # leave the node with unit Metropolis acceptance probability.
            self.current_log_abs_psi = log_abs_psi.clone().detach()
        return self.current_log_abs_psi

    def _step_all_electrons(self) -> float:
        """
        全电子一体化移动 (All-electron move)。
        一次性对每个 walker 的所有电子提议高斯位移。
        """
        # 提议新坐标 r' = r + sigma * eta
        noise = torch.randn_like(self.walkers) * self.step_size
        proposed_walkers = self.walkers + noise

        with torch.no_grad():
            log_abs_prop, _ = self.model.forward_log_psi(proposed_walkers)
            # 严格防范非有限波函数：若出现 NaN 或 Inf，直接判定为非法提议并拒绝 (严格符合 Born 概率测度)
            prop_is_valid = torch.isfinite(log_abs_prop)
            safe_log_prop = torch.nan_to_num(log_abs_prop, nan=-1e20, neginf=-1e20, posinf=-1e20)

            # 接受概率对数: log(alpha) = 2.0 * (log|Psi(r')| - log|Psi(r)|)
            log_ratio = 2.0 * (safe_log_prop - self.current_log_abs_psi)
            ratio = torch.exp(torch.clamp(log_ratio, max=0.0))  # 截断上限为 1.0

            # 抽取均匀分布随机数 u ~ U(0, 1)
            uniform_rand = torch.rand_like(ratio)
            accepted = (uniform_rand < ratio) & prop_is_valid  # (num_walkers,)

            # 更新接受的 walkers 与对数幅值
            accepted_mask = accepted.unsqueeze(-1).unsqueeze(-1)  # (num_walkers, 1, 1)
            self.walkers = torch.where(accepted_mask, proposed_walkers, self.walkers)
            self.current_log_abs_psi = torch.where(accepted, safe_log_prop, self.current_log_abs_psi)

            acc_rate = accepted.float().mean().item()
            self.accepted_steps += int(accepted.sum().item())

        return acc_rate

    def _step_one_electron(self) -> float:
        """
        逐电子依次移动 (One-electron move)。
        对每个电子轮流提议位移并独立进行 MH 接受-拒绝判定，在大体系下维持高接受率。
        """
        total_electrons = max(self.system.total_electrons, 1)
        total_accepted = 0.0
        total_accepted_count = 0

        with torch.no_grad():
            for i in range(self.system.total_electrons):
                # 仅对第 i 个电子提议位移
                noise_i = torch.randn(
                    self.num_walkers, 3, device=self.device, dtype=self.dtype
                ) * self.step_size

                proposed_walkers = self.walkers.clone()
                proposed_walkers[:, i, :] = proposed_walkers[:, i, :] + noise_i

                log_abs_prop, _ = self.model.forward_log_psi(proposed_walkers)
                prop_is_valid = torch.isfinite(log_abs_prop)
                safe_log_prop = torch.nan_to_num(log_abs_prop, nan=-1e20, neginf=-1e20, posinf=-1e20)

                log_ratio = 2.0 * (safe_log_prop - self.current_log_abs_psi)
                ratio = torch.exp(torch.clamp(log_ratio, max=0.0))

                uniform_rand = torch.rand_like(ratio)
                accepted = (uniform_rand < ratio) & prop_is_valid  # (num_walkers,)

                accepted_mask_i = accepted.unsqueeze(-1)  # (num_walkers, 1)
                self.walkers[:, i, :] = torch.where(
                    accepted_mask_i, proposed_walkers[:, i, :], self.walkers[:, i, :]
                )
                self.current_log_abs_psi = torch.where(
                    accepted, safe_log_prop, self.current_log_abs_psi
                )

                total_accepted += accepted.float().mean().item()
                total_accepted_count += int(accepted.sum().item())

        acc_rate = total_accepted / total_electrons
        self.accepted_steps += int(round(total_accepted_count / total_electrons))
        return acc_rate

    def step(self) -> float:
        """
        执行单步 Metropolis-Hastings MCMC 采样移动。
        根据当前接受率与目标接受率动态调整 step_size。

        Returns:
            float: 本次步进的平均接受率 (0.0 ~ 1.0)。
        """
        if self.move_mode == "all":
            acc_rate = self._step_all_electrons()
        else:
            acc_rate = self._step_one_electron()

        self.last_acceptance_rate = acc_rate
        self.total_steps += 1

        # 自适应更新步长: 根据接受率偏离程度指数调整 (使用纯标量算术，避免 CPU-GPU 张量同步开销)
        if self.adapt_step_size:
            diff = acc_rate - self.target_acceptance
            # 步长因子限制在 [-0.5, 0.5] 避免步长剧烈跳跃
            factor = max(-0.5, min(0.5, self.adapt_rate * diff))
            new_step = self.step_size * math.exp(factor)
            # 限制步长边界在 [1e-3, 10.0]
            self.step_size = max(0.001, min(10.0, new_step))

        return acc_rate

    def sample(self, num_steps: int = 10) -> torch.Tensor:
        """
        进行多次连续 MCMC 迭代采样以实现充分去相关，并返回最新构型副本。

        Args:
            num_steps (int): 执行的 MCMC 步数 (默认 10 步)。

        Returns:
            torch.Tensor: 采样得到的电子空间构型副本，Shape: `(num_walkers, total_electrons, 3)`。
        """
        for _ in range(num_steps):
            self.step()
        return self.walkers.clone().detach()

    def burn_in(self, num_steps: int = 100) -> float:
        """
        执行预热 (Burn-in) 弛豫，消除随机初始构型的不良偏差，使 Walker 进入高概率波函数区域。

        Args:
            num_steps (int): 预热步数。

        Returns:
            float: 预热阶段的平均接受率。
        """
        acc_list = []
        for _ in range(num_steps):
            acc = self.step()
            acc_list.append(acc)
        return float(sum(acc_list) / max(len(acc_list), 1))
