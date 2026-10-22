"""
批量并行 Metropolis-Hastings (MCMC) 采样器
==========================================
依据玻恩概率密度 p(r) ∝ |Psi(r)|² 采样电子空间分布构型，
支持全电子一体化移动、逐电子移动以及目标接受率自适应步长调节。
"""

import math
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn

try:
    from .hamiltonian import MolecularSystem
except (ImportError, ValueError):
    from hamiltonian import MolecularSystem


class MCMCSampler:
    """
    批量并行 Metropolis-Hastings MCMC 采样器。

    核心机制：
    1. 构型提议：高斯随机游走 r' = r + sigma * N(0, I)
    2. 接受准则：alpha = min(1, |Psi(r')|² / |Psi(r)|²) = min(1, exp(2 * (ln|Psi(r')| - ln|Psi(r)|)))
    3. 自适应步长：根据当前接受率动态缩放提议步长 sigma，稳定在目标接受率 (通常 50%)
    4. 对数缓存：缓存当前构型的 ln|Psi|，提议时仅需一次前向计算
    """

    def __init__(
        self,
        model: nn.Module,
        system: MolecularSystem,
        num_walkers: int = 1024,
        step_size: float = 0.5,
        target_acceptance: Optional[float] = None,
        adapt_step_size: bool = True,
        adapt_rate: float = 0.05,
        move_mode: str = "all",
        init_width: float = 0.8,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        initial_walkers: Optional[torch.Tensor] = None,
        forward_log_psi_fn: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        """
        初始化 MCMC 采样器。

        Args:
            model (nn.Module): FermiNet 波函数网络，具备 forward_log_psi 接口。
            system (MolecularSystem): 分子体系定义 (包含核坐标与电子配置)。
            num_walkers (int): 并行 MCMC Walker (构型) 数量。
            step_size (float): 高斯提议分布的标准差 sigma。
            target_acceptance (Optional[float]): 目标接受率。若为 None，全电子模式默认为理论最优值 0.234，逐电子模式默认为 0.50。
            adapt_step_size (bool): 是否动态自适应调节 step_size。
            adapt_rate (float): 步长自适应调节的学习率/步长因子。
            move_mode (str): 移动模式，可选 'all' (全电子) 或 'one_by_one' (逐电子)。
            init_width (float): 初始构型绕核高斯分布的标准差。
            device (Optional[torch.device]): 运行设备。若为 None 则跟随 system.nuclear_coords.device。
            dtype (torch.dtype): 数据精度，默认 torch.float32。
            initial_walkers (Optional[torch.Tensor]): 可选的用户指定初始构型，Shape: `(num_walkers, N, 3)`。
            forward_log_psi_fn (Optional[Callable]): 可选的波函数对数模前向计算函数 (如编译加速后的函数)。默认使用 model.forward_log_psi。
        """
        self.model = model
        self.forward_log_psi_fn = (
            forward_log_psi_fn if forward_log_psi_fn is not None else self.model.forward_log_psi
        )
        self.system = system
        self.num_walkers = num_walkers
        self.step_size = float(step_size)
        assert move_mode in ("all", "one_by_one"), f"不支持的移动模式: {move_mode}"
        self.move_mode = move_mode
        if target_acceptance is None:
            self.target_acceptance = 0.234 if self.move_mode == "all" else 0.50
        else:
            self.target_acceptance = float(target_acceptance)
        self.adapt_step_size = adapt_step_size
        self.adapt_rate = float(adapt_rate)
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

        # 统计指标跟踪 (宏观步数与精确微观移动分离)
        self.total_steps = 0
        self.accepted_steps = 0
        self.proposed_moves: int = 0
        self.accepted_moves: int = 0
        self.last_acceptance_rate = 0.0

    @property
    def micro_acceptance_rate(self) -> float:
        """真实累计微观移动接受率 (无舍入失真)。"""
        return float(self.accepted_moves) / max(self.proposed_moves, 1)

    @property
    def acceptance_rate(self) -> float:
        """向后兼容属性，反映真实累计微观移动接受率。"""
        return self.micro_acceptance_rate

    @property
    def cumulative_acceptance_rate(self) -> float:
        """真实累计微观移动接受率。"""
        return self.micro_acceptance_rate

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
            log_abs_psi, _ = self.forward_log_psi_fn(self.walkers)
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

    @staticmethod
    def _metropolis_accept(
        current_log_psi: torch.Tensor,
        prop_log_psi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        无分支安全 Metropolis 接受判定核函数 (DRY 原则重构)。

        Args:
            current_log_psi: 当前构型的对数波函数值, shape (W,)
            prop_log_psi: 建议移动构型的对数波函数值, shape (W,)

        Returns:
            Tuple[accepted, updated_log_psi]:
                accepted: 布尔张量, shape (W,)
                updated_log_psi: 更新后的对数波函数值, shape (W,)
        """
        valid = torch.isfinite(prop_log_psi) & ~torch.isnan(prop_log_psi)
        cur_is_node = torch.isneginf(current_log_psi)
        safe_current = torch.where(
            torch.isfinite(current_log_psi), current_log_psi, torch.zeros_like(current_log_psi)
        )
        safe_prop = torch.where(valid, prop_log_psi, torch.zeros_like(prop_log_psi))
        log_ratio = torch.where(
            cur_is_node & valid,
            torch.zeros_like(prop_log_psi),
            2.0 * (safe_prop - safe_current),
        )
        ratio = torch.where(valid, torch.exp(torch.clamp(log_ratio, max=0.0)), torch.zeros_like(prop_log_psi))
        accepted = valid & (torch.rand_like(ratio) < ratio)
        updated_log_psi = torch.where(accepted, prop_log_psi, current_log_psi)
        return accepted, updated_log_psi

    def _step_all_electrons(self) -> float:
        """全电子一体化移动 (All-electron moves)。"""
        noise = torch.randn_like(self.walkers) * self.step_size
        prop_walkers = self.walkers + noise

        with torch.no_grad():
            log_prop, _ = self.forward_log_psi_fn(prop_walkers)
            accepted, self.current_log_abs_psi = self._metropolis_accept(
                self.current_log_abs_psi, log_prop
            )

            acc_mask = accepted.unsqueeze(-1).unsqueeze(-1)
            self.walkers = torch.where(acc_mask, prop_walkers, self.walkers)

            acc_rate = accepted.float().mean().item()
            accepted_count = int(accepted.sum().item())
            self.accepted_steps += accepted_count

            # 微观移动统计 (每个全电子移动包含 total_electrons 个微观坐标位移)
            total_electrons = self.system.total_electrons
            self.proposed_moves += self.num_walkers * total_electrons
            self.accepted_moves += accepted_count * total_electrons

        return acc_rate

    def _step_one_electron(self) -> float:
        """逐电子轮换移动 (One-by-one moves, 优化单缓冲区复用)。"""
        total_electrons = max(self.system.total_electrons, 1)
        total_accepted = 0.0
        total_accepted_count = 0

        with torch.no_grad():
            # 仅在步长开始时分配一次缓冲区，避免循环内 N 次全量克隆 (RED-17 修复)
            prop_walkers = self.walkers.clone()
            for i in range(self.system.total_electrons):
                noise_i = torch.randn(
                    self.num_walkers, 3, device=self.device, dtype=self.dtype
                ) * self.step_size

                prop_walkers[:, i, :] = self.walkers[:, i, :] + noise_i
                log_prop, _ = self.forward_log_psi_fn(prop_walkers)
                accepted, self.current_log_abs_psi = self._metropolis_accept(
                    self.current_log_abs_psi, log_prop
                )

                acc_mask_i = accepted.unsqueeze(-1)
                self.walkers[:, i, :] = torch.where(
                    acc_mask_i, prop_walkers[:, i, :], self.walkers[:, i, :]
                )
                # 恢复未接受的坐标切片，保持 prop_walkers 与 self.walkers 严格对齐
                prop_walkers[:, i, :] = self.walkers[:, i, :]

                total_accepted += accepted.float().mean().item()
                total_accepted_count += int(accepted.sum().item())

        acc_rate = total_accepted / total_electrons
        self.accepted_steps += int(round(total_accepted_count / total_electrons))

        # 微观移动统计 (共计尝试 N * num_walkers 次微观移动，实际接受 total_accepted_count 次)
        self.proposed_moves += self.num_walkers * self.system.total_electrons
        self.accepted_moves += total_accepted_count
        return acc_rate


    def step(self) -> float:
        """执行单步 MCMC 移动，并动态自适应更新步长。"""
        acc_rate = self._step_all_electrons() if self.move_mode == "all" else self._step_one_electron()
        self.last_acceptance_rate = acc_rate
        self.total_steps += 1

        if self.adapt_step_size:
            diff = acc_rate - self.target_acceptance
            factor = max(-0.5, min(0.5, self.adapt_rate * diff))
            self.step_size = max(0.001, min(10.0, self.step_size * math.exp(factor)))

        return acc_rate

    def sample(self, num_steps: int = 10, adapt: bool = False) -> torch.Tensor:
        """执行多步连续 MCMC 迭代并返回最新构型副本。

        默认在正式抽样期间冻结步长调整，保障马尔可夫链时齐性与细致平衡。
        """
        prev_adapt = self.adapt_step_size
        self.adapt_step_size = adapt
        try:
            for _ in range(num_steps):
                self.step()
        finally:
            self.adapt_step_size = prev_adapt
        return self.walkers.clone().detach()

    def burn_in(self, num_steps: int = 100) -> float:
        """预热 (Burn-in) 消除随机初始偏差，返回平均接受率。"""
        acc_list = [self.step() for _ in range(num_steps)]
        return float(sum(acc_list) / max(len(acc_list), 1))


# 别名兼容
MetropolisHastings = MCMCSampler

