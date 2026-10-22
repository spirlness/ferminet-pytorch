"""
ferminet_pytorch.vmc_driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~
面向 PyTorch 2.x 的现代变分量子蒙特卡洛 (VMC) 四层正交解耦驱动引擎。
包含四大正交支柱:
1. VariationalState: 变分波函数态与几何上下文生命周期管理
2. HamiltonianComposite: 复合哈密顿量物理算符 (动能 + 势能)
3. MetropolisSampler: 蒙特卡洛几何构型采样器
4. VMCDriver: 变分优化主循环驱动器 (学习率调度、两阶段解冻、统计指标分析)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .geometry import GeometricContext
from .hamiltonian import MolecularSystem, compute_local_energy, KineticMode
from .loss import VMCLoss, clip_local_energy
from .mcmc import MCMCSampler


class VariationalState:
    """
    变分态容器: 管理 Walker 构型分布、共享几何上下文以及波函数前向评估。
    """
    def __init__(
        self,
        model: nn.Module,
        system: MolecularSystem,
        num_walkers: int = 1024,
        init_walkers: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: Optional[float] = None,
    ) -> None:
        self.model = model
        self.system = system
        self.num_walkers = num_walkers
        self.eps = getattr(model, "eps", 1e-24) if eps is None else eps
        if hasattr(model, 'eps') and self.eps != model.eps:
            raise ValueError('VariationalState feature smoothing must match model.eps')

        # 确定设备与数据类型
        first_param = next(model.parameters(), None)
        self.device = device or (first_param.device if first_param is not None else torch.device("cpu"))
        self.dtype = dtype or (first_param.dtype if first_param is not None else torch.float32)

        # 初始化 Walker 构型 (B, N, 3)
        if init_walkers is not None:
            self.walkers = init_walkers.to(device=self.device, dtype=self.dtype).clone()
            self.num_walkers = self.walkers.shape[0]
        else:
            self.walkers = self._initialize_walkers()

        self._geom_ctx: Optional[GeometricContext] = None

    def _initialize_walkers(self) -> torch.Tensor:
        """围绕原子核高斯分布初始化电子构型。"""
        n_elec = self.system.total_electrons
        n_nuc = self.system.num_nuclei
        coords = self.system.nuclear_coords.to(device=self.device, dtype=self.dtype)

        # 随机分配电子到原子核邻近
        center_idx = torch.randint(0, n_nuc, (self.num_walkers, n_elec), device=self.device)
        centers = coords[center_idx]  # (B, N, 3)
        noise = torch.randn(self.num_walkers, n_elec, 3, device=self.device, dtype=self.dtype) * 0.5
        return centers + noise

    @property
    def current_walkers(self) -> torch.Tensor:
        return self.walkers

    def update_walkers(self, new_walkers: torch.Tensor) -> None:
        """更新 Walker 构型并重置几何上下文缓存。"""
        self.walkers = new_walkers
        self._geom_ctx = None

    def get_geometric_context(self) -> GeometricContext:
        """获取或构建当前构型的不可变几何上下文。"""
        if self._geom_ctx is None or self._geom_ctx.r.data_ptr() != self.walkers.data_ptr():
            self._geom_ctx = GeometricContext.compute(
                r=self.walkers,
                nuclear_coords=self.system.nuclear_coords,
                eps=self.eps,
            )
        return self._geom_ctx

    def evaluate_wavefunction(
        self,
        r: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向计算对数波函数 (log|Psi|, sign)。
        """
        eval_r = self.walkers if r is None else r
        return self.model.forward_log_psi(eval_r)


class HamiltonianComposite:
    """
    复合物理哈密顿量算符: 封装局域动能与静电势能求解。
    """
    def __init__(
        self,
        system: MolecularSystem,
        kinetic_mode: KineticMode = KineticMode.REVERSE,
        num_probes: int = 1,
    ) -> None:
        self.system = system
        self.kinetic_mode = kinetic_mode
        self.num_probes = num_probes

    def compute_local_energy(
        self,
        model: nn.Module,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        统一计算 E_L(r) = T_L(r) + V(r)。
        """
        return compute_local_energy(
            model=model,
            system=self.system,
            r=r,
            create_graph=create_graph,
            geom_ctx=geom_ctx,
            kinetic_mode=self.kinetic_mode,
            num_probes=self.num_probes,
        )


class MetropolisSampler:
    """
    纯几何采样器: 驱动 Metropolis-Hastings 链更新变分态。
    """
    def __init__(
        self,
        variational_state: VariationalState,
        step_size: float = 0.05,
        target_acceptance: float = 0.50,
        move_mode: str = "all",
    ) -> None:
        self.vstate = variational_state
        self.mcmc = MCMCSampler(
            model=self.vstate.model,
            system=self.vstate.system,
            num_walkers=self.vstate.num_walkers,
            step_size=step_size,
            target_acceptance=target_acceptance,
            move_mode=move_mode,
            initial_walkers=self.vstate.walkers,
            device=self.vstate.device,
            dtype=self.vstate.dtype,
        )


    @property
    def step_size(self) -> float:
        return self.mcmc.step_size

    def burn_in(self, num_steps: int = 100) -> float:
        rate = self.mcmc.burn_in(num_steps)
        self.vstate.update_walkers(self.mcmc.walkers)
        return rate

    def step(self, decorrelation_steps: int = 1) -> float:
        """执行指定去相关步数的 MCMC 采样。"""
        self.mcmc.sample(num_steps=decorrelation_steps, adapt=False)
        acc_rate = self.mcmc.last_acceptance_rate
        self.vstate.update_walkers(self.mcmc.walkers)
        return acc_rate


class VMCDriver:
    """
    变分量子蒙特卡洛 (VMC) 高层驱动器。
    协调 VariationalState、HamiltonianComposite 与 MetropolisSampler 完成端到端优化。
    """
    def __init__(
        self,
        variational_state: VariationalState,
        hamiltonian: HamiltonianComposite,
        sampler: MetropolisSampler,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        loss_fn: Optional[VMCLoss] = None,
        clip_factor: float = 5.0,
        clip_method: str = "tanh",
        decorrelation_steps: int = 10,
    ) -> None:
        self.vstate = variational_state
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or VMCLoss(clip_factor=clip_factor, clip_method=clip_method)
        self.decorrelation_steps = decorrelation_steps

        self.best_energy: float = float("inf")
        self.history: List[Dict[str, Any]] = []

    def step(self) -> Dict[str, Any]:
        """
        执行单步 VMC 迭代:
        1. 几何采样采样推进
        2. 共享几何上下文构建
        3. 独立坐标图计算动能/势能，再以参数图计算代理损失
        4. 变分代理损失反向传播与优化器更新
        """
        t0 = time.perf_counter()

        # 1. 采样推进
        acc_rate = self.sampler.step(decorrelation_steps=self.decorrelation_steps)

        # 2. 获取当前构型的常数几何上下文与独立求导坐标
        walkers = self.vstate.current_walkers
        geom_ctx_walkers = self.vstate.get_geometric_context()

        r_eval = walkers.clone().detach().requires_grad_(True)
        geom_ctx_eval = GeometricContext.compute(
            r=r_eval,
            nuclear_coords=self.vstate.system.nuclear_coords,
            eps=self.vstate.eps,
        )

        # 3. 局域能量计算 (动能 + 势能复用 geom_ctx_eval)
        e_local, t_local, v_local = self.hamiltonian.compute_local_energy(
            model=self.vstate.model,
            r=r_eval,
            geom_ctx=geom_ctx_eval,
            create_graph=False,
        )

        # 4. 参数梯度前向变分损失: 复用无梯度构型上下文 geom_ctx_walkers 计算变分代理损失
        log_abs_psi, _ = self.vstate.model.forward_log_psi(walkers, geom_ctx=geom_ctx_walkers)
        loss, stats = self.loss_fn.forward(log_abs_psi, e_local.detach())

        # 5. 反向更新
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.sampler.mcmc.recompute_log_psi()

        if self.scheduler is not None:
            self.scheduler.step()

        dt = time.perf_counter() - t0

        # 6. 整合物理度量
        metrics = {
            "epoch_time": dt,
            "loss": stats["loss"],
            "energy_mean": stats["energy_mean"],
            "energy_std": stats["energy_std"],
            "energy_median": stats["energy_median"],
            "kinetic_mean": float(t_local.mean().item()),
            "potential_mean": float(v_local.mean().item()),
            "mcmc_acceptance": acc_rate,
            "mcmc_step_size": self.sampler.step_size,
        }

        if stats["energy_mean"] < self.best_energy:
            self.best_energy = stats["energy_mean"]

        self.history.append(metrics)
        return metrics

    def run(
        self,
        num_epochs: int,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        端到端循环优化驱动。
        """
        for epoch in range(1, num_epochs + 1):
            metrics = self.step()
            if callback is not None:
                callback(epoch, metrics)
        return self.history
