"""
FermiNet 变分量子蒙特卡洛 (VMC) 综合可视化模块
=============================================
提供高质量、学术期刊级别 (Publication-ready) 的数据可视化工具库，涵盖：
1. 训练过程动态演化跟踪:
   - 能量收敛曲线 (原始能量 <E_raw>、截断能量 <E_clip>、理论参考基准线如 H2 -1.17447 Ha, LiH -8.0705 Ha)
   - 能量方差 Var(E) 与均值标准误 SEM 演化曲线 (直观展现零方差原理 Zero-Variance Principle)
   - MCMC 采样健康度跟踪 (接受率 Acc %、自适应步长 step size 随 epoch 演化)
   - 训练全景综合看板 (Training Dashboard)
2. 训练结果与物理状态解析:
   - 电子空间几率构型分布 (3D 空间散点与 2D 投影图，区分自旋向上与自旋向下电子，标注核位置)
   - 波函数截面与电子密度剖面 (沿核间轴 1D 对数波函数与几率密度、展示核处 Cusp 尖峰，以及 2D 成键截面热力图)
   - 局域能量分布直方图与统计偏度 (Local Energy Histogram、核密度估计、中位数/偏度/峰度信息卡片)
   - 物理状态综合看板 (Physical Results Dashboard)
3. 软件工程支持:
   - 严格在无 GUI 交互环境下安全执行 (强制切换 matplotlib 后端为 'Agg')
   - 支持高分辨率图片导出 (支持 PNG, PDF, SVG 等格式)
   - 支持与训练脚本 (train_vmc.py) 无缝一键集成
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

# 强制使用无 GUI 交互的 Agg 后端，确保在无头服务器或自动化脚本中绝无阻塞崩溃风险
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # 显式导入注册 3D 投影支持

try:
    from .hamiltonian import MolecularSystem
except (ImportError, ValueError):
    from hamiltonian import MolecularSystem


# ==============================================================================
# 1. 学术级绘图全局配色与样式设定 (Nature / Science 风格)
# ==============================================================================

# 经典学术配色方案
COLOR_PALETTE = {
    "raw_energy": "#1f77b4",       # 原始能量 (Steel Blue)
    "clipped_energy": "#2ca02c",   # 截断能量 (Forest Green)
    "ref_energy": "#d62728",       # 理论参考能 (Crimson Red)
    "sem_band": "#aec7e8",         # SEM 误差阴影带 (Light Blue)
    "variance": "#9467bd",         # 能量方差 (Muted Purple)
    "sem": "#ff7f0e",              # 均值标准误 (Amber Orange)
    "acceptance": "#17becf",       # MCMC 接受率 (Cyan)
    "step_size": "#8c564b",        # MCMC 步长 (Sienna Brown)
    "target_line": "#7f7f7f",      # 目标基准线 (Gray)
    "spin_up": "#2980b9",          # 自旋向上电子 alpha (Sky Blue)
    "spin_down": "#e74c3c",        # 自旋向下电子 beta (Coral Red)
    "nucleus": "#2c3e50",          # 原子核标记 (Dark Charcoal)
    "density_map": "viridis",      # 2D 密度热力图色卡
    "wavefunction": "#34495e",     # 波函数曲线 (Navy)
    "cusp_marker": "#e67e22",      # 核 Cusp 尖峰标线 (Orange)
}

# 常见分子体系的理论精确基态能量基准 (单位: Hartree / a.u.)
REFERENCE_ENERGIES: Dict[str, float] = {
    "h": -0.50000,          # 氢原子 (精确非相对论能量)
    "h2": -1.17447,         # 氢分子 (R=1.4 a.u. FCI / 实验精确解)
    "he": -2.90372,         # 氦原子 (Pekeris 精确解)
    "li": -7.47806,         # 锂原子
    "lih": -8.07050,        # 氢化锂分子 (R=3.015 a.u. FCI 精确基态能)
    "be": -14.66736,        # 铍原子
}


@dataclass(frozen=True)
class ReferenceEnergy:
    """与特定电荷、自旋和核几何绑定的参考能量记录。"""

    key: str
    label: str
    energy: float
    nuclear_charges: Tuple[float, ...]
    n_up: int
    n_down: int
    pair_signature: Tuple[Tuple[float, float, float], ...]
    method: str
    source: str


REFERENCE_SYSTEMS: Dict[str, ReferenceEnergy] = {
    "h": ReferenceEnergy("h", "H", -0.50000, (1.0,), 1, 0, (), "exact", "Hydrogenic solution"),
    "he": ReferenceEnergy("he", "He", -2.90372, (2.0,), 1, 1, (), "high-accuracy nonrelativistic", "Pekeris"),
    "li": ReferenceEnergy("li", "Li", -7.47806, (3.0,), 2, 1, (), "high-accuracy nonrelativistic", "reference literature"),
    "be": ReferenceEnergy("be", "Be", -14.66736, (4.0,), 2, 2, (), "high-accuracy nonrelativistic", "reference literature"),
    "h2": ReferenceEnergy(
        "h2", "$H_2$", -1.17447, (1.0, 1.0), 1, 1,
        ((1.0, 1.0, 1.4),), "Born-Oppenheimer nonrelativistic, R=1.4 Bohr", "FermiNet benchmark",
    ),
    "lih": ReferenceEnergy(
        "lih", "LiH", -8.07050, (1.0, 3.0), 2, 2,
        ((1.0, 3.0, 3.015),), "Born-Oppenheimer reference, R=3.015 Bohr", "benchmark literature",
    ),
}

_CHARGE_ATOL = 1e-6
_GEOMETRY_ATOL_BOHR = 5e-4


def _system_pair_signature(system: Any) -> Tuple[Tuple[float, float, float], ...]:
    """生成不依赖平移、旋转或核排列顺序的带电荷核间距离签名。"""
    coords = system.nuclear_coords.detach().to(device="cpu", dtype=torch.float64)
    charges = system.nuclear_charges.detach().to(device="cpu", dtype=torch.float64)
    pairs: List[Tuple[float, float, float]] = []
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            z_low, z_high = sorted((float(charges[i]), float(charges[j])))
            distance = float(torch.linalg.vector_norm(coords[i] - coords[j]))
            pairs.append((z_low, z_high, distance))
    return tuple(sorted(pairs))


def get_system_reference(system: Optional[Any]) -> Optional[ReferenceEnergy]:
    """严格匹配体系参考记录；无法验证电荷、自旋和几何时返回 ``None``。"""
    if system is None:
        return None
    if isinstance(system, str):
        return REFERENCE_SYSTEMS.get(system.strip().lower())
    required = ("nuclear_coords", "nuclear_charges", "n_up", "n_down")
    if not all(hasattr(system, name) for name in required):
        return None

    charges = tuple(sorted(float(z) for z in system.nuclear_charges.detach().cpu()))
    signature = _system_pair_signature(system)
    for reference in REFERENCE_SYSTEMS.values():
        if system.n_up != reference.n_up or system.n_down != reference.n_down:
            continue
        if len(charges) != len(reference.nuclear_charges):
            continue
        if any(abs(actual - expected) > _CHARGE_ATOL for actual, expected in zip(charges, reference.nuclear_charges)):
            continue
        if len(signature) != len(reference.pair_signature):
            continue
        if all(
            abs(actual_z1 - expected_z1) <= _CHARGE_ATOL
            and abs(actual_z2 - expected_z2) <= _CHARGE_ATOL
            and abs(actual_r - expected_r) <= _GEOMETRY_ATOL_BOHR
            for (actual_z1, actual_z2, actual_r), (expected_z1, expected_z2, expected_r)
            in zip(signature, reference.pair_signature)
        ):
            return reference
    return None


def apply_academic_style() -> None:
    """
    设定统一的现代学术期刊级 Matplotlib 渲染样式参数。
    采用清晰的无衬线字体体系、纤细边框与柔和网格辅助线。
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Calibri"],
        "axes.edgecolor": "#2c3e50",
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": "#bdc3c7",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.4,
        "legend.fontsize": 9.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#bdc3c7",
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# 立即应用全局学术样式
apply_academic_style()


# ==============================================================================
# 2. 统计量与分子体系分析辅助函数
# ==============================================================================

def compute_distribution_stats(
    values: Union[torch.Tensor, np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    计算输入样本序列的全面描述性统计量（包括均值、中位数、标准差、MAD、偏度及超额峰度）。
    包含严格的数值零方差与异常值兜底保护。

    Args:
        values: 任意可转换为一维数值数组的数据 (PyTorch Tensor, NumPy array 或 Python list)。

    Returns:
        Dict[str, float]: 包含 mean, median, std, var, mad, skewness, kurtosis 等统计指标的字典。
    """
    if isinstance(values, torch.Tensor):
        vals = values.detach().cpu().flatten().numpy()
    else:
        vals = np.asarray(values, dtype=np.float64).flatten()

    # 滤除非有限值 (NaN, Inf)
    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "var": 0.0,
            "mad": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    clean_vals = vals[finite_mask]
    n = clean_vals.size
    mean = float(np.mean(clean_vals))
    median = float(np.median(clean_vals))
    var = float(np.var(clean_vals))
    std = float(np.sqrt(var))
    mad = float(np.median(np.abs(clean_vals - median)))

    # 计算三阶中心矩与偏度 (Skewness)
    # S = E[(X - mu)^3] / sigma^3
    if std > 1e-12 and n > 2:
        m3 = float(np.mean((clean_vals - mean) ** 3))
        skewness = m3 / (std ** 3)
    else:
        skewness = 0.0

    # 计算四阶中心矩与超额峰度 (Excess Kurtosis, 正态分布基准为 0)
    # K = E[(X - mu)^4] / sigma^4 - 3.0
    if std > 1e-12 and n > 3:
        m4 = float(np.mean((clean_vals - mean) ** 4))
        kurtosis = (m4 / (std ** 4)) - 3.0
    else:
        kurtosis = 0.0

    return {
        "count": n,
        "mean": mean,
        "median": median,
        "std": std,
        "var": var,
        "mad": mad,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "min": float(np.min(clean_vals)),
        "max": float(np.max(clean_vals)),
    }


def get_system_reference_energy(system: Optional[Any]) -> Optional[float]:
    """
    根据分子体系信息自动推断理论参考基态能量 (Hartree)。

    Args:
        system: MolecularSystem 实例或体系名称字符串。

    Returns:
        Optional[float]: 匹配到的参考基态能量；若未匹配则返回 None。
    """
    reference = get_system_reference(system)
    return reference.energy if reference is not None else None


def get_system_label(system: Optional[Any]) -> str:
    """获取分子体系的友好显示标签 (如 'H2', 'LiH')。"""
    if system is None:
        return "Molecule"
    if isinstance(system, str):
        return system.upper()
    reference = get_system_reference(system)
    if reference is not None:
        return reference.label
    if hasattr(system, "nuclear_charges") and hasattr(system, "total_electrons"):
        charges = ",".join(f"{float(z):g}" for z in system.nuclear_charges.detach().cpu())
        return f"Molecule (Z=[{charges}], N={system.total_electrons}, spin=({system.n_up},{system.n_down}))"
    return "Molecule"


# ==============================================================================
# 3. 训练过程可视化 (Training Process Visualization)
# ==============================================================================

def plot_energy_convergence(
    history: Dict[str, List[float]],
    system_name: str = "H2",
    ref_energy: Optional[float] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    绘制能量收敛曲线 (原始能量 <E_raw>、截断能量 <E_clip>、理论参考能量线)。

    Args:
        history (Dict[str, List[float]]): 训练历史记录字典。
        system_name (str): 体系名称，用于标题和标注。
        ref_energy (Optional[float]): 理论参考能量基准线 (Hartree)。
        save_path (Optional[str]): 图片保存路径。
        ax (Optional[plt.Axes]): 可选的目标 Matplotlib 坐标系。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        created_fig = True
    else:
        fig = ax.get_figure()

    e_raw = np.array(history.get("energy_raw", []))
    e_clip = np.array(history.get("energy_clipped", []))
    epochs = np.arange(1, len(e_raw) + 1)

    if len(epochs) == 0:
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # 1. 绘制截断能量曲线
    if len(e_clip) == len(epochs):
        ax.plot(
            epochs,
            e_clip,
            color=COLOR_PALETTE["clipped_energy"],
            label=r"Clipped Energy $\langle E_{\mathrm{clip}} \rangle$",
            linewidth=1.8,
            linestyle="-",
            alpha=0.9,
        )

    # 2. 绘制原始能量曲线。训练期批次相关，不能把 IID 代理量画成置信带。
    line_raw = ax.plot(
        epochs,
        e_raw,
        color=COLOR_PALETTE["raw_energy"],
        label=r"Raw Energy $\langle E_{\mathrm{raw}} \rangle$",
        linewidth=2.0,
        marker="o" if len(epochs) <= 25 else None,
        markersize=4,
    )
    # 3. 绘制理论参考能量基准线
    final_e = e_raw[-1] if len(e_raw) > 0 else 0.0
    if ref_energy is not None:
        ax.axhline(
            y=ref_energy,
            color=COLOR_PALETTE["ref_energy"],
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label=f"Ref ({system_name}): {ref_energy:.5f} Ha",
        )
        # 标注与参考能量的最终残差 (化学精度判据)
        delta_e_ha = abs(final_e - ref_energy)
        delta_e_kcal = delta_e_ha * 627.509  # 1 Ha ≈ 627.509 kcal/mol
        info_text = (
            f"Final $\\langle E \\rangle$: {final_e:.4f} Ha\n"
            f"$\\Delta E$: {delta_e_ha * 1e3:.2f} mHa\n"
            f"({delta_e_kcal:.2f} kcal/mol)"
        )
        # 放置在左下角，避免与后期收敛到右下角基准线的曲线碰撞
        ax.text(
            0.04,
            0.08,
            info_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="#bdc3c7"),
        )

    # 动态设定 y 轴范围，为基线与阴影预留呼吸空间
    all_energy_pts = np.concatenate([e_raw, e_clip]) if len(e_clip) > 0 else e_raw
    if len(all_energy_pts) > 0:
        y_min = min(float(np.min(all_energy_pts)), ref_energy if ref_energy is not None else float(np.min(all_energy_pts)))
        y_max = float(np.max(all_energy_pts))
        y_span = max(y_max - y_min, 0.05)
        ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.10 * y_span)

    ax.set_title(f"{system_name} Ground State Energy Convergence")
    ax.set_xlabel("Optimization Epoch")
    ax.set_ylabel(r"Energy $\langle E \rangle$ (Hartree)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    if created_fig and save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_variance_and_sem(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
) -> plt.Figure:
    """
    绘制能量方差 Var(E) 与单批次 IID SEM 代理量变化曲线。
    体现量子蒙特卡洛著名的零方差原理 (Zero-Variance Principle:
    当变分试探波函数 Psi 逼近哈密顿量真实本征态时，局域能量涨落与方差严格趋近于 0)。

    Args:
        history (Dict[str, List[float]]): 训练历史记录。
        save_path (Optional[str]): 保存路径。
        ax (Optional[plt.Axes]): 可选的目标 Matplotlib 坐标系。
        log_scale (bool): 是否使用对数纵坐标展示。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        created_fig = True
    else:
        fig = ax.get_figure()

    var_e = np.array(history.get("energy_var", []))
    sem_e = np.array(history.get("energy_iid_sem_proxy", []))
    epochs = np.arange(1, len(var_e) + 1)

    if len(epochs) == 0:
        ax.text(0.5, 0.5, "No variance data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # 主轴绘制 Var(E)
    ax.plot(
        epochs,
        var_e,
        color=COLOR_PALETTE["variance"],
        label=r"Energy Variance $\mathrm{Var}(E)$",
        linewidth=2.0,
        marker="s" if len(epochs) <= 25 else None,
        markersize=4,
    )

    # 辅轴绘制 IID 代理量；它不是相关 MCMC 时间序列的可信误差条。
    ax2 = ax.twinx()
    ax2.plot(
        epochs,
        sem_e,
        color=COLOR_PALETTE["sem"],
        label=r"IID Batch SEM Proxy",
        linewidth=1.8,
        linestyle="-.",
        marker="^" if len(epochs) <= 25 else None,
        markersize=4,
    )

    if log_scale:
        finite_var = var_e[np.isfinite(var_e)]
        finite_sem = sem_e[np.isfinite(sem_e)]
        if len(finite_var) > 0 and np.all(finite_var > 0):
            ax.set_yscale("log")
            v_min, v_max = float(np.min(finite_var)), float(np.max(finite_var))
            # 让 Var 位于偏上方，留出呼吸空间
            ax.set_ylim(v_min * 0.4, v_max * 3.0)
        if len(finite_sem) > 0 and np.all(finite_sem > 0):
            ax2.set_yscale("log")
            s_min, s_max = float(np.min(finite_sem)), float(np.max(finite_sem))
            # 让 SEM 位于偏下方，防止两条曲线直接黏连重合
            ax2.set_ylim(s_min * 0.15, s_max * 1.5)

    ax.set_title("Zero-Variance Principle Convergence")
    ax.set_xlabel("Optimization Epoch")
    ax.set_ylabel(r"$\mathrm{Var}(E)\ (\mathrm{Ha}^2)$", color=COLOR_PALETTE["variance"])
    ax2.set_ylabel(r"IID SEM Proxy $(\mathrm{Ha})$", color=COLOR_PALETTE["sem"])
    ax.tick_params(axis="y", labelcolor=COLOR_PALETTE["variance"])
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["sem"])
    ax.grid(True, linestyle="--", alpha=0.4)

    # 合并图例
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    if created_fig and save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_mcmc_diagnostics(
    history: Dict[str, List[float]],
    target_acc: float = 0.5,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    绘制 MCMC 采样健康度跟踪曲线 (接受率 Acc %、自适应移动步长 Step Size)。

    Args:
        history (Dict[str, List[float]]): 包含 'acceptance_rate' 和 'step_size' 的字典。
        target_acc (float): 目标接受率 (通常 0.5 即 50%)。
        save_path (Optional[str]): 保存路径。
        ax (Optional[plt.Axes]): 可选目标坐标系。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        created_fig = True
    else:
        fig = ax.get_figure()

    acc = np.array(history.get("acceptance_rate", [])) * 100.0  # 转为百分比
    step = np.array(history.get("step_size", []))
    epochs = np.arange(1, len(acc) + 1)

    if len(epochs) == 0:
        ax.text(0.5, 0.5, "No MCMC diagnostics data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # 左轴：接受率 Acc %
    l1 = ax.plot(
        epochs,
        acc,
        color=COLOR_PALETTE["acceptance"],
        label="Acceptance Rate (%)",
        linewidth=2.0,
    )
    # 目标参考线
    ax.axhline(
        y=target_acc * 100.0,
        color=COLOR_PALETTE["target_line"],
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label=f"Target Acc ({target_acc * 100:.0f}%)",
    )

    # 右轴：自适应步长
    ax2 = ax.twinx()
    l2 = ax2.plot(
        epochs,
        step,
        color=COLOR_PALETTE["step_size"],
        label=r"Adaptive Step Size $\sigma_{\mathrm{prop}}$",
        linewidth=1.8,
        linestyle="-.",
    )

    ax.set_title("MCMC Sampler Health & Adaptive Step")
    ax.set_xlabel("Optimization Epoch")
    ax.set_ylabel("Acceptance Rate (%)", color=COLOR_PALETTE["acceptance"])
    ax2.set_ylabel(r"Proposal Step Size (Bohr)", color=COLOR_PALETTE["step_size"])
    ax.tick_params(axis="y", labelcolor=COLOR_PALETTE["acceptance"])
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["step_size"])

    # 动态预留轴呼吸空间
    acc_min, acc_max = float(np.min(acc)), float(np.max(acc))
    acc_span = max(acc_max - acc_min, 10.0)
    ax.set_ylim([max(0.0, acc_min - 0.15 * acc_span), min(100.0, acc_max + 0.15 * acc_span)])

    if len(step) > 0:
        s_min, s_max = float(np.min(step)), float(np.max(step))
        s_span = max(s_max - s_min, 0.05)
        ax2.set_ylim([s_min - 0.20 * s_span, s_max + 0.25 * s_span])

    ax.grid(True, linestyle="--", alpha=0.4)

    # 汇总图例 (放置于右上角，彻底防范被下降的步长曲线划过)
    lines = l1 + [ax.lines[-1]] + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right")

    if created_fig and save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_training_dashboard(
    history: Dict[str, List[float]],
    system_name: str = "H2",
    ref_energy: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    组合绘制 VMC 训练过程全景三合一综合看板 (1x3 布局: 能量收敛、方差衰减、MCMC 健康度)。

    Args:
        history (Dict[str, List[float]]): 训练记录字典。
        system_name (str): 体系名称。
        ref_energy (Optional[float]): 理论参考能。
        save_path (Optional[str]): 保存路径。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.0))
    fig.suptitle(f"FermiNet VMC Training Process Overview — {system_name}", fontsize=14, y=1.02)

    # Panel 1: 能量收敛
    plot_energy_convergence(
        history=history,
        system_name=system_name,
        ref_energy=ref_energy,
        ax=axes[0],
    )

    # Panel 2: 方差与 SEM
    plot_variance_and_sem(
        history=history,
        ax=axes[1],
        log_scale=True,
    )

    # Panel 3: MCMC 诊断
    plot_mcmc_diagnostics(
        history=history,
        ax=axes[2],
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


# ==============================================================================
# 4. 训练结果与物理状态可视化 (Training Results & Physical State Visualization)
# ==============================================================================

def plot_electron_density_3d(
    walkers: Union[torch.Tensor, np.ndarray],
    nuclear_coords: Union[torch.Tensor, np.ndarray],
    nuclear_charges: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_up: Optional[int] = None,
    n_down: Optional[int] = None,
    max_points: int = 1500,
    save_path: Optional[str] = None,
    ax: Optional[Axes3D] = None,
) -> plt.Figure:
    """
    绘制电子空间构型的三维立体散点分布 (3D Electron Spatial Probability Distribution)。
    区分自旋向上与自旋向下电子，并立体标注原子核空间位置。

    Args:
        walkers: MCMC 采样的电子空间构型，Shape: `(B, N, 3)`。
        nuclear_coords: 原子核空间三维坐标，Shape: `(M, 3)`。
        nuclear_charges: 原子核电荷量，Shape: `(M,)`。
        n_up (Optional[int]): 自旋向上电子数 (默认自动按 N // 2 估计)。
        n_down (Optional[int]): 自旋向下电子数。
        max_points (int): 绘制电子的最大采样点数量 (防止过密与卡顿)。
        save_path (Optional[str]): 保存路径。
        ax (Optional[Axes3D]): 可选的目标 3D 坐标轴。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(8.0, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.get_figure()

    # 格式化转换
    if isinstance(walkers, torch.Tensor):
        w = walkers.detach().cpu().numpy()
    else:
        w = np.asarray(walkers)

    if isinstance(nuclear_coords, torch.Tensor):
        nuc = nuclear_coords.detach().cpu().numpy()
    else:
        nuc = np.asarray(nuclear_coords)

    if nuclear_charges is not None:
        if isinstance(nuclear_charges, torch.Tensor):
            charges = nuclear_charges.detach().cpu().numpy().flatten()
        else:
            charges = np.asarray(nuclear_charges).flatten()
    else:
        charges = np.ones(nuc.shape[0], dtype=np.float64)

    b_size, n_elec, _ = w.shape
    if n_up is None:
        n_up = (n_elec + 1) // 2
    if n_down is None:
        n_down = n_elec - n_up

    # 抽取自旋向上与自旋向下电子坐标并下采样
    up_coords = w[:, :n_up, :].reshape(-1, 3)
    down_coords = w[:, n_up:n_up + n_down, :].reshape(-1, 3)

    if len(up_coords) > max_points:
        idx_up = np.random.choice(len(up_coords), size=max_points, replace=False)
        up_coords = up_coords[idx_up]

    if len(down_coords) > max_points:
        idx_down = np.random.choice(len(down_coords), size=max_points, replace=False)
        down_coords = down_coords[idx_down]

    # 1. 绘制自旋向上电子散点 (Alpha)
    if len(up_coords) > 0:
        ax.scatter(
            up_coords[:, 0],
            up_coords[:, 1],
            up_coords[:, 2],
            c=COLOR_PALETTE["spin_up"],
            alpha=0.35,
            s=12,
            edgecolors="none",
            label=f"Spin Up $e^-$ ($\\alpha$, {n_up})",
        )

    # 2. 绘制自旋向下电子散点 (Beta)
    if len(down_coords) > 0:
        ax.scatter(
            down_coords[:, 0],
            down_coords[:, 1],
            down_coords[:, 2],
            c=COLOR_PALETTE["spin_down"],
            alpha=0.35,
            s=12,
            edgecolors="none",
            label=f"Spin Down $e^-$ ($\\beta$, {n_down})",
        )

    # 3. 绘制原子核位置 (大尺寸球体)
    for i, (coord, q) in enumerate(zip(nuc, charges)):
        ax.scatter(
            [coord[0]],
            [coord[1]],
            [coord[2]],
            c=COLOR_PALETTE["nucleus"],
            s=180 + q * 40,
            marker="o",
            edgecolors="black",
            linewidth=1.5,
            label="Nuclei" if i == 0 else "",
            zorder=10,
        )
        element_sym = "H" if abs(q - 1.0) < 0.2 else ("Li" if abs(q - 3.0) < 0.2 else f"Z={int(q)}")
        ax.text(
            coord[0] + 0.25,
            coord[1] + 0.25,
            coord[2] + 0.35,
            f"{element_sym} ($Z={q:.0f}$)",
            fontsize=9.5,
            fontweight="bold",
            color="#2c3e50",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="#bdc3c7", linewidth=0.6),
            zorder=20,
        )

    ax.view_init(elev=20, azim=35)
    ax.set_title("3D Electron Probability Density (Walkers)", pad=15)
    ax.set_xlabel("$x$ (Bohr)")
    ax.set_ylabel("$y$ (Bohr)")
    ax.set_zlabel("$z$ (Bohr)")
    ax.legend(loc="upper right")

    if created_fig and save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_electron_density_2d(
    walkers: Union[torch.Tensor, np.ndarray],
    nuclear_coords: Union[torch.Tensor, np.ndarray],
    n_up: Optional[int] = None,
    n_down: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制电子构型沿主要成键核间轴 (z 轴) 的投影密度直方图与 x-z 截面散点分布。

    Args:
        walkers: 电子坐标构型，Shape: `(B, N, 3)`。
        nuclear_coords: 原子核坐标，Shape: `(M, 3)`。
        n_up (Optional[int]): 自旋向上电子数。
        n_down (Optional[int]): 自旋向下电子数。
        save_path (Optional[str]): 图片保存路径。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    fig, (ax_hist, ax_proj) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    if isinstance(walkers, torch.Tensor):
        w = walkers.detach().cpu().numpy()
    else:
        w = np.asarray(walkers)

    if isinstance(nuclear_coords, torch.Tensor):
        nuc = nuclear_coords.detach().cpu().numpy()
    else:
        nuc = np.asarray(nuclear_coords)

    b_size, n_elec, _ = w.shape
    if n_up is None:
        n_up = (n_elec + 1) // 2
    if n_down is None:
        n_down = n_elec - n_up

    # 提取 z 坐标
    z_up = w[:, :n_up, 2].flatten()
    z_down = w[:, n_up:n_up + n_down, 2].flatten()
    z_all = w[:, :, 2].flatten()

    # 子图 1: 沿 z 轴的单粒子电子密度投影直方图
    bins = np.linspace(float(np.min(z_all)) - 0.5, float(np.max(z_all)) + 0.5, 50)
    if len(z_up) > 0:
        ax_hist.hist(
            z_up,
            bins=bins,
            density=True,
            alpha=0.5,
            color=COLOR_PALETTE["spin_up"],
            label=r"Spin Up $\alpha$",
        )
    if len(z_down) > 0:
        ax_hist.hist(
            z_down,
            bins=bins,
            density=True,
            alpha=0.5,
            color=COLOR_PALETTE["spin_down"],
            label=r"Spin Down $\beta$",
        )

    # 标记核的 z 位置
    for idx, n_coord in enumerate(nuc):
        ax_hist.axvline(
            x=n_coord[2],
            color=COLOR_PALETTE["cusp_marker"],
            linestyle="--",
            linewidth=1.8,
            label="Nucleus Position" if idx == 0 else "",
        )

    ax_hist.set_title("Electron Density Profile along Nuclear Axis ($z$)")
    ax_hist.set_xlabel("$z$ Coordinate (Bohr)")
    ax_hist.set_ylabel("Probability Density $\\rho(z)$")
    ax_hist.grid(True, linestyle="--", alpha=0.4)
    ax_hist.legend()

    # 子图 2: x-z 平面投影散点与核位置
    x_up = w[:, :n_up, 0].flatten()
    x_down = w[:, n_up:n_up + n_down, 0].flatten()

    max_pts = 1000
    if len(x_up) > max_pts:
        sub_idx = np.random.choice(len(x_up), size=max_pts, replace=False)
        x_up, z_up_sub = x_up[sub_idx], z_up[sub_idx]
    else:
        z_up_sub = z_up

    if len(x_down) > max_pts:
        sub_idx = np.random.choice(len(x_down), size=max_pts, replace=False)
        x_down, z_down_sub = x_down[sub_idx], z_down[sub_idx]
    else:
        z_down_sub = z_down

    if len(x_up) > 0:
        ax_proj.scatter(
            z_up_sub,
            x_up,
            c=COLOR_PALETTE["spin_up"],
            alpha=0.4,
            s=14,
            label=r"$\alpha$ Electrons",
        )
    if len(x_down) > 0:
        ax_proj.scatter(
            z_down_sub,
            x_down,
            c=COLOR_PALETTE["spin_down"],
            alpha=0.4,
            s=14,
            label=r"$\beta$ Electrons",
        )

    # 绘制核位置
    for idx, n_coord in enumerate(nuc):
        ax_proj.scatter(
            [n_coord[2]],
            [n_coord[0]],
            c=COLOR_PALETTE["nucleus"],
            s=150,
            marker="X",
            label="Nucleus" if idx == 0 else "",
            zorder=10,
        )

    ax_proj.set_title("2D Projection onto $x-z$ Plane")
    ax_proj.set_xlabel("$z$ Coordinate (Bohr)")
    ax_proj.set_ylabel("$x$ Coordinate (Bohr)")
    ax_proj.grid(True, linestyle="--", alpha=0.4)
    ax_proj.legend()

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_wavefunction_profile(
    model: nn.Module,
    system: Any,
    axis: str = "z",
    grid_range: Tuple[float, float] = (-3.0, 3.0),
    num_points: int = 200,
    walkers: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    沿核间轴探测波函数截面剖面 (1D Wavefunction Profile & Cusp Condition)。
    固定其它 N-1 个电子位置，将第 1 个电子沿着指定轴扫描，
    绘制 log|Psi(r)| 与概率密度 |Psi(r)|^2，清晰展示在原子核处的尖峰 (Kato Cusp) 特征。

    Args:
        model (nn.Module): FermiNet 深度波函数网络。
        system: MolecularSystem 分子体系。
        axis (str): 扫描轴向 ('x', 'y' 或 'z'，默认 'z')。
        grid_range (Tuple[float, float]): 扫描范围 (单位: Bohr，默认 [-3.0, 3.0])。
        num_points (int): 扫描网格采样点数 (默认 200)。
        walkers (Optional[torch.Tensor]): 可选的参考电子构型。
        save_path (Optional[str]): 保存路径。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    model.eval()
    device = system.nuclear_coords.device
    dtype = system.nuclear_coords.dtype
    total_electrons = system.total_electrons
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

    # 构建基准构型 (Base Configuration): 其余电子固定
    base_conf = torch.zeros((1, total_electrons, 3), dtype=dtype, device=device)
    if walkers is not None and walkers.shape[0] > 0:
        base_conf[0] = walkers[0].clone().to(device=device, dtype=dtype)
    else:
        # 若无给定 walkers，将各电子分别放置在各原子核邻域
        for i in range(total_electrons):
            nuc_idx = i % system.num_nuclei
            base_conf[0, i] = system.nuclear_coords[nuc_idx] + torch.randn(3, device=device, dtype=dtype) * 0.1

    # 生成沿指定轴向扫描的 1D 网格
    axis_vals = np.linspace(grid_range[0], grid_range[1], num_points)
    batch_scan = base_conf.repeat(num_points, 1, 1)

    for i, val in enumerate(axis_vals):
        batch_scan[i, 0, axis_idx] = float(val)

    # 纯前向推理波函数
    with torch.no_grad():
        log_psi, _ = model.forward_log_psi(batch_scan)
        log_psi_np = log_psi.cpu().numpy().flatten()

    # 归一化几率密度 (相对最大峰值归一)
    prob_density = np.exp(2.0 * (log_psi_np - np.max(log_psi_np)))

    # 绘制双子图面板
    fig, (ax_log, ax_prob) = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)

    # 上子图: 对数波函数 ln|Psi|
    ax_log.plot(
        axis_vals,
        log_psi_np,
        color=COLOR_PALETTE["wavefunction"],
        linewidth=2.0,
        label=r"$\ln|\Psi_1(" + axis + r")|$",
    )
    # 标记核坐标
    nuc_np = system.nuclear_coords.detach().cpu().numpy()
    for i, n_coord in enumerate(nuc_np):
        val_nuc = n_coord[axis_idx]
        ax_log.axvline(
            x=val_nuc,
            color=COLOR_PALETTE["cusp_marker"],
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label="Nucleus (Cusp)" if i == 0 else "",
        )

    ax_log.set_title(f"Wavefunction Profile & Electron-Nuclear Cusp along {axis.upper()}-Axis")
    ax_log.set_ylabel(r"$\ln|\Psi|$ (Log Amplitude)")
    ax_log.grid(True, linestyle="--", alpha=0.4)
    ax_log.legend(loc="upper right")

    # 下子图: 几率密度 |Psi|^2
    ax_prob.plot(
        axis_vals,
        prob_density,
        color=COLOR_PALETTE["raw_energy"],
        linewidth=2.0,
        label=r"$|\Psi|^2$ (Relative Density)",
    )
    ax_prob.fill_between(axis_vals, 0, prob_density, color=COLOR_PALETTE["sem_band"], alpha=0.4)
    for n_coord in nuc_np:
        val_nuc = n_coord[axis_idx]
        ax_prob.axvline(
            x=val_nuc,
            color=COLOR_PALETTE["cusp_marker"],
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
        )

    ax_prob.set_xlabel(f"Electron Coordinate along ${axis}$-axis (Bohr)")
    ax_prob.set_ylabel(r"Normalized $|\Psi|^2$")
    ax_prob.grid(True, linestyle="--", alpha=0.4)
    ax_prob.legend(loc="upper right")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_wavefunction_2d_slice(
    model: nn.Module,
    system: Any,
    plane: str = "xz",
    grid_range_1: Tuple[float, float] = (-3.0, 3.0),
    grid_range_2: Tuple[float, float] = (-3.0, 3.0),
    num_points: int = 60,
    walkers: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制电子波函数在二维平面上的切片截面等高线/热力图 (2D Wavefunction Slice & Bonding Density)。
    直观展现两原子核之间的共价键/离子键电子云重叠与空间对称性。

    Args:
        model (nn.Module): FermiNet 模型。
        system: MolecularSystem。
        plane (str): 切片平面 ('xz', 'yz' 或 'xy'，默认 'xz')。
        grid_range_1: 横轴坐标范围 (Bohr)。
        grid_range_2: 纵轴坐标范围 (Bohr)。
        num_points: 每个维度的网格离散点数。
        walkers: 参考电子构型。
        save_path: 保存路径。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    model.eval()
    device = system.nuclear_coords.device
    dtype = system.nuclear_coords.dtype
    total_electrons = system.total_electrons

    axes_map = {
        "xz": (0, 2, "x", "z"),
        "yz": (1, 2, "y", "z"),
        "xy": (0, 1, "x", "y"),
    }
    idx1, idx2, name1, name2 = axes_map[plane.lower()]

    # 网格离散化
    v1 = np.linspace(grid_range_1[0], grid_range_1[1], num_points)
    v2 = np.linspace(grid_range_2[0], grid_range_2[1], num_points)
    V1, V2 = np.meshgrid(v1, v2)

    # 展平为 (num_points^2, 3)
    grid_coords = np.zeros((num_points * num_points, 3), dtype=np.float32)
    grid_coords[:, idx1] = V1.flatten()
    grid_coords[:, idx2] = V2.flatten()

    # 基准电子构型
    base_conf = torch.zeros((1, total_electrons, 3), dtype=dtype, device=device)
    if walkers is not None and walkers.shape[0] > 0:
        base_conf[0] = walkers[0].clone().to(device=device, dtype=dtype)
    else:
        for i in range(total_electrons):
            nuc_idx = i % system.num_nuclei
            base_conf[0, i] = system.nuclear_coords[nuc_idx]

    batch_confs = base_conf.repeat(num_points * num_points, 1, 1)
    batch_confs[:, 0, :] = torch.from_numpy(grid_coords).to(device=device, dtype=dtype)

    # 分批前向评估防止显存溢出
    log_psi_list = []
    chunk_size = 512
    with torch.no_grad():
        for start_idx in range(0, batch_confs.shape[0], chunk_size):
            chunk = batch_confs[start_idx:start_idx + chunk_size]
            lp, _ = model.forward_log_psi(chunk)
            log_psi_list.append(lp.cpu().numpy())

    log_psi_all = np.concatenate(log_psi_list, axis=0)
    density_2d = np.exp(2.0 * (log_psi_all - np.max(log_psi_all))).reshape(num_points, num_points)

    # 绘图
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    contour = ax.contourf(V1, V2, density_2d, levels=30, cmap="viridis", alpha=0.9)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r"Relative Probability Density $|\Psi|^2$", rotation=270, labelpad=15)

    # 绘制原子核投影位置
    nuc_np = system.nuclear_coords.detach().cpu().numpy()
    for i, n_coord in enumerate(nuc_np):
        ax.scatter(
            [n_coord[idx1]],
            [n_coord[idx2]],
            c="#e74c3c",
            s=160,
            marker="o",
            edgecolors="white",
            linewidth=1.5,
            label="Nuclei" if i == 0 else "",
            zorder=10,
        )

    ax.set_title(f"2D Electron Density Slice on ${name1}-{name2}$ Plane")
    ax.set_xlabel(f"${name1}$ Coordinate (Bohr)")
    ax.set_ylabel(f"${name2}$ Coordinate (Bohr)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_local_energy_distribution(
    local_energies: Union[torch.Tensor, np.ndarray, List[float]],
    ref_energy: Optional[float] = None,
    clip_factor: float = 5.0,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    绘制局域能量分布直方图与统计偏度分析 (Local Energy Histogram & Skewness/Kurtosis)。
    展示能量样本的集中度、偏度 (Skewness) 以及截断截击区域。

    Args:
        local_energies: 局域能量样本序列 E_L(r)。
        ref_energy (Optional[float]): 理论参考能。
        clip_factor (float): MAD 截断因子。
        save_path (Optional[str]): 保存路径。
        ax (Optional[plt.Axes]): 可选的目标坐标系。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        created_fig = True
    else:
        fig = ax.get_figure()

    stats = compute_distribution_stats(local_energies)
    if isinstance(local_energies, torch.Tensor):
        e_vals = local_energies.detach().cpu().numpy().flatten()
    else:
        e_vals = np.asarray(local_energies).flatten()

    e_vals = e_vals[np.isfinite(e_vals)]
    if len(e_vals) == 0:
        ax.text(0.5, 0.5, "No finite local energy data", ha="center", va="center", transform=ax.transAxes)
        return fig

    # 过滤极端异常离群点以优化直方图显示
    med = stats["median"]
    mad = stats["mad"]
    display_range = 6.0 * max(mad, 0.1)
    plot_mask = (e_vals >= med - display_range) & (e_vals <= med + display_range)
    display_vals = e_vals[plot_mask] if np.any(plot_mask) else e_vals

    # 绘制直方图
    counts, bin_edges, _ = ax.hist(
        display_vals,
        bins=min(50, max(15, len(display_vals) // 20)),
        density=True,
        color=COLOR_PALETTE["raw_energy"],
        alpha=0.65,
        edgecolor="white",
        linewidth=0.8,
        label="Local Energy $E_L$",
    )

    # 绘制均值线与中位数线
    ax.axvline(
        x=stats["mean"],
        color=COLOR_PALETTE["clipped_energy"],
        linestyle="-",
        linewidth=2.0,
        label=f"Mean $\\mu$: {stats['mean']:.4f} Ha",
    )
    ax.axvline(
        x=stats["median"],
        color=COLOR_PALETTE["variance"],
        linestyle="--",
        linewidth=1.8,
        label=f"Median: {stats['median']:.4f} Ha",
    )

    # 绘制理论参考能线
    if ref_energy is not None:
        ax.axvline(
            x=ref_energy,
            color=COLOR_PALETTE["ref_energy"],
            linestyle=":",
            linewidth=2.0,
            label=f"Ref Energy: {ref_energy:.4f} Ha",
        )

    # 绘制平滑截断 MAD 阈值边界
    clip_low = med - clip_factor * mad
    clip_high = med + clip_factor * mad
    ax.axvspan(
        max(float(np.min(display_vals)), clip_low),
        min(float(np.max(display_vals)), clip_high),
        color=COLOR_PALETTE["sem_band"],
        alpha=0.15,
        label=f"MAD Clip Range ({clip_factor:.0f}$\\times$MAD)",
    )

    # 统计信息小卡片
    info_box = (
        f"Count: {stats['count']}\n"
        f"Mean: {stats['mean']:.4f} Ha\n"
        f"Std Dev: {stats['std']:.4f}\n"
        f"MAD: {stats['mad']:.4f}\n"
        f"Skewness: {stats['skewness']:+.3f}\n"
        f"Kurtosis: {stats['kurtosis']:+.3f}"
    )
    ax.text(
        0.04,
        0.95,
        info_box,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.92, edgecolor="#bdc3c7"),
    )

    # 预留顶部呼吸空间，防止直方图最高峰贴近右上角图例
    if len(counts) > 0:
        max_density = float(np.max(counts))
        ax.set_ylim(0.0, max_density * 1.25)

    ax.set_title("Local Energy Distribution & Error Statistics")
    ax.set_xlabel("Local Energy $E_L(r)$ (Hartree)")
    ax.set_ylabel("Probability Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    if created_fig and save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_physical_results_dashboard(
    walkers: torch.Tensor,
    system: Any,
    local_energies: torch.Tensor,
    model: Optional[nn.Module] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    组合绘制训练结果物理状态全景综合看板 (包含 3D 电子空间分布、核轴密度剖面、局域能量直方图统计)。

    Args:
        walkers: 电子构型 (B, N, 3)。
        system: MolecularSystem。
        local_energies: 最终采样的一批局域能量值 (B,)。
        model: FermiNet 模型。
        save_path: 保存路径。

    Returns:
        plt.Figure: Matplotlib 图表对象。
    """
    fig = plt.figure(figsize=(18.0, 5.5))
    system_label = get_system_label(system)
    fig.suptitle(f"FermiNet Quantum Physical State Dashboard — {system_label}", fontsize=14, y=1.02)

    # Panel 1: 3D 电子几率密度空间分布
    ax1 = fig.add_subplot(131, projection="3d")
    plot_electron_density_3d(
        walkers=walkers,
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        ax=ax1,
    )

    # Panel 2: 沿核轴的波函数剖面 (若有模型) 或 2D 投影密度
    ax2 = fig.add_subplot(132)
    if model is not None:
        # 简化版 1D 剖面
        model.eval()
        device = system.nuclear_coords.device
        dtype = system.nuclear_coords.dtype
        z_vals = np.linspace(-3.0, 3.0, 150)
        base = walkers[0:1].clone().to(device=device, dtype=dtype)
        batch = base.repeat(150, 1, 1)
        for idx, z in enumerate(z_vals):
            batch[idx, 0, 2] = float(z)
        with torch.no_grad():
            lp, _ = model.forward_log_psi(batch)
            lp_np = lp.cpu().numpy().flatten()
            density = np.exp(2.0 * (lp_np - np.max(lp_np)))

        ax2.plot(z_vals, density, color=COLOR_PALETTE["raw_energy"], linewidth=2.0, label=r"$|\Psi(z)|^2$")
        ax2.fill_between(z_vals, 0, density, color=COLOR_PALETTE["sem_band"], alpha=0.4)
        for idx_nc, n_coord in enumerate(system.nuclear_coords.detach().cpu().numpy()):
            ax2.axvline(
                n_coord[2],
                color=COLOR_PALETTE["cusp_marker"],
                linestyle="--",
                linewidth=1.6,
                label="Nucleus (Cusp)" if idx_nc == 0 else "",
            )
        ax2.set_title("Wavefunction Density along Nuclear Axis")
        ax2.set_xlabel("$z$ Coordinate (Bohr)")
        ax2.set_ylabel("Relative Density")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()
    else:
        # 退化为 walkers 轴投影直方图
        z_all = walkers[:, :, 2].detach().cpu().numpy().flatten()
        ax2.hist(z_all, bins=40, density=True, color=COLOR_PALETTE["raw_energy"], alpha=0.6)
        for idx_nc, n_coord in enumerate(system.nuclear_coords.detach().cpu().numpy()):
            ax2.axvline(
                n_coord[2],
                color=COLOR_PALETTE["cusp_marker"],
                linestyle="--",
                linewidth=1.6,
                label="Nucleus" if idx_nc == 0 else "",
            )
        ax2.set_title("Walkers Nuclear Axis Projection")
        ax2.set_xlabel("$z$ Coordinate (Bohr)")
        ax2.set_ylabel("Histogram Density")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()

    # Panel 3: 局域能量直方图与统计偏度
    ax3 = fig.add_subplot(133)
    ref_e = get_system_reference_energy(system)
    plot_local_energy_distribution(
        local_energies=local_energies,
        ref_energy=ref_e,
        ax=ax3,
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


# ==============================================================================
# 5. 一键全自动综合可视化导出器 (One-Stop All Visualizations Exporter)
# ==============================================================================

def plot_all_visualizations(
    history: Dict[str, Any],
    system: Any,
    model: Optional[nn.Module] = None,
    walkers: Optional[torch.Tensor] = None,
    local_energies: Optional[torch.Tensor] = None,
    plot_dir: str = "assets/plots",
    prefix: str = "",
) -> Dict[str, str]:
    """
    一键全自动生成并保存全部训练过程与物理状态可视化图表。

    生成文件包括：
    1. training_dashboard.png  — 训练过程全景综合看板 (能量收敛、方差衰减、MCMC 健康度)
    2. energy_convergence.png  — 详细能量收敛曲线 (带 SEM 误差带与理论基准线)
    3. variance_sem.png        — 能量方差与 SEM 演化曲线 (零方差原理验证)
    4. mcmc_diagnostics.png    — MCMC 采样接受率与自适应步长
    5. electron_density_3d.png — 3D 电子几率密度空间分布 (区分自旋向上/向下与核位置)
    6. electron_density_2d.png — 2D 投影密度与核轴一维分布直方图
    7. local_energy_dist.png   — 最终局域能量直方图分布与偏度统计卡片
    8. wavefunction_1d.png     — 沿核轴 1D 对数波函数与核 Cusp 尖峰 (若 model 有效)
    9. wavefunction_2d.png     — 2D 成键截面电子密度热力图 (若 model 有效)
    10. physical_dashboard.png — 物理状态全景综合看板

    Args:
        history (Dict[str, Any]): 训练历史记录。
        system: MolecularSystem 物理体系。
        model (Optional[nn.Module]): FermiNet 深度波函数网络。
        walkers (Optional[torch.Tensor]): MCMC 最终平衡态电子构型。
        local_energies (Optional[torch.Tensor]): 最终步局域能量序列。
        plot_dir (str): 图表输出目录。
        prefix (str): 文件名前缀。

    Returns:
        Dict[str, str]: 生成的图表文件字典，键为图表类型名，值为保存的绝对文件路径。
    """
    os.makedirs(plot_dir, exist_ok=True)
    saved_files: Dict[str, str] = {}
    system_name = get_system_label(system)
    ref_energy = get_system_reference_energy(system)

    def _path(filename: str) -> str:
        name = f"{prefix}_{filename}" if prefix else filename
        return os.path.abspath(os.path.join(plot_dir, name))

    # A. 训练过程可视化
    if history and len(history.get("energy_raw", [])) > 0:
        p_dash = _path("training_dashboard.png")
        plot_training_dashboard(history, system_name=system_name, ref_energy=ref_energy, save_path=p_dash)
        saved_files["training_dashboard"] = p_dash

        p_energy = _path("energy_convergence.png")
        plot_energy_convergence(history, system_name=system_name, ref_energy=ref_energy, save_path=p_energy)
        saved_files["energy_convergence"] = p_energy

        p_var = _path("variance_sem.png")
        plot_variance_and_sem(history, save_path=p_var)
        saved_files["variance_sem"] = p_var

        p_mcmc = _path("mcmc_diagnostics.png")
        plot_mcmc_diagnostics(history, save_path=p_mcmc)
        saved_files["mcmc_diagnostics"] = p_mcmc

    # B. 电子密度分布可视化 (Walkers)
    if walkers is not None and system is not None:
        p_3d = _path("electron_density_3d.png")
        plot_electron_density_3d(
            walkers=walkers,
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            save_path=p_3d,
        )
        saved_files["electron_density_3d"] = p_3d

        p_2d = _path("electron_density_2d.png")
        plot_electron_density_2d(
            walkers=walkers,
            nuclear_coords=system.nuclear_coords,
            n_up=system.n_up,
            n_down=system.n_down,
            save_path=p_2d,
        )
        saved_files["electron_density_2d"] = p_2d

    # C. 局域能量直方图统计
    if local_energies is not None:
        p_el = _path("local_energy_dist.png")
        plot_local_energy_distribution(local_energies, ref_energy=ref_energy, save_path=p_el)
        saved_files["local_energy_dist"] = p_el

    # D. 波函数截面与等高线扫描 (若提供了模型)
    if model is not None and system is not None:
        try:
            p_wf1 = _path("wavefunction_1d.png")
            plot_wavefunction_profile(model=model, system=system, walkers=walkers, save_path=p_wf1)
            saved_files["wavefunction_1d"] = p_wf1

            p_wf2 = _path("wavefunction_2d.png")
            plot_wavefunction_2d_slice(model=model, system=system, walkers=walkers, save_path=p_wf2)
            saved_files["wavefunction_2d"] = p_wf2
        except Exception as e:
            # 兼容极端边界退化体系
            print(f"[Warning] Failed to generate wavefunction profile: {e}")

    # E. 物理状态全景综合看板
    if walkers is not None and system is not None and local_energies is not None:
        p_phys = _path("physical_dashboard.png")
        plot_physical_results_dashboard(
            walkers=walkers,
            system=system,
            local_energies=local_energies,
            model=model,
            save_path=p_phys,
        )
        saved_files["physical_dashboard"] = p_phys

    return saved_files
