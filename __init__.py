"""
FermiNet (Fermi Neural Network) PyTorch 实现
用于量子化学多电子薛定谔方程求解的基础波函数神经网络架构
"""

# 基础几何与特征嵌入
from .geometry import GeometricContext, smooth_distance
from .features import ElectronFeatureEmbedding

# 等变交互层 (经典两流与现代化无分支层)
from .layers import FermiInteractionLayer, FermiNetLayer
from .layers_modern import FermiInteractionLayerModern
from .psiformer_attention import PsiformerAttentionBlock

# 原子核指数衰减包络
from .envelope import NuclearEnvelope
from .envelope_modern import NuclearEnvelopeModern

# Kato 尖峰 Jastrow 因子与自旋纯态投影
from .jastrow import KatoCuspJastrow
from .spin import SpinProjector

# 深度波函数模型与解耦流水线
from .model import FermiNet
from .composable_model import (
    ComposableFermiNet,
    OrbitalEnvelopeProjector,
    BranchlessCofactorDeterminant,
)

# 物理算符、哈密顿量与现代动能求解器
from .kinetic_modern import KineticEnergyCalculator, KineticMode
from .hamiltonian import (
    MolecularSystem,
    compute_local_energy,
    compute_kinetic_energy,
    diagnose_cusp_divergence,
)

# 蒙特卡洛采样与变分损失
from .mcmc import MCMCSampler
from .loss import clip_local_energy, vmc_surrogate_loss, VMCLoss

# 现代化四层正交解耦架构
from .vmc_driver import (
    VariationalState,
    HamiltonianComposite,
    MetropolisSampler,
    VMCDriver,
)

# 向后兼容适配器
from .legacy_adapters import (
    FermiInteractionLayerLegacyAdapter,
    NuclearEnvelopeLegacyAdapter,
    FermiNetLegacyAdapter,
)

# 变分训练引擎与工具函数
from .train_vmc import (
    train_vmc,
    train_two_stage_jastrow,
    evaluate_frozen_wavefunction,
    compute_autocorrelation_time,
    build_cosine_scheduler,
    build_h2_system,
    build_lih_system,
)

__all__ = [
    # 几何与特征
    "GeometricContext",
    "smooth_distance",
    "ElectronFeatureEmbedding",
    # 等变层
    "FermiInteractionLayer",
    "FermiNetLayer",
    "FermiInteractionLayerModern",
    "PsiformerAttentionBlock",
    # 包络与物理因子
    "NuclearEnvelope",
    "NuclearEnvelopeModern",
    "KatoCuspJastrow",
    "SpinProjector",
    # 深度模型
    "FermiNet",
    "ComposableFermiNet",
    "OrbitalEnvelopeProjector",
    "BranchlessCofactorDeterminant",
    # 物理与动能算符
    "KineticEnergyCalculator",
    "KineticMode",
    "MolecularSystem",
    "compute_local_energy",
    "compute_kinetic_energy",
    "diagnose_cusp_divergence",
    # 采样与损失
    "MCMCSampler",
    "clip_local_energy",
    "vmc_surrogate_loss",
    "VMCLoss",
    # 四层驱动器
    "VariationalState",
    "HamiltonianComposite",
    "MetropolisSampler",
    "VMCDriver",
    # 历史兼容适配器
    "FermiInteractionLayerLegacyAdapter",
    "NuclearEnvelopeLegacyAdapter",
    "FermiNetLegacyAdapter",
    # 训练工具
    "train_vmc",
    "train_two_stage_jastrow",
    "evaluate_frozen_wavefunction",
    "compute_autocorrelation_time",
    "build_cosine_scheduler",
    "build_h2_system",
    "build_lih_system",
]
