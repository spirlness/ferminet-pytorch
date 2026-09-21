"""
FermiNet (Fermi Neural Network) PyTorch 实现
用于量子化学多电子薛定谔方程求解的基础波函数神经网络架构
"""

from .features import ElectronFeatureEmbedding
from .layers import FermiInteractionLayer
from .envelope import NuclearEnvelope
from .model import FermiNet
from .hamiltonian import MolecularSystem, compute_local_energy
from .mcmc import MCMCSampler
from .mcmc_stats import estimate_mcmc_statistics
from .loss import clip_local_energy, vmc_surrogate_loss, VMCLoss
from .train_vmc import train_vmc, build_h2_system, build_lih_system

__all__ = [
    "ElectronFeatureEmbedding",
    "FermiInteractionLayer",
    "NuclearEnvelope",
    "FermiNet",
    "MolecularSystem",
    "compute_local_energy",
    "MCMCSampler",
    "estimate_mcmc_statistics",
    "clip_local_energy",
    "vmc_surrogate_loss",
    "VMCLoss",
    "train_vmc",
    "build_h2_system",
    "build_lih_system",
]
