"""Editable-install namespace shim for the deliberately flat source tree.

Wheels map the explicitly listed runtime modules directly into this package via
Hatch's ``sources`` prefix.  An editable install instead exposes this small
directory and extends its package search path to the flat checked-out modules.
"""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent.parent))

from .features import ElectronFeatureEmbedding
from .layers import FermiInteractionLayer
from .envelope import NuclearEnvelope
from .model import FermiNet
from .hamiltonian import MolecularSystem, compute_local_energy
from .mcmc import MCMCSampler
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
    "clip_local_energy",
    "vmc_surrogate_loss",
    "VMCLoss",
    "train_vmc",
    "build_h2_system",
    "build_lih_system",
]
