"""Small molecule geometries, in Hartree atomic units."""

import torch
from .hamiltonian import MolecularSystem


def build_h2_system(bond_length=1.4, device=None, dtype=torch.float32):
    coords = torch.tensor(
        [[0.0, 0.0, -bond_length / 2], [0.0, 0.0, bond_length / 2]],
        device=device,
        dtype=dtype,
    )
    return MolecularSystem(coords, torch.ones(2, device=device, dtype=dtype), 1, 1)


def build_lih_system(bond_length=3.015, device=None, dtype=torch.float32):
    coords = torch.tensor(
        [[0.0, 0.0, -bond_length / 2], [0.0, 0.0, bond_length / 2]],
        device=device,
        dtype=dtype,
    )
    return MolecularSystem(
        coords, torch.tensor([3.0, 1.0], device=device, dtype=dtype), 2, 2
    )
