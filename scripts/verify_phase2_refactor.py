"""Offline audit against the archived phase-one implementations.

The conversion below belongs only to this audit, not the runtime/checkpoint API.
Run from the checkout: python -m scripts.verify_phase2_refactor
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.model import FermiNet

ROOT = Path(__file__).resolve().parents[1]


def map_parameters(old, new, architecture):
    source = old.state_dict()
    target = new.state_dict()
    for name, value in source.items():
        key = name
        for spin in ("up", "down"):
            key = key.replace(f"projector_{spin}.linear.", f"linear_orb_{spin}.")
            key = key.replace(f"projector_{spin}.envelope.", f"envelope_{spin}.")
        key = key.replace("res_proj_single", "res_proj_one").replace(
            "res_proj_double", "res_proj_two"
        )
        key = key.replace("linear_double", "linear_two")
        if key in target:
            target[key] = value
    if architecture == "composable":
        for i, layer in enumerate(old.layers):
            blocks = []
            for name, width in (
                ("proj_h1_self", layer.d_in_single),
                ("proj_h1_mean_up", layer.d_in_single),
                ("proj_h1_mean_down", layer.d_in_single),
                ("proj_h2_mean_up", layer.d_in_double),
                ("proj_h2_mean_down", layer.d_in_double),
            ):
                projection = getattr(layer, name)
                blocks.append(
                    projection.weight
                    if projection is not None
                    else torch.zeros(layer.d_out_single, width, dtype=torch.float64)
                )
            target[f"layers.{i}.linear_one.weight"] = torch.cat(blocks, dim=1)
            target[f"layers.{i}.linear_one.bias"] = layer.proj_h1_self.bias
    new.load_state_dict(target)


def main():
    torch.set_num_threads(1)
    source = ROOT / "physics_evidence/phase2_baseline/ferminet_pytorch"
    spec = importlib.util.spec_from_file_location(
        "_phase2_baseline",
        source / "__init__.py",
        submodule_search_locations=[str(source)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    records = []
    for architecture in ("classic", "composable", "psiformer"):
        for up, down in ((1, 0), (1, 1), (2, 2)):
            torch.manual_seed(91)
            nuclei = torch.tensor(
                [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float64
            )
            charges = torch.ones(2, dtype=torch.float64)
            options = dict(
                num_determinants=2,
                hidden_dim_one=16,
                hidden_dim_two=4,
                num_layers=2,
                use_cusp_jastrow=True,
                spin_adapted=up == down,
            )
            if architecture == "psiformer":
                options["interaction_type"] = "psiformer"
            old_cls = (
                package.FermiNet
                if architecture == "classic"
                else package.ComposableFermiNet
            )
            old = old_cls(nuclei, charges, up, down, **options).double()
            new = FermiNet(nuclei, charges, up, down, **options).double()
            map_parameters(old, new, architecture)
            points = torch.randn(3, up + down, 3, dtype=torch.float64) + 0.3
            values = []
            for model, system_cls, energy_fn in (
                (old, package.MolecularSystem, package.compute_local_energy),
                (new, MolecularSystem, compute_local_energy),
            ):
                r = points.clone().requires_grad_()
                log, sign = model.forward_log_psi(r)
                coordinate_gradient = torch.autograd.grad(log.sum(), r)[0]
                energy = energy_fn(
                    model,
                    system_cls(nuclei, charges, up, down),
                    points.clone().requires_grad_(),
                    create_graph=True,
                )[0]
                head = (
                    model.linear_orb_up
                    if architecture == "classic" or model is new
                    else model.projector_up.linear
                )
                parameter_gradient = torch.autograd.grad(energy.sum(), head.weight)[0]
                values.append(
                    [
                        v.detach()
                        for v in (
                            log,
                            sign,
                            coordinate_gradient,
                            energy,
                            parameter_gradient,
                        )
                    ]
                )
            differences = {}
            for name, expected, actual in zip(
                ("log", "sign", "coordinate_gradient", "energy", "parameter_gradient"),
                values[0],
                values[1],
            ):
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
                differences[name] = float((actual - expected).abs().max())
            records.append(
                dict(
                    architecture=architecture,
                    n_up=up,
                    n_down=down,
                    passed=True,
                    max_absolute_differences=differences,
                )
            )
    report = dict(
        records=records,
        torch=torch.__version__,
        device="cpu",
        dtype="float64",
        baseline_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.glob("*.py")
        },
        current_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "ferminet_pytorch").glob("*.py")
        },
    )
    path = ROOT / "physics_evidence/phase2_equivalence.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Passed {len(records)} archived-model comparisons; evidence: {path}")


if __name__ == "__main__":
    main()
