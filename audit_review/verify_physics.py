"""Independent numerical checks for the FermiNet scientific review.

Run from the repository root with the selected PyTorch environment.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envelope import NuclearEnvelope
from hamiltonian import MolecularSystem, compute_local_energy
from loss import vmc_surrogate_loss
from model import FermiNet
from visualize import get_system_label, get_system_reference_energy


OUTPUT = ROOT / "audit_review" / "results.json"


class Hydrogen1s(nn.Module):
    """Exact hydrogen 1s log amplitude in atomic units."""

    def forward_log_psi(self, r: torch.Tensor):
        radius = torch.linalg.vector_norm(r[:, 0, :], dim=-1)
        return -radius, torch.ones_like(radius)


class QuadraticLogPsi(nn.Module):
    """log|psi| = -1/2 x^T A x for an analytic Hessian."""

    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward_log_psi(self, r: torch.Tensor):
        x = r.reshape(r.shape[0], -1)
        logp = -0.5 * torch.einsum("bi,ij,bj->b", x, self.matrix, x)
        return logp, torch.ones_like(logp)


def tensor_values(x: torch.Tensor):
    return [float(v) for v in x.detach().cpu().reshape(-1)]


def main() -> None:
    torch.manual_seed(20260921)
    np.random.seed(20260921)
    started = time.perf_counter()
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, object] = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    }

    # Exact hydrogen 1s: local energy is -0.5 Ha away from the nucleus.
    h_system = MolecularSystem(
        torch.zeros(1, 3, dtype=dtype, device=device),
        torch.ones(1, dtype=dtype, device=device),
        1,
        0,
    )
    radii = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=dtype, device=device)
    r_h = torch.zeros(4, 1, 3, dtype=dtype, device=device)
    r_h[:, 0, 0] = radii
    r_h.requires_grad_(True)
    e_h, t_h, v_h = compute_local_energy(Hydrogen1s().to(device), h_system, r_h)
    results["hydrogen_1s"] = {
        "radii": tensor_values(radii),
        "energy": tensor_values(e_h),
        "kinetic": tensor_values(t_h),
        "potential": tensor_values(v_h),
        "max_abs_error_from_minus_half": float((e_h + 0.5).abs().max().detach()),
    }

    # Quantify the Hamiltonian modification introduced by sqrt(r^2 + eps).
    near = torch.tensor([1e-6, 1e-5, 1e-4, 1e-3], dtype=dtype, device=device)
    r_near = torch.zeros(4, 1, 3, dtype=dtype, device=device)
    r_near[:, 0, 0] = near
    softened = h_system.compute_potential_energy(r_near)
    exact_coulomb = -1.0 / near
    at_nucleus = h_system.compute_potential_energy(torch.zeros(1, 1, 3, dtype=dtype, device=device))
    results["softened_coulomb"] = {
        "radii": tensor_values(near),
        "implemented": tensor_values(softened),
        "exact": tensor_values(exact_coulomb),
        "relative_error": tensor_values((softened - exact_coulomb).abs() / exact_coulomb.abs()),
        "implemented_at_nucleus": float(at_nucleus.item()),
    }

    # Exact reverse Laplacian versus torch.func JVP; record any fallback warning.
    matrix = torch.tensor(
        [[1.7, 0.4, -0.2], [0.4, 1.2, 0.3], [-0.2, 0.3, 0.9]],
        dtype=dtype,
        device=device,
    )
    quad = QuadraticLogPsi(matrix).to(device)
    zero_system = MolecularSystem(
        torch.zeros(1, 3, dtype=dtype, device=device),
        torch.zeros(1, dtype=dtype, device=device),
        1,
        0,
    )
    r_q = torch.tensor(
        [[[0.3, -0.2, 0.5]], [[-0.7, 0.4, 0.1]]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    e_exact, t_exact, _ = compute_local_energy(quad, zero_system, r_q, laplacian_method="exact")
    r_jvp = r_q.detach().clone().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e_jvp, t_jvp, _ = compute_local_energy(quad, zero_system, r_jvp, laplacian_method="exact_jvp")
    x_q = r_q.detach().reshape(2, 3)
    grad_q = -(x_q @ matrix.T)
    analytic_t = 0.5 * torch.trace(matrix) - 0.5 * torch.sum(grad_q**2, dim=1)
    results["laplacian_crosscheck"] = {
        "analytic_kinetic": tensor_values(analytic_t),
        "exact_kinetic": tensor_values(t_exact),
        "jvp_kinetic": tensor_values(t_jvp),
        "exact_max_abs_error": float((t_exact - analytic_t).abs().max()),
        "jvp_max_abs_error": float((t_jvp - analytic_t).abs().max()),
        "jvp_fallback_warning": any("回退" in str(w.message) for w in caught),
        "warnings": [str(w.message) for w in caught],
    }

    # Hutchinson is unbiased for the trace, but a one-probe local estimate is noisy.
    hutch_values = []
    for _ in range(500):
        r_probe = r_q[:1].detach().clone().requires_grad_(True)
        _, t_probe, _ = compute_local_energy(
            quad,
            zero_system,
            r_probe,
            laplacian_method="hutchinson",
            num_hutchinson_samples=1,
            hutchinson_distribution="rademacher",
        )
        hutch_values.append(float(t_probe.item()))
    hutch = np.asarray(hutch_values)
    target_t = float(analytic_t[0].item())
    results["hutchinson_500_repetitions"] = {
        "target_kinetic": target_t,
        "mean": float(hutch.mean()),
        "std": float(hutch.std(ddof=1)),
        "mean_error": float(hutch.mean() - target_t),
        "standard_error_of_repetition_mean": float(hutch.std(ddof=1) / math.sqrt(len(hutch))),
        "min": float(hutch.min()),
        "max": float(hutch.max()),
    }

    # Translation invariance should hold; simultaneous rotation is not built into this network.
    nuclei = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=dtype, device=device)
    charges = torch.ones(2, dtype=dtype, device=device)
    net = FermiNet(nuclei, charges, 1, 1, num_determinants=4, hidden_dim_one=16, hidden_dim_two=8).to(device, dtype)
    coords = torch.tensor(
        [[[0.2, -0.4, -0.8], [-0.3, 0.5, 0.9]], [[0.7, 0.1, -0.2], [-0.6, -0.4, 0.3]]],
        dtype=dtype,
        device=device,
    )
    base, _ = net.forward_log_psi(coords)
    shift = torch.tensor([1.3, -0.9, 0.4], dtype=dtype, device=device)
    old_nuclei = net.nuclear_coords.detach().clone()
    net.nuclear_coords = old_nuclei + shift
    translated, _ = net.forward_log_psi(coords + shift)
    theta = math.pi / 3
    rotation = torch.tensor(
        [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    net.nuclear_coords = old_nuclei @ rotation.T
    rotated, _ = net.forward_log_psi(coords @ rotation.T)
    net.nuclear_coords = old_nuclei
    results["symmetry"] = {
        "translation_max_abs_logpsi_change": float((base - translated).abs().max()),
        "rotation_max_abs_logpsi_change": float((base - rotated).abs().max()),
        "base": tensor_values(base),
        "rotated": tensor_values(rotated),
    }

    # The smoothed radial envelope has zero derivative exactly at a nucleus.
    env = NuclearEnvelope(1, 1, 1, eps=1e-8).to(device, dtype)
    with torch.no_grad():
        env.decay_rates_raw.fill_(math.log(math.expm1(1.0)))
        env.weights.fill_(1.0)
    r_zero = torch.zeros(1, 1, 3, dtype=dtype, device=device, requires_grad=True)
    env_value = env(r_zero, torch.zeros(1, 3, dtype=dtype, device=device)).sum()
    env_grad = torch.autograd.grad(env_value, r_zero)[0]
    results["envelope_at_nucleus"] = {
        "value": float(env_value.item()),
        "cartesian_gradient": tensor_values(env_grad),
        "gradient_norm": float(torch.linalg.vector_norm(env_grad).item()),
        "expected_unsmoothed_radial_right_derivative_alpha1": -1.0,
    }

    # A determinant coefficient at exactly zero receives no gradient and cannot revive.
    zero_weight_net = FermiNet(nuclei, charges, 1, 1, num_determinants=2, hidden_dim_one=8, hidden_dim_two=4).to(device, dtype)
    with torch.no_grad():
        zero_weight_net.det_weights.copy_(torch.tensor([1.0, 0.0], dtype=dtype, device=device))
    logp, _ = zero_weight_net.forward_log_psi(coords[:1])
    zero_weight_net.zero_grad(set_to_none=True)
    logp.sum().backward()
    results["zero_determinant_weight"] = {
        "weights": tensor_values(zero_weight_net.det_weights),
        "gradients": tensor_values(zero_weight_net.det_weights.grad),
    }

    # Same-sample centering produces the known (B-1)/B covariance factor.
    covariance_bias = {}
    for batch in (2, 4, 8, 32):
        estimates = []
        for _ in range(20000):
            observable = torch.randn(batch, dtype=dtype)
            energy = observable.clone()
            estimates.append(float(torch.mean((energy - energy.mean()) * observable)))
        observed = float(np.mean(estimates))
        covariance_bias[str(batch)] = {
            "observed": observed,
            "population_target": 1.0,
            "theory": (batch - 1) / batch,
        }
    results["same_batch_centering_bias"] = covariance_bias

    # Clipping transforms the score-function coefficient and therefore changes its gradient.
    log_score = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, requires_grad=True)
    energies = torch.tensor([-2.0, -1.0, 0.0, 1.0, 100.0], dtype=dtype)
    loss_none, _ = vmc_surrogate_loss(log_score, energies, clip_method="none")
    grad_none = torch.autograd.grad(loss_none, log_score, retain_graph=True)[0]
    loss_tanh, _ = vmc_surrogate_loss(log_score, energies, clip_method="tanh")
    grad_tanh = torch.autograd.grad(loss_tanh, log_score)[0]
    results["clipping_changes_gradient"] = {
        "gradient_no_clip": tensor_values(grad_none),
        "gradient_tanh": tensor_values(grad_tanh),
        "max_abs_difference": float((grad_none - grad_tanh).abs().max()),
    }

    # Naive SEM ignores serial correlation; show the analytic AR(1) inflation factor.
    rho = 0.9
    n = 200000
    noise = np.random.randn(n) * math.sqrt(1.0 - rho**2)
    series = np.empty(n)
    series[0] = noise[0]
    for i in range(1, n):
        series[i] = rho * series[i - 1] + noise[i]
    naive_sem = float(series.std(ddof=1) / math.sqrt(n))
    block_size = 500
    block_means = series[: n // block_size * block_size].reshape(-1, block_size).mean(axis=1)
    block_sem = float(block_means.std(ddof=1) / math.sqrt(len(block_means)))
    results["correlated_sem"] = {
        "rho": rho,
        "n": n,
        "naive_sem": naive_sem,
        "block_sem_500": block_sem,
        "ratio_block_to_naive": block_sem / naive_sem,
        "asymptotic_inflation": math.sqrt((1 + rho) / (1 - rho)),
    }

    # Reference lookup ignores charges and geometry.
    non_h2 = MolecularSystem(
        torch.tensor([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]], dtype=dtype),
        torch.tensor([2.0, 1.0], dtype=dtype),
        1,
        1,
    )
    results["reference_inference_counterexample"] = {
        "charges": tensor_values(non_h2.nuclear_charges),
        "separation_bohr": 4.0,
        "label": get_system_label(non_h2),
        "reference_energy": get_system_reference_energy(non_h2),
    }

    if device.type == "cuda":
        torch.cuda.synchronize()
    results["elapsed_seconds"] = time.perf_counter() - started
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
