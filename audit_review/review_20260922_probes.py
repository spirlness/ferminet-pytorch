"""Independent review probes; run with uv run python audit_review/review_20260922_probes.py.

Does not modify production code. JSON records observed behavior, not assertions
that the behavior is correct. Small deterministic CPU/float64 experiments.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hamiltonian import MolecularSystem, compute_local_energy
from model import FermiNet
from mcmc import MCMCSampler
from mcmc_stats import estimate_mcmc_statistics
from train_vmc import build_h2_system, build_lih_system, train_vmc
from loss import vmc_surrogate_loss

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(20260922)
results = {"source_commit": subprocess.check_output(
               ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "environment": {"python": sys.version, "torch": torch.__version__,
                           "cuda_available": torch.cuda.is_available()}}


def small(system, **kwargs):
    return FermiNet(system.nuclear_coords, system.nuclear_charges,
                    system.n_up, system.n_down, num_determinants=2,
                    hidden_dim_one=8, hidden_dim_two=4, num_layers=2,
                    **kwargs).double()


def singular_branch():
    system = build_h2_system(dtype=torch.float64)
    model = small(system)
    with torch.no_grad():
        model.linear_orb_up.weight[0].zero_()
        model.linear_orb_up.bias[0].zero_()
    r = torch.tensor([[[0.3, 0.2, -0.5], [-0.2, 0.4, 0.8]]], requires_grad=True)
    lp, sign = model.forward_log_psi(r)
    observed = {"log_psi": lp.tolist(), "sign": sign.tolist()}
    try:
        grad = torch.autograd.grad(lp.sum(), r, retain_graph=True)[0]
        observed["gradient_finite"] = bool(torch.isfinite(grad).all())
    except RuntimeError as exc:
        observed["gradient_error"] = str(exc)
    up, down = model.forward_orbitals(r)
    direct = (model.det_weights * up[..., 0, 0] * down[..., 0, 0]).sum(dim=1)
    direct_grad = torch.autograd.grad(direct.abs().log().sum(), r)[0]
    observed["direct_gradient_finite"] = bool(torch.isfinite(direct_grad).all())
    observed["direct_log_psi"] = direct.abs().log().tolist()
    observed["direct_gradient"] = direct_grad.tolist()
    try:
        e, _, _ = compute_local_energy(model, system, r.detach().requires_grad_(True))
        observed["local_energy"] = e.tolist()
    except RuntimeError as exc:
        observed["local_energy_error"] = str(exc)
    return observed


def checkpoint_groups():
    system = build_lih_system(dtype=torch.float64)
    source = small(system)
    with torch.no_grad():
        for envelope in (source.envelope_up, source.envelope_down):
            envelope.decay_rates_raw[..., 0].fill_(-1.)
            envelope.decay_rates_raw[..., 1].fill_(2.)
    swapped = MolecularSystem(system.nuclear_coords.clone(),
                              system.nuclear_charges.flip(0), 2, 2)
    restored = small(swapped)
    status = restored.load_state_dict(source.state_dict(), strict=True)
    restored.check_system_consistency(system)
    r = torch.randn(4, 4, 3)
    lp1, s1 = source.forward_log_psi(r)
    lp2, s2 = restored.forward_log_psi(r)
    return {"load_status": str(status), "consistency_guard_passed": True,
            "state_dicts_equal": all(torch.equal(v, restored.state_dict()[k])
                                     for k, v in source.state_dict().items()),
            "source_groups": source.envelope_up.nucleus_group_index.tolist(),
            "restored_groups": restored.envelope_up.nucleus_group_index.tolist(),
            "max_log_psi_difference": float((lp1-lp2).abs().max()),
            "max_local_energy_difference": float((
                compute_local_energy(source, system, r.clone().requires_grad_())[0] -
                compute_local_energy(restored, system, r.clone().requires_grad_())[0]
            ).abs().max())}


def spin_mismatch():
    system = build_h2_system(dtype=torch.float64)
    other_spin = MolecularSystem(system.nuclear_coords, system.nuclear_charges, 2, 0)
    model = small(other_spin)
    with warnings.catch_warnings(record=True):
        out = train_vmc(system=system, model=model, num_walkers=8, num_epochs=1,
                        burn_in_steps=2, mcmc_steps_per_epoch=1, evaluation_steps=4,
                        evaluation_mcmc_steps=1, evaluation_burn_in_steps=1,
                        dtype=torch.float64, device="cpu", verbose=False)
    from visualize import get_system_reference
    return {"model_spins": [model.n_up, model.n_down],
            "system_spins": [system.n_up, system.n_down],
            "training_accepted": True, "energy": out["final_energy"],
            "reference_selected": get_system_reference(system).energy}


def anisotropic_rotation():
    system = MolecularSystem(torch.zeros(1, 3), torch.ones(1), 1, 0)
    model = small(system, rotation_invariant=True, anisotropic_envelope=True)
    with torch.no_grad():
        model.envelope_up.decay_rates_raw[..., 0].fill_(0.)
        model.envelope_up.decay_rates_raw[..., 1].fill_(2.)
        model.envelope_up.decay_rates_raw[..., 2].fill_(4.)
    a, _ = model.forward_log_psi(torch.tensor([[[1., 0., 0.]]]))
    b, _ = model.forward_log_psi(torch.tensor([[[0., 1., 0.]]]))
    return {"rotation_invariant_flag": model.rotation_invariant,
            "log_psi_x": a.tolist(), "log_psi_y": b.tolist(),
            "difference": float((a-b).abs().max())}


def geyer_pairs():
    # A reversible product of two stationary Gaussian AR(1) processes.
    # Covariance rho(t)=w*a**t+(1-w)*b**t; g is analytically known.
    time, walkers, a, b, w = 32768, 32, .98, -.8, .05
    generator = torch.Generator().manual_seed(171)
    noise = torch.randn(time, walkers, 2, generator=generator)
    state = torch.randn(walkers, 2, generator=generator)
    rates = torch.tensor([a, b])
    scale = torch.sqrt(1-rates**2)
    series = torch.empty(time, walkers)
    for t in range(time):
        state = state*rates + scale*noise[t]
        series[t] = math.sqrt(w)*state[:, 0] + math.sqrt(1-w)*state[:, 1]
    stats = estimate_mcmc_statistics(series)
    expected_g = w*(1+a)/(1-a)+(1-w)*(1+b)/(1-b)
    from mcmc_stats import _mean_autocorrelation
    rho = _mean_autocorrelation(series)
    pairs = []
    for t in range(0, len(rho)-1, 2):
        pair = float(rho[t]+rho[t+1])
        if pair <= 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    corrected_g = -1+2*sum(pairs)
    return {"expected_g": expected_g, "estimated_g": stats["statistical_inefficiency"],
            "same_data_correct_pairing_g": corrected_g,
            "status": stats["autocorrelation_status"], "r_hat": stats["r_hat"],
            "sem_underestimate_factor": math.sqrt(expected_g/stats["statistical_inefficiency"]),
            "rho1_plus_rho2": w*(a+a*a)+(1-w)*(b+b*b)}


def between_chain_offsets():
    generator = torch.Generator().manual_seed(88)
    offsets = torch.randn(64, generator=generator)*10
    noise = torch.randn(4096, 64, generator=generator)*.01
    stats = estimate_mcmc_statistics(noise + offsets)
    return {"stats": stats, "independent_chain_mean_sem": float(offsets.std()/8),
            "reported_sem": stats["energy_sem"]}


def normalization_scale():
    system = build_h2_system(dtype=torch.float64)
    model = small(system)
    r = torch.randn(5, 2, 3)
    e1 = compute_local_energy(model, system, r.clone().requires_grad_())[0]
    lp1 = model.forward_log_psi(r)[0]
    with torch.no_grad():
        model.det_weights.mul_(1e-45)
    e2 = compute_local_energy(model, system, r.clone().requires_grad_())[0]
    lp2 = model.forward_log_psi(r)[0]
    return {"max_energy_change": float((e1-e2).abs().max()),
            "max_log_shift_error": float((lp2-lp1-math.log(1e-45)).abs().max())}


def positive_infinite_cache():
    class Model(torch.nn.Module):
        def forward_log_psi(self, r):
            lp = -r.square().sum(dim=(-1, -2))
            lp = torch.where(r.square().sum(dim=(-1, -2)) == 0,
                             torch.full_like(lp, float("inf")), lp)
            return lp, torch.ones_like(lp)
    system = MolecularSystem(torch.zeros(1, 3), torch.ones(1), 1, 0)
    sampler = MCMCSampler(Model(), system, num_walkers=4, adapt_step_size=False,
                          dtype=torch.float64, initial_walkers=torch.zeros(4, 1, 3))
    sampler.sample(20)
    return {"last_acceptance_rate": sampler.last_acceptance_rate,
            "all_walkers_unchanged": bool((sampler.walkers == 0).all()),
            "cached_log_psi": sampler.current_log_abs_psi.tolist()}


def analytic_baselines():
    class Hydrogen(torch.nn.Module):
        def forward_log_psi(self, r):
            lp = -torch.linalg.vector_norm(r, dim=-1).sum(dim=-1)
            return lp, torch.ones_like(lp)
    system = MolecularSystem(torch.zeros(1, 3), torch.ones(1), 1, 0)
    r = torch.tensor([[[.3, .4, .5]], [[1., 2., 3.]], [[1e-5, 0., 0.]]])
    hydrogen = {}
    for method in ("exact", "exact_jvp"):
        e, _, _ = compute_local_energy(Hydrogen(), system,
                                        r.clone().requires_grad_(), laplacian_method=method)
        hydrogen[method] = {"energies": e.tolist(), "max_error": float((e+.5).abs().max().detach())}
    theta = torch.tensor(1.2, requires_grad=True)
    x = torch.tensor([.1, .3, .7, 1.1])
    energies = torch.tensor([-1., -.5, .2, 1.])
    loss, _ = vmc_surrogate_loss(theta*x, energies)
    loss.backward()
    expected = 2*((energies-energies.mean())*(x-x.mean())).sum()/(len(x)-1)
    return {"hydrogen": hydrogen, "covariance_gradient_error": float((theta.grad-expected).abs())}


def direct_wavefunction_kinetic():
    torch.manual_seed(54)
    system = build_lih_system(dtype=torch.float64)
    model = small(system)
    r = (torch.randn(4, 4, 3)+.3).requires_grad_()
    up, down = model.forward_orbitals(r)
    psi = (torch.linalg.det(up)*torch.linalg.det(down)*model.det_weights).sum(dim=1)
    grad = torch.autograd.grad(psi.sum(), r, create_graph=True)[0].flatten(1)
    lap = torch.zeros(len(r))
    for d in range(12):
        second = torch.autograd.grad(grad[:, d].sum(), r, retain_graph=True)[0]
        lap += second.flatten(1)[:, d]
    direct_t = -.5*lap/psi
    observed = {}
    for method in ("exact", "exact_jvp"):
        _, t, _ = compute_local_energy(model, system, r.detach().requires_grad_(),
                                       laplacian_method=method)
        observed[method+"_max_abs_error"] = float((t-direct_t).abs().max().detach())
    # Independent scalar-loop Coulomb summation in Python doubles.
    coords = system.nuclear_coords.tolist()
    charges = system.nuclear_charges.tolist()
    distance = lambda x, y: math.sqrt(sum((a-b)**2 for a, b in zip(x, y)))
    expected = []
    for electrons in r.tolist():
        en = -sum(z/distance(e, c) for e in electrons for z, c in zip(charges, coords))
        ee = sum(1/distance(electrons[i], electrons[j]) for i in range(4) for j in range(i+1, 4))
        nn = charges[0]*charges[1]/distance(coords[0], coords[1])
        expected.append(en+ee+nn)
    observed["potential_max_abs_error"] = float((system.compute_potential_energy(r)-torch.tensor(expected)).abs().max().detach())
    return observed


def atomic_parity():
    torch.manual_seed(93)
    system = MolecularSystem(torch.zeros(1, 3), torch.tensor([5.]), 3, 2)
    model = small(system)
    r = torch.randn(6, 5, 3)
    log1, sign1 = model.forward_log_psi(r)
    log2, sign2 = model.forward_log_psi(-r)
    return {"system": "B: Z=5, n_up=3, n_down=2", "rotation_invariant": model.rotation_invariant,
            "all_nonzero": bool((sign1 != 0).all()),
            "inversion_log_difference": float((log1-log2).abs().max().detach()),
            "inversion_signs_same": bool(torch.equal(sign1, sign2)),
            "required_for_odd_parity": "psi(-r) = -psi(r); default distances force psi(-r) = psi(r)"}


def wheel_probe():
    dist = Path(tempfile.gettempdir()) / "ferminet_review_dist_20260922"
    wheels = list(dist.glob("*.whl"))
    if not wheels:
        return {"skipped": "Run uv build --out-dir <TEMP>/ferminet_review_dist_20260922 first"}
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    with tempfile.TemporaryDirectory(prefix="ferminet_wheel_review_") as install_root:
        install = subprocess.run(["uv", "pip", "install", "--no-deps", "--target", install_root,
                                  str(wheel)], capture_output=True, text=True)
        code = ("import sys; sys.path.insert(0, " + repr(install_root) +
                "); import ferminet_pytorch; print(ferminet_pytorch.__file__)")
        proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=tempfile.gettempdir(),
                               capture_output=True, text=True)
    return {"wheel": str(wheel), "first_20_members": names[:20],
            "package_members": [n for n in names if n.startswith("ferminet_pytorch/")],
            "install_exit_code": install.returncode,
            "import_exit_code": proc.returncode, "stderr": proc.stderr}


if __name__ == "__main__":
    for name, probe in [("analytic_baselines", analytic_baselines),
                        ("singular_branch", singular_branch),
                        ("checkpoint_groups", checkpoint_groups),
                        ("spin_mismatch", spin_mismatch),
                        ("anisotropic_rotation", anisotropic_rotation),
                        ("geyer_pairs", geyer_pairs),
                        ("between_chain_offsets", between_chain_offsets),
                        ("normalization_scale", normalization_scale),
                        ("positive_infinite_cache", positive_infinite_cache),
                        ("direct_wavefunction_kinetic", direct_wavefunction_kinetic),
                        ("atomic_parity", atomic_parity),
                        ("wheel", wheel_probe)]:
        try:
            results[name] = probe()
        except Exception as exc:
            results[name] = {"error": type(exc).__name__, "message": str(exc)}
    destination = Path(__file__).with_name("review_20260922_results.json")
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))
