"""Locate rare local energies and cross-check clipped descent at fixed weights.

Every replica starts independently. Energy, geometry and gradient blocks retain
their replica/time axes. No optimizer is created and no observations are removed.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch

from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.loss import clip_local_energy
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.vmc_statistics import energy_statistics

RULES = ("raw", "current", "paper_hard")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), allow_nan=False), flush=True)


def gradient_vectors(model, positions, energies):
    """Current training coefficients and an unbiased IID sample-covariance raw rule.

    The B/(B-1) factor corrects the raw empirical centering for independent walkers.
    The clipped rule matches training exactly, including its batch-dependent scale.
    """
    energies = energies.detach()
    current, median, _ = clip_local_energy(energies, 5, "tanh")
    mean_ad = (energies - median).abs().mean()
    paper = energies.clamp(median - 5 * mean_ad, median + 5 * mean_ad)
    size = len(energies)
    coefficients = (
        (energies - energies.mean()) * size / (size - 1),
        current - current.mean(),
        paper - paper.mean(),
    )
    log, _ = model.forward_log_psi(positions.detach())
    parameters = list(model.parameters())
    vectors = []
    for index, coefficient in enumerate(coefficients):
        loss = 2 * (coefficient.detach() * log).mean()
        grads = torch.autograd.grad(
            loss, parameters, allow_unused=True, retain_graph=index < 2
        )
        vector = torch.cat([
            (torch.zeros_like(p) if g is None else g).flatten()
            for p, g in zip(parameters, grads)
        ]).detach()
        if not torch.isfinite(vector).all():
            raise FloatingPointError("Nonfinite frozen gradient")
        vectors.append(vector)
    return torch.stack(vectors)


def off_diagonal_dot(a, b):
    """U statistic: independent replicas only, never a same-batch dot product."""
    count = len(a)
    dots = a @ b.T
    return float((dots.sum() - dots.diagonal().sum()) / (count * (count - 1)))


def cross_gradient_statistics(replica_gradients):
    """Small-replica jackknife intervals are approximate, not certificates."""
    x = replica_gradients.double().cpu()
    count = len(x)
    if count < 4 or x.ndim != 3 or x.shape[1] != len(RULES):
        raise ValueError("Need >=4 independent replicas, three rules, parameter axis")
    # Student-t critical values for df=n-1. Larger n falls back conservatively.
    critical = {4: 3.182446, 5: 2.776445, 6: 2.570582, 7: 2.446912,
                8: 2.364624, 9: 2.306004, 10: 2.262157}.get(count, 3.182446)

    def estimate(a, b):
        value = off_diagonal_dot(a, b)
        leave = []
        for i in range(count):
            keep = [j for j in range(count) if j != i]
            leave.append(off_diagonal_dot(a[keep], b[keep]))
        values = torch.tensor(leave, dtype=torch.float64)
        se = math.sqrt(float((count-1)/count * (values-values.mean()).square().sum()))
        return dict(mean=value, jackknife_se=se,
                    ci95_approx=[value-critical*se, value+critical*se],
                    leave_one_replica_out=leave)

    raw = x[:, 0]
    result = {
        "replicas": count,
        "raw_signal_squared": estimate(raw, raw),
        "rules": {},
        "interval_method": "Replica jackknife with t critical value; approximate under heavy tails, only four independent replicas in the default run.",
        "criterion": "For direction -g_rule, first-order raw energy change is -g_population dot g_rule. Positive cross dot supports descent; negative supports ascent.",
    }
    for index, rule in enumerate(RULES[1:], 1):
        g = x[:, index]
        dots = estimate(raw, g)
        lower, upper = dots["ci95_approx"]
        signal_resolved = result["raw_signal_squared"]["ci95_approx"][0] > 0
        verdict = (
            "descent_supported" if lower > 0 and signal_resolved else
            "ascent_supported" if upper < 0 and signal_resolved else "inconclusive"
        )
        aggregate_cosine = float(torch.nn.functional.cosine_similarity(raw.mean(0), g.mean(0), dim=0))
        loo = []
        for i in range(count):
            reference = raw[[j for j in range(count) if j != i]].mean(0)
            loo.append(dict(replica=i, dot=float(reference @ g[i]),
                            cosine=float(torch.nn.functional.cosine_similarity(reference, g[i], dim=0))))
        result["rules"][rule] = dict(
            cross_dot=dots, verdict=verdict, aggregate_cosine=aggregate_cosine,
            leave_one_replica_out_direction=loo,
            rule_signal_squared=estimate(g, g),
            mean_gradient_norm=float(g.mean(0).norm()),
            mean_raw_gradient_norm=float(raw.mean(0).norm()),
        )
    result["between_replica_raw_cosines"] = [
        float(torch.nn.functional.cosine_similarity(raw[i], raw[j], dim=0))
        for i in range(count) for j in range(i+1, count)
    ]
    return result


@torch.no_grad()
def geometry_observations(model, positions):
    distances = (positions[:, :, None] - model.nuclear_coords[None, None]).norm(dim=-1)
    nearest = distances.min(-1).values
    up, down = model.forward_orbitals(torch.cat((positions, positions.flip(1))))
    terms = up[..., 0, 0] * down[..., 0, 0] * model.det_weights
    direct, swapped = terms.chunk(2)
    terms = .5 * torch.cat((direct, swapped), dim=1)
    cancellation = terms.abs().sum(1) / terms.sum(1).abs()
    log, sign = model.forward_log_psi(positions)
    return dict(
        min_en=nearest.min(1).values,
        max_nearest_nucleus=nearest.max(1).values,
        ee=(positions[:, 0] - positions[:, 1]).norm(dim=-1),
        cancellation=cancellation, log_abs_psi=log, sign=sign,
    )


def direct_kinetic_h2(model, positions):
    """Independent 1x1 orbital product and explicit singlet amplitude derivative."""
    r = positions.detach().clone().requires_grad_(True)

    def base(coords):
        up, down = model.forward_orbitals(coords)
        amplitude = (up[..., 0, 0]*down[..., 0, 0]*model.det_weights).sum(-1)
        return amplitude * model.jastrow(coords, model.nuclear_coords).exp()

    amplitude = .5 * (base(r) + base(r.flip(1)))
    gradient = torch.autograd.grad(amplitude.sum(), r, create_graph=True)[0].reshape(-1, 6)
    laplacian = torch.zeros_like(amplitude)
    for dimension in range(6):
        second = torch.autograd.grad(gradient[:, dimension].sum(), r, retain_graph=dimension < 5)[0]
        laplacian += second.reshape(-1, 6)[:, dimension]
    return (-.5 * laplacian / amplitude).detach()


def summarize_geometry(data, reference=-1.174475714220):
    e = data["energy"].flatten().double()
    residual = e-reference
    squared = (e-e.mean()).square()
    flags = {
        "near_nucleus_lt_0.05": data["min_en"].flatten() < .05,
        "near_electron_lt_0.1": data["ee"].flatten() < .1,
        "far_nearest_nucleus_gt_3": data["max_nearest_nucleus"].flatten() > 3,
        "cancellation_gt_1000": data["cancellation"].flatten() > 1000,
    }
    extreme = residual.abs() > .5
    regions = {}
    for name, flag in flags.items():
        regions[name] = dict(count=int(flag.sum()),
            extreme_count=int((flag & extreme).sum()),
            variance_share=float(squared[flag].sum()/squared.sum()),
            signed_error_contribution_mHa=float(residual[flag].sum()/len(e)*1000))
    k = max(1, int(len(e)*.0001))
    indices = squared.topk(k).indices
    top_flags = {name: int(flag[indices].sum()) for name, flag in flags.items()}
    nearest_bins = []
    distance = data["max_nearest_nucleus"].flatten()
    for lower, upper in zip([0,1,2,3,4,5,6], [1,2,3,4,5,6,float("inf")]):
        flag = (distance >= lower) & (distance < upper)
        nearest_bins.append(dict(lower=lower, upper=None if math.isinf(upper) else upper,
            count=int(flag.sum()), extreme_count=int((flag & extreme).sum()),
            variance_share=float(squared[flag].sum()/squared.sum()),
            signed_error_contribution_mHa=float(residual[flag].sum()/len(e)*1000)))
    return dict(samples=len(e), extreme_threshold_abs_error_Ha=.5,
                extreme_count=int(extreme.sum()), regions_overlap=True, regions=regions,
                top_0_01_percent=dict(count=k, variance_share=float(squared[indices].sum()/squared.sum()), flags=top_flags),
                far_distance_bins=nearest_bins,
                sign_counts={str(s): int((data["sign"]==s).sum()) for s in [-1,0,1]})


def load_model(path, case, device):
    expected = case["checkpoint_sha256"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Checkpoint identity mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["model_state_dict"]
    coords, charges = state["nuclear_coords"].double(), state["nuclear_charges"].double()
    model = FermiNet(coords, charges, 1, 1, num_determinants=8,
        hidden_dim_one=64, hidden_dim_two=16, num_layers=3,
        use_cusp_jastrow=True, spin_adapted=True, interaction_type=case["interaction"]).double()
    model.load_state_dict(state)
    model = model.to(device).eval()
    system = MolecularSystem(coords.to(device), charges.to(device), 1, 1)
    return model, system


def sample_replica(model, system, case_name, index, seed, args, output):
    start = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sampler = MCMCSampler(model, system, num_walkers=args.walkers,
        step_size=.5, device=args.device, dtype=torch.float64)
    sampler.burn_in(args.burn_in)
    sampler.adapt_step_size = False
    proposal = sampler.step_size
    fields = {key: [] for key in ["positions","energy","kinetic","potential",
        "min_en","max_nearest_nucleus","ee","cancellation","log_abs_psi","sign"]}
    gradient_blocks, accumulator = [], None
    batch_directions = []
    for record in range(args.records):
        r = sampler.sample(args.decorrelation, adapt=False)
        energy, kinetic, potential = compute_local_energy(model, system, r.clone().requires_grad_(True))
        energy, kinetic, potential = energy.detach(), kinetic.detach(), potential.detach()
        if not torch.isfinite(energy).all():
            raise FloatingPointError("Nonfinite energy; diagnostic stopped without discarding observations")
        geometry = geometry_observations(model, r)
        values = dict(positions=r, energy=energy, kinetic=kinetic, potential=potential, **geometry)
        for name, value in values.items():
            fields[name].append(value.detach().cpu())
        gradient = gradient_vectors(model, r, energy)
        accumulator = gradient.clone() if accumulator is None else accumulator + gradient
        batch_directions.append(dict(
            record=record,
            current_cosine=float(torch.nn.functional.cosine_similarity(gradient[0],gradient[1],dim=0)),
            current_dot_raw=float(gradient[0] @ gradient[1]),
            raw_norm=float(gradient[0].norm()), current_norm=float(gradient[1].norm()),
        ))
        if (record+1) % args.block_records == 0:
            gradient_blocks.append((accumulator/args.block_records).cpu())
            accumulator = None
        if (record+1) % 16 == 0 or record+1 == args.records:
            emit("sampling", case=case_name, replica=index, record=record+1,
                records=args.records, seconds=round(time.perf_counter()-start,2),
                extreme_count=sum(int(((v+1.174475714220).abs()>.5).sum()) for v in fields["energy"]))
    assert proposal == sampler.step_size
    data = {name:torch.stack(values) for name,values in fields.items()}
    gradients = torch.stack(gradient_blocks)
    torch.save(data, output/f"replica_{index}_observations.pt")
    torch.save(gradients, output/f"replica_{index}_gradient_blocks.pt")
    stats = energy_statistics(data["energy"])
    result = dict(seed=seed, records=args.records, walkers=args.walkers,
        step_size=proposal, production_adaptation=False,
        cumulative_acceptance=sampler.acceptance_rate,
        seconds=time.perf_counter()-start, statistics=stats,
        geometry=summarize_geometry(data), batch_directions=batch_directions)
    write_json(output/f"replica_{index}.json", result)
    return data, gradients, result


def run_case(case, args, output):
    name = f"{case['interaction']}_seed{case['seed']}"
    folder = output/name
    folder.mkdir(exist_ok=True)
    model, system = load_model(args.checkpoints/name/"final_checkpoint.pt", case, args.device)
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    datasets, gradients, replicas = [], [], []
    for index in range(args.replicas):
        prefix = folder/f"replica_{index}.json"
        if prefix.exists():
            datasets.append(torch.load(folder/f"replica_{index}_observations.pt",weights_only=True))
            gradients.append(torch.load(folder/f"replica_{index}_gradient_blocks.pt",weights_only=True))
            replicas.append(json.loads(prefix.read_text()))
        else:
            seed = args.seed + 10000*(case['interaction']=='psiformer') + 100*case['seed']+index
            data, blocks, result = sample_replica(model,system,name,index,seed,args,folder)
            datasets.append(data);gradients.append(blocks);replicas.append(result)
    data = {key:torch.stack([d[key] for d in datasets]) for key in datasets[0]}
    gradient_blocks = torch.stack(gradients)
    gradient_stats = cross_gradient_statistics(gradient_blocks.mean(1))
    residual = data['energy'].flatten()+1.174475714220
    indices = residual.abs().topk(min(64,len(residual))).indices
    extremes = {key:value.reshape(-1,*value.shape[3:])[indices] for key,value in data.items()}
    # Geometry/energy values have three axes; positions have a fourth/fifth axis.
    r = extremes['positions'].to(args.device)
    independent = direct_kinetic_h2(model,r)
    log_e,_,_ = compute_local_energy(model,system,r.clone().requires_grad_(True))
    low = copy.deepcopy(model).float()
    low_system = copy.deepcopy(system).to(dtype=torch.float32)
    low_e,_,_ = compute_local_energy(low,low_system,r.float().requires_grad_(True))
    audit = dict(direct_kinetic_max_error_Ha=float((independent.cpu()-extremes['kinetic']).abs().max()),
        fp32_vs_fp64_energy_max_difference_Ha=float((low_e.detach().double()-log_e.detach()).abs().max()),
        fp32_vs_fp64_energy_rms_difference_Ha=float((low_e.detach().double()-log_e.detach()).square().mean().sqrt()))
    extremes['flat_index']=indices
    torch.save(extremes,folder/'extreme_configurations.pt')
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    result=dict(case=name,checkpoint_sha256=case['checkpoint_sha256'],
        frozen_model_unchanged=True,optimizer_updates=0,
        total_samples=int(data['energy'].numel()),
        geometry=summarize_geometry(data),cross_gradients=gradient_stats,
        extreme_audit=audit,
        replicas=[{k:v for k,v in r.items() if k!='batch_directions'} for r in replicas],
        limits=['Four independent replicas limit inference; jackknife intervals are approximate.',
                'Raw-gradient sample covariance may have heavy-tail noise.',
                'Gradient descent signs refer to -gradient, not the Adam-preconditioned update.',
                'Geometry flags may overlap and do not alone identify a causal mechanism.'])
    write_json(folder/'result.json',result)
    emit('case_completed',case=name,geometry=result['geometry'],cross_gradients=gradient_stats,extreme_audit=audit)
    del model, data, datasets, initial, low
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',type=Path,default=Path('campaign_runs/continue_diagnosis_20261023/ferminet_results'))
    parser.add_argument('--summary',type=Path,default=Path('campaign_runs/continue_logs_20261023_2020/ferminet_results/summary.json'))
    parser.add_argument('--output',type=Path,default=Path('campaign_runs/frozen_tail_gradient_study_20261023'))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--replicas',type=int,default=4)
    parser.add_argument('--records',type=int,default=128)
    parser.add_argument('--walkers',type=int,default=512)
    parser.add_argument('--burn-in',type=int,default=1000)
    parser.add_argument('--decorrelation',type=int,default=20)
    parser.add_argument('--block-records',type=int,default=16)
    parser.add_argument('--seed',type=int,default=710000)
    parser.add_argument('--cases',nargs='*')
    args=parser.parse_args()
    if args.replicas<4 or args.walkers<2 or args.records<1 or args.records%args.block_records:
        parser.error('Require >=4 replicas, >=2 walkers, positive records divisible by block size')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity=args.output/'plan.json'
    if identity.exists() and json.loads(identity.read_text())!=settings:
        raise ValueError('Diagnostic resume requires exactly the same plan and script')
    write_json(identity,settings)
    write_json(args.output/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
    cases=json.loads(args.summary.read_text())['cases']
    if args.cases:
        cases=[c for c in cases if f"{c['interaction']}_seed{c['seed']}" in args.cases]
    results=[]
    for case in cases:
        target=args.output/f"{case['interaction']}_seed{case['seed']}"/'result.json'
        if target.exists():
            results.append(json.loads(target.read_text()))
        else:
            results.append(run_case(case,args,args.output))
        write_json(args.output/'summary.json',dict(cases=results,all_completed=len(results)==len(cases)))
    emit('study_completed',cases=len(results))


if __name__=='__main__':
    main()
