"""Read-only diagnostics of archived checkpoints; no optimizer or MCMC updates."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.loss import clip_local_energy

OUT = Path(__file__).resolve().parent
DATA = ROOT / 'campaign_runs/continue_diagnosis_20261023'


def scalar(x):
    return float(x.detach().cpu())


def quantiles(x):
    return torch.quantile(x.detach().double().cpu().flatten(), torch.tensor([0., .5, .9, .99, 1.], dtype=torch.float64)).tolist()


def flat_grad(model, coords, centered_energy):
    log, _ = model.forward_log_psi(coords)
    loss = 2 * (centered_energy.detach() * log).mean()
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, grads)])


def compare_grad(a, b):
    return dict(cosine=scalar(F.cosine_similarity(a, b, dim=0)), norm_ratio=scalar(a.norm()/b.norm()), relative_difference=scalar((a-b).norm()/b.norm()))


def direct_h2_amplitude(model, r):
    # H2 spin blocks are 1x1; avoid signed-log-determinant and spin-logsum paths.
    def base(coords):
        up, down = model.forward_orbitals(coords)
        amplitude = (up[..., 0, 0] * down[..., 0, 0] * model.det_weights).sum(-1)
        return amplitude * model.jastrow(coords, model.nuclear_coords).exp()
    return .5 * (base(r) + base(r.flip(1)))


def direct_kinetic(model, r):
    r = r.detach().clone().requires_grad_(True)
    psi = direct_h2_amplitude(model, r)
    g = torch.autograd.grad(psi.sum(), r, create_graph=True)[0].reshape(-1, 6)
    lap = torch.zeros_like(psi)
    for d in range(6):
        lap += torch.autograd.grad(g[:, d].sum(), r, retain_graph=d<5)[0].reshape(-1, 6)[:, d]
    return (-.5 * lap / psi).detach()


def run_case(path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Checkpoint SHA mismatch: {path}")
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['model_state_dict']
    name = path.parent.name
    kind = name.split('_seed')[0]
    coords, charges = state['nuclear_coords'].double(), state['nuclear_charges'].double()
    model = FermiNet(coords, charges, 1, 1, num_determinants=8, hidden_dim_one=64, hidden_dim_two=16, num_layers=3, use_cusp_jastrow=True, spin_adapted=True, interaction_type=kind).double()
    model.load_state_dict(state)
    model = model.cuda().eval()
    system = MolecularSystem(coords.cuda(), charges.cuda(), 1, 1)
    r = saved['walkers'].detach().double().cuda()
    before = {k: v.detach().clone() for k,v in model.state_dict().items()}
    e, t, v = compute_local_energy(model, system, r.clone().requires_grad_(True))
    e, t, v = e.detach(), t.detach(), v.detach()
    current, median, mad = clip_local_energy(e, 5, 'tanh')
    meanad = (e-median).abs().mean()
    paper = e.clamp(median-5*meanad, median+5*meanad)
    raw_grad = flat_grad(model, r, e-e.mean())
    current_grad = flat_grad(model, r, current-current.mean())
    paper_grad = flat_grad(model, r, paper-paper.mean())
    delta = max(5*scalar(mad), .001, .01*abs(scalar(median)))
    history = saved['history_rows']
    floor_counts = {}
    for label, rows in [('all',history),('continuation',history[20000:]),('last5000',history[-5000:])]:
        floor_counts[label] = dict(records=len(rows), active=sum(5*x['energy_mad'] < max(.001,.01*abs(x['energy_median'])) for x in rows))
    hist = history[-5000:]
    distances = (r[:,:,None,:] - model.nuclear_coords[None,None,:,:]).norm(dim=-1)
    ee = (r[:,0]-r[:,1]).norm(dim=-1)
    with torch.no_grad():
        log,sign = model.forward_log_psi(r)
        midpoint = model.nuclear_coords.mean(0)
        axis = F.normalize(model.nuclear_coords[1]-model.nuclear_coords[0],dim=0)
        centered = r-midpoint
        rotated = midpoint + torch.cross(axis.expand_as(centered), centered, dim=-1) + (centered*axis).sum(-1,keepdim=True)*axis
        inversion_log, inversion_sign = model.forward_log_psi(2*midpoint-r)
        rotate_log, rotate_sign = model.forward_log_psi(rotated)
        swap_log, swap_sign = model.forward_log_psi(r.flip(1))
    indices = torch.argsort(e)
    extremes = []
    for i in torch.cat((indices[:5], indices[-5:])).tolist():
        extremes.append(dict(index=i, energy=scalar(e[i]), kinetic=scalar(t[i]), potential=scalar(v[i]), min_en=scalar(distances[i].min()), max_nearest_nucleus=scalar(distances[i].min(-1).values.max()), ee=scalar(ee[i]), logabs=scalar(log[i]), sign=scalar(sign[i])))
    k_direct = direct_kinetic(model, r[:16])
    # Far-field probes are deterministic configurations, NOT distribution averages.
    radii = torch.tensor([1.,2.,3.,4.,6.,8.,12.],device='cuda',dtype=torch.float64)
    base = r[0:1].expand(len(radii),-1,-1).clone()
    direction = torch.cross(axis, torch.tensor([1.,0.,0.],device='cuda',dtype=torch.float64),dim=0)
    if direction.norm()<.1:
        direction = torch.cross(axis, torch.tensor([0.,1.,0.],device='cuda',dtype=torch.float64),dim=0)
    direction=F.normalize(direction,dim=0)
    base[:,0]=midpoint+radii[:,None]*direction
    probe = base.detach().requires_grad_(True)
    probe_log,probe_sign=model.forward_log_psi(probe)
    slope=(torch.autograd.grad(probe_log.sum(),probe)[0][:,0]*direction).sum(-1)
    probe_e,_,_=compute_local_energy(model,system,base.detach().requires_grad_(True))
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    trajectory=torch.load(path.with_name('evaluation_trajectory.pt'),map_location='cpu',weights_only=True).double()
    mu=trajectory.mean();dev=trajectory-mu
    abs_sorted=dev.abs().flatten().sort(descending=True).values
    tail={}
    for frac in [.0001,.001,.01]:
        n=max(1,int(trajectory.numel()*frac))
        tail[str(frac)] = dict(count=n, variance_share=scalar(abs_sorted[:n].square().sum()/dev.square().sum()))
    result=dict(case=name, checkpoint_sha256=digest, epoch=saved['epoch'], no_updates=True, device=torch.cuda.get_device_name(),dtype='float64',parameters=sum(p.numel() for p in model.parameters()),
        clipping=dict(energy_mean=scalar(e.mean()),median=scalar(median),median_abs_deviation=scalar(mad),mean_abs_deviation_about_median=scalar(meanad), current_width=delta, paper_width=5*scalar(meanad),fraction_outside_current=scalar(((e-median).abs()>delta).double().mean()),fraction_outside_paper=scalar(((e-median).abs()>5*meanad).double().mean()),current_mean_shift_mHa=1000*scalar(current.mean()-e.mean()),paper_mean_shift_mHa=1000*scalar(paper.mean()-e.mean()),current_vs_raw_gradient=compare_grad(current_grad,raw_grad),paper_vs_raw_gradient=compare_grad(paper_grad,raw_grad),current_vs_paper_gradient=compare_grad(current_grad,paper_grad),raw_gradient_norm=scalar(raw_grad.norm()),current_gradient_norm=scalar(current_grad.norm()),floor_counts=floor_counts),
        parameters_physical=dict(b_en=model.jastrow.b_en.detach().cpu().tolist(),b_ee=scalar(model.jastrow.b_ee_antiparallel),envelope_alpha_up=F.softplus(model.envelope_up.decay_rates_raw).detach().cpu().flatten().tolist(),envelope_alpha_down=F.softplus(model.envelope_down.decay_rates_raw).detach().cpu().flatten().tolist()),
        symmetry=dict(inversion_abs_logdiff_quantiles=quantiles((inversion_log-log).abs()),rotation_abs_logdiff_quantiles=quantiles((rotate_log-log).abs()),swap_log_max=scalar((swap_log-log).abs().max()),sign_counts={str(s):int((sign==s).sum()) for s in [-1,0,1]},inversion_sign_changes=int((sign!=inversion_sign).sum()),rotation_sign_changes=int((sign!=rotate_sign).sum())),
        kinetic_direct_h2_max_error=scalar((k_direct-t[:16]).abs().max()),saved_walker_extremes=extremes,
        far_field_probe=dict(radii=radii.cpu().tolist(),radial_log_slope=slope.detach().cpu().tolist(),energy=probe_e.detach().cpu().tolist(),sign=probe_sign.detach().cpu().tolist()),
        frozen_trajectory=dict(shape=list(trajectory.shape),finite=bool(torch.isfinite(trajectory).all()),mean=scalar(mu),variance=scalar(trajectory.var()),extrema_quantiles=quantiles(trajectory),excess_kurtosis=scalar(dev.pow(4).mean()/dev.square().mean().square()-3),tail_variance_share=tail),
        limitations=['Gradient comparisons use one stored training ensemble of 512 walkers, not a converged population gradient.','Spatial symmetry and radial probes diagnose fixed configurations, not energy contributions.','Extreme energy trajectories contain no paired electron coordinates.','No optimizer, sampler transition, or model parameter changes were performed.'])
    (OUT/(name+'_diagnostic.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['case','parameters','clipping','symmetry','kinetic_direct_h2_max_error','far_field_probe','frozen_trajectory']}),flush=True)
    del model,saved,trajectory
    torch.cuda.empty_cache()
    return result


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    summary=json.loads((ROOT/'campaign_runs/continue_logs_20261023_2020/ferminet_results/summary.json').read_text())
    cases={f"{c['interaction']}_seed{c['seed']}":c for c in summary['cases']}
    results=[]
    paths = sorted((DATA/'ferminet_results').glob('*/final_checkpoint.pt'))
    if len(paths) != 6:
        raise ValueError('All six downloaded final checkpoints are required')
    for path in paths:
        results.append(run_case(path,cases[path.parent.name]['checkpoint_sha256']))
    (OUT/'diagnostics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
