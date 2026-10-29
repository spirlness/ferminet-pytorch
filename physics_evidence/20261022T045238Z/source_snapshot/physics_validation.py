"""Reproducible, time-bounded physical evidence; no precision-based stopping.

Run: python physics_validation.py --device cuda --budget-seconds 1500
Every checkpoint and full production trajectory is retained for independent audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback

import torch

try:
    from .composable_model import ComposableFermiNet
    from .model import FermiNet
    from .hamiltonian import MolecularSystem
    from .evaluation import evaluate_frozen_wavefunction
    from .train_vmc import build_h2_system, build_cosine_scheduler
    from .vmc_driver import VariationalState, HamiltonianComposite, MetropolisSampler, VMCDriver
except ImportError:
    from composable_model import ComposableFermiNet
    from model import FermiNet
    from hamiltonian import MolecularSystem
    from evaluation import evaluate_frozen_wavefunction
    from train_vmc import build_h2_system, build_cosine_scheduler
    from vmc_driver import VariationalState, HamiltonianComposite, MetropolisSampler, VMCDriver


def provenance(root):
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    def git(*args):
        return subprocess.run(['git', *args], cwd=root, capture_output=True,
                              text=True, encoding='utf-8', check=True).stdout
    return dict(commit=git('rev-parse','HEAD').strip(), diff_sha256=hashlib.sha256(
        git('diff','HEAD').encode()).hexdigest(), status=git('status','--short'),
        source_sha256=files, python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run_case(case, output, device, deadline):
    name, architecture, seed, species, train_limit, eval_limit = case
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if species == 'He':
        system = MolecularSystem(torch.zeros(1,3,device=device,dtype=torch.float64),
            torch.tensor([2.],device=device,dtype=torch.float64),1,1)
    else:
        system = build_h2_system(bond_length=3. if species == 'H2_stretched' else 1.4,
                                 device=device, dtype=torch.float64)
    cls = FermiNet if architecture == 'classic' else ComposableFermiNet
    config = dict(num_determinants=4, hidden_dim_one=32, hidden_dim_two=8,
                  num_layers=2, use_cusp_jastrow=True, spin_adapted=True,
                  envelope_type='gaussian')
    if architecture == 'psiformer':
        config['interaction_type'] = 'psiformer'
    model = cls(system.nuclear_coords, system.nuclear_charges,1,1,**config).to(device=device,dtype=torch.float64)
    state = VariationalState(model,system,num_walkers=256)
    sampler = MetropolisSampler(state,step_size=.5)
    sampler.burn_in(50)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    driver = VMCDriver(state,HamiltonianComposite(system),sampler,optimizer,
                       scheduler=build_cosine_scheduler(optimizer,600,min_ratio=.01),
                       decorrelation_steps=10)
    start = time.perf_counter()
    for epoch in range(600):
        if time.perf_counter() >= min(deadline, start+train_limit):
            break
        metrics = driver.step()
        if epoch % 100 == 0:
            print(name,'epoch',epoch+1,'energy',metrics['energy_mean'],flush=True)
    train_seconds = time.perf_counter()-start
    checkpoint = output / (name+'.pt')
    torch.save({'state_dict':model.state_dict(),'config':config,'seed':seed,
                'nuclear_coords':system.nuclear_coords.cpu(),
                'nuclear_charges':system.nuclear_charges.cpu(),
                'n_up':1,'n_down':1},checkpoint)
    history_path = output / (name+'_training.json')
    write_json(history_path,driver.history)
    result = dict(name=name,architecture=architecture,species=species,seed=seed,config=config,
        dtype='float64',walkers=256,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint=checkpoint.name,epochs=len(driver.history),train_seconds=train_seconds,
        training_stop='epochs' if len(driver.history)==600 else 'time_budget',
        training_history=history_path.name,learning_rate=1e-3,scheduler_epochs=600,
        burn_in_steps=50,selection='final_checkpoint_no_energy_selection')
    remaining = deadline-time.perf_counter()
    if remaining > 5:
        result['evaluation'] = evaluate_frozen_wavefunction(model,system,
            num_walkers=256,num_steps=512,burn_in_steps=50,decorrelation_steps=10,
            device=device,dtype=torch.float64,seed=seed+10000,verbose=False,
            max_seconds=min(eval_limit,max(1.,remaining-5)),
            trajectory_path=str(output/(name+'_trajectory.pt')))
    else:
        result['evaluation'] = {'diagnostics_status':'not_run_budget_exhausted'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--budget-seconds',type=float,default=1500.)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.budget_seconds <= 0 or args.budget_seconds > 1800:
        parser.error('budget must be in (0, 1800] seconds')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    root = Path(__file__).resolve().parent
    output = args.output or root/'physics_evidence'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(environment=provenance(root),budget_seconds=args.budget_seconds,
                  started_utc=datetime.now(timezone.utc).isoformat(),cases=[],seed_summary={})
    cases = [(f'{arch}_H2_seed{seed}',arch,seed,'H2',100.,70.)
             for arch in ('classic','composable') for seed in (11,22,33)]
    cases += [('psiformer_H2_seed11','psiformer',11,'H2',60.,40.),
              ('composable_He_seed11','composable',11,'He',60.,40.),
              ('composable_stretched_seed11','composable',11,'H2_stretched',60.,40.)]
    start = time.perf_counter()
    deadline = start+args.budget_seconds
    for case in cases:
        if time.perf_counter() >= deadline-10:
            report['cases'].append({'name':case[0],'status':'not_run_budget_exhausted'})
            continue
        try:
            item = run_case(case,output,device,deadline)
            item['status'] = 'completed'
        except Exception:
            item = {'name':case[0],'status':'failed','error':traceback.format_exc()}
        report['cases'].append(item)
        report['elapsed_seconds'] = time.perf_counter()-start
        write_json(output/'results.json',report)
        print(case[0],item.get('evaluation',item.get('error')),flush=True)
    for architecture in ('classic','composable'):
        runs = [v for v in report['cases'] if v.get('architecture')==architecture
                and v.get('species')=='H2' and 'energy_mean' in v.get('evaluation',{})]
        if len(runs) == 3:
            means = torch.tensor([v['evaluation']['energy_mean'] for v in runs],dtype=torch.float64)
            within = math.sqrt(sum(v['evaluation']['energy_sem']**2 for v in runs))/3
            between = float(means.std(unbiased=True))/math.sqrt(3)
            # Three independent training runs, df=2 Student t critical value.
            sem = max(within,between)
            half = 4.3026527299*sem
            report['seed_summary'][architecture] = dict(mean_of_checkpoint_energies=float(means.mean()),
                sem=max(within,between),ci95=[float(means.mean())-half,float(means.mean())+half],
                method='t_df2_max_within_between_SEM',
                all_diagnostics_passed=all(v['evaluation']['diagnostics_status']=='passed' for v in runs))
    report['elapsed_seconds'] = time.perf_counter()-start
    write_json(output/'results.json',report)
    print('Evidence saved:',output,flush=True)


if __name__ == '__main__':
    main()
