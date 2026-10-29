"""Attribute raw-minus-clipped gradient differences to disjoint spatial regions.

Reuses saved energies/coordinates. No MCMC, energy recomputation or model updates.
CPU execution can run independently of the CUDA sampling process.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch

from ferminet_pytorch.loss import clip_local_energy
from .diagnose_frozen_tails import load_model, off_diagonal_dot, write_json

REGIONS = ('far_gt3', 'near_nucleus_lt005_nonfar', 'near_nucleus_005_to02_nonfar',
           'near_electron_lt01_other', 'near_electron_01_to03_other', 'remainder')


def region_masks(data, record):
    far = data['max_nearest_nucleus'][record] > 3
    near = (data['min_en'][record] < .05) & ~far
    moderate = (data['min_en'][record] < .2) & ~far & ~near
    ee = (data['ee'][record] < .1) & ~far & ~near & ~moderate
    ee_moderate = (data['ee'][record] < .3) & ~far & ~near & ~moderate & ~ee
    return (far, near, moderate, ee, ee_moderate,
            ~(far | near | moderate | ee | ee_moderate))


def region_gradient_vectors(model,positions,energy,masks):
    """Center the score so attribution is invariant to wavefunction normalization."""
    parameters=list(model.parameters())
    clipped,_,_=clip_local_energy(energy)
    size=len(energy)
    coefficient=(energy-energy.mean())*size/(size-1)-(clipped-clipped.mean())
    log,_=model.forward_log_psi(positions)
    log=log-log.mean()
    zero=log.new_zeros(sum(p.numel() for p in parameters))
    vectors=[zero for _ in masks]
    active=[i for i,mask in enumerate(masks) if bool(mask.any())]
    for index,i in enumerate(active):
        mask=masks[i]
        loss=2*(coefficient*mask*log).mean()
        grads=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=index<len(active)-1)
        vectors[i]=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,grads)]).detach()
    return torch.stack(vectors)


def attribute(root, folder, case, checkpoints):
    model, _ = load_model(checkpoints/folder.name/'final_checkpoint.pt',case,'cpu')
    parameters = list(model.parameters())
    replicas, checks = [], []
    start = time.perf_counter()
    for index in range(json.loads((root/'plan.json').read_text())['replicas']):
        data = torch.load(folder/f'replica_{index}_observations.pt',weights_only=True)
        accumulator = torch.zeros(len(REGIONS),sum(p.numel() for p in parameters),dtype=torch.float64)
        for record,positions in enumerate(data['positions']):
            accumulator+=region_gradient_vectors(model,positions,data['energy'][record],region_masks(data,record))
        accumulator/=len(data['energy'])
        saved=torch.load(folder/f'replica_{index}_gradient_blocks.pt',weights_only=True).mean(0)
        reference=saved[0]-saved[1]
        error=float((accumulator.sum(0)-reference).norm()/reference.norm())
        if error>1e-9:
            raise ValueError('Region gradients do not reconstruct the original gradient difference')
        checks.append(error);replicas.append(accumulator)
        print(json.dumps(dict(case=folder.name,replica=index,seconds=time.perf_counter()-start,reconstruction_relative_error=error)),flush=True)
    x=torch.stack(replicas)
    total=x.sum(1)
    denominator=off_diagonal_dot(total,total)
    rows=[]
    for i,name in enumerate(REGIONS):
        numerator=off_diagonal_dot(x[:,i],total)
        fraction=numerator/denominator if denominator>0 else None
        leave=[]
        if fraction is not None:
            for j in range(len(x)):
                keep=[k for k in range(len(x)) if k!=j]
                d=off_diagonal_dot(total[keep],total[keep])
                leave.append(off_diagonal_dot(x[keep,i],total[keep])/d if d>0 else None)
        interval=None
        if leave and all(v is not None for v in leave):
            values=torch.tensor(leave,dtype=torch.float64)
            se=math.sqrt(float((len(x)-1)/len(x)*(values-values.mean()).square().sum()))
            interval=[fraction-3.182446*se,fraction+3.182446*se]
        rows.append(dict(region=name,cross_dot_with_total_bias=numerator,
            signed_projection_fraction=fraction,ci95_approx=interval))
    torch.save(x,folder/'region_gradient_bias.pt')
    result=dict(schema_version=3,case=folder.name,regions_disjoint=True,regions=rows,
        score_centering='Per-batch centered parameter score; invariant to parameter-dependent wavefunction normalization.',
        reconstruction_relative_errors=checks,seconds=time.perf_counter()-start,
        limits=['Projection fractions can be negative or above one when regions cancel.',
                'Only four independent replicas; jackknife intervals are approximate.',
                'This attributes estimator differences at fixed weights, not a causal training ablation.'])
    write_json(folder/'region_bias_attribution.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--cases',nargs='*')
    args=parser.parse_args()
    torch.set_num_threads(1)
    plan=json.loads((args.root/'plan.json').read_text())
    summary=json.loads(Path(plan['summary']).read_text())
    cases={f"{c['interaction']}_seed{c['seed']}":c for c in summary['cases']}
    for folder in sorted(p.parent for p in args.root.glob('*/result.json')):
        if args.cases and folder.name not in args.cases:
            continue
        previous=folder/'region_bias_attribution.json'
        if previous.exists():
            version=json.loads(previous.read_text()).get('schema_version',1)
            if version==3:
                continue
            shutil.copyfile(previous,folder/f'region_bias_attribution_v{version}.json')
            shutil.copyfile(folder/'region_gradient_bias.pt',folder/f'region_gradient_bias_v{version}.pt')
        attribute(args.root,folder,cases[folder.name],Path(plan['checkpoints']))


if __name__=='__main__':
    main()
