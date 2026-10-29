"""Summarize frozen-ensemble gradient bias and exact extreme configurations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import compute_local_energy
from .diagnose_frozen_tails import RULES, off_diagonal_dot, write_json, load_model, cross_gradient_statistics


def bias_statistics(blocks):
    """Use independent replicas for estimating squared systematic differences."""
    x = blocks.double().mean(1)
    count = len(x)
    critical = {4:3.182446, 5:2.776445, 6:2.570582, 7:2.446912,
                8:2.364624}.get(count, 3.182446)

    def squared_signal(vectors):
        value = off_diagonal_dot(vectors, vectors)
        leave = torch.tensor([
            off_diagonal_dot(vectors[[j for j in range(count) if j != i]],
                             vectors[[j for j in range(count) if j != i]])
            for i in range(count)
        ], dtype=torch.float64)
        se = math.sqrt(float((count-1)/count*(leave-leave.mean()).square().sum()))
        return dict(mean=value, ci95_approx=[value-critical*se,value+critical*se],
                    jackknife_se=se)

    result = {}
    for rule_index, rule in enumerate(RULES[1:],1):
        difference = x[:,0]-x[:,rule_index]
        blocks_loo = []
        for i in range(count):
            ref = x[[j for j in range(count) if j != i],0].mean(0)
            for k,block in enumerate(blocks[i]):
                g = block[rule_index].double()
                blocks_loo.append(dict(replica=i,block=k,dot=float(ref@g),
                    cosine=float(torch.nn.functional.cosine_similarity(ref,g,dim=0))))
        result[rule] = dict(
            raw_minus_clipped_squared_signal=squared_signal(difference),
            difference_between_replica_cosines=[
                float(torch.nn.functional.cosine_similarity(difference[i],difference[j],dim=0))
                for i in range(count) for j in range(i+1,count)
            ],
            temporal_blocks_against_independent_raw_reference=blocks_loo,
            note='Time blocks share reference vectors and may be autocorrelated; their signs are descriptive, not independent Bernoulli trials.',
        )
    return result


def parameter_group_statistics(blocks, interaction):
    """Named groups retain the exact model parameter order of the saved vectors."""
    model=FermiNet(torch.tensor([[0.,0.,-.7],[0.,0.,.7]],dtype=torch.float64),
        torch.ones(2,dtype=torch.float64),1,1,num_determinants=8,
        hidden_dim_one=64,hidden_dim_two=16,num_layers=3,
        use_cusp_jastrow=True,spin_adapted=True,interaction_type=interaction).double()
    groups={name:[] for name in ['interactions','orbitals','envelopes','determinant_weights','jastrow']}
    start=0
    for name,parameter in model.named_parameters():
        group=('jastrow' if name.startswith('jastrow.') else
               'envelopes' if name.startswith('envelope_') else
               'orbitals' if name.startswith('linear_orb_') else
               'determinant_weights' if name=='det_weights' else 'interactions')
        groups[group].extend(range(start,start+parameter.numel()))
        start+=parameter.numel()
    if start!=blocks.shape[-1]:
        raise ValueError('Saved gradient vector does not match model parameter order')
    means=blocks.double().mean(1)
    result={}
    for name,indices in groups.items():
        raw=means[:,0,indices];clipped=means[:,1,indices]
        checks=cross_gradient_statistics(means[:,:,indices])
        result[name]=dict(parameters=len(indices),
            cross_raw_dot_current=off_diagonal_dot(raw,clipped),
            raw_squared_signal=off_diagonal_dot(raw,raw),
            difference_squared_signal=off_diagonal_dot(raw-clipped,raw-clipped),
            aggregate_cosine=float(torch.nn.functional.cosine_similarity(raw.mean(0),clipped.mean(0),dim=0)),
            current_cross_dot_ci95_approx=checks['rules']['current']['cross_dot']['ci95_approx'],
            raw_squared_signal_ci95_approx=checks['raw_signal_squared']['ci95_approx'],
            current_verdict=checks['rules']['current']['verdict'])
    return result


def derivative_path_probe(folder,case,checkpoints):
    """Local derivative-path ablation at identical amplitude values, not new energies."""
    identity=dict(schema_version=1,checkpoint_sha256=case['checkpoint_sha256'],
        extremes_sha256=hashlib.sha256((folder/'extreme_configurations.pt').read_bytes()).hexdigest(),
        torch_version=torch.__version__)
    cache=folder/'derivative_path_probe.json'
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved.get('identity')==identity:
            return saved
    model,system=load_model(checkpoints/folder.name/'final_checkpoint.pt',case,'cpu')
    data=torch.load(folder/'extreme_configurations.pt',weights_only=True)
    positions=data['positions'][:16]
    with torch.no_grad():
        original_log,_=model.forward_log_psi(positions)
    results={}
    for mode in ['full','envelope_coordinate_path_stopped','jastrow_coordinate_path_stopped']:
        handles=[]
        try:
            if mode.startswith('envelope'):
                for envelope in (model.envelope_up,model.envelope_down):
                    handles.append(envelope.register_forward_pre_hook(lambda module,inputs:(inputs[0].detach(),)))
            elif mode.startswith('jastrow'):
                handles.append(model.jastrow.register_forward_hook(lambda module,inputs,value:value.detach()))
            log,_=model.forward_log_psi(positions)
            torch.testing.assert_close(log.detach(),original_log,atol=0,rtol=0)
            _,kinetic,_=compute_local_energy(model,system,positions.clone().requires_grad_(True))
            results[mode]=kinetic.detach().tolist()
        finally:
            for handle in handles:
                handle.remove()
    result=dict(identity=identity,kinetic_derivative_path_probes=results,
        note='Amplitude values are identical; stopping a coordinate derivative path changes the local jet, not a globally defined trial wavefunction. These ablations are not variational energies or training experiments.')
    write_json(cache,result)
    return result


def tail_coefficient_statistics(folder,plan):
    data=[torch.load(folder/f'replica_{i}_observations.pt',weights_only=True)
          for i in range(plan['replicas'])]
    energy=torch.stack([d['energy'] for d in data])
    median=energy.median(-1,keepdim=True).values
    mad=(energy-median).abs().median(-1,keepdim=True).values
    width=torch.maximum(5*mad,torch.maximum(torch.full_like(mad,.001),.01*median.abs()))
    clipped=median+width*torch.tanh((energy-median)/width)
    coefficient=clipped-clipped.mean(-1,keepdim=True)
    raw=(energy-energy.mean(-1,keepdim=True))*plan['walkers']/(plan['walkers']-1)
    top=(energy-energy.mean()).square().flatten().topk(max(1,int(energy.numel()*.0001))).indices
    ratios=(coefficient.flatten()[top]/raw.flatten()[top]).abs()
    positions=torch.stack([d['positions'] for d in data]).reshape(-1,6)[top]
    return dict(top_count=len(top),unique_top_positions=len(torch.unique(positions,dim=0)),
        median_absolute_clipped_to_raw_coefficient_ratio=float(ratios.median()),
        min_absolute_clipped_to_raw_coefficient_ratio=float(ratios.min()),
        max_absolute_clipped_to_raw_coefficient_ratio=float(ratios.max()),
        note='Ratios compare covariance coefficients on exactly the same original 512-walker batches, before score multiplication or Adam.')


def analyze(root,probes=False):
    rows = []
    plan = json.loads((root/'plan.json').read_text())
    cases={f"{c['interaction']}_seed{c['seed']}":c for c in json.loads(Path(plan['summary']).read_text())['cases']}
    for folder in sorted(p.parent for p in root.glob('*/result.json')):
        r = json.loads((folder/'result.json').read_text())
        blocks = torch.stack([
            torch.load(folder/f'replica_{i}_gradient_blocks.pt',weights_only=True)
            for i in range(plan['replicas'])
        ])
        r['bias_analysis'] = bias_statistics(blocks)
        r['parameter_groups'] = parameter_group_statistics(blocks,folder.name.split('_seed')[0])
        r['tail_coefficients'] = tail_coefficient_statistics(folder,plan)
        extremes = torch.load(folder/'extreme_configurations.pt',weights_only=True)
        examples = []
        for i in range(min(8,len(extremes['energy']))):
            index = int(extremes['flat_index'][i])
            examples.append(dict(
                replica=index//(plan['records']*plan['walkers']),
                record=(index//plan['walkers'])%plan['records'],
                walker=index%plan['walkers'],
                positions_bohr=extremes['positions'][i].tolist(),
                **{name:float(extremes[name][i]) for name in ['energy','kinetic','potential','min_en','max_nearest_nucleus','ee','cancellation','log_abs_psi','sign']},
            ))
        r['extreme_examples'] = examples
        if probes:
            r['derivative_path_probe']=derivative_path_probe(folder,cases[folder.name],Path(plan['checkpoints']))
        rows.append(r)
        write_json(folder/'analysis.json',r)
    write_json(root/'analysis.json',dict(cases=rows,completed_cases=len(rows),plan=plan))
    for row in rows:
        g=row['cross_gradients'];top=row['geometry']['top_0_01_percent']
        print(json.dumps(dict(case=row['case'],samples=row['total_samples'],top_tail=top,
            raw_signal=g['raw_signal_squared'], current=g['rules']['current'],
            bias_squared=row['bias_analysis']['current']['raw_minus_clipped_squared_signal'],
            extreme_example=row['extreme_examples'][0],audit=row['extreme_audit'])),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--derivative-probes',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    analyze(args.root,args.derivative_probes)


if __name__=='__main__':
    main()
