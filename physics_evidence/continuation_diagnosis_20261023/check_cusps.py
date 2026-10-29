"""Check trained checkpoint coalescence limits on fixed non-nodal probes."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem,compute_local_energy
torch.set_num_threads(1)
out=Path(__file__).resolve().parent
data=ROOT/'campaign_runs/continue_diagnosis_20261023'
directions=torch.cat([torch.eye(3),-torch.eye(3)]).double().cuda()
radii=[1e-2,1e-3,1e-4,1e-5,1e-6]
results=[]
paths=sorted((data/'ferminet_results').glob('*/final_checkpoint.pt'))
if len(paths)!=6:
 raise ValueError('All six downloaded final checkpoints are required')
for p in paths:
 s=torch.load(p,map_location='cpu',weights_only=True)['model_state_dict']
 coords,charges=s['nuclear_coords'].double(),s['nuclear_charges'].double()
 m=FermiNet(coords,charges,1,1,num_determinants=8,hidden_dim_one=64,hidden_dim_two=16,num_layers=3,use_cusp_jastrow=True,spin_adapted=True,interaction_type=p.parent.name.split('_seed')[0]).double()
 m.load_state_dict(s);m=m.cuda().eval()
 system=MolecularSystem(coords.cuda(),charges.cuda(),1,1)
 case={'case':p.parent.name,'probes':{}}
 for mode in ['en0','en1','ee']:
  curves=[]
  for distance in radii:
   r=torch.zeros((6,2,3),dtype=torch.float64,device='cuda')
   if mode.startswith('en'):
    r[:,0]=m.nuclear_coords[int(mode[-1])]+distance*directions
    r[:,1]=torch.tensor([.3,.4,1.3],dtype=torch.float64,device='cuda')
   else:
    center=torch.tensor([.3,.4,.2],dtype=torch.float64,device='cuda')
    r[:,0]=center-.5*distance*directions;r[:,1]=center+.5*distance*directions
   r.requires_grad_(True)
   log,sign=m.forward_log_psi(r)
   grad=torch.autograd.grad(log.sum(),r)[0]
   derivative=(grad[:,0]*directions).sum(-1) if mode.startswith('en') else (.5*(grad[:,1]-grad[:,0])*directions).sum(-1)
   weights=sign*(log-log.max()).exp()
   slope=float((weights*derivative).sum()/weights.sum())
   e,_,_=compute_local_energy(m,system,r.detach().requires_grad_(True))
   curves.append({'r':distance,'spherical_log_slope':slope,'energy_mean':float(e.mean()),'energy_abs_max':float(e.abs().max()),'nonzero':bool((sign!=0).all())})
  case['probes'][mode]=curves
 results.append(case);print(json.dumps(case),flush=True)
 (out/'cusp_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
