"""Compare execution against ae0bc0d at identical fixed states, without selecting energies."""
import copy, importlib.util, json, pathlib, time
import torch
from scripts.factorial_campaign import make_driver, parent_path, legacy_parent_identity
from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction

ROOT=pathlib.Path('campaign_runs/factorial_io_baseline_20261025')
torch.set_num_threads(1)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False

def old_module(name):
 spec=importlib.util.spec_from_file_location('ferminet_pytorch._baseline_'+name,ROOT/(name+'.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
old_eval=old_module('evaluation').evaluate_frozen_wavefunction
old_loss=old_module('loss').VMCLoss
plan=json.loads(pathlib.Path('configs/factorial_smoke.json').read_text())
parent_output=pathlib.Path('campaign_runs/factorial_parent_local/ferminet_results')
checks=[]
for arm in 'ABCD':
 pair=[]
 for old in (True,False):
  driver=make_driver(plan,11,arm,'cuda:0')
  driver.fork_from_checkpoint(parent_path(plan,11,parent_output),expected_sha256=plan['fork']['checkpoint_sha256']['11'],parent_model_identity=legacy_parent_identity(driver))
  driver.sampler.adapt_step_size=False
  if old:driver.loss_fn=old_loss(driver.config.clip_factor,driver.config.clip_method,max_invalid_fraction=0.)
  started=time.perf_counter();driver.run(10);torch.cuda.synchronize()
  pair.append((driver,time.perf_counter()-started,torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()))
 baseline,new=pair
 assert baseline[0].training_history==new[0].training_history
 for key,value in baseline[0].model.state_dict().items():
  torch.testing.assert_close(value,new[0].model.state_dict()[key],rtol=0,atol=0)
 torch.testing.assert_close(baseline[0].sampler.walkers,new[0].sampler.walkers,rtol=0,atol=0)
 assert torch.equal(baseline[2],new[2]) and torch.equal(baseline[3],new[3])
 checks.append(dict(arm=arm,updates=10,parameters_walkers_history_rng_exact=True,old_seconds=baseline[1],new_seconds=new[1]))
 print(json.dumps(checks[-1]),flush=True)

# The C arm's fixed, already archived checkpoint; no optimizer updates during sampling comparison.
saved=pathlib.Path('campaign_runs/factorial_local_20261025/ferminet_seed11_C/final_checkpoint.pt')
driver=make_driver(plan,11,'C','cuda:0');driver.load_checkpoint(saved)
model=copy.deepcopy(driver.model).double();system=copy.deepcopy(driver.system).to(dtype=torch.float64)
outputs=[]
for index,use_old in enumerate((True,False,False,True)):
 folder=ROOT/f'eval_optimized_{index}';folder.mkdir(exist_ok=False)
 function=old_eval if use_old else evaluate_frozen_wavefunction
 torch.cuda.synchronize();started=time.perf_counter()
 result=function(model,system,num_walkers=512,num_steps=16,burn_in_steps=100,decorrelation_steps=20,energy_batch_size=128,
  device='cuda:0',dtype=torch.float64,seed=987001,verbose=False,
  observations_path=str(folder/'observations'),trajectory_path=str(folder/'energy.pt'),records_per_shard=128)
 torch.cuda.synchronize();seconds=time.perf_counter()-started
 data=torch.load(folder/'observations/observations_0000.pt',weights_only=True)
 outputs.append((use_old,seconds,data))
 if len(outputs)>1:
  for key,value in outputs[0][2].items():
   torch.testing.assert_close(value,data[key],rtol=1e-12 if key=='log_abs' else 0,atol=2e-12 if key=='log_abs' else 0)
 print(json.dumps(dict(evaluation=index,old=use_old,seconds=seconds,physical_observations_exact=True)),flush=True)
result=dict(baseline_commit='ae0bc0dc542b932004b7aa75a0766c10af3610c3',training=checks,evaluation=[dict(old=v[0],seconds=v[1]) for v in outputs],
 physical_observations_exact=True,evaluation_speed_ratio=sum(v[1] for v in outputs if v[0])/sum(v[1] for v in outputs if not v[0]),
 log_abs_tolerance=dict(atol=2e-12,rtol=1e-12),scope='Local Windows RTX 3060 measurement. Two temporary ten-update paths per arm were discarded; no scientific checkpoint selection or formal campaign changes.')
(ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
