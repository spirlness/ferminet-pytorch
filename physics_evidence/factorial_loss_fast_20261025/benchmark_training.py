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

