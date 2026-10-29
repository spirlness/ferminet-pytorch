"""Post-download audit using only the fixed experiment's statistical functions."""
import hashlib,json,pathlib,sys
import torch
sys.path.insert(0,str(pathlib.Path.cwd()))
from ferminet_pytorch.vmc_statistics import energy_statistics
from ferminet_pytorch.evaluation_observations import summarize_observations
from scripts.factorial_campaign import seed_for
from scripts.factorial_pipeline import read_plan

torch.set_num_threads(1)
main=pathlib.Path(r'C:\Users\Administrator\.gemini\antigravity\scratch\ferminet_pytorch')
base=main/'campaign_runs/factorial_final_9b001c5/ferminet_results'
plan=read_plan(False);checks=[];checkpoint_rows=[];forks={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(condition,message):
 if not condition:raise ValueError(message)
for seed in plan['seeds']:
 for arm,settings in plan['arms'].items():
  folder=base/f'ferminet_seed{seed}_{arm}';c=torch.load(folder/'final_checkpoint.pt',map_location='cpu',weights_only=False)
  require(c['epoch']==10000 and len(c['history_rows'])==10000,'Incorrect updates/history')
  require(all(len(v)==10000 for v in c['history'].values()),'Incorrect history lengths')
  require(c['scheduler_state_dict']['last_epoch']==10000,'Incorrect scheduler epoch')
  require(all(float(v['step'])==10000 for v in c['optimizer_state_dict']['state'].values()),'Optimizer updates differ')
  require(c['sampler_settings']['adapt_step_size'] is False and c['sampler_settings']['total_steps']==101000,'Sampler burn/production/adaptation differs')
  require(c['walkers'].shape==(512,2,3) and c['walkers'].dtype==torch.float32,'Training walkers differ')
  require(c['config']['dtype']==torch.float32 and c['kinetic_mode']=='reverse' and not c['config']['compile_sampler'],'Training numeric/execution identity differs')
  require(c['model_identity']['envelope_type']==settings['envelope_type'] and c['model_identity']['envelope_scale']==2.0,'Envelope identity differs')
  require(c['loss_settings']['clip_method']==settings['clip_method'],'Clipping identity differs')
  require(c['stage']==2 and not c['config']['two_stage_jastrow'],'Jastrow not in required training stage')
  require(c['experiment_origin']['parent_epoch']==50000 and c['experiment_origin']['parent_checkpoint_sha256']==plan['fork']['checkpoint_sha256'][str(seed)],'Parent identity differs')
  require(c['experiment_origin']['optimizer_state_preserved'] is False,'Parent optimizer retained')
  require(len(c['optimizer_state_dict']['param_groups'])==2 and c['scheduler_state_dict']['base_lrs']==[1e-4,1e-5],'Learning rates differ')
  require(all(v.dtype==torch.float32 for v in c['model_state_dict'].values() if v.is_floating_point()),'Training parameters are not FP32')
  initial=torch.load(folder/'fork_initial_state.pt',map_location='cpu',weights_only=True)
  if seed in forks:
   require(all(torch.equal(v,forks[seed][k]) for k,v in initial.items()),'Within-seed starting walkers/RNG differ')
  else:forks[seed]=initial
  row=json.loads((folder/'result.json').read_text())
  require(row['precision_before']['passed'] and row['precision_after']['passed'] and row['evaluation']['extreme_precision_audit']['passed'],'Numerical audit failed')
  checkpoint_rows.append({'case_id':folder.name,'updates':c['epoch'],'history_rows':len(c['history_rows']),'optimizer_states':len(c['optimizer_state_dict']['state']),'sampler_total_steps':c['sampler_settings']['total_steps'],'sha256':sha(folder/'final_checkpoint.pt'),'all_checks_passed':True})
  del c
for seed,initial in forks.items():
 parent_path=main/f'campaign_runs/factorial_parent_local/ferminet_results/ferminet_seed{seed}/final_checkpoint.pt'
 require(sha(parent_path)==plan['fork']['checkpoint_sha256'][str(seed)],'Local parent SHA differs')
 parent=torch.load(parent_path,map_location='cpu',weights_only=False)
 require(torch.equal(parent['walkers'],initial['walkers']) and torch.equal(parent['rng_state'],initial['rng']),'Fork differs from verified parent walkers/CPU RNG')
 del parent
campaign=json.loads((base/'campaign.json').read_text())
require(all(sha(pathlib.Path.cwd()/name)==digest for name,digest in campaign['source_sha256'].items()),'Pinned source hashes differ')
coords=torch.tensor([[0.,0.,-0.7],[0.,0.,0.7]],dtype=torch.float32).double()
for manifest_path in sorted(base.rglob('observations/manifest.json')):
 folder=manifest_path.parent;ev=folder.parent;manifest=json.loads(manifest_path.read_text());result=json.loads((ev/'result.json').read_text())
 is_zero=ev.name.startswith('zero_seed');count=512 if is_zero else 2048
 require(manifest['complete'] and manifest['time_records']==count and manifest['num_walkers']==512,'Observation count/complete mismatch')
 require(manifest['seed']==result['seed'],'Observation seed differs')
 energy=torch.load(ev/'energy.pt',map_location='cpu',weights_only=True)
 require(energy.dtype==torch.float64 and energy.shape==(count,512) and torch.isfinite(energy).all(),'Energy shape/dtype/nonfinite mismatch')
 expected=0;max_decomposition=0.
 for shard in manifest['shards']:
  target=folder/shard['file'];require(sha(target)==shard['sha256'],'Shard SHA mismatch')
  d=torch.load(target,map_location='cpu',weights_only=True);n=shard['records']
  require(shard['start_record']==expected and torch.equal(d['time_index'],torch.arange(expected,expected+n)),'Time identity differs')
  require(torch.equal(d['walker_id'],torch.arange(512)),'Walker identity differs')
  for field in ['positions','energy','kinetic','potential','log_abs','sign']:
   shape=(n,512,2,3) if field=='positions' else (n,512)
   require(d[field].shape==shape and d[field].dtype==torch.float64 and torch.isfinite(d[field]).all(),f'Field {field} shape/dtype/finite differs')
  require(torch.equal(d['energy'],energy[expected:expected+n]),'Shard energy differs from time/walker trajectory')
  delta=float((d['energy']-d['kinetic']-d['potential']).abs().max());max_decomposition=max(max_decomposition,delta)
  require(torch.allclose(d['energy'],d['kinetic']+d['potential'],rtol=0.,atol=2e-12),'Energy decomposition differs')
  require(((d['sign']==1)|(d['sign']==-1)).all(),'Invalid wavefunction sign')
  expected+=n
 require(expected==count,'Incomplete shard records')
 stats=energy_statistics(energy)
 for key in ['energy_mean','energy_sem','energy_var','tau_int','effective_sample_size','split_rhat']:
  require(abs(stats[key]-result[key])<=1e-12,'Recomputed statistics differ: '+key)
 require(stats['diagnostics_status']==result['diagnostics_status'],'Diagnostics differ')
 tail=summarize_observations(folder,coords)
 for region,metrics in tail['regions'].items():
  for key,value in metrics.items():
   old=result['tail_summary']['regions'][region][key]
   require(abs(value-old)<1e-9,'Tail summary differs')
 require(tail['extreme_count']==result['tail_summary']['extreme_count'] and abs(tail['top_variance_share']-result['tail_summary']['top_variance_share'])<1e-12,'Extreme summary differs')
 flat=energy.flatten();center=flat.mean();variance_total=float((flat-center).square().sum());reference=result['exact_energy']
 extremes=(flat-reference).abs()>0.5;top=(flat-center).abs().topk(max(1,int(flat.numel()*0.0001))).indices
 contributions={}
 for name,selection in [('extreme_abs_reference_error_gt_0.5_ha',extremes),('top_0.01_percent_absolute_centered_energy',top)]:
  values=flat[selection]
  contributions[name]={'count':values.numel(),'signed_error_contribution_mha':float(1000*(values-reference).sum()/flat.numel()),'variance_share':float((values-center).square().sum())/variance_total if variance_total else 0.0}
 checks.append({'sample_contributions':contributions,'evaluation':str(ev.relative_to(base)),'time_records':count,'walkers':512,'shards':len(manifest['shards']),'dtype':'float64','all_finite':True,'time_walker_identity_verified':True,'energy_trajectory_exact':True,'max_energy_decomposition_residual_ha':max_decomposition,'statistics_and_tails_reproduced':True,'seed':manifest['seed']})
 print('Audited',ev.name if is_zero else ev.parent.name,flush=True)
require(len(checks)==18,'Expected 12 final and 6 zero evaluations')
require(len(set(x['seed'] for x in checks))==18,'Evaluation seeds are not distinct')
result={'source_commit':'9b001c5bbd4c143058248bbc0c6e3585412ae27c','all_checks_passed':True,'source_files_verified':len(campaign['source_sha256']),'parent_walkers_and_cpu_rng_verified':True,'checkpoints':checkpoint_rows,'fork_walkers_and_rng_equal_within_seed':True,'evaluations':checks,'final_samples':12*2048*512,'zero_samples':6*512*512,'limits':'Saved states and observations are audited; no new training or sampling was performed.'}
(main/'campaign_runs/factorial_final_raw_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('ALL RAW AUDITS PASSED',flush=True)
