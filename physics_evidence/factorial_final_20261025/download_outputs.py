"""Read-only Kaggle output download; atomic local writes and published SHA checks."""
import concurrent.futures,hashlib,json,pathlib,time,requests
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
root=pathlib.Path('campaign_runs/factorial_final_9b001c5').resolve()
api=KaggleApi();api.authenticate();items=[];token=None;log=None
with api.build_kaggle_client() as client:
 while True:
  req=ApiListKernelSessionOutputRequest();req.user_name='hwbwhzbsh';req.kernel_slug='ferminet-factorial-accuracy';api._set_paging(req,100,token)
  response=client.kernels.kernels_api_client.list_kernel_session_output(req)
  items.extend((x.file_name,x.url) for x in response.files or [])
  if log is None:log=response.log
  token=response.next_page_token
  if not token:break
manifest=json.loads((root/'ferminet_results/archive_manifest.json').read_text())
print('Listed',len(items),'published files; manifest',len(manifest),flush=True)
def download(item):
 name,url=item;target=root/name;expected=manifest.get(name.removeprefix('ferminet_results/'))
 if target.is_file() and expected and hashlib.sha256(target.read_bytes()).hexdigest()==expected:return name,'verified-existing'
 for attempt in range(3):
  try:
   response=requests.get(url,timeout=(15,60));response.raise_for_status();data=response.content
   if expected and hashlib.sha256(data).hexdigest()!=expected:raise ValueError('SHA mismatch')
   target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.parallel-part');tmp.write_bytes(data);tmp.replace(target)
   return name,'downloaded'
  except Exception:
   if attempt==2:raise
   time.sleep(2)
# Fetch result metadata first while preserving every raw data file.
items.sort(key=lambda x:(not x[0].endswith(('.json','.log')),x[0]))
failures=[];done=0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 futures={pool.submit(download,item):item[0] for item in items}
 for future in concurrent.futures.as_completed(futures):
  done+=1
  try:future.result()
  except Exception as exc:failures.append({'file':futures[future],'error':str(exc)});print('Download retry needed:',futures[future],type(exc).__name__,flush=True)
  if done%20==0:print('Processed',done,'/',len(items),flush=True)
if log:
 (root/'ferminet-factorial-accuracy.log').write_text(log,encoding='utf-8')
(root/'download_inventory.json').write_text(json.dumps({'kernel':'hwbwhzbsh/ferminet-factorial-accuracy','files':[x[0] for x in items],'failures':failures},indent=2)+'\n',encoding='utf-8')
print('Finished; failures:',len(failures),flush=True)
if failures:raise SystemExit(1)
