import datetime,json,pathlib,time,requests
from kaggle.api.kaggle_api_extended import KaggleApi
root=pathlib.Path('campaign_runs/factorial_log_check_20261025')
root.mkdir(exist_ok=True)
original_send=requests.Session.send
def bounded_send(self,request,**kwargs):
 if kwargs.get('timeout') is None:kwargs['timeout']=(15,15)
 return original_send(self,request,**kwargs)
requests.Session.send=bounded_send
api=KaggleApi();api.authenticate()
started=time.monotonic();events=[];failure=None
stream=api.kernels_logs_stream('hwbwhzbsh/ferminet-factorial-accuracy')
try:
 with (root/'events.jsonl').open('w',encoding='utf-8') as raw, (root/'console.log').open('w',encoding='utf-8') as decoded:
  for event in stream:
   events.append(event);raw.write(json.dumps(event,ensure_ascii=False)+'\n');raw.flush()
   decoded.write(event.get('data',''));decoded.flush()
   if time.monotonic()-started>=35:break
except Exception as exc:failure=type(exc).__name__+': '+str(exc)
finally:stream.close()
status=api.kernels_status('hwbwhzbsh/ferminet-factorial-accuracy')
summary=dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=str(status.status),failure_message=status.failure_message,event_count=len(events),stream_error=failure)
(root/'snapshot.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
