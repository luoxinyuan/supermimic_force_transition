"""Prepare all downloaded BONES G1 motions, then flat-ground filter and EE cache."""
import json,os,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
py=str(ROOT/'.venv/bin/python');report=ROOT/'reports/ceer_full';report.mkdir(exist_ok=True)
pid=int((report/'extract.pid').read_text())
while Path(f'/proc/{pid}/stat').exists():
 if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
 time.sleep(5)
files=list((ROOT/'data/raw/bones_seed/g1/csv').rglob('*.csv'))
print('extracted CSV count',len(files),flush=True)
if len(files)!=142220:raise RuntimeError('Unexpected raw CSV count; inspect extraction')
raw='data/processed/bones_seed_raw_50hz.npz';filtered='data/processed/bones_seed_filtered_50hz.npz'
def run(args,log):
 print('START',args,flush=True)
 with open(report/log,'w') as f:subprocess.run([py,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',args,flush=True)
if '--resume-raw' not in sys.argv: run(['code/scripts/prepare_motions.py','--format','bones','--fps-in','120','--input','data/raw/bones_seed/g1/csv','--output',raw,'--workers','16'],'convert.log')
import numpy as np
d=np.load(raw);names=d['names'];lengths=d['lengths']
if len(names)!=len(files):raise RuntimeError('Some input clips skipped; inspect conversion log')
for key in ['qpos','qvel']:
 values=d[key]
 for start in range(0,len(values),1000000):
  if not np.isfinite(values[start:start+1000000]).all():raise RuntimeError('Nonfinite '+key)
 del values
print('raw frames',int(lengths.sum()),flush=True)
d.close()
run(['experiments/ceer_full/filter_batched.py','--input',raw,'--output',filtered,'--report','reports/ceer_full/filter.json'],'filter.log')
run(['experiments/ceer_full/cache_interface.py','--motions',filtered],'cache_full.log')
d=np.load(filtered);n=int(d['lengths'].sum())
result={'ready':True,'raw_clips':len(files),'filtered_clips':len(d['names']),'frames':n,'hours':n/50/3600,'motions':filtered,'cache':filtered+'.ceer_wrist.npy','source_revision':'2f59b2077b9da34dd4e43618e705c7cb962c9a66'}
(report/'data_ready.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
