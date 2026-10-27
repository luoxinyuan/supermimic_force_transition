"""Start one detached three-GPU full-data run after preflight artifacts pass."""
import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
for file in ['reports/ceer_full/smoke.json','reports/ceer_full/interface_test.json']:
 if not json.loads(Path(file).read_text())['passed']:raise RuntimeError(file)
if not Path('runs/ceer_compiled_smoke/DONE').exists():raise RuntimeError('Compiled full-batch smoke unfinished')
data=json.loads(Path('reports/ceer_full/data_ready.json').read_text())
if not data['ready']:raise RuntimeError('Data not ready')
out=Path('runs/ceer_full_seed0');out.mkdir(parents=True,exist_ok=True)
if (out/'process.pid').exists():
 pid=int((out/'process.pid').read_text())
 if Path(f'/proc/{pid}').exists():raise RuntimeError(f'Run already active: {pid}')
if (out/'last.pt').exists():raise RuntimeError('Existing checkpoint: explicitly resume with train.sh')
source=out/'source_snapshot';source.mkdir(exist_ok=True)
for tree in ['code/supermimic','experiments/ceer_full']:
 shutil.copytree(tree,source/tree,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for key in ['motions','cache']:
 with open(data[key],'rb') as stream:data[key+'_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
(out/'data_manifest.json').write_text(json.dumps(data,indent=2))
shutil.copy('reports/ceer_full/filter.json',out/'filter.json')
shutil.copy('reports/ceer_full/filter_clips.txt',out/'motion_names.txt')
paths=list((source/'code').rglob('*.py'))+list((source/'experiments').rglob('*.py'))
hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(out/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
log=open(out/'console.log','a');p=subprocess.Popen(['bash','experiments/ceer_full/train.sh'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(out/'process.pid').write_text(str(p.pid))
record={'pid':p.pid,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_gpus':[2,3,4],'world_size':3,'num_envs_per_gpu':4096,'total_envs':12288,'iterations':320000,'wall_hours_limit':48,'data':data,'command_interface':'ceer_style_root_hands_v1','from_scratch':True,'status':'launched_pending_initial_updates'}
(out/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
