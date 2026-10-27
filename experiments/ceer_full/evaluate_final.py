"""Evaluate exactly last.pt on every retained motion, after the training job exits."""
import argparse,json,time
from pathlib import Path
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.evaluate import evaluate,make_eval_env,load_supermimic_policy
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);a=ap.parse_args()
policy,ck=load_supermimic_policy(a.checkpoint,'cuda');cfg=ck['cfg'];assert cfg['command_interface']=='ceer'
lib=MotionLibrary(cfg['motions'],'cuda');env=make_eval_env(lib,min(1024,lib.num_clips),'cuda',command_interface='ceer')
t0=time.time();result=evaluate(lambda o,g:policy(o,g),lib,env,verbose=True)
result.update(checkpoint=a.checkpoint,iteration=ck['it'],command_interface='ceer',motion_count=lib.num_clips,evaluation_seconds=time.time()-t0,training_iterations_completed=ck['it']>=cfg['iterations'])
Path(a.checkpoint).parent.joinpath('evaluation_full_final.json').write_text(json.dumps(result)+'\n')
print(json.dumps({k:v for k,v in result.items() if not k.startswith('per_clip')}),flush=True)
