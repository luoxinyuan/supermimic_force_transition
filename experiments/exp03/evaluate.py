"""Held-out closed-loop sensorless evaluation; teacher labels are audit-only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import EnvCfg
from supermimic.task import TaskCfg
from supermimic.force_transition.env import TransitionEnv
from supermimic.force_transition.policy import ForceTeacher
from supermimic.force_transition.student import ForceStudent

ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='runs/exp03_student/last.pt');args=ap.parse_args()
torch.manual_seed(18271)
ck=torch.load('runs/exp02_wm_refine/last.pt',map_location='cuda',weights_only=False)
t=ForceTeacher().cuda();t.load_state_dict(ck['policy']);t.eval()
s=ForceStudent(t).cuda();s.load_state_dict(torch.load(args.checkpoint,map_location='cuda',weights_only=False)['student']);s.eval()
lib=MotionLibrary(ck['cfg']['motions'],'cuda');n=64;ids=torch.arange(n,device='cuda');mode=ids%2
schedule=torch.zeros(n,4,device='cuda');schedule[:,0]=torch.randint(30,100,(n,),device='cuda');schedule[:,1]=torch.randint(140,190,(n,),device='cuda');schedule[:,2]=(15+5*(ids//4%4))*torch.where(mode==0,-1,1);schedule[ids%4<2,2]=0;schedule[:,3]=10000
allmetrics={};hist={}
for name,p in [('teacher',t),('student',s)]:
 env=TransitionEnv(lib,EnvCfg(num_envs=n,compile=False,domain_rand=False,obs_noise=False,reset_noise=False),TaskCfg(),'cuda',delay=ck['cfg']['force_delay']);env.reset();env.reset_envs(ids,frames=lib.starts[mode]);env._finish_resets(ids,force=True);env.schedule.copy_(schedule)
 hidden=None;failed=torch.zeros(n,dtype=torch.bool,device='cuda');zs=[];ks=[];action_errors=[];latent_errors=[]
 with torch.no_grad():
  for step in range(500):
   obs=env.obs_hist.flatten(1)
   if name=='student':
    latent,hidden=p.sequence(obs[:,None],hidden);action=p.decode(obs,latent[:,0])
    # Read privileged values strictly AFTER choosing the student action, for metrics only.
    _,goal=env.policy_inputs();tz=t.latent(obs,goal);ta=t(obs,goal)
    valid=~failed
    if valid.any():
     action_errors.append((action[valid]-ta[valid]).square().mean().item());latent_errors.append((latent[:,0][valid]-tz[valid]).square().mean().item())
   else: _,goal=env.policy_inputs();action=p(obs,goal)
   st,info=env.step(action,auto_reset=False);failed|=info['terminated'];zs.append(st['pos'][:,2].cpu().numpy());ks.append(st['q'][:,[3,9]].mean(-1).cpu().numpy())
 z=np.stack(zs);k=np.stack(ks);hist[name+'_height']=z;hist[name+'_knee']=k
 metrics={'failures':int(failed.sum()),'episodes':n,'cases':{}}
 for m,label in [(0,'stand_to_squat'),(1,'squat_to_stand')]:
  base=(ids%2==m)&(ids%4<2);force=(ids%2==m)&(ids%4>=2);b=base.cpu().numpy();f=force.cpu().numpy()
  dz=float(z[-100:,f].mean()-z[-100:,b].mean());err=float(np.abs(z[-100:,b].mean(0)-(.782827 if m==0 else .706031)).mean())
  metrics['cases'][label]={'height_change_m':dz,'no_force_endpoint_drift_m':err,'force_height_m':float(z[-100:,f].mean()),'no_force_height_m':float(z[-100:,b].mean()),'failures':int(failed[base|force].sum()),'passed':bool(dz*(-1 if m==0 else 1)>.045 and err<.025 and not failed[base|force].any())}
 if action_errors:metrics.update(action_rmse=float(np.sqrt(np.mean(action_errors))),latent_rmse=float(np.sqrt(np.mean(latent_errors))))
 allmetrics[name]=metrics
hist['schedule']=schedule.cpu().numpy();np.savez('reports/exp03/heldout.npz',**hist)
allmetrics['passed']=all(v['passed'] for v in allmetrics['student']['cases'].values());allmetrics['checkpoint']=args.checkpoint;allmetrics['protocol']={'seed':18271,'forces_N':[15,20,25,30],'seconds':10,'randomized_onset_and_duration':True,'noise':False,'domain_randomization':False,'independent_physical_replicas':False}
Path('reports/exp03/heldout.json').write_text(json.dumps(allmetrics,indent=2));print(json.dumps(allmetrics,indent=2))
