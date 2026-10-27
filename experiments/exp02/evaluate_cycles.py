"""Continuous bidirectional cycles with one frozen checkpoint; held-out pulse timing."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import EnvCfg
from supermimic.task import TaskCfg,state_to_flat
from supermimic.force_transition.env import TransitionEnv
from supermimic.force_transition.policy import ForceTeacher

p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/exp02_teacher_seed0/ckpt_0002000.pt');p.add_argument('--out',default='reports/exp02/cycles_2000');a=p.parse_args()
torch.manual_seed(20261027)
ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
policy=ForceTeacher().cuda();policy.load_state_dict(ck['policy']);policy.eval()
lib=MotionLibrary(ck['cfg']['motions'],'cuda')
env=TransitionEnv(lib,EnvCfg(num_envs=64,compile=False,domain_rand=False,obs_noise=False,reset_noise=False),TaskCfg(),'cuda',delay=ck['cfg']['force_delay'])
env.reset();ids=torch.arange(64,device='cuda');case=ids%32;mode=(case//16).long()
env.reset_envs(ids,frames=lib.starts[mode]);env._finish_resets(ids,force=True)
mags=torch.tensor([10.,15.,25.,30.],device='cuda')[(case%16)//4]
env.schedule[:,0]=50;env.schedule[:,1]=175;env.schedule[:,3]=450
env.schedule[:,2]=mags*torch.where(mode==0,-1.,1.);env.schedule[32:,2]=0
failed=torch.zeros(64,dtype=torch.bool,device='cuda');states=[];forces=[];failures=[]
with torch.no_grad():
    for t in range(900):
        obs,g=env.policy_inputs();action=policy(obs,g);c=env.contacts()
        s,info=env.step(action,auto_reset=False)
        failed|=info['terminated'];states.append(state_to_flat(s).cpu().numpy());forces.append(c[:,0,6].cpu().numpy());failures.append(failed.cpu().numpy())
states=np.stack(states);forces=np.stack(forces);failures=np.stack(failures)
rows=[]
for m in range(2):
    for j,mag in enumerate((10,15,25,30)):
        sl=slice(m*16+j*4,m*16+j*4+4);control=slice(m*16+j*4+32,m*16+j*4+36)
        z1=float(states[350:425,sl,2].mean());zc1=float(states[350:425,control,2].mean())
        z2=float(states[800:,sl,2].mean());zc2=float(states[800:,control,2].mean())
        nfail=int(failures[-1,sl].sum());sign=-1 if m==0 else 1
        row={'start':'standing' if m==0 else 'squatting','force_N':mag,'first_hold_height':z1,'first_hold_control_height':zc1,'return_hold_height':z2,'return_control_height':zc2,'failed':nfail,'passed':bool(sign*(z1-zc1)>.045 and abs(z2-zc2)<.025 and nfail==0)}
        rows.append(row)
report={'checkpoint':a.checkpoint,'rows':rows,'passed_25N':all(r['passed'] for r in rows if r['force_N']==25),'all_magnitudes_passed':all(r['passed'] for r in rows),'protocol':{'first_pulse_s':[1.,4.5],'reverse_pulse_s':[9.,12.5],'total_s':18.,'one_continuous_episode':True,'four_deterministic_replicas':True}}
out=Path(a.out);out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');np.savez(out.with_suffix('.npz'),state=states,force_z=forces,failed=failures)
print(json.dumps(report,indent=2))
