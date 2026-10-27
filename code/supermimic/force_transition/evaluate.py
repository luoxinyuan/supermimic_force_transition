"""Paired real-simulator tests of ONE policy; includes passive/latent ablations."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
import torch
from ..motion_lib import MotionLibrary
from ..sim_env import EnvCfg
from ..task import TaskCfg,state_to_flat
from .policy import ForceTeacher
from .env import TransitionEnv
from .forces import TEACHER_DIM

@torch.no_grad()
def evaluate_teacher(policy,cfg,device='cuda',out=None,steps=500):
    # For each starting pose: no force, force, force with blinded teacher, latent only.
    n=32
    lib=MotionLibrary(cfg.motions,device)
    env=TransitionEnv(lib,EnvCfg(num_envs=n,compile=False,domain_rand=False,obs_noise=False,reset_noise=False),TaskCfg(),device,delay=cfg.force_delay)
    env.reset();ids=torch.arange(n,device=device)
    mode=(ids//16).long();group=(ids%16)//4
    env.reset_envs(ids,frames=lib.starts[mode]);env._finish_resets(ids,force=True)
    env.schedule[:,0]=50;env.schedule[:,1]=175;env.schedule[:,2]=torch.where(mode==0,-25.,25.)
    env.schedule[group==0,2]=0;env.schedule[:,3]=10000
    failed=torch.zeros(n,dtype=torch.bool,device=device)
    hist={'height':[],'knee':[],'action':[],'force_z':[],'joint_rmse':[],'failed':[],'state':[]}
    target=1-mode
    for t in range(steps):
        obs,g=env.policy_inputs()
        g[group==2,:TEACHER_DIM]=0 # Physics still sees force; teacher gets no force/history.
        action=policy(obs,g)
        contacts=env.contacts()
        # Group 3: actual no-force physics, but same causal sensor signal as the forced trial.
        # This counterfactual measures active response of the neural policy.
        physics=contacts.clone();physics[group==3,:,4:7]=0
        s,info=env.step(action,contacts=contacts,physics_contacts=physics,auto_reset=False)
        failed|=info['terminated']
        hist['state'].append(state_to_flat(s).cpu().numpy())
        hist['height'].append(s['pos'][:,2].cpu().numpy())
        hist['knee'].append(s['q'][:,[3,9]].mean(-1).cpu().numpy())
        hist['action'].append(action.cpu().numpy())
        hist['force_z'].append(contacts[:,0,6].cpu().numpy())
        hist['joint_rmse'].append((s['q']-env.task.pose_q[target]).square().mean(-1).sqrt().cpu().numpy())
        hist['failed'].append(failed.cpu().numpy())
    h={k:np.stack(v) for k,v in hist.items()}
    result={}
    for m,name in enumerate(('stand_to_squat','squat_to_stand')):
        item={}
        for g,label in enumerate(('no_force','force','blinded','latent_only')):
            sl=slice(m*16+g*4,m*16+g*4+4)
            z=h['height'][-100:,sl].mean(0);qerr=h['joint_rmse'][-100:,sl].mean(0)
            item[label]={'final_height':float(z.mean()),'failed':int(h['failed'][-1,sl].sum()),'destination_joint_rmse':float(qerr.mean()),'final_knee':float(h['knee'][-100:,sl].mean())}
        dz=item['force']['final_height']-item['no_force']['final_height']
        knee_delta=item['force']['final_knee']-item['no_force']['final_knee']
        adiff=float(np.sqrt(np.mean((h['action'][75:225,m*16+4:m*16+8]-h['action'][75:225,m*16:m*16+4])**2)))
        sign=-1 if m==0 else 1
        item['force_minus_no_force_height']=dz;item['action_difference_rms']=adiff
        active_delta=sign*(item['force']['final_height']-item['blinded']['final_height'])
        latent_delta=sign*(item['latent_only']['final_height']-item['no_force']['final_height'])
        item['active_height_advantage_over_blinded']=active_delta
        item['latent_only_height_change']=latent_delta
        item['passed']=bool(sign*dz>.045 and (-sign)*knee_delta>.25 and adiff>.03 and
                            active_delta>.03 and latent_delta>.04 and
                            item['force']['failed']==0 and item['no_force']['failed']==0 and
                            item['latent_only']['failed']==0 and item['force']['destination_joint_rmse']<.18)
        result[name]=item
    result['passed']=all(result[k]['passed'] for k in ('stand_to_squat','squat_to_stand'))
    result['protocol']={'steps':steps,'dt':.02,'force_onset_s':1.,'force_end_s':4.5,'magnitude_N':25,'repeats_per_case':4,'paired_deterministic':True}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        np.savez(p.with_suffix('.npz'),**h)
    return result

if __name__=='__main__':
    from .train import ForceTrainCfg
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',default='reports/exp02/evaluation');a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False);cfg=ForceTrainCfg(**ck['cfg'])
    policy=ForceTeacher().cuda();policy.load_state_dict(ck['policy']);policy.eval()
    print(json.dumps(evaluate_teacher(policy,cfg,out=a.out),indent=2))
