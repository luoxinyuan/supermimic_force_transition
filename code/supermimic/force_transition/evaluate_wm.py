"""Fresh held-out simulator trajectories; 1/4/8/16-step forced vs force-blind WM."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..motion_lib import MotionLibrary
from ..sim_env import EnvCfg
from ..task import TaskCfg,state_to_flat,flat_to_state
from ..networks import WorldModel,state_error
from .env import TransitionEnv
from .policy import ForceTeacher
from .forces import features

@torch.no_grad()
def evaluate_wm(checkpoint,out):
    torch.manual_seed(9137)
    ck=torch.load(checkpoint,map_location='cuda',weights_only=False)
    lib=MotionLibrary(ck['cfg']['motions'],'cuda')
    env=TransitionEnv(lib,EnvCfg(num_envs=32,compile=False,domain_rand=False,obs_noise=False,reset_noise=False),TaskCfg(),'cuda',delay=ck['cfg']['force_delay'])
    env.reset();ids=torch.arange(32,device='cuda');mode=ids%2
    env.reset_envs(ids,frames=lib.starts[mode]);env._finish_resets(ids,force=True)
    env.schedule[:,0]=0;env.schedule[:,1]=200;env.schedule[:,3]=10000
    # Includes both directions at both poses, and nominal trajectories.
    magnitudes=torch.tensor([0.,12.,-12.,22.,-22.,28.,-28.,0.],device='cuda').repeat_interleave(4)
    env.schedule[:,2]=magnitudes
    policy=ForceTeacher().cuda();policy.load_state_dict(ck['policy'])
    wm=WorldModel(env.task,env.task.features_fast(env.state).shape[-1],57).cuda();wm.load_state_dict(ck['wm'])
    states=[];actions=[];contexts=[];alive=[]
    valid=torch.ones(32,dtype=torch.bool,device='cuda')
    for t in range(220):
        obs,g=env.policy_inputs();a=policy(obs,g);c=env.contacts()
        states.append(state_to_flat(env.state));actions.append(a);contexts.append(torch.cat((env.xi,features(c)),-1));alive.append(valid.clone())
        _,info=env.step(a,auto_reset=False);valid&=~info['terminated']
    states=torch.stack(states);actions=torch.stack(actions);contexts=torch.stack(contexts);alive=torch.stack(alive)
    starts=torch.randint(30,180,(256,),device='cuda');en=torch.randint(0,32,(256,),device='cuda')
    valid_windows=alive[starts+16,en]
    assert int(valid_windows.sum())>=32,'Too few surviving held-out windows'
    result={'checkpoint':str(checkpoint),'seed':9137,'valid_windows':int(valid_windows.sum()),'horizons':{}}
    preds=[flat_to_state(states[starts,en]),flat_to_state(states[starts,en])]
    for h in range(1,17):
        ctx=contexts[starts+h-1,en];blind=ctx.clone();blind[:,33:]=0
        preds=[wm.step(preds[0],actions[starts+h-1,en],ctx),wm.step(preds[1],actions[starts+h-1,en],blind)]
        if h in (1,4,8,16):
            truth=flat_to_state(states[starts+h,en]);entry={}
            for name,pred in zip(('force_aware','force_blind'),preds):
                err=state_error(pred,truth)
                for group,select in [('all',valid_windows),('forced',valid_windows&(magnitudes[en]!=0)),('zero_force',valid_windows&(magnitudes[en]==0))]:
                    e=err[select]
                    if not len(e):continue
                    entry[name+'_'+group]={'normalized_mse':float((e/wm.out_norm.std).square().mean()),'root_position_rmse_m':float(e[:,:3].square().mean().sqrt()),'root_rotation_rmse_rad':float(e[:,3:6].square().mean().sqrt()),'joint_position_rmse_rad':float(e[:,12:41].square().mean().sqrt()),'joint_velocity_rmse_rad_s':float(e[:,41:].square().mean().sqrt())}
            result['horizons'][str(h)]=entry
    one=result['horizons']['1'];sixteen=result['horizons']['16']
    ratio=one['force_aware_forced']['normalized_mse']/one['force_blind_forced']['normalized_mse']
    result['one_step_forced_mse_ratio']=ratio
    result['force_conditioning_passed']=bool(ratio<.95)
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
    np.savez(p.with_suffix('.npz'),states=states.cpu().numpy(),actions=actions.cpu().numpy(),contexts=contexts.cpu().numpy(),alive=alive.cpu().numpy())
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',default='reports/exp02/wm_validation.json');a=ap.parse_args()
    print(json.dumps(evaluate_wm(a.checkpoint,a.out),indent=2))
