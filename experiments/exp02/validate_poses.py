"""Validate endpoint stability using the reproduced tracker; collect bootstrap states."""
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.evaluate import load_supermimic_policy
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import TrackingSimEnv, EnvCfg
from supermimic.task import TaskCfg, state_to_flat

lib=MotionLibrary('data/processed/standing_squatting.npz','cuda')
env=TrackingSimEnv(lib,EnvCfg(num_envs=64,compile=False,obs_noise=False,reset_noise=False,domain_rand=False,episode_length_s=20.),TaskCfg(),'cuda')
policy,_=load_supermimic_policy('runs/exp01_lafan40_seed0/ckpt_0036000.pt','cuda')
env.reset(); ids=torch.arange(64,device='cuda'); modes=ids%2
env.reset_envs(ids,frames=lib.starts[modes]);env._finish_resets(ids,force=True)
failed=torch.zeros(64,dtype=torch.bool,device='cuda')
records=[]; heights=[]; errors=[]
with torch.no_grad():
    for t in range(500):
        obs,goal=env.policy_inputs(); action=policy(obs,goal)
        if t%5==0:
            records.append({'state':state_to_flat(env.state).cpu(),'obs':obs.cpu(),'action':action.cpu(),'mode':modes.cpu()})
        s,info=env.step(action)
        failed |= info['terminated']
        heights.append(s['pos'][:,2].cpu())
        errors.append((s['q']-lib.q[lib.starts[modes]]).square().mean(-1).sqrt().cpu())
heights=torch.stack(heights);errors=torch.stack(errors)
report={'passed':not bool(failed.any()),'seconds':10,'failed':failed.cpu().tolist(),'mean_final_height':[float(heights[-100:,i::2].mean()) for i in range(2)],'max_pose_rmse_last2s':[float(errors[-100:,i::2].max()) for i in range(2)]}
Path('reports/exp02/poses_validation.json').write_text(json.dumps(report,indent=2)+'\n')
torch.save(records,'data/processed/pose_bootstrap.pt')
np.savez('reports/exp02/poses_trajectory.npz',height=heights.numpy(),joint_rmse=errors.numpy())
print(json.dumps(report));assert report['passed'], 'Static endpoints did not survive 10 s'
