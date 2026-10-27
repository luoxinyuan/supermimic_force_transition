"""Compare force-task physics at zero force against unchanged tracking simulator."""
import json
from pathlib import Path
import numpy as np
import mujoco
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import TrackingSimEnv,EnvCfg
from supermimic.task import TaskCfg
from supermimic.force_transition.env import TransitionEnv

lib=MotionLibrary('data/processed/standing_squatting.npz','cuda')
cfg=EnvCfg(num_envs=8,compile=False,domain_rand=False,obs_noise=False,reset_noise=False)
base=TrackingSimEnv(lib,cfg,TaskCfg(),'cuda');force=TransitionEnv(lib,cfg,TaskCfg(),'cuda')
control=TrackingSimEnv(lib,cfg,TaskCfg(),'cuda')
ids=torch.arange(8,device='cuda');frames=lib.starts[ids%2]
for env in (base,force,control):
    env.reset();env.reset_envs(ids,frames=frames);env._finish_resets(ids,force=True)
errors={};control_errors={}
for t in range(10):
    action=torch.zeros(8,29,device='cuda')
    a,_=base.step(action);b,_=force.step(action,contacts=torch.zeros(8,3,8,device='cuda'),auto_reset=False);c,_=control.step(action)
    for k in a:
        errors[k]=max(errors.get(k,0.),float((a[k]-b[k]).abs().max()))
        control_errors[k]=max(control_errors.get(k,0.),float((a[k]-c[k]).abs().max()))
        # GPU contact solves are not bitwise deterministic across simulator instances.
        torch.testing.assert_close(a[k],b[k],atol=2e-3 if k in ("v","w","qd") else 2e-5,rtol=2e-5)
# Geometry and joint-limit checks on designed endpoints, actual MuJoCo model.
m=force.mj_model;d=mujoco.MjData(m);qa=force.q_adr.cpu().numpy()
report=[]
for i in range(2):
    d.qpos[:]=np.r_[lib.root_pos[lib.starts[i]].cpu(),lib.root_quat[lib.starts[i]].cpu(),lib.q[lib.starts[i]].cpu()]
    mujoco.mj_forward(m,d)
    q=d.qpos[qa];ranges=m.jnt_range[1:]
    assert np.all(q>=ranges[:,0]) and np.all(q<=ranges[:,1])
    feet=force.foot_geoms.cpu().numpy();x=d.geom_xpos[feet,:2]
    # Foot collision geoms are capsules: include half-length along local z, not just radius.
    axis=d.geom_xmat[feet].reshape(-1,3,3)[:,:2,2]
    extent=np.abs(axis)*m.geom_size[feet,1,None]+m.geom_size[feet,0,None]
    com=d.subtree_com[force._find_body('pelvis'),:2]
    assert np.all(com>=(x-extent).min(0)) and np.all(com<=(x+extent).max(0))
    penetration=min([float(c.dist) for c in d.contact[:d.ncon]]+[0.])
    assert penetration>-.005
    robot_root=force._find_body("pelvis")
    self_contacts=sum(m.body_rootid[m.geom_bodyid[c.geom1]]==robot_root and m.body_rootid[m.geom_bodyid[c.geom2]]==robot_root for c in d.contact[:d.ncon])
    assert self_contacts==0
    report.append({'mode':i,'root_height':float(d.qpos[2]),'com_xy':com.tolist(),'min_contact_distance':penetration,'self_contacts':int(self_contacts),'contacts':int(d.ncon)})
record={'passed':True,'zero_force_max_abs_errors':errors,'two_unmodified_simulators_max_abs_errors':control_errors,'geometry':report}
Path('reports/exp02/zero_force_validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
