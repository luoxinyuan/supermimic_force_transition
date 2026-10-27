"""Physical wrench semantics, independence, replay alignment and causal visibility."""
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import EnvCfg
from supermimic.task import TaskCfg,state_to_flat
from supermimic.force_transition.env import TransitionEnv
from supermimic.force_transition.forces import apply_contacts,wrench_from_contacts,features,scheduled_contacts
from supermimic.force_transition.replay import ForceReplay

torch.manual_seed(4)
lib=MotionLibrary('data/processed/standing_squatting.npz','cuda')
env=TransitionEnv(lib,EnvCfg(num_envs=8,compile=False,domain_rand=False,obs_noise=False,reset_noise=False),TaskCfg(),'cuda')
env.reset();ids=torch.arange(8,device='cuda')
env.reset_envs(ids,frames=lib.starts[torch.zeros(8,dtype=torch.long,device='cuda')]);env._finish_resets(ids,force=True)
# Lift off ground to isolate direction/magnitude response.
env.sim.data.qpos[:,2]+=1.;env.sim.forward();env.state=env.read_state()
c=torch.zeros(8,3,8,device='cuda');c[:,0,0]=env.torso_body;c[:,0,3]=.15;c[:,0,6]=torch.tensor([0,10,-10,20,0,0,0,0],device='cuda');c[:,0,7]=1
w=wrench_from_contacts(c,env.sim.data.xpos,env.sim.data.xmat,env.sim.data.xipos)
# CPU MuJoCo: point force equals COM force plus r cross F.
m=env.mj_model;d=mujoco.MjData(m);d.qpos[:]=env.sim.data.qpos[1].cpu().numpy();mujoco.mj_forward(m,d)
b=env.torso_body;F=np.array([0.,0.,10.]);P=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([0.,0.,.15])
qa=np.zeros(m.nv);qb=np.zeros(m.nv)
mujoco.mj_applyFT(m,d,F,np.zeros(3),P,b,qa)
mujoco.mj_applyFT(m,d,F,w[1,b,3:].cpu().numpy(),d.xipos[b],b,qb)
np.testing.assert_allclose(qa,qb,atol=2e-6)
# Offset must change moment, multiple slots must sum and be permutation invariant.
c2=c.clone();c2[:,0,1]=.1
w2=wrench_from_contacts(c2,env.sim.data.xpos,env.sim.data.xmat,env.sim.data.xipos)
assert (w2[1,b,3:]-w[1,b,3:]).norm()>.5
multi=c.clone();multi[:,1]=c2[:,0]
torch.testing.assert_close(wrench_from_contacts(multi,env.sim.data.xpos,env.sim.data.xmat,env.sim.data.xipos),w+w2)
torch.testing.assert_close(wrench_from_contacts(multi[:,[1,2,0]],env.sim.data.xpos,env.sim.data.xmat,env.sim.data.xipos),w+w2)
before=env.policy_inputs()[1].clone()
# Changing an upcoming force does not alter current policy inputs.
env.schedule[:,2]=30
torch.testing.assert_close(env.policy_inputs()[1],before)
s,info=env.step(torch.zeros(8,29,device='cuda'),contacts=c,auto_reset=False)
v=s['v'][:,2]
assert v[3]>v[1]>v[0]>v[2],v
for key in s:torch.testing.assert_close(s[key][0],s[key][4],atol=2e-5,rtol=2e-5)
assert env.policy_inputs()[1][1,6]!=0 and before[1,6]==0
# Replay force must be time-varying, never broadcast the first force over a window.
buf=ForceReplay(8,16,90,'cuda')
for t in range(10):
    ctx=env.context();ctx[:,39]=t;ctx[:,40]=1 # first force Fz at absolute context index 33+6
    x=state_to_flat(s);buf.add(x,torch.zeros(8,29,device='cuda'),x,ctx,torch.full((8,),t,device='cuda'),torch.zeros(8,dtype=torch.long,device='cuda'))
_,_,_,xi,mask=buf.sample_windows(32,4)
torch.testing.assert_close(xi[:,1:,39]-xi[:,:-1,39],torch.full((32,3),1/30,device='cuda'))
assert mask.all()
# Reward labels and future schedules cannot change any policy input.
obs0,g0=env.policy_inputs()
env.target=1-env.target;env.schedule[:,0]+=100
obs1,g1=env.policy_inputs()
torch.testing.assert_close(obs0,obs1);torch.testing.assert_close(g0,g1)
# Configurable delay also delays the accumulated force memory (no bypass).
env.delay=2;env.force_queue=torch.zeros(8,2,3,8,device='cuda');env.force_memory.zero_()
env.step(torch.zeros(8,29,device='cuda'),contacts=c,auto_reset=False)
assert not env.policy_inputs()[1][:,:27].any()
env.step(torch.zeros(8,29,device='cuda'),contacts=c,auto_reset=False)
assert env.policy_inputs()[1][1,6]!=0
report={'passed':True,'point_force_generalized_force_max_error':float(np.abs(qa-qb).max()),'vertical_velocities':v.cpu().tolist(),'causal_delay_checked':True,'time_varying_replay_checked':True,'multi_contact_sum_and_permutation_checked':True}
Path('reports/exp02/force_validation.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez('reports/exp02/force_response.npz',state=state_to_flat(s).cpu().numpy(),contacts=c.cpu().numpy())
print(json.dumps(report,indent=2))
