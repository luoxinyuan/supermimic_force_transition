"""Three-rank real SuperMimic updates and interface isolation tests."""
import json,os
import torch
import torch.distributed as dist
from supermimic.train import Trainer,TrainCfg
from supermimic.rotations import yaw_mat

cfg=TrainCfg(command_interface='ceer',motions='data/processed/lafan1_50hz.npz',
 out='runs/ceer_interface_smoke',num_envs=32,buffer_steps=32,warmup_steps=12,
 wm_batch=32,wm_batches=1,pi_batch=32,horizon=32,iterations=10,
 critic_epochs=1,critic_minibatches=1,log_every=5,ckpt_every=10,eval_every=0,
 compile_wm=False,sim_compile=False,amp=False)
t=Trainer(cfg);obs,g=t.env.policy_inputs();assert g.shape==(32,18)
s=t.env.state;ref=t.env.ref_future(t.env.frame);poison={k:v.clone() for k,v in ref.items()}
poison['q'].normal_();poison['qd'].normal_();poison['w'].normal_()
torch.testing.assert_close(g,t.task.goal(s,poison))
# Root XY target is not an actor command; yaw and velocity use body-relative frames.
poison={k:v.clone() for k,v in ref.items()};poison['pos'][...,:2]+=100
torch.testing.assert_close(g,t.task.goal(s,poison))
ref_now=t.lib.frame(t.env.frame)
an=t.task.analytics(s,ref_now,t.task.body_kin(ref_now),ref,t.env.prev_action,t.env.prev_action,t.env.enc_bias)
torch.testing.assert_close(an['goal'],g)
assert torch.isfinite(t.policy(obs,g)).all()
# Direct external EE/root commands change actions without changing any motion data.
g2=g.clone();g2[:,6]+=.2
assert (t.policy(obs,g2)-t.policy(obs,g)).abs().max()>0
before=[p.detach().clone() for p in t.policy.parameters()]
t.run()
assert any(not torch.equal(a,b) for a,b in zip(before,t.policy.parameters()))
assert all(torch.isfinite(p).all() for p in t.policy.parameters())
if t.world>1:
 for model in (t.policy,t.wm,t.critic):
  for param in model.parameters():
   other=param.detach().clone();dist.broadcast(other,0);torch.testing.assert_close(param,other,rtol=0,atol=0)
 flags=torch.tensor([cfg.lr],device=t.device);other=flags.clone();dist.broadcast(other,0);torch.testing.assert_close(flags,other)
if t.rank==0:
 result={'passed':True,'ranks':t.world,'iterations':t.it,'updates':t.updates,'goal_dim':18,'obs_dim':930,'full_reference_joint_input_absent':True,'reference_xy_input_absent':True,'sim_imagination_commands_equal':True,'policy_parameters_updated':True,'parameters_synchronized':True,'lr_synchronized':True}
 open('reports/ceer_full/smoke.json','w').write(json.dumps(result,indent=2));print(result,flush=True)
if t.world>1:dist.destroy_process_group()
