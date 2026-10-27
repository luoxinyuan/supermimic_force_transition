"""Zero-force prior equivalence, trainable force latent and differentiable WM."""
import json
from pathlib import Path
import torch
from supermimic.evaluate import load_supermimic_policy
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import EnvCfg
from supermimic.task import TaskCfg
from supermimic.networks import WorldModel
from supermimic.force_transition.env import TransitionEnv
from supermimic.force_transition.policy import ForceTeacher

ck=torch.load('runs/exp02_smoke_v2/last.pt',map_location='cuda',weights_only=False)
lib=MotionLibrary('data/processed/standing_squatting.npz','cuda')
env=TransitionEnv(lib,EnvCfg(num_envs=8,compile=False,domain_rand=False,obs_noise=False,reset_noise=False),TaskCfg(),'cuda')
env.reset();ids=torch.arange(8,device='cuda');mode=ids%2
env.reset_envs(ids,frames=lib.starts[mode]);env._finish_resets(ids,force=True)
policy=ForceTeacher().cuda();policy.load_state_dict(ck['policy'])
base,_=load_supermimic_policy('runs/exp01_lafan40_seed0/ckpt_0036000.pt','cuda')
obs,g=env.policy_inputs()
with torch.no_grad():
    action=policy(obs,g)
    selected=torch.where(mode[:,None]==0,g[:,27:757],g[:,757:])
    baseline=base(obs,selected)
    torch.testing.assert_close(action,baseline,atol=2e-5,rtol=2e-5)
# Reward labels/schedules not consulted in inference.
env.target=1-env.target;env.schedule[:,:]=100
obs2,g2=env.policy_inputs();torch.testing.assert_close(g,g2)
# Force encoder has a gradient through frozen decoder.
g=g.clone();g[:,6]=-.8;g[:,7]=1
policy(obs,g).square().mean().backward()
encoder_grad=sum(float(p.grad.abs().sum()) for p in policy.encoder.parameters() if p.grad is not None)
assert encoder_grad>0 and all(p.grad is None for p in policy.decoder.parameters())
wm=WorldModel(env.task,env.task.features_fast(env.state).shape[-1],57).cuda();wm.load_state_dict(ck['wm'])
s={k:v.clone().requires_grad_() for k,v in env.state.items()}
a=action.detach().clone().requires_grad_();xi=torch.cat((env.xi,torch.zeros(8,24,device='cuda')),-1).requires_grad_()
pred=wm.step(s,a,xi)
loss=sum(v.square().mean() for v in pred.values());loss.backward()
assert torch.isfinite(a.grad).all() and a.grad.norm()>0
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in s.values())
assert xi.grad[:,33:].abs().sum()>0
record={'passed':True,'zero_force_action_max_difference':float((action-baseline).abs().max()),'force_encoder_gradient_l1':encoder_grad,'world_model_action_gradient_norm':float(a.grad.norm()),'force_input_gradient_l1':float(xi.grad[:,33:].abs().sum()),'frozen_decoder':True}
Path('reports/exp02/model_validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
