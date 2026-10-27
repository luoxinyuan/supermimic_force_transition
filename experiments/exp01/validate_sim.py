"""Assert joint ordering, state roundtrip, FK and finite batched stepping."""
import json
from pathlib import Path
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.rotations import quat_to_mat
from supermimic.sim_env import EnvCfg, TrackingSimEnv
from supermimic.task import TaskCfg

root = Path(__file__).resolve().parents[2]
torch.manual_seed(0)
lib = MotionLibrary(str(root / 'data/processed/lafan1_50hz.npz'), 'cuda')
env = TrackingSimEnv(lib, EnvCfg(num_envs=40, compile=False, reset_noise=False, obs_noise=False), TaskCfg(), 'cuda')
readme = (root / 'data/raw/lafan1_g1/README.md').read_text()
section = readme.split('G1: (30 FPS)')[1].split('H1_2:')[0]
expected = [s.strip() for s in section.splitlines() if s.strip().endswith('_joint')]
assert env.joint_names == expected, (env.joint_names, expected)
env.reset()
ids = torch.arange(40, device='cuda')
env.reset_envs(ids, frames=lib.starts)
env._finish_resets(ids, force=True)
ref = lib.frame(lib.starts)
for key in ('pos', 'q', 'v', 'w', 'qd'):
    torch.testing.assert_close(env.state[key], ref[key], atol=2e-5, rtol=2e-5)
env.sim.forward()
s = env.read_state()
P, R, V, W = env.task.body_kin(s)
names = [nm.split('/')[-1] for nm in env.robot.body_names]
perm = [names.index(nm) for nm in env.kin.body_names]
rob = env.robot.data
pairs = {'position': (P, rob.body_link_pos_w[:, perm]),
         'rotation': (R, quat_to_mat(rob.body_link_quat_w[:, perm])),
         'linear_velocity': (V, rob.body_link_lin_vel_w[:, perm]),
         'angular_velocity': (W, rob.body_link_ang_vel_w[:, perm])}
errors = {}
for name, (actual, expected_value) in pairs.items():
    errors[name] = float((actual - expected_value).abs().max())
    torch.testing.assert_close(actual, expected_value, atol=5e-5, rtol=5e-5)
for _ in range(20):
    state, info = env.step(torch.zeros(40, 29, device='cuda'))
    assert all(torch.isfinite(x).all() for x in state.values())
    assert torch.isfinite(info['reward']).all()
obs, goal = env.policy_inputs()
assert obs.shape == (40, 930) and goal.shape == (40, 730)
report = {'passed': True, 'joint_names': env.joint_names, 'fk_max_abs_errors': errors,
          'observation_shape': list(obs.shape), 'goal_shape': list(goal.shape),
          'context_shape': list(env.xi.shape), 'finite_steps': 20, 'environments': 40}
(root / 'reports/exp01_sim.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
