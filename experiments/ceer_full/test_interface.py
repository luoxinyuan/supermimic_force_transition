"""Independent MuJoCo FK and frame-invariance validation of sparse commands."""
import json
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from supermimic.ceer.interface import CEERTask,HAND_OFFSETS
from supermimic.rotations import yaw_mat

raw=np.load('data/processed/lafan1_50hz.npz');qpos=raw['qpos'];cache=np.load('data/processed/lafan1_50hz.npz.ceer_wrist.npy',mmap_mode='r')
m=Entity(get_g1_robot_cfg()).spec.compile();d=mujoco.MjData(m)
bodies=[next(i for i in range(m.nbody) if m.body(i).name.split('/')[-1]==name) for name in ['left_wrist_yaw_link','right_wrist_yaw_link']]
maxpos=0.;maxrot=0.
for frame in np.random.default_rng(7).integers(len(qpos),size=128):
 d.qpos[:]=qpos[frame];mujoco.mj_kinematics(m,d)
 root=d.xmat[1].reshape(3,3);rootpos=d.xpos[1]
 for j,b in enumerate(bodies):
  R=d.xmat[b].reshape(3,3);p=d.xpos[b]+R@np.array(HAND_OFFSETS[j]);local=root.T@(p-rootpos)
  maxpos=max(maxpos,float(np.max(np.abs(local-cache[frame,j*3:j*3+3]))))
  expected=root.T@R;actual=Rotation.from_rotvec(cache[frame,6+j*3:9+j*3]).as_matrix()
  maxrot=max(maxrot,float(np.max(np.abs(expected-actual))))
assert maxpos<2e-5 and maxrot<2e-5,(maxpos,maxrot)
# Common world translation/yaw must not change public command.
B=4;R=yaw_mat(torch.tensor([.3,.5,-.2,1.]));refR=yaw_mat(torch.tensor([.8,.1,.7,2.]))
s={'rot':R};ref={'rot':refR[:,None],'pos':torch.randn(B,1,3),'v':torch.randn(B,1,3),'ceer_wrist':torch.randn(B,1,12)}
g=CEERTask.goal(None,s,ref);Y=yaw_mat(torch.ones(B)*.7)
s2={'rot':Y@R};ref2={k:v.clone() for k,v in ref.items()};ref2['rot']=(Y@refR)[:,None];ref2['v']=(Y@ref['v'][:,0,:,None]).squeeze(-1)[:,None];ref2['pos'][...,:2]+=100
assert torch.allclose(g,CEERTask.goal(None,s2,ref2),atol=1e-6)
result={'passed':True,'mujoco_fk_frames':128,'max_hand_position_error_m':maxpos,'max_hand_rotation_matrix_error':maxrot,'world_translation_yaw_invariant':True,'command_dim':18}
open('reports/ceer_full/interface_test.json','w').write(json.dumps(result,indent=2));print(result)
