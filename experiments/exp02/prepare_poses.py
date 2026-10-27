"""Design symmetric endpoint poses with flat feet and COM centered over feet."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import TrackingSimEnv, EnvCfg
from supermimic.task import TaskCfg

root = Path.cwd()
env = TrackingSimEnv(MotionLibrary('data/processed/lafan1_50hz.npz','cuda'), EnvCfg(num_envs=1,compile=False),TaskCfg(),'cuda')
m=env.mj_model
d=mujoco.MjData(m)
q0=env.q0.cpu().numpy()
qa=env.q_adr.cpu().numpy()
foot=env.foot_geoms.cpu().numpy()
root_body=env._find_body('pelvis')
poses=[]
for knee in (0.3,1.2):
    def place(hip):
        q=q0.copy()
        for side in ('left','right'):
            for joint,val in [('hip_pitch',float(hip[0])),('knee',knee),('ankle_pitch',-float(hip[0])-knee),('hip_roll',0.),('hip_yaw',0.),('ankle_roll',0.)]:
                q[env.joint_names.index(f'{side}_{joint}_joint')]=val
        d.qpos[:]=0;d.qpos[2]=0.8;d.qpos[3]=1;d.qpos[qa]=q
        mujoco.mj_forward(m,d)
        bottom=(d.geom_xpos[foot,2]-m.geom_size[foot,0]).min()
        d.qpos[2]-=bottom
        mujoco.mj_forward(m,d)
        return d.subtree_com[root_body,0]-d.geom_xpos[foot,0].mean()
    result=least_squares(lambda x:[place(x)],[-knee/2],bounds=([-1.2],[0.1]))
    place(result.x)
    poses.append(d.qpos.copy())
    print(knee, 'height',d.qpos[2],'hip',result.x,'com_error',result.fun)
poses=np.array(poses)
length=1001
qpos=np.repeat(poses,length,axis=0).astype('float32')
qvel=np.zeros((2*length,35),dtype='float32')
Path('data/processed').mkdir(parents=True,exist_ok=True)
np.savez('data/processed/standing_squatting.npz',qpos=qpos,qvel=qvel,starts=np.array([0,length]),lengths=np.array([length,length]),names=np.array(['standing','squatting']),fps=np.array(50.))
record={'joint_names':env.joint_names,'qpos':poses.tolist(),'names':['standing','squatting'],'source':'symmetric joint design + flat feet + COM x alignment; not a prescribed transition trajectory','requires_dynamic_validation':True}
Path('experiments/exp02/poses.json').write_text(json.dumps(record,indent=2)+'\n')
