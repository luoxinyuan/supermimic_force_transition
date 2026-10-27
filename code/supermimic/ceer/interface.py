"""CEER public command semantics, with SuperMimic's own actor/training algorithm.

18 = root height, body-local vx/vy, yaw-relative heading xy, reserved force
conditioning (constant 5), left/right hand root-local xyz, left/right rotvec.
No reference joints, phase, clip ID or privileged state enters the actor goal.
"""
import torch
from ..task import Task
from ..rotations import heading_mat, mtv, mtm

GOAL_DIM=18
# Fixed wrist_yaw -> hand_mimic transforms read from feet_reach's G1 USD.
HAND_OFFSETS=((.115995705,.000018297129,.0000071705485),
              (.115995705,-.000018295808,.0000071705485))

class CEERTask(Task):
    goal_dim=GOAL_DIM
    def goal(self,s,ref_future):
        ref={k:v[...,0,:] if k!='rot' else v[...,0,:,:] for k,v in ref_future.items()}
        vel=mtv(s['rot'],ref['v'])[...,:2]
        heading=mtm(heading_mat(s['rot']),heading_mat(ref['rot']))[..., :2,0]
        return torch.cat((ref['pos'][...,2:3],vel,heading,
                          torch.full_like(ref['pos'][...,2:3],5.),ref['ceer_wrist']),-1)

    def install_cache(self,lib):
        import numpy as np
        from pathlib import Path
        cache=Path(str(lib.path)+'.ceer_wrist.npy')
        if not cache.exists():
            raise FileNotFoundError(f'Build CEER wrist cache first: {cache}')
        wrist=np.load(cache,mmap_mode='r')
        if wrist.shape!=(lib.num_frames,12):raise ValueError('CEER cache/frame mismatch')
        lib.ceer_wrist=torch.tensor(wrist,device=self.device)
