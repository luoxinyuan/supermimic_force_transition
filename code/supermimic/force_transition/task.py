"""Endpoint rewards only; no intermediate reference motion or policy target label."""
import torch
from ..task import Task

class TransitionTask(Task):
    def __init__(self, original, poses):
        super().__init__(original.kin, original.q0, torch.stack((original.qmin, original.qmax),-1),
                         original.action_scale, original.cfg, original.device, original.compile_sizes)
        self.pose_q = poses[:, 7:36]
        self.pose_z = poses[:, 2]
        self.feet = self.kin.body_index(['left_ankle_roll_link','right_ankle_roll_link'])

    def public_context(self,s):
        # Both fixed endpoint descriptions, invariant to reward mode and schedule.
        parts=[]
        B=s['q'].shape[0]
        for i in range(2):
            ref={'pos':s['pos'].new_zeros(B,10,3), 'rot':torch.eye(3,device=self.device).expand(B,10,3,3),
                 'q':self.pose_q[i].expand(B,10,29),'qd':s['q'].new_zeros(B,10,29),
                 'v':s['pos'].new_zeros(B,10,3),'w':s['pos'].new_zeros(B,10,3)}
            ref['pos'][...,2]=self.pose_z[i]
            parts.append(self.goal(s,ref))
        return torch.cat(parts,-1)

    def distance(self, s, target):
        q = self.pose_q[target.long()]
        # Legs receive most weight; arm posture remains softly constrained.
        joint = (s['q'] - q).square()
        return joint[..., :12].mean(-1) + .1 * joint[..., 12:].mean(-1) + 4 * (s['pos'][...,2] - self.pose_z[target.long()]).square()

    def transition(self, s, next_s, target, action, previous):
        dist = self.distance(next_s, target)
        progress = self.distance(s, target) - dist
        P,R,V,W = self.body_kin(next_s)
        feet=P[...,self.feet,:]; foot_vel=V[...,self.feet,:]
        tilt=(1-next_s['rot'][...,2,2]).clamp_min(0)
        limits=(torch.relu(next_s['q']-self.qmax)+torch.relu(self.qmin-next_s['q'])).sum(-1)
        # Continuous foot-height/slip surrogates remain differentiable in imagination.
        terms={'target':2.5*torch.exp(-dist/.15), 'progress':15*progress,
               'upright':torch.exp(-tilt/.12), 'balance':torch.exp(-next_s['pos'][...,:2].square().sum(-1)/.04),
               'feet':.5*torch.exp(-(feet[...,2]-.035).square().mean(-1)/.0025),
               'slip':-.05*foot_vel[...,:2].square().sum(-1).mean(-1),
               'velocity':-.005*next_s['qd'].square().mean(-1)-.05*next_s['v'].square().sum(-1),
               'action_rate':-.1*(action-previous).square().sum(-1),
               'effort':-.002*action.square().mean(-1), 'limits':-2*limits}
        reward=sum(terms.values())*.02
        z=next_s['pos'][...,2]
        hard=(z<.40)|(z>1.05)|(tilt>.5)|(next_s['pos'][...,:2].norm(dim=-1)>.6)
        soft=1-(1-torch.sigmoid((.42-z)/.02))*(1-torch.sigmoid((tilt-.45)/.04))
        return reward,hard,soft,terms
