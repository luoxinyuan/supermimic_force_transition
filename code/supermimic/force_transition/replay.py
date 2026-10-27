"""Reuse ring buffer and observation reconstruction; sample TIME-VARYING force inputs."""
import torch
from ..replay import ReplayBuffer
from .forces import features

class ForceReplay(ReplayBuffer):
    def sample_windows(self,batch,k):
        assert self.size>=k
        env=torch.randint(0,self.n,(batch,),device=self.device)
        age=torch.randint(k-1,self.size,(batch,),device=self.device)
        idx=self._age_index(age[:,None]-torch.arange(k,device=self.device))
        e=env[:,None].expand(-1,k)
        ep=self.ep[idx,e]
        valid=(ep==ep[:,:1]).all(-1)&(ep[:,0]>=0)
        context=self.xi[idx,e]
        # First 33 are physical context. Force is the one applied to THIS transition.
        xi=torch.cat((context[...,:33],features(context[...,33:57].reshape(batch,k,3,8))),-1)
        return self.s[idx[:,0],env],self.a[idx,e],self.s1[idx,e],xi,valid
