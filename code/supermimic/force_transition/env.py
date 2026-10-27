"""Same G1 / PD / 50 Hz dynamics, separate endpoint transition task."""
import torch
import warp as wp
import mujoco_warp as mjwarp
from ..sim_env import TrackingSimEnv
from ..rotations import mat_to_quat
from .task import TransitionTask
from .forces import apply_contacts, scheduled_contacts, latch, update_memory, teacher_features

class TransitionEnv(TrackingSimEnv):
    def __init__(self, lib, cfg, task_cfg, device, delay=1, force_min=5., force_max=30., **kwargs):
        if delay<1: raise ValueError('Teacher force delay must be at least one control step')
        super().__init__(lib,cfg,task_cfg,device,**kwargs)
        poses=torch.cat((lib.root_pos[lib.starts],lib.root_quat[lib.starts],lib.q[lib.starts]),-1)
        self.task=TransitionTask(self.task,poses)
        self.delay=delay;self.force_min=force_min;self.force_max=force_max
        n=self.num_envs
        self.target=torch.zeros(n,device=device)
        self.initial_mode=torch.zeros(n,dtype=torch.long,device=device)
        self.schedule=torch.zeros(n,4,device=device)
        self.force_queue=torch.zeros(n,delay,3,8,device=device)
        self.force_memory=torch.zeros(n,3,device=device)
        self.max_ep_len=600
        self.reward_fn=torch.compile(self.task.transition,dynamic=False) if cfg.compile else self.task.transition

    def reset_envs(self, ids, frames=None):
        if not len(ids):return
        n=len(ids);dev=self.device
        mode=torch.randint(0,2,(n,),device=dev) if frames is None else (frames>=self.lib.starts[1]).long()
        super().reset_envs(ids,frames=self.lib.starts[mode])
        self.initial_mode[ids]=mode;self.target[ids]=mode.float()
        self.force_queue[ids]=0;self.force_memory[ids]=0
        onset=torch.randint(40,101,(n,),device=dev)
        duration=torch.randint(100,176,(n,),device=dev)
        magnitude=self.force_min+torch.rand(n,device=dev)*(self.force_max-self.force_min)
        magnitude*=torch.where(mode==0,-1.,1.)
        magnitude*= (torch.rand(n,device=dev)>.25).float()
        second=onset+duration+torch.randint(75,126,(n,),device=dev)
        second=torch.where(torch.rand(n,device=dev)>.5,second,torch.full_like(second,10000))
        self.schedule[ids]=torch.stack((onset,duration,magnitude,second),-1)

    def policy_inputs(self):
        return self.obs_hist.flatten(1),torch.cat((teacher_features(self.force_queue[:,0],self.force_memory),self.task.public_context(self.state)),-1)

    def contacts(self):
        return scheduled_contacts(self.ep_len,self.schedule,self.torso_body)

    def context(self):
        # Privileged replay metadata. Never passed wholesale to policy.
        return torch.cat((self.xi,self.contacts().flatten(1),self.schedule,self.target[:,None],
                          self.force_memory,self.force_queue.flatten(1),self.ep_len[:,None].float()),-1)

    def step(self,action,contacts=None,auto_reset=True,ablate_physics=False,physics_contacts=None):
        force=self.contacts() if contacts is None else contacts
        old=self.state
        self.sim.data.ctrl[:]=(self.action_scale*action+self.q0-self.enc_bias)[:,self.act_to_joint]
        for _ in range(self.decimation):
            # Update geometry only: full forward would run an extra constraint solve.
            with wp.ScopedDevice(self.sim.wp_device):
                mjwarp.kinematics(self.sim.wp_model,self.sim.wp_data)
            applied = force if physics_contacts is None else physics_contacts
            apply_contacts(self.sim,torch.zeros_like(applied) if ablate_physics else applied)
            self.sim.step()
        s_next=self.read_state()
        target=latch(self.target,force)
        r,hard,soft,terms=self.reward_fn(old,s_next,target,action,self.prev_action)
        self.target=target
        self.force_queue=torch.cat((self.force_queue[:,1:],force[:,None]),1)
        self.force_memory=update_memory(self.force_memory,self.force_queue[:,0])
        self.ep_len+=1;self.ep_ret+=r
        done=hard|(self.ep_len>=self.max_ep_len)
        info={'reward':r,'terminated':hard,'done':done,'terms':terms,'force':force.clone(),'target':target.clone(),'step':self.ep_len.clone()}
        obs=self.task.obs_clean(s_next,action,self.enc_bias)+self._noise(self.num_envs)
        self.obs_hist=torch.cat((self.obs_hist[:,1:],obs[:,None]),1)
        self.prev_action=action.clone();self.state=s_next
        ids=done.nonzero().squeeze(-1) if auto_reset else torch.empty(0,dtype=torch.long,device=self.device)
        if len(ids):
            info.update(done_ep_return=self.ep_ret[ids].clone(),done_ep_len=self.ep_len[ids].clone(),
                        done_norm_return=self.ep_ret[ids]/(self.ep_len[ids]*.1),done_terminated=hard[ids])
            self.reset_envs(ids)
        self._finish_resets(ids)
        return s_next,info
