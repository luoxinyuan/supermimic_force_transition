"""Force-aware Teacher using SuperMimic's WM, replay, BPTT, critic and checkpointing."""
import argparse
import dataclasses
import json
import os
import time
import torch
from ..train import Trainer, TrainCfg
from ..networks import WorldModel, Policy, Critic, state_delta_c, state_error, apply_delta_c
from ..task import XI_DIM, OBS_DIM, state_to_flat, flat_to_state
from ..sim_env import EnvCfg
from .env import TransitionEnv
from .replay import ForceReplay
from .policy import ForceTeacher
from .forces import features, teacher_features, scheduled_contacts, update_memory, latch, TEACHER_DIM

@dataclasses.dataclass
class ForceTrainCfg(TrainCfg):
    motions:str='data/processed/standing_squatting.npz'
    out:str='runs/exp02_teacher_seed0'
    pretrained:str='runs/exp01_lafan40_seed0/ckpt_0036000.pt'
    bootstrap:str='data/processed/pose_bootstrap.pt'
    bootstrap_steps:int=0
    num_envs:int=1024
    buffer_steps:int=512
    wm_batch:int=512
    wm_batches:int=5
    pi_batch:int=512
    warmup_steps:int=128
    eval_every:int=2000
    ckpt_every:int=2000
    wall_hours:float=2.
    force_delay:int=1
    force_min:float=5.
    force_max:float=30.
    wm_only_updates:int=100
    sim_compile:bool=True

class ForceTrainer(Trainer):
    def __init__(self,cfg,device='cuda'):
        # Base trainer owns the original optimizer loop, counters, logging and DDP.
        super().__init__(dataclasses.replace(cfg,resume=''),device)
        self.cfg=cfg
        if not cfg.sim_compile:self.task.compile_sizes.clear()
        self.wm=WorldModel(self.task,self.feat_dim,XI_DIM+24).to(device)
        self.policy=ForceTeacher(self.obs_dim).to(device)
        self.critic_in=self.feat_dim+TEACHER_DIM+1+XI_DIM+1
        self.critic=Critic(self.critic_in).to(device)
        self.opt_wm=torch.optim.Adam(self.wm.net.parameters(),lr=cfg.lr,fused=True)
        self.opt_pi=torch.optim.Adam(list(self.policy.encoder.parameters())+list(self.policy.decoder.parameters()),lr=cfg.lr,betas=(.9,.95),fused=True)
        self.opt_v=torch.optim.Adam(self.critic.net.parameters(),lr=cfg.lr,fused=True)
        self._reward=torch.compile(self.task.transition,dynamic=False) if cfg.sim_compile else self.task.transition
        if cfg.resume:
            self.load(cfg.resume)
        else:
            self.initialize_baseline(cfg.pretrained)
            if cfg.bootstrap_steps: raise ValueError("Static behavior cloning is superseded by the frozen baseline stabilizer")
            self.save(os.path.join(cfg.out,"bootstrap.pt"))
        self._wm_loss=torch.compile(self.wm_window_loss,dynamic=False) if cfg.compile_wm else self.wm_window_loss

    def _make_env(self):
        c=self.cfg
        return TransitionEnv(self.lib,EnvCfg(num_envs=c.num_envs,compile=c.sim_compile,domain_rand=False,reset_noise=False),self.tcfg,self.device,
                             delay=c.force_delay,force_min=c.force_min,force_max=c.force_max,gamma=c.gamma)

    def _make_buffer(self):
        return ForceReplay(self.cfg.num_envs,self.cfg.buffer_steps,66+24*self.cfg.force_delay,self.device)

    def initialize_baseline(self,path):
        ck=torch.load(path,map_location=self.device,weights_only=False)
        old=ck['wm'];new=self.wm.state_dict()
        with torch.no_grad():
            for key,value in old.items():
                if key=='net.0.weight':
                    new[key].zero_();new[key][:,:value.shape[1]].copy_(value)
                elif key in ('in_norm.mean','in_norm.var'):
                    new[key][:value.shape[0]].copy_(value)
                else:new[key].copy_(value)
            self.wm.load_state_dict(new)
            self.policy.reference_encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in ck['policy'].items() if k.startswith('encoder.')})
            self.policy.goal_norm.load_state_dict({k.removeprefix('goal_norm.'):v for k,v in ck['policy'].items() if k.startswith('goal_norm.')})
            self.policy.knee_q0.copy_(self.env.q0[[3,9]].mean())
            self.policy.knee_stand.copy_(self.task.pose_q[0,[3,9]].mean())
            self.policy.knee_squat.copy_(self.task.pose_q[1,[3,9]].mean())
            self.policy.decoder.load_state_dict({k.removeprefix('decoder.'):v for k,v in ck['policy'].items() if k.startswith('decoder.')})
            self.policy.obs_norm.load_state_dict({k.removeprefix('obs_norm.'):v for k,v in ck['policy'].items() if k.startswith('obs_norm.')})
        print('Initialized force WM (zero new columns) and decoder from',path,flush=True)

    @torch.no_grad()
    def collect(self,random=False):
        env=self.env
        obs,g=env.policy_inputs()
        # Force features have physical fixed scales; avoid normalizing sparse force history.
        a=self.policy(obs,g)+self.cfg.action_noise*torch.randn(env.num_envs,29,device=self.device)
        s=state_to_flat(env.state);ctx=env.context();frame=env.frame.clone();ep=env.ep_id.clone()
        s1,info=env.step(a)
        self.buf.add(s,a,state_to_flat(s1),ctx,frame,ep)
        self.env_steps+=env.num_envs
        if 'done_ep_return' in info:
            self.ep_returns.append(info['done_ep_return'].mean().item())
            self.ep_norm_returns.append(info['done_norm_return'].mean().item())
            self.ep_terms.append(info['done_terminated'].float().mean().item())
            self.ep_lens.append(info['done_ep_len'].float().mean().item())
            self.ep_step_rewards.append((info['done_ep_return']/(.1*info['done_ep_len'])).mean().item())

    def update_wm(self):
        # Same k-step normalized autoregressive WM loss and optimizer as Trainer.
        cfg=self.cfg;stats={'wm_loss':0.,'wm_err1':0.,'wm_errk':0.}
        for bi in range(cfg.wm_batches):
            s0,A,S1,xi,mask=self.buf.sample_windows(cfg.wm_batch,cfg.k)
            if not bool(mask.any()):continue
            if bi==0 and self.updates%cfg.norm_update_every==0:
                with torch.no_grad():
                    prev=flat_to_state(s0);xs=[];ds=[]
                    for j in range(cfg.k):
                        nxt=flat_to_state(S1[:,j]);xs.append(self.wm.inputs(prev,A[:,j],xi[:,j])[mask]);ds.append(state_delta_c(prev,nxt)[mask]);prev=nxt
                    self.wm.in_norm.update(torch.cat(xs));self.wm.out_norm.update(torch.cat(ds))
            loss,e1,ek=self._wm_loss(s0,A,S1,xi,mask.float())
            self.opt_wm.zero_grad(set_to_none=True);loss.backward()
            self.sync_grads(list(self.wm.net.parameters()))
            torch.nn.utils.clip_grad_norm_(self.wm.net.parameters(),1.);self.opt_wm.step()
            for key,val in zip(stats,(loss,e1,ek)):stats[key]+=float(val.detach())/cfg.wm_batches
        return stats

    def wm_window_loss(self,s0,A,S1,xi,mask):
        s=flat_to_state(s0);errs=[]
        for j in range(self.cfg.k):
            s=self.wm.step(s,A[:,j],xi[:,j])
            errs.append((state_error(s,flat_to_state(S1[:,j]))/self.wm.out_norm.std).square().mean(-1))
        denom=mask.sum().clamp_min(1)
        return (torch.stack(errs).sum(0)*mask).sum()/denom,(errs[0].detach()*mask).sum()/denom,(errs[-1].detach()*mask).sum()/denom

    def update_policy(self):
        cfg=self.cfg
        if self.updates<cfg.wm_only_updates:return {'pi_warmup':1.}
        B,H=cfg.pi_batch,cfg.horizon
        s0,ctx,_,hist,previous=self.buf.sample_starts(B,self.tcfg.history,self.task)
        xi=ctx[:,:33];schedule=ctx[:,57:61];target=ctx[:,61];memory=ctx[:,62:65]
        queue=ctx[:,65:-1].reshape(B,cfg.force_delay,3,8);elapsed=ctx[:,-1]
        s=flat_to_state(s0);hist=hist+self.task.sample_obs_noise((B,self.tcfg.history))
        for p in self.wm.net.parameters():p.requires_grad_(False)
        survival=torch.ones(B,device=self.device);J=torch.zeros_like(survival)
        rec={'x':[],'r':[],'d':[],'m':[],'c':[]}
        valid=torch.stack([(elapsed+t<self.env.max_ep_len).float() for t in range(H+1)],1)
        for t in range(H):
            g=teacher_features(queue[:,0],memory)
            feats=self.task.features_fast(s)
            rem=(self.env.max_ep_len-elapsed-t).clamp(0,100)/100
            rec['x'].append(self.critic_input(feats,torch.cat((g,target[:,None]),-1),xi,rem).detach())
            a=self.policy(hist.flatten(1),torch.cat((g,self.task.public_context(s)),-1))
            # Future schedule is passed only to dynamics/reward, never policy inputs.
            force=scheduled_contacts(elapsed+t,schedule,self.env.torso_body)
            s1=self.wm.step(s,a,torch.cat((xi,features(force)),-1),feats=feats)
            target=latch(target,force)
            r,_,d,terms=self._reward(s,s1,target,a,previous)
            m=valid[:,t+1];J=J+cfg.gamma**t*survival*r*m
            for key,val in [('r',r*m),('d',d),('m',m),('c',survival)]:rec[key].append(val.detach())
            survival=survival*(1-d)
            obs=self.task.obs_clean(s1,a,xi[:,1:30])+self.task.sample_obs_noise((B,))
            hist=torch.cat((hist[:,1:],obs[:,None]),1)
            queue=torch.cat((queue[:,1:],force[:,None]),1);memory=update_memory(memory,queue[:,0])
            s,previous=s1,a
        g=teacher_features(queue[:,0],memory)
        xH=self.critic_input(self.task.features_fast(s),torch.cat((g,target[:,None]),-1),xi,(self.env.max_ep_len-elapsed-H).clamp(0,100)/100)
        if cfg.use_critic:J=J+cfg.gamma**H*survival*self.critic(xH)*valid[:,H]
        loss=-J.mean()/H
        self.opt_pi.zero_grad(set_to_none=True);loss.backward()
        params=[p for group in self.opt_pi.param_groups for p in group['params']]
        self.sync_grads(params);gnorm=torch.nn.utils.clip_grad_norm_(params,1.);torch.nn.utils.clip_grad_value_(params,.1);self.opt_pi.step()
        for p in self.wm.net.parameters():p.requires_grad_(True)
        self.critic.zero_grad(set_to_none=True)
        stats={'pi_J':J.mean().item(),'pi_loss':loss.item(),'pi_gnorm':gnorm.item(),'img_survival':survival.mean().item()}
        if cfg.use_critic:stats.update(self.update_critic(rec,xH.detach(),valid))
        return stats

    def run_eval(self):
        from .evaluate import evaluate_teacher
        self.policy.eval()
        result=evaluate_teacher(self.policy,self.cfg,self.device)
        self.policy.train()
        record={'it':self.it,'hours':self.train_time/3600,**result}
        with open(os.path.join(self.cfg.out,'eval.jsonl'),'a') as f:f.write(json.dumps(record)+'\n')
        print('FORCE EVAL',json.dumps(record),flush=True)


def parse_args():
    ap=argparse.ArgumentParser()
    for f in dataclasses.fields(ForceTrainCfg):
        t=f.type if not isinstance(f.type,str) else eval(f.type)
        ap.add_argument('--'+f.name.replace('_','-'),type=(lambda x:x.lower() in ('true','1','yes')) if t is bool else t,default=f.default)
    return ForceTrainCfg(**vars(ap.parse_args()))

if __name__=='__main__':
    cfg=parse_args();print(cfg,flush=True);ForceTrainer(cfg).run()
