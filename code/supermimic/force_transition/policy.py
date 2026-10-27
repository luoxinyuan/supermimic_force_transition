"""Frozen baseline stabilizer with a trainable force-latent residual.

Both fixed endpoints are always present, in the same order. The nominal mixture is
computed from measured knee posture, never from a target label or force schedule.
Only the residual force encoder is optimized by BPTT; gradients pass through the
frozen original decoder. At zero force/history this is exactly the nominal policy.
"""
import torch
from torch import nn
from ..networks import Policy, mlp, run_mlp
from .forces import TEACHER_DIM

PUBLIC_DIM=2*730
GOAL_DIM=TEACHER_DIM+PUBLIC_DIM

class ForceTeacher(Policy):
    def __init__(self,obs_dim=930):
        super().__init__(obs_dim,730)
        self.reference_encoder=self.encoder
        self.encoder=mlp(TEACHER_DIM,(128,128),32,out_gain=.01)
        self.register_buffer('knee_q0',torch.tensor(.3))
        self.register_buffer('knee_stand',torch.tensor(.3))
        self.register_buffer('knee_squat',torch.tensor(1.2))
        for module in (self.reference_encoder,self.decoder):
            module.requires_grad_(False)

    def latent(self,obs,goal):
        force=goal[...,:TEACHER_DIM]
        stand=goal[...,TEACHER_DIM:TEACHER_DIM+730]
        squat=goal[...,TEACHER_DIM+730:]
        z0=torch.tanh(run_mlp(self.reference_encoder,self.goal_norm(stand)))
        z1=torch.tanh(run_mlp(self.reference_encoder,self.goal_norm(squat)))
        knee=obs[...,-93:][...,[3,9]].mean(-1)+self.knee_q0
        alpha=((knee-self.knee_stand)/(self.knee_squat-self.knee_stand)).clamp(0,1)
        alpha=(3*alpha-1).clamp(0,1)
        zbase=z0*(1-alpha[...,None])+z1*alpha[...,None]
        residual=torch.tanh(run_mlp(self.encoder,force))-torch.tanh(run_mlp(self.encoder,torch.zeros_like(force)))
        z=(zbase+residual).clamp(-1,1)
        return z

    def forward(self,obs,goal):
        return run_mlp(self.decoder,torch.cat((self.latent(obs,goal),self.obs_norm(obs)),-1))
